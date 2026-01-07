import argparse
import math
import os
import random
import json
from dataclasses import dataclass
from typing import Iterable, List

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        try:
            torch.mps.manual_seed(seed)  # type: ignore[attr-defined]
        except Exception:
            pass


def detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class WikiTextExample:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor


class JsonlMaskedExample:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


class WikiTextDataset(Dataset):
    """
    Mirror operators/finetune_ops/data/wikitext2_dataset:
    - concatenate lines with EOS between samples
    - fixed-length chunks; create chunk only when seq_len+1 tokens available
    - stride = seq_len when stride <=0
    - labels == input_ids; HF does the shift internally
    """

    def __init__(
        self,
        path: str,
        tokenizer: GPT2TokenizerFast,
        seq_len: int,
        stride: int = -1,
        eos_token_id: int = 50256,
        data_fraction: float = 1.0,
        insert_eos_between_lines: bool = True,
        drop_last: bool = True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.stride = seq_len if stride <= 0 else stride
        tokens = self._load_tokens(
            path, tokenizer, eos_token_id, insert_eos_between_lines
        )
        if data_fraction < 1.0:
            keep = max(seq_len + 1, int(len(tokens) * data_fraction))
            tokens = tokens[:keep]
        self.chunks = self._chunk(tokens, drop_last)

    def _load_tokens(
        self,
        path: str,
        tokenizer: GPT2TokenizerFast,
        eos_token_id: int,
        insert_eos_between_lines: bool,
    ) -> List[int]:
        toks: List[int] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if line == "":
                    if insert_eos_between_lines:
                        toks.append(eos_token_id)
                    continue
                ids = tokenizer.encode(line, add_special_tokens=False)
                toks.extend(ids)
                if insert_eos_between_lines:
                    toks.append(eos_token_id)
        return toks

    def _chunk(self, tokens: List[int], drop_last: bool) -> List[torch.Tensor]:
        chunks: List[torch.Tensor] = []
        n = len(tokens)
        need = self.seq_len + 1
        for start in range(0, n - need + 1, self.stride):
            window = tokens[start : start + self.seq_len]
            chunks.append(torch.tensor(window, dtype=torch.long))
        if not drop_last and (n >= need):
            last_start = (n - need) // self.stride * self.stride
            if last_start + self.seq_len < n and last_start + need > n:
                window = tokens[-self.seq_len :]
                chunks.append(torch.tensor(window, dtype=torch.long))
        return chunks

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> WikiTextExample:
        ids = self.chunks[idx]
        attn = torch.ones_like(ids, dtype=torch.long)
        return WikiTextExample(ids, attn)


def collate_batch(batch: List[WikiTextExample]) -> dict:
    input_ids = torch.stack([b.input_ids for b in batch], dim=0)
    attention_mask = torch.stack([b.attention_mask for b in batch], dim=0)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": input_ids.clone()}


class JsonlMaskedDataset(Dataset):
    """
    JSONL dataset {"ids": [...], "mask": [...]} with masked labels (no shift).
    Labels equal token ids where mask==1, otherwise -100, matching C++ JSONL mode.
    """

    def __init__(self, path: str, seq_len: int, pad_id: int):
        self.samples: List[JsonlMaskedExample] = []
        self.seq_len = seq_len
        self.pad_id = pad_id
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    ids = rec.get("ids", [])
                    mask = rec.get("mask", [])
                    if not isinstance(ids, list) or not isinstance(mask, list):
                        continue
                    if len(ids) != len(mask) or not ids:
                        continue
                    ids = ids[:seq_len]
                    mask = mask[:seq_len]
                    if len(ids) < seq_len:
                        pad_n = seq_len - len(ids)
                        ids = ids + [pad_id] * pad_n
                        mask = mask + [0] * pad_n
                    ids_t = torch.tensor(ids, dtype=torch.long)
                    attn = torch.ones_like(ids_t, dtype=torch.long)
                    labels = torch.full_like(ids_t, -100)
                    mask_t = torch.tensor(mask, dtype=torch.long)
                    labels = torch.where(mask_t > 0, ids_t, labels)
                    self.samples.append(JsonlMaskedExample(ids_t, attn, labels))
                except Exception:
                    continue

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> JsonlMaskedExample:
        return self.samples[idx]


def collate_jsonl(batch: List[JsonlMaskedExample]) -> dict:
    input_ids = torch.stack([b.input_ids for b in batch], dim=0)
    attention_mask = torch.stack([b.attention_mask for b in batch], dim=0)
    labels = torch.stack([b.labels for b in batch], dim=0)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def lr_schedule(step: int, total_steps: int, base_lr: float, warmup_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(max(1, warmup_steps))
    s = max(0, step - warmup_steps)
    d = max(1, total_steps - warmup_steps)
    min_lr = 0.1 * base_lr
    cosv = 0.5 * (1.0 + math.cos(math.pi * float(s) / float(d)))
    return min_lr + (base_lr - min_lr) * cosv


def count_tokens(attention_mask: torch.Tensor) -> int:
    return int(attention_mask.sum().item())


def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device, max_batches: int) -> float:
    model.eval()
    losses: List[float] = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            losses.append(out.loss.item())
    model.train()
    if not losses:
        return float("inf")
    return math.exp(sum(losses) / len(losses))


def cycle(iterable: Iterable):
    while True:
        for item in iterable:
            yield item


def main():
    parser = argparse.ArgumentParser(description="PyTorch GPT-2 full finetune (alignment)")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--pretrained_dir", type=str, required=True)
    parser.add_argument("--jsonl_train", type=str, default="")
    parser.add_argument("--jsonl_valid", type=str, default="")
    parser.add_argument("--jsonl_test", type=str, default="")
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--resume_from", type=str, default="")
    parser.add_argument("--eval_out", type=str, default="")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    parser.add_argument("--data_fraction", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--eval_interval", type=int, default=0)
    parser.add_argument("--eval_batches", type=int, default=50)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--save_every", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = detect_device()

    tok = GPT2TokenizerFast.from_pretrained(args.pretrained_dir)
    tok.padding_side = "right"
    tok.pad_token = tok.eos_token

    use_jsonl = bool(args.jsonl_train)
    if use_jsonl:
        train_dataset: Dataset = JsonlMaskedDataset(args.jsonl_train, args.seq_len, tok.pad_token_id)
        valid_dataset: Dataset = (
            JsonlMaskedDataset(args.jsonl_valid, args.seq_len, tok.pad_token_id)
            if args.jsonl_valid
            else train_dataset
        )
        collate_fn = collate_jsonl
    else:
        train_dataset = WikiTextDataset(
            os.path.join(args.data_dir, "wiki.train.raw"),
            tok,
            seq_len=args.seq_len,
            stride=-1,
            eos_token_id=tok.eos_token_id,
            data_fraction=args.data_fraction,
            insert_eos_between_lines=True,
            drop_last=True,
        )
        valid_dataset = WikiTextDataset(
            os.path.join(args.data_dir, "wiki.valid.raw"),
            tok,
            seq_len=args.seq_len,
            stride=-1,
            eos_token_id=tok.eos_token_id,
            data_fraction=args.data_fraction,
            insert_eos_between_lines=True,
            drop_last=False,
        )
        collate_fn = collate_batch

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    eval_loader = DataLoader(
        valid_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
    )

    total_train_seqs = len(train_dataset)
    steps_per_epoch = (total_train_seqs + args.batch_size * args.grad_accum_steps - 1) // (
        args.batch_size * args.grad_accum_steps
    )
    if args.epochs > 0:
        args.steps = steps_per_epoch * args.epochs
    elif args.steps <= 0:
        args.steps = steps_per_epoch

    model = GPT2LMHeadModel.from_pretrained(args.pretrained_dir)
    # Align with C++ implementation: disable dropout during finetune
    model.config.attn_pdrop = 0.0
    model.config.embd_pdrop = 0.0
    model.config.resid_pdrop = 0.0
    model.config.summary_first_dropout = 0.0
    if args.resume_from:
        sd = torch.load(args.resume_from, map_location="cpu")
        model.load_state_dict(sd)

    model.to(device)
    model.train()

    optim = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("========== PyTorch GPT-2 Full Finetune (alignment) ==========")
    print(f"device={device}, steps={args.steps}, steps_per_epoch={steps_per_epoch}, "
          f"bs={args.batch_size}, accum={args.grad_accum_steps}, seq_len={args.seq_len}")

    data_iter = cycle(train_loader)
    total_tokens = 0

    for step in range(args.steps):
        accum_loss = 0.0  # sum of per-micro losses (unscaled)
        accum_tokens = 0

        for acc in range(args.grad_accum_steps):
            batch = next(data_iter)
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            loss = loss / float(args.grad_accum_steps)
            loss.backward()
            accum_loss += loss.item() * args.grad_accum_steps  # restore micro loss for logging
            accum_tokens += count_tokens(batch["attention_mask"])

        lr_cur = lr_schedule(step, args.steps, args.lr, args.warmup_steps)
        for g in optim.param_groups:
            g["lr"] = lr_cur

        if args.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)

        optim.step()
        optim.zero_grad(set_to_none=True)

        total_tokens += accum_tokens

        if (step + 1) % max(1, args.log_interval) == 0:
            avg_loss = accum_loss / float(args.grad_accum_steps)
            ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
            print(f"[Train] step {step+1}/{args.steps} | lr {lr_cur:.6f} | loss {avg_loss:.4f} | ppl {ppl:.2f} | tokens {accum_tokens}")

        if args.eval_interval > 0 and (step + 1) % args.eval_interval == 0:
            val_ppl = evaluate(model, eval_loader, device, args.eval_batches)
            print(f"[Eval] step {step+1} | valid_ppl {val_ppl:.2f} | total_tokens {total_tokens}")
            if args.eval_out:
                with open(args.eval_out, "a", encoding="utf-8") as f:
                    f.write(f'{{"step":{step+1},"valid_ppl":{val_ppl},"total_tokens":{total_tokens}}}\n')

        if args.save_every > 0 and (step + 1) % args.save_every == 0 and args.output_path:
            out_path = args.output_path
            if out_path.endswith(".pt"):
                torch.save(model.state_dict(), out_path)
            else:
                os.makedirs(out_path, exist_ok=True)
                model.save_pretrained(out_path)
            print(f"[Checkpoint] saved to {out_path}")

    if args.output_path:
        out_path = args.output_path
        if out_path.endswith(".pt"):
            torch.save(model.state_dict(), out_path)
        else:
            os.makedirs(out_path, exist_ok=True)
            model.save_pretrained(out_path)
        print(f"[Save] final weights -> {out_path}")


if __name__ == "__main__":
    main()
