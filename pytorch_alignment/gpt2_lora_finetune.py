import argparse
import math
import os
import random
import json
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Optional

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

try:
    from peft import LoraConfig, get_peft_model, PeftModel
except ImportError as e:  # pragma: no cover - import-time guard
    raise SystemExit("Please install peft to run this script: pip install peft") from e


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    Minimal dataset that mirrors operators/finetune_ops/data/wikitext2_dataset.{h,cpp}:
    - concatenates lines with EOS between samples
    - builds fixed-length chunks (drop_last=true for train, false for eval)
    - labels are identical to input_ids; loss does the shift (logits[:-1] vs labels[1:])
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
        tokens: List[int] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if line == "":
                    if insert_eos_between_lines:
                        tokens.append(eos_token_id)
                    continue
                ids = tokenizer.encode(
                    line, add_special_tokens=False
                )
                tokens.extend(ids)
                if insert_eos_between_lines:
                    tokens.append(eos_token_id)
        return tokens

    def _chunk(self, tokens: List[int], drop_last: bool) -> List[torch.Tensor]:
        # Align with C++ version exactly:
        # C++ uses: for s in range(0, N - (S+1) + 1, stride) where need = S+1
        # This means: for s in range(0, N - S, stride) with condition s + S + 1 <= N
        # HuggingFace does the shift internally (logits[:-1] vs labels[1:])
        # So we need seq_len tokens per chunk, but only create chunks where seq_len+1 tokens are available
        chunks: List[torch.Tensor] = []
        n = len(tokens)
        need = self.seq_len + 1  # Align with C++: need seq_len+1 tokens available
        for start in range(0, n - need + 1, self.stride):
            # Take seq_len tokens for input (HuggingFace handles the shift internally)
            window = tokens[start : start + self.seq_len]
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
    JSONL dataset with fields {"ids": [...], "mask": [...]} (masked causal LM).
    Matches C++ JSONL loader: labels are the token itself when mask=1, else -100.
    No shift is applied here; HF model will apply the causal shift internally.
    """

    def __init__(self, path: str, seq_len: int, pad_id: int):
        self.seq_len = seq_len
        self.pad_id = pad_id
        self.samples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    ids = rec.get("ids", [])
                    mask = rec.get("mask", [])
                    if not isinstance(ids, list) or not isinstance(mask, list):
                        continue
                    if len(ids) != len(mask):
                        continue
                    if len(ids) == 0:
                        continue
                    ids = ids[:seq_len]
                    mask = mask[:seq_len]
                    # pad if shorter than seq_len
                    if len(ids) < seq_len:
                        pad_n = seq_len - len(ids)
                        ids = ids + [pad_id] * pad_n
                        mask = mask + [0] * pad_n
                    ids_t = torch.tensor(ids, dtype=torch.long)
                    attn = torch.ones_like(ids_t, dtype=torch.long)
                    labels = torch.full_like(ids_t, fill_value=-100)
                    mask_t = torch.tensor(mask, dtype=torch.long)
                    labels = torch.where(mask_t > 0, ids_t, labels)
                    self.samples.append((ids_t, attn, labels))
                except Exception:
                    continue

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> JsonlMaskedExample:
        ids, attn, labels = self.samples[idx]
        return JsonlMaskedExample(ids, attn, labels)


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
    parser = argparse.ArgumentParser(description="PyTorch GPT-2 LoRA finetune (alignment build)")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--pretrained_dir", type=str, required=True)
    parser.add_argument("--jsonl_train", type=str, default="")
    parser.add_argument("--jsonl_valid", type=str, default="")
    parser.add_argument("--jsonl_test", type=str, default="")
    parser.add_argument("--lora_out", type=str, default="")
    parser.add_argument("--resume_from", type=str, default="")
    parser.add_argument("--eval_out", type=str, default="")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--data_fraction", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--eval_interval", type=int, default=0)
    parser.add_argument("--eval_batches", type=int, default=50)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--save_every", type=int, default=0)
    parser.add_argument("--ema_beta", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            data_fraction=1.0,
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

    model = GPT2LMHeadModel.from_pretrained(args.pretrained_dir)
    lora_cfg = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        target_modules=["c_attn", "c_proj"],  # fused QKV + proj
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    if args.resume_from:
        model = PeftModel.from_pretrained(model, args.resume_from, is_trainable=True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = math.ceil(len(train_loader) / max(1, args.grad_accum_steps))
    total_steps = args.steps if args.steps > 0 else steps_per_epoch * max(1, args.epochs)
    warmup_steps = args.warmup_steps

    print("\n========== PyTorch GPT-2 LoRA Finetune (alignment) ==========")
    print(f"Train sequences: {len(train_dataset)}, Valid sequences: {len(valid_dataset)}")
    print(f"Total steps: {total_steps}, steps/epoch: {steps_per_epoch}, grad_accum: {args.grad_accum_steps}")
    print(f"LoRA rank/alpha/dropout: {args.rank}/{args.alpha}/{args.lora_dropout}")

    model.to(device)
    model.train()

    ema_loss = None
    token_counter = 0
    train_iter = cycle(train_loader)

    for step in range(total_steps):
        accum_loss = 0.0
        accum_tokens = 0
        optimizer.zero_grad()
        for _ in range(max(1, args.grad_accum_steps)):
            batch = next(train_iter)
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            (loss / max(1, args.grad_accum_steps)).backward()
            accum_loss += loss.item()
            accum_tokens += count_tokens(batch["attention_mask"])

        if args.clip_grad_norm and args.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(trainable, args.clip_grad_norm)

        cur_lr = lr_schedule(step, total_steps, args.lr, warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = cur_lr
        optimizer.step()

        avg_loss = accum_loss / float(max(1, args.grad_accum_steps))
        token_counter += accum_tokens
        if ema_loss is None:
            ema_loss = avg_loss
        else:
            beta = max(0.0, min(0.9999, args.ema_beta))
            ema_loss = beta * ema_loss + (1.0 - beta) * avg_loss

        if (step + 1) % max(1, args.log_interval) == 0:
            ppl = math.exp(avg_loss)
            print(
                f"[Train] step {step + 1}/{total_steps} "
                f"lr {cur_lr:.6f} loss {avg_loss:.4f} ppl {ppl:.2f} "
                f"ema_loss {ema_loss:.4f} tokens {accum_tokens}"
            )

        if args.eval_interval > 0 and (step + 1) % args.eval_interval == 0:
            valid_ppl = evaluate(model, eval_loader, device, args.eval_batches)
            print(
                f"[Eval] step {step + 1}/{total_steps} valid_ppl {valid_ppl:.2f} "
                f"ema_loss {ema_loss:.4f} total_tokens {token_counter}"
            )
            if args.eval_out:
                rec = {
                    "step": step + 1,
                    "valid_ppl": valid_ppl,
                    "ema_loss": ema_loss,
                    "total_tokens": token_counter,
                }
                with open(args.eval_out, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")

        if args.save_every > 0 and args.lora_out and (step + 1) % args.save_every == 0:
            ckpt_dir = f"{args.lora_out}_step{step + 1}"
            os.makedirs(ckpt_dir, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            tok.save_pretrained(ckpt_dir)
            print(f"[Checkpoint] saved {ckpt_dir}")

    if args.lora_out:
        os.makedirs(args.lora_out, exist_ok=True)
        model.save_pretrained(args.lora_out)
        tok.save_pretrained(args.lora_out)
        print(f"[Save] LoRA adapter written to {args.lora_out}")

    print("\n[DONE] Training finished")
    print(f"Total steps: {total_steps}, total tokens: {token_counter}, final EMA loss: {ema_loss:.4f}")


if __name__ == "__main__":
    main()
