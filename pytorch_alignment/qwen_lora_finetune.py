import argparse
import json
import math
import os
import random
from typing import Iterable, List

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import LoraConfig, get_peft_model, PeftModel
except ImportError as e:  # pragma: no cover - import-time guard
    raise SystemExit("Please install peft to run this script: pip install peft") from e


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class WikiTextDataset(Dataset):
    """
    Mirrors operators/finetune_ops/data/wikitext2_dataset for alignment:
    - concat lines with EOS between samples
    - fixed-length chunks; drop_last for train, keep tail for eval
    - labels identical to input_ids; HF handles shift internally
    """

    def __init__(
        self,
        path: str,
        tokenizer,
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
        tokenizer,
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
        if not drop_last and n >= need:
            last_start = (n - need) // self.stride * self.stride
            if last_start + self.seq_len < n and last_start + need > n:
                window = tokens[-self.seq_len :]
                chunks.append(torch.tensor(window, dtype=torch.long))
        return chunks

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int):
        ids = self.chunks[idx]
        attn = torch.ones_like(ids, dtype=torch.long)
        return {"input_ids": ids, "attention_mask": attn, "labels": ids.clone()}


class JsonlMaskedDataset(Dataset):
    """
    JSONL dataset {"ids": [...], "mask": [...]} (masked causal LM, no shift here).
    Matches the C++ JSONL mode used for MMLU finetuning.
    """

    def __init__(self, path: str, seq_len: int, pad_id: int):
        self.samples = []
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
                    self.samples.append({"input_ids": ids_t, "attention_mask": attn, "labels": labels})
                except Exception:
                    continue

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def collate_batch(batch: List[dict]) -> dict:
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}


def cycle(loader: Iterable):
    while True:
        for item in loader:
            yield item


def make_scheduler(step: int, total_steps: int, warmup_steps: int, base_lr: float, mode: str) -> float:
    step_1indexed = step + 1
    if warmup_steps > 0 and step_1indexed <= warmup_steps:
        return base_lr * float(step_1indexed) / float(max(1, warmup_steps))
    remain = max(1, total_steps - warmup_steps)
    progress = float(step_1indexed - warmup_steps) / float(remain)
    progress = min(max(progress, 0.0), 1.0)
    if mode == "cosine":
        return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (1.0 - progress)


def evaluate(model, dataloader: DataLoader, device: torch.device, max_batches: int) -> float:
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


def main():
    parser = argparse.ArgumentParser(description="PyTorch Qwen2.5-0.5B LoRA finetune (alignment)")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data/wikitext2/wikitext-2-raw")
    parser.add_argument("--jsonl_train", type=str, default="")
    parser.add_argument("--jsonl_valid", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="./qwen_lora_pt")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--lr_scheduler", type=str, default="cosine", choices=["cosine", "linear"])
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--data_fraction", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--target_mode", type=str, default="qv", choices=["qv", "full"])
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=0)
    parser.add_argument("--eval_batches", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from", type=str, default="")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = AutoTokenizer.from_pretrained(args.model_dir, padding_side="right")
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    use_jsonl = bool(args.jsonl_train)
    if use_jsonl:
        train_dataset: Dataset = JsonlMaskedDataset(args.jsonl_train, args.seq_len, tok.pad_token_id)
        eval_dataset: Dataset = (
            JsonlMaskedDataset(args.jsonl_valid, args.seq_len, tok.pad_token_id)
            if args.jsonl_valid
            else train_dataset
        )
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
        eval_dataset = WikiTextDataset(
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
        batch_size=max(1, args.batch),
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=max(1, args.batch),
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
    )

    model = AutoModelForCausalLM.from_pretrained(args.model_dir)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"] if args.target_mode == "full" else ["q_proj", "v_proj"]
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    if args.resume_from:
        model = PeftModel.from_pretrained(model, args.resume_from, is_trainable=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)

    steps_per_epoch = math.ceil(len(train_loader) / max(1, args.grad_accum))
    total_steps = args.steps if args.steps > 0 else steps_per_epoch * max(1, args.epochs)
    warmup_steps = args.warmup_steps

    print("\n========== PyTorch Qwen2.5-0.5B LoRA Finetune (alignment) ==========")
    print(f"Train sequences: {len(train_dataset)}, Eval sequences: {len(eval_dataset)}")
    print(f"Total steps: {total_steps}, steps/epoch: {steps_per_epoch}, grad_accum: {args.grad_accum}")
    print(f"LoRA rank/alpha/dropout: {args.lora_r}/{args.lora_alpha}/{args.lora_dropout}")
    print(f"Targets: {','.join(target_modules)}")

    model.to(device)
    model.train()

    ema_loss = None
    token_counter = 0
    train_iter = cycle(train_loader)

    for step in range(total_steps):
        accum_loss = 0.0
        accum_tokens = 0
        optimizer.zero_grad()
        for _ in range(max(1, args.grad_accum)):
            batch = next(train_iter)
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            (loss / max(1, args.grad_accum)).backward()
            accum_loss += loss.item()
            accum_tokens += int(batch["attention_mask"].sum().item())

        if args.max_grad_norm and args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)

        lr_cur = make_scheduler(step, total_steps, warmup_steps, args.learning_rate, args.lr_scheduler)
        for g in optimizer.param_groups:
            g["lr"] = lr_cur
        optimizer.step()

        avg_loss = accum_loss / float(max(1, args.grad_accum))
        token_counter += accum_tokens
        if ema_loss is None:
            ema_loss = avg_loss
        else:
            beta = 0.9
            ema_loss = beta * ema_loss + (1.0 - beta) * avg_loss

        if (step + 1) % max(1, args.logging_steps) == 0:
            ppl = math.exp(avg_loss)
            print(
                f"[Train] step {step + 1}/{total_steps} "
                f"lr {lr_cur:.6f} loss {avg_loss:.4f} ppl {ppl:.2f} "
                f"ema_loss {ema_loss:.4f} tokens {accum_tokens}"
            )

        if args.eval_steps > 0 and (step + 1) % args.eval_steps == 0:
            valid_ppl = evaluate(model, eval_loader, device, args.eval_batches)
            print(
                f"[Eval] step {step + 1}/{total_steps} valid_ppl {valid_ppl:.2f} "
                f"ema_loss {ema_loss:.4f} total_tokens {token_counter}"
            )

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print(f"\n🎉 Qwen LoRA training done. Saved adapter to {args.output_dir}")
    print(f"Total steps {total_steps}, total tokens {token_counter}, final EMA loss {ema_loss:.4f}")


if __name__ == "__main__":
    main()

