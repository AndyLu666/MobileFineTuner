#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal PyTorch LoRA fine-tuning entrypoint for Gemma 3 270M on WikiText-2.

This mirrors the C++ train_lora_gemma binary:
  * Reads wiki.train.raw / wiki.valid.raw from --data_dir
  * Packs tokens into fixed-length sequences (seq_len, stride=seq_len)
  * Injects LoRA on attention projections (default) via PEFT
  * Supports gradient accumulation, warmup, eval, and embedding dumps

Example:
python scripts/train_gemma3_lora_torch.py \
    --model_dir /Users/tony/Documents/FT/gemma-3-270m \
    --data_dir /Users/tony/Documents/FT/data/wikitext2/wikitext-2-raw \
    --output_dir ./torch_runs/gemma_attn_lora \
    --batch_size 4 \
    --grad_accum 1 \
    --seq_len 128 \
    --epochs 1 \
    --lr 1e-5 \
    --warmup_ratio 0.1 \
    --targets attn \
    --dump_embedding 1
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)

try:
    from peft import LoraConfig, get_peft_model
except ImportError as exc:  # pragma: no cover - dependency hint
    raise SystemExit(
        "peft is required. Install with `pip install peft` "
        "or refer to https://github.com/huggingface/peft."
    ) from exc


@dataclass
class TrainerArgs:
    model_dir: str
    data_dir: str
    output_dir: str
    seq_len: int = 128
    stride: int = -1
    batch_size: int = 4
    grad_accum: int = 1
    lr: float = 2e-5
    weight_decay: float = 0.0
    warmup_ratio: float = 0.05
    epochs: int = 1
    max_steps: int = -1
    logging_steps: int = 1
    eval_steps: int = 0
    max_grad_norm: float = 1.0
    data_fraction: float = 1.0
    targets: str = "attn"  # attn / full / light
    dump_embedding: bool = False
    dump_embedding_step: int = 1
    dump_embedding_dir: str = "./torch_debug"
    device: Optional[str] = None
    use_fast_tokenizer: bool = False


class PackedWikiText(Dataset):
    """Simple tokenizer->chunk dataset."""

    def __init__(
        self,
        path: str,
        tokenizer: AutoTokenizer,
        seq_len: int,
        stride: int,
        fraction: float,
    ):
        tokens = self._load_tokens(path, tokenizer)
        min_tokens = seq_len + 1
        limit = max(min_tokens, int(len(tokens) * max(0.0, min(1.0, fraction))))
        limit = min(limit, len(tokens))
        tokens = tokens[:limit]

        self.seq_len = seq_len
        self.stride = seq_len if stride <= 0 else stride
        self.inputs, self.labels = self._pack(tokens, seq_len, self.stride)

    @staticmethod
    def _load_tokens(path: str, tokenizer: AutoTokenizer) -> List[int]:
        eos = tokenizer.eos_token_id
        tokens: List[int] = []
        with open(path, "r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.rstrip("\n")
                if line:
                    tokens.extend(tokenizer.encode(line, add_special_tokens=False))
                tokens.append(eos)
        if not tokens or tokens[-1] != eos:
            tokens.append(eos)
        return tokens

    @staticmethod
    def _pack(tokens: List[int], seq_len: int, stride: int) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs: List[List[int]] = []
        labels: List[List[int]] = []
        total = len(tokens) - seq_len - 1
        for start in range(0, max(0, total + 1), stride):
            end = start + seq_len + 1
            if end >= len(tokens):
                break
            chunk = tokens[start:end]
            inputs.append(chunk[:-1])
            labels.append(chunk[1:])
        inp_tensor = torch.tensor(inputs, dtype=torch.long)
        lbl_tensor = torch.tensor(labels, dtype=torch.long)
        return inp_tensor, lbl_tensor

    def __len__(self) -> int:
        return self.inputs.size(0)

    def __getitem__(self, idx: int):
        return {
            "input_ids": self.inputs[idx],
            "labels": self.labels[idx],
        }


def parse_args() -> TrainerArgs:
    parser = argparse.ArgumentParser(description="PyTorch Gemma3 LoRA trainer (WikiText-2).")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--stride", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--data_fraction", type=float, default=1.0)
    parser.add_argument("--targets", type=str, default="attn", choices=["attn", "full", "light"])
    parser.add_argument("--dump_embedding", type=int, default=0)
    parser.add_argument("--dump_embedding_step", type=int, default=1)
    parser.add_argument("--dump_embedding_dir", type=str, default="./torch_debug")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--use_fast_tokenizer", action="store_true", help="Use fast tokenizer backend (default: slow).")
    args = parser.parse_args()

    return TrainerArgs(
        model_dir=args.model_dir,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        seq_len=args.seq_len,
        stride=args.stride,
        batch_size=args.batch_size,
        grad_accum=max(1, args.grad_accum),
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        epochs=args.epochs,
        max_steps=args.max_steps,
        logging_steps=max(1, args.logging_steps),
        eval_steps=args.eval_steps,
        max_grad_norm=args.max_grad_norm,
        data_fraction=args.data_fraction,
        targets=args.targets,
        dump_embedding=bool(args.dump_embedding),
        dump_embedding_step=max(1, args.dump_embedding_step),
        dump_embedding_dir=args.dump_embedding_dir,
        device=args.device,
        use_fast_tokenizer=args.use_fast_tokenizer,
    )


def select_targets(name: str) -> List[str]:
    if name == "attn":
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    if name == "light":
        return ["q_proj", "v_proj"]
    return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def prepare_dataloaders(args: TrainerArgs, tokenizer: AutoTokenizer) -> Tuple[DataLoader, DataLoader]:
    train_path = Path(args.data_dir) / "wiki.train.raw"
    valid_path = Path(args.data_dir) / "wiki.valid.raw"
    train_ds = PackedWikiText(str(train_path), tokenizer, args.seq_len, args.stride, args.data_fraction)
    valid_ds = PackedWikiText(str(valid_path), tokenizer, args.seq_len, args.stride, min(1.0, args.data_fraction))
    print(f"[Data] Train sequences: {len(train_ds)}, Eval sequences: {len(valid_ds)}")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    eval_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
    return train_loader, eval_loader


def maybe_dump_embedding(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    step: int,
    args: TrainerArgs,
    already_dumped: bool,
) -> bool:
    if not args.dump_embedding or already_dumped or step != args.dump_embedding_step:
        return already_dumped
    base = getattr(model, "model", None) or getattr(model, "base_model", None)
    if base is None or not hasattr(base, "embed_tokens"):
        print("[EmbeddingDump] Warning: model does not expose embed_tokens.")
        return True
    with torch.no_grad():
        embeds = base.embed_tokens(input_ids.to(model.device)).detach().cpu()
    flat = embeds.view(-1, embeds.size(-1))
    mean = flat.mean().item()
    std = flat.std(unbiased=False).item()
    min_val = flat.min().item()
    max_val = flat.max().item()
    print(
        f"[EmbeddingDump] step {step} shape={tuple(embeds.shape)} "
        f"mean={mean:.6f} std={std:.6f} min={min_val:.6f} max={max_val:.6f}"
    )
    tokens_to_show = min(4, embeds.size(1))
    dims_to_show = min(8, embeds.size(2))
    for t in range(tokens_to_show):
        sl = embeds[0, t, :dims_to_show].tolist()
        formatted = ", ".join(f"{v:.4f}" for v in sl)
        print(f"  hidden[0,{t},0:{dims_to_show}] = [{formatted}]")
    dump_dir = Path(args.dump_embedding_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)
    dump_path = dump_dir / f"embedding_step{step}.pt"
    torch.save(embeds, dump_path)
    print(f"  [EmbeddingDump] wrote tensor to {dump_path}")
    return True


def evaluate(model, loader, device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            inputs = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            loss = model(input_ids=inputs, labels=labels).loss
            total_loss += loss.item()
            count += 1
    model.train()
    if count == 0:
        return 0.0, float("inf")
    mean_loss = total_loss / count
    ppl = math.exp(min(mean_loss, 100))  # clamp to avoid overflow
    print(f"[Eval] loss={mean_loss:.4f} ppl={ppl:.2f}")
    return mean_loss, ppl


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    random.seed(42)
    torch.manual_seed(42)

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=args.use_fast_tokenizer)
    except Exception as exc:
        if args.use_fast_tokenizer:
            raise
        print(f"[Tokenizer] Fast tokenizer load failed ({exc}); falling back to slow tokenizer.")
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_loader, eval_loader = prepare_dataloaders(args, tokenizer)

    print("[Model] Loading Gemma3ForCausalLM from", args.model_dir)
    config = AutoConfig.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        config=config,
        torch_dtype=torch.float32,
        device_map=None,
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=select_targets(args.targets),
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    updates_per_epoch = (len(train_loader) // args.grad_accum)
    total_updates = args.epochs * max(1, updates_per_epoch)
    if args.max_steps > 0:
        total_updates = min(total_updates, args.max_steps)
    warmup_steps = max(1, int(total_updates * args.warmup_ratio)) if total_updates > 0 else 0
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates if total_updates > 0 else updates_per_epoch * args.epochs,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    global_step = 0
    micro_step = 0
    accum_loss = 0.0
    dumped = False

    model.train()
    for epoch in range(args.epochs):
        print(f"\n=== Torch Gemma Epoch {epoch+1}/{args.epochs} ===")
        for batch in train_loader:
            micro_step += 1
            inputs = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            dumped = maybe_dump_embedding(model, inputs, micro_step, args, dumped)

            outputs = model(input_ids=inputs, labels=labels)
            loss = outputs.loss / args.grad_accum
            loss.backward()
            accum_loss += outputs.loss.item()

            if micro_step % args.grad_accum != 0:
                continue

            torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            avg_loss = accum_loss / args.grad_accum
            accum_loss = 0.0
            ppl = math.exp(min(avg_loss, 100))
            if global_step % args.logging_steps == 0:
                lr = scheduler.get_last_lr()[0]
                print(f"[Step {global_step}] loss={avg_loss:.4f} ppl={ppl:.2f} lr={lr:.6g}")

            if args.eval_steps and global_step % args.eval_steps == 0:
                evaluate(model, eval_loader, device)

            if 0 < args.max_steps <= global_step:
                break

        if 0 < args.max_steps <= global_step:
            print(f"[Trainer] Reached max_steps={args.max_steps}, stopping.")
            break

    if args.eval_steps == 0:
        evaluate(model, eval_loader, device)

    save_path = Path(args.output_dir) / "lora_adapter"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"[Save] LoRA adapter saved to {save_path}")


if __name__ == "__main__":
    main()
