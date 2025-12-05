#!/usr/bin/env python3
import argparse
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def save_np(name: str, arr: torch.Tensor, dump_dir: Path):
    np.save(dump_dir / f"{name}.npy", arr.detach().cpu().numpy())
    print(f"[dump] {name}: {list(arr.shape)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--bin_path", required=True)
    ap.add_argument("--meta_path", required=True)
    ap.add_argument("--dump_dir", required=True)
    ap.add_argument("--seq_len", type=int, default=32)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--target_modules", nargs="+", default=["q_proj", "o_proj"])
    ap.add_argument("--lora_rank", type=int, default=8)
    ap.add_argument("--lora_alpha", type=float, default=32.0)
    ap.add_argument("--lora_dropout", type=float, default=0.0)
    ap.add_argument("--use_ids_from", type=str, default="")
    args = ap.parse_args()

    meta = json.loads(Path(args.meta_path).read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dump_dir = Path(args.dump_dir)
    ensure_dir(dump_dir)
    ensure_dir(dump_dir / "grads")

    if args.use_ids_from:
        base = Path(args.use_ids_from)
        input_ids = torch.tensor(np.load(base / "input_ids.npy"), dtype=torch.long, device=device)
        attention_mask = torch.tensor(np.load(base / "attention_mask.npy"), dtype=torch.float32, device=device)
        labels = torch.tensor(np.load(base / "labels.npy"), dtype=torch.long, device=device)
    else:
        tokens = np.fromfile(args.bin_path, dtype=np.int32)
        span = tokens[args.start: args.start + args.seq_len + 1]
        input_ids = torch.tensor(span[:-1], dtype=torch.long, device=device).unsqueeze(0)
        labels = torch.tensor(span[1:], dtype=torch.long, device=device).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=torch.float32, attn_implementation="eager"
    ).to(device)
    for p_name in ["attention_dropout", "hidden_dropout", "hidden_dropout_prob", "embedding_dropout"]:
        if hasattr(model.config, p_name):
            setattr(model.config, p_name, 0.0)
    lora_cfg = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=args.target_modules, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_cfg)
    model.train()

    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=None, use_cache=False)
    logits = out.logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    vocab = shift_logits.size(-1)
    loss = F.cross_entropy(shift_logits.view(-1, vocab), shift_labels.view(-1), ignore_index=-100, reduction="sum")
    loss.backward()

    for name, param in model.named_parameters():
        if "lora_B" in name and param.grad is not None:
            save_np(f"grads/{name.replace('.', '_')}", param.grad, dump_dir)

    print(f"[done] grads saved to {dump_dir}")


if __name__ == "__main__":
    main()


