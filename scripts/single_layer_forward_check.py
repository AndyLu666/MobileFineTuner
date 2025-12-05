#!/usr/bin/env python3
"""
Single-layer forward mirror recomputation for Gemma3 + LoRA.
Reads layer input from C++ dumps, runs ONLY one decoder layer in PyTorch with identical weights,
reconstructs attention path (q/k/v -> scores/probs -> context -> o_proj), residual + LN, and MLP,
then dumps the same intermediate tensors for 1:1 comparison.
"""

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM


def save_np(path: Path, t: torch.Tensor):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, t.detach().cpu().float().numpy())
    print(f"[dump] {path.name}: {list(t.shape)} -> {path}")


def build_causal_mask(S: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    # strictly upper-triangular masked to -1e10
    m = torch.triu(torch.ones((S, S), dtype=dtype, device=device), diagonal=1)
    m = torch.where(m > 0.5, torch.full_like(m, -1e10, dtype=dtype), torch.zeros_like(m, dtype=dtype))
    return m.unsqueeze(0).unsqueeze(0)  # [1,1,S,S]


def repeat_kv(v: torch.Tensor, H: int) -> torch.Tensor:
    # v: [B,S,kvH*Hd] with kvH=1 → reshape and repeat to [B,H,S,Hd]
    B, S, D = v.shape
    Hd = D // 1
    kvH = 1
    v = v.view(B, S, kvH, Hd).permute(0, 2, 1, 3).contiguous()  # [B,1,S,Hd]
    if H > 1:
        v = v.repeat(1, H, 1, 1)
    return v


def apply_rope_with_model(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # q,k: [B,H,S,Hd]; cos,sin: [B,S,Hd]
    # match HF apply_rotary_pos_emb broadcast semantics
    cos = cos.unsqueeze(1)  # [B,1,S,Hd]
    sin = sin.unsqueeze(1)
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--cpp_dir", required=True, help="C++ dump dir to read layer input and compare against")
    p.add_argument("--dump_dir", required=True, help="PT dump dir for single-layer recompute")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--device", default="auto", choices=["auto","cpu","cuda"])
    p.add_argument("--target_modules", nargs="+", default=["q_proj","o_proj"])
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=float, default=32.0)
    args = p.parse_args()

    device = torch.device("cuda" if (args.device=="auto" and torch.cuda.is_available()) else (args.device if args.device!="auto" else "cpu"))
    cpp_dir = Path(args.cpp_dir)
    out_dir = Path(args.dump_dir)
    L = args.layer

    # Load layer input from C++ dump
    if L == 0:
        inp_np = np.load(cpp_dir / "hidden_states_emb.npy")
    else:
        inp_np = np.load(cpp_dir / f"hidden_after_mlp_norm_l{L-1}.npy")
    x0 = torch.tensor(inp_np, dtype=torch.float32, device=device)  # [B,S,hidden]
    B, S, Hsz = x0.shape

    # Load model in float32 and attach LoRA the same way as alignment
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=torch.float32, attn_implementation="eager").to(device)
    # disable dropouts
    for name in ["attention_dropout", "hidden_dropout", "hidden_dropout_prob", "embedding_dropout"]:
        if hasattr(model.config, name):
            setattr(model.config, name, 0.0)
    lora_cfg = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=0.0,
        target_modules=args.target_modules, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_cfg)
    model.eval()

    # Access base model under PEFT wrapper
    try:
        base = model.get_base_model()
    except AttributeError:
        base = model
    text_model = getattr(base, "model", base)
    layers = text_model.layers
    attn = layers[L].self_attn

    # Norms for this layer
    ln_in = layers[L].input_layernorm
    ln_post_attn = layers[L].post_attention_layernorm
    ln_pre_ffn = layers[L].pre_feedforward_layernorm
    ln_post_ffn = layers[L].post_feedforward_layernorm

    # Prepare masks and RoPE embeddings
    pos_ids = torch.arange(S, device=device).unsqueeze(0)  # [1,S]
    # Position embeddings for this layer type
    layer_type = model.config.layer_types[L]
    cos, sin = text_model.rotary_emb(x0, pos_ids, layer_type)  # [B,S,Hd]
    causal = build_causal_mask(S, device=device, dtype=torch.float32)  # [1,1,S,S]
    scale = float(model.config.query_pre_attn_scalar) ** -0.5

    # 1) pre-attn LN
    x1 = ln_in(x0)                            # hidden_before_attn
    save_np(out_dir / f"hidden_before_attn_l{L}.npy", x1)

    # 2) q/k/v projections (pre-reshape)
    q_lin = attn.q_proj(x1)                   # [B,S,H*Hd]
    k_lin = attn.k_proj(x1)                   # [B,S,kvH*Hd]
    v_lin = attn.v_proj(x1)                   # [B,S,kvH*Hd]
    save_np(out_dir / f"q_proj_out_l{L}.npy", q_lin)
    save_np(out_dir / f"k_proj_out_l{L}.npy", k_lin)
    save_np(out_dir / f"v_proj_out_l{L}.npy", v_lin)

    # 3) reshape to heads and apply RMSNorm on q/k
    H = model.config.num_attention_heads
    kvH = model.config.num_key_value_heads
    Hd = model.config.head_dim
    q = q_lin.view(B, S, H, Hd).permute(0, 2, 1, 3).contiguous()     # [B,H,S,Hd]
    k = k_lin.view(B, S, kvH, Hd).permute(0, 2, 1, 3).contiguous()   # [B,kvH,S,Hd]
    # RMSNorm per-head
    q = attn.q_norm(q)
    k = attn.k_norm(k)
    save_np(out_dir / f"q_norm_out_l{L}.npy", q)
    save_np(out_dir / f"k_norm_out_l{L}.npy", k)

    # 4) RoPE
    q_rot, k_rot = apply_rope_with_model(q, k, cos, sin)

    # 5) scores/probs
    k_full = k_rot.repeat(1, H // kvH, 1, 1)           # [B,H,S,Hd]
    scores = torch.matmul(q_rot, k_full.transpose(-2, -1)) * scale  # [B,H,S,S]
    scores = scores + causal
    probs = torch.softmax(scores, dim=-1)
    save_np(out_dir / f"attn_scores_l{L}.npy", scores)
    save_np(out_dir / f"attn_probs_l{L}.npy", probs)

    # 6) context and o_proj
    v_full = repeat_kv(v_lin, H)                       # [B,H,S,Hd]
    context = torch.matmul(probs, v_full)              # [B,H,S,Hd]
    context = context.permute(0, 2, 1, 3).contiguous().view(B, S, H * Hd)
    save_np(out_dir / f"attn_context_l{L}.npy", context)
    attn_out = attn.o_proj(context)                    # [B,S,hidden]
    save_np(out_dir / f"attn_out_raw_l{L}.npy", attn_out)
    save_np(out_dir / f"hidden_after_attn_l{L}.npy", attn_out)

    # 7) post-attn norm + residual add
    attn_norm = ln_post_attn(attn_out)
    save_np(out_dir / f"hidden_after_attn_norm_l{L}.npy", attn_norm)
    attn_add = x0 + attn_norm
    save_np(out_dir / f"hidden_after_attn_add_l{L}.npy", attn_add)

    # 8) pre-FFN norm + MLP + post-FFN norm + residual add
    pre_ffn = ln_pre_ffn(attn_add)
    save_np(out_dir / f"hidden_before_mlp_norm_l{L}.npy", pre_ffn)
    mlp_out = layers[L].mlp(pre_ffn)
    save_np(out_dir / f"hidden_after_mlp_l{L}.npy", mlp_out)
    mlp_norm = ln_post_ffn(mlp_out)
    save_np(out_dir / f"hidden_after_mlp_norm_l{L}.npy", mlp_norm)
    final_out = attn_add + mlp_norm
    save_np(out_dir / f"hidden_states_out_l{L}.npy", final_out)

    print("[done] Single-layer forward recompute finished.")


if __name__ == "__main__":
    main()


