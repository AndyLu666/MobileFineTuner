#!/usr/bin/env python3
"""
T2: Residual + RMSNorm chain microtest (no attention/MLP math).
Given layer input x0 and mocked attn_out/ffn_out from C++ dumps, apply HF Gemma3 RMSNorms and residual adds,
compare with C++ dumped tensors to isolate formula/ordering differences.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM


def save(name: str, t: torch.Tensor, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{name}.npy", t.detach().cpu().float().numpy())


def rdiff(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64); b = b.astype(np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def run_for_layer(model, cpp_dir: Path, out_dir: Path, L: int):
    device = torch.device("cpu")
    layers = model.model.layers
    ln_in = layers[L].input_layernorm
    ln_post_attn = layers[L].post_attention_layernorm
    ln_pre_ffn = layers[L].pre_feedforward_layernorm
    ln_post_ffn = layers[L].post_feedforward_layernorm

    # x0: previous layer output
    if L == 0:
        x0 = np.load(cpp_dir / "hidden_states_emb.npy")
    else:
        # Reconstruct previous layer output: x0 = attn_add_{L-1} + post_ffn_norm_{L-1}
        attn_add_prev = np.load(cpp_dir / f"hidden_after_attn_add_l{L-1}.npy")
        post_ffn_norm_prev = np.load(cpp_dir / f"hidden_after_mlp_norm_l{L-1}.npy")
        x0 = attn_add_prev + post_ffn_norm_prev
    x0_t = torch.tensor(x0, dtype=torch.float32, device=device)

    # attn_out from C++ dump (mock self_attn output before post-attn LN)
    attn_raw = np.load(cpp_dir / f"attn_out_raw_l{L}.npy")
    attn_raw_t = torch.tensor(attn_raw, dtype=torch.float32, device=device)

    # ffn_out from C++ dump (mock MLP output before post-ffn LN)
    ffn_out = np.load(cpp_dir / f"hidden_after_mlp_l{L}.npy")
    ffn_out_t = torch.tensor(ffn_out, dtype=torch.float32, device=device)

    # Chain in PT
    x1 = ln_in(x0_t)
    attn_norm = ln_post_attn(attn_raw_t)
    attn_add = x0_t + attn_norm
    pre_ffn = ln_pre_ffn(attn_add)
    ffn_norm = ln_post_ffn(ffn_out_t)
    x_next = attn_add + ffn_norm

    # Save PT reconstructions
    save(f"pt_hidden_before_attn_l{L}", x1, out_dir)
    save(f"pt_hidden_after_attn_norm_l{L}", attn_norm, out_dir)
    save(f"pt_hidden_after_attn_add_l{L}", attn_add, out_dir)
    save(f"pt_hidden_before_mlp_norm_l{L}", pre_ffn, out_dir)
    save(f"pt_hidden_after_mlp_norm_l{L}", ffn_norm, out_dir)
    save(f"pt_hidden_states_out_l{L}", x_next, out_dir)

    # Compare to C++
    keys = [
        ("hidden_before_attn_l{}", x1),
        ("hidden_after_attn_norm_l{}", attn_norm),
        ("hidden_after_attn_add_l{}", attn_add),
        ("hidden_before_mlp_norm_l{}", pre_ffn),
        ("hidden_after_mlp_norm_l{}", ffn_norm),
    ]
    for k, t in keys:
        name = k.format(L)
        try:
            cpp = np.load(cpp_dir / f"{name}.npy")
            pt = t.detach().cpu().numpy().astype(np.float32)
            print(f"{name:28s} rel_diff={rdiff(cpp, pt):.3e}")
        except Exception as e:
            print(f"{name:28s} MISSING ({e})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--cpp_dir", required=True)
    ap.add_argument("--dump_dir", required=True)
    ap.add_argument("--layer", type=int, required=True)
    args = ap.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=torch.float32)
    model.eval()

    run_for_layer(model, Path(args.cpp_dir), Path(args.dump_dir), args.layer)


if __name__ == "__main__":
    main()


