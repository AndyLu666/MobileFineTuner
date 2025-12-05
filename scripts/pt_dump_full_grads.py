#!/usr/bin/env python3
"""
PT script to dump both forward activations AND backward gradients for alignment.
This enables direct comparison with C++ intermediate gradients.
Uses hooks to capture gradients during the actual backward pass.
"""
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


def rms_norm_manual(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    x_f = x.float()
    inv_rms = torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + eps)
    y_f = x_f * inv_rms * (1.0 + w.float())
    return y_f.to(dtype=x.dtype)


def build_causal_mask(S: int, device) -> torch.Tensor:
    m = torch.triu(torch.ones((S, S), dtype=torch.float32, device=device), diagonal=1)
    m = torch.where(m > 0.5, torch.full_like(m, -1e10), torch.zeros_like(m))
    return m.unsqueeze(0).unsqueeze(0)


def build_padding_mask(attn_mask: torch.Tensor) -> torch.Tensor:
    S = attn_mask.shape[1]
    m = torch.zeros((1, 1, 1, S), dtype=torch.float32, device=attn_mask.device)
    m[0, 0, 0] = torch.where(attn_mask[0] < 0.5, torch.tensor(-1e10, device=attn_mask.device), torch.tensor(0.0, device=attn_mask.device))
    return m


def apply_rope(x: torch.Tensor, rope_theta: float) -> torch.Tensor:
    # x: [B,H,S,D]
    b, h, S, D = x.shape
    out = x.clone()
    idx = torch.arange(D // 2, device=x.device, dtype=x.dtype)
    freq = 1.0 / (torch.tensor(rope_theta, dtype=x.dtype, device=x.device) ** (2.0 * idx / D))
    for pos in range(S):
        angle = pos * freq
        cosv = torch.cos(angle)
        sinv = torch.sin(angle)
        x1 = out[:, :, pos, 0::2].clone()
        x2 = out[:, :, pos, 1::2].clone()
        out[:, :, pos, 0::2] = x1 * cosv - x2 * sinv
        out[:, :, pos, 1::2] = x1 * sinv + x2 * cosv
    return out


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
    ap.add_argument("--align_layers", nargs="+", default=["0", "1", "last"])
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
    model.train()  # Need train mode for gradients

    # Get base model layers
    base = getattr(model, "base_model", model)
    core = base
    for _ in range(4):
        if hasattr(core, "embed_tokens") and hasattr(core, "layers"):
            break
        if hasattr(core, "model"):
            core = core.model
        else:
            break
    base = core
    layers = base.layers
    last_idx = len(layers) - 1
    align_layers = []
    for t in args.align_layers:
        if t == "last":
            align_layers.append(last_idx)
        else:
            try:
                align_layers.append(int(t))
            except ValueError:
                pass
    align_layers = [i for i in align_layers if 0 <= i <= last_idx]

    # Configs
    H = model.config.num_attention_heads
    H_kv = getattr(model.config, "num_key_value_heads", 1)
    D = getattr(model.config, "head_dim", model.config.hidden_size // H)
    eps = float(model.config.rms_norm_eps)
    qk_scale = float(model.config.query_pre_attn_scalar) ** -0.5
    causal_mask = build_causal_mask(input_ids.shape[1], device)
    pad_mask = build_padding_mask(attention_mask)

    # Storage for activations and gradients via hooks
    activations = {}
    gradients = {}

    def make_fwd_hook(name):
        def hook(module, inp, out):
            activations[name] = out.detach().clone()
        return hook

    def make_bwd_hook(name):
        def hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                gradients[name] = grad_output[0].detach().clone()
        return hook

    def make_tensor_bwd_hook(name):
        def hook(grad):
            gradients[name] = grad.detach().clone()
        return hook

    # Register hooks for aligned layers
    handles = []
    for idx in align_layers:
        layer = layers[idx]
        # q_proj output
        handles.append(layer.self_attn.q_proj.register_forward_hook(make_fwd_hook(f"q_proj_out_l{idx}")))
        handles.append(layer.self_attn.q_proj.register_full_backward_hook(make_bwd_hook(f"q_proj_out_l{idx}")))
        # k_proj output
        handles.append(layer.self_attn.k_proj.register_forward_hook(make_fwd_hook(f"k_proj_out_l{idx}")))
        handles.append(layer.self_attn.k_proj.register_full_backward_hook(make_bwd_hook(f"k_proj_out_l{idx}")))
        # v_proj output
        handles.append(layer.self_attn.v_proj.register_forward_hook(make_fwd_hook(f"v_proj_out_l{idx}")))
        handles.append(layer.self_attn.v_proj.register_full_backward_hook(make_bwd_hook(f"v_proj_out_l{idx}")))
        # o_proj output
        handles.append(layer.self_attn.o_proj.register_forward_hook(make_fwd_hook(f"attn_out_raw_l{idx}")))
        handles.append(layer.self_attn.o_proj.register_full_backward_hook(make_bwd_hook(f"attn_out_raw_l{idx}")))
        # input_layernorm output (hidden_before_attn)
        handles.append(layer.input_layernorm.register_forward_hook(make_fwd_hook(f"hidden_before_attn_l{idx}")))
        handles.append(layer.input_layernorm.register_full_backward_hook(make_bwd_hook(f"hidden_before_attn_l{idx}")))

    # Forward pass
    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=None, use_cache=False)
    logits = out.logits
    save_np("logits", logits, dump_dir)

    # Compute loss and backward
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    vocab = shift_logits.size(-1)
    loss = F.cross_entropy(shift_logits.view(-1, vocab), shift_labels.view(-1), ignore_index=-100, reduction="sum")
    print(f"[loss] {loss.item():.6f}")
    save_np("loss_scalar", torch.tensor([loss.item()]), dump_dir)
    
    loss.backward()

    # Remove hooks
    for h in handles:
        h.remove()

    # Dump forward activations
    for name, tensor in activations.items():
        save_np(name, tensor, dump_dir)

    # Dump gradients
    for name, tensor in gradients.items():
        save_np(f"grads/{name}", tensor, dump_dir)

    # Dump LoRA gradients
    for name, param in model.named_parameters():
        if "lora" in name and param.grad is not None:
            save_np(f"grads/{name.replace('.', '_')}", param.grad, dump_dir)

    # Dump LoRA A/B weights to weights_after_step (for C++ to load and match A)
    weights_after = dump_dir / "weights_after_step"
    ensure_dir(weights_after)
    for idx, layer in enumerate(layers):
        for mod in ["q_proj", "o_proj"]:
            if hasattr(layer.self_attn, mod):
                lin = getattr(layer.self_attn, mod)
                # PEFT stores lora_A/lora_B under .lora_A/.lora_B with key 'default'
                try:
                    wA = lin.lora_A["default"].weight.detach().float().cpu().numpy()
                    wB = lin.lora_B["default"].weight.detach().float().cpu().numpy()
                    np.save(weights_after / f"base_model_model_model_layers_{idx}_self_attn_{mod}_lora_A_default_weight.npy", wA)
                    np.save(weights_after / f"base_model_model_model_layers_{idx}_self_attn_{mod}_lora_B_default_weight.npy", wB)
                    print(f"[dump] weights_after_step {mod} L{idx}: A{list(wA.shape)} B{list(wB.shape)}")
                except Exception as e:
                    print(f"[warn] dump lora weights failed for layer {idx} {mod}: {e}")

    # Now do manual reconstruction to get intermediate tensors that hooks can't capture
    print("\n[manual reconstruction for intermediate tensors]")
    with torch.no_grad():
        emb = base.embed_tokens(input_ids)
        save_np("hidden_states_emb", emb, dump_dir)
        resid_cur = emb

        # Iterate all layers sequentially; only dump for align_layers
        for idx in range(last_idx + 1):
            dump_this = (idx in align_layers)
            if dump_this:
                save_np(f"hidden_states_in_l{idx}", resid_cur, dump_dir)
            
            # input layernorm
            norm_in = rms_norm_manual(resid_cur, layers[idx].input_layernorm.weight.detach(), eps)
            if dump_this:
                save_np(f"hidden_before_attn_l{idx}", norm_in, dump_dir)
            
            # q/k/v proj
            q_lin = layers[idx].self_attn.q_proj(norm_in)
            k_lin = layers[idx].self_attn.k_proj(norm_in)
            v_lin = layers[idx].self_attn.v_proj(norm_in)
            if dump_this:
                save_np(f"q_proj_out_l{idx}", q_lin, dump_dir)
                save_np(f"k_proj_out_l{idx}", k_lin, dump_dir)
                save_np(f"v_proj_out_l{idx}", v_lin, dump_dir)
            
            # heads + q/k norm
            Bsz, Slen, _ = q_lin.shape
            q_heads = q_lin.view(Bsz, Slen, H, D).permute(0, 2, 1, 3).contiguous()
            k_heads = k_lin.view(Bsz, Slen, H_kv, D).permute(0, 2, 1, 3).contiguous()
            v_heads = v_lin.view(Bsz, Slen, H_kv, D).permute(0, 2, 1, 3).contiguous()
            qn = rms_norm_manual(q_heads, layers[idx].self_attn.q_norm.weight.detach(), eps)
            kn = rms_norm_manual(k_heads, layers[idx].self_attn.k_norm.weight.detach(), eps)
            if dump_this:
                save_np(f"q_norm_out_l{idx}", qn, dump_dir)
                save_np(f"k_norm_out_l{idx}", kn, dump_dir)
            
            # RoPE
            is_sliding = False
            if hasattr(model.config, "layer_types") and idx < len(model.config.layer_types):
                is_sliding = (model.config.layer_types[idx] == "sliding_attention")
            rope_theta = float(model.config.rope_local_base_freq) if is_sliding else float(model.config.rope_theta)
            q = apply_rope(qn, rope_theta)
            k = apply_rope(kn, rope_theta)
            
            # repeat kv
            if k.shape[1] == 1 and H > 1:
                k_full = k.repeat(1, H, 1, 1)
                v_full = v_heads.repeat(1, H, 1, 1)
            else:
                k_full = k
                v_full = v_heads
            
            # scores/probs
            scores = torch.matmul(q, k_full.transpose(-1, -2)) * qk_scale
            scores_masked = scores + causal_mask + pad_mask
            if dump_this:
                save_np(f"attn_scores_l{idx}", scores_masked, dump_dir)
            
            probs = torch.softmax(scores_masked, dim=-1)
            if dump_this:
                save_np(f"attn_probs_l{idx}", probs, dump_dir)
            
            # context
            context = torch.matmul(probs, v_full).permute(0, 2, 1, 3).contiguous().view(Bsz, Slen, H * D)
            if dump_this:
                save_np(f"attn_context_l{idx}", context, dump_dir)
            
            # o_proj
            attn_out = layers[idx].self_attn.o_proj(context)
            if dump_this:
                save_np(f"attn_out_raw_l{idx}", attn_out, dump_dir)
                save_np(f"hidden_after_attn_l{idx}", attn_out, dump_dir)
            
            # post-attn norm + residual
            attn_norm = rms_norm_manual(attn_out, layers[idx].post_attention_layernorm.weight.detach(), eps)
            attn_add = resid_cur + attn_norm
            if dump_this:
                save_np(f"hidden_after_attn_norm_l{idx}", attn_norm, dump_dir)
                save_np(f"hidden_after_attn_add_l{idx}", attn_add, dump_dir)
            
            # pre-ffn norm
            pre_ffn = rms_norm_manual(attn_add, layers[idx].pre_feedforward_layernorm.weight.detach(), eps)
            if dump_this:
                save_np(f"hidden_before_mlp_norm_l{idx}", pre_ffn, dump_dir)
            
            # mlp
            mlp_out = layers[idx].mlp(pre_ffn)
            mlp_norm = rms_norm_manual(mlp_out, layers[idx].post_feedforward_layernorm.weight.detach(), eps)
            final_out = attn_add + mlp_norm
            if dump_this:
                save_np(f"hidden_after_mlp_l{idx}", mlp_out, dump_dir)
                save_np(f"hidden_after_mlp_norm_l{idx}", mlp_norm, dump_dir)
                save_np(f"hidden_states_out_l{idx}", final_out, dump_dir)

            resid_cur = final_out

    print(f"\n[done] dumps saved to {dump_dir}")


if __name__ == "__main__":
    main()
