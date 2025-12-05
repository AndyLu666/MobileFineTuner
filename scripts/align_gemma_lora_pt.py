#!/usr/bin/env python3
"""
Gemma + LoRA PyTorch baseline用于对齐：
1) 从离线 pretokenized bin 取一小批序列 (batch=1 默认)。
2) HF + PEFT 挂 LoRA (attn_only)，关闭 dropout/AMP。
3) dump 中间激活、logits、loss、LoRA 梯度/权重到 dump_dir 以供 C++ 对比。
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_tokens(bin_path: Path, start: int, seq_len: int) -> np.ndarray:
    data = np.fromfile(bin_path, dtype=np.int32)
    end = start + seq_len + 1  # need +1 for next-token labels
    if end > data.size:
        raise ValueError(f"request beyond file size: start={start}, seq_len={seq_len}, total={data.size}")
    return data[start:end]


def load_meta(meta_path: Path) -> Dict:
    with meta_path.open("r") as f:
        return json.load(f)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def save_np(name: str, arr: torch.Tensor, dump_dir: Path):
    npy_path = dump_dir / f"{name}.npy"
    np.save(npy_path, arr.detach().cpu().numpy())
    print(f"[dump] {name}: {list(arr.shape)} -> {npy_path}")


def main():
    parser = argparse.ArgumentParser(description="Gemma+LoRA PyTorch baseline for C++ alignment.")
    parser.add_argument("--model_dir", default="gemma-3-270m")
    parser.add_argument("--bin_path", default="data/wikitext2/pretokenized_gemma/wt2_gemma_tokens.bin")
    parser.add_argument("--meta_path", default="data/wikitext2/pretokenized_gemma/meta.json")
    parser.add_argument("--dump_dir", default="logs/align_pt")
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--start", type=int, default=0, help="token offset in the concatenated stream")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--do_adam_step", action="store_true", help="apply one AdamW step (weight_decay=0.0) after backward and dump updated weights")
    parser.add_argument("--target_modules", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj"])
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=32.0)
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="set 0 for deterministic alignment")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_ids_from", type=str, default="", help="if set, load input_ids/attention_mask/labels npy from this dir")
    parser.add_argument(
        "--align_layers",
        nargs="+",
        default=["0", "1", "last"],
        help='which layers to hook/dump; use indices or "last"; empty -> no hooks',
    )
    parser.add_argument(
        "--minimal_mode",
        action="store_true",
        help="disable all activation hooks/dumps (only dump logits/loss/LoRA grads)",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    # 自动模式：优先 cuda，否则退回 cpu，避免把 "auto" 直接传给 torch.device 导致错误
    if args.device == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = args.device
    device = torch.device(device_str)

    dump_dir = Path(args.dump_dir)
    ensure_dir(dump_dir)

    meta = load_meta(Path(args.meta_path))
    print(f"[meta] total_tokens={meta.get('total_tokens')} eos={meta.get('eos_token_id')} pad={meta.get('pad_token_id')}")

    if args.use_ids_from:
        base = Path(args.use_ids_from)
        input_ids_np = np.load(base / "input_ids.npy")
        attn_np = np.load(base / "attention_mask.npy")
        labels_np = np.load(base / "labels.npy")
        input_ids = torch.tensor(input_ids_np, dtype=torch.long, device=device)
        attention_mask = torch.tensor(attn_np, dtype=torch.float32, device=device)
        labels = torch.tensor(labels_np, dtype=torch.long, device=device)
        assert list(input_ids.shape)[0] == 1, "expected batch=1"
        assert list(input_ids.shape) == list(attention_mask.shape) == list(labels.shape), "ids/mask/labels shape mismatch"
    # 注意：为避免某些环境下缩进解析异常，这里使用单行条件赋值避免 else 块
    if not args.use_ids_from: token_span = load_tokens(Path(args.bin_path), args.start, args.seq_len); input_ids = torch.tensor(token_span[:-1], dtype=torch.long, device=device).unsqueeze(0); labels = torch.tensor(token_span[1:], dtype=torch.long, device=device).unsqueeze(0); attention_mask = torch.ones_like(input_ids)
    print(f"[data] input_ids[0,:5]={input_ids[0, :5].tolist()} shape={list(input_ids.shape)}")

    # tokenizer 未在对齐流程中使用，避免在某些环境下加载失败
    tokenizer = None

    # ★ 强制使用 eager attention，这样 output_attentions=True 才起作用
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    ).to(device)

    # 关闭 dropout
    for p_name in ["attention_dropout", "hidden_dropout", "hidden_dropout_prob", "embedding_dropout"]:
        if hasattr(model.config, p_name):
            setattr(model.config, p_name, 0.0)

    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.train()  # 需梯度

    # 取得实际的 base model（经过 PEFT 包装后）
    base = getattr(model, "base_model", model)
    core = base
    for _ in range(3):
        if hasattr(core, "embed_tokens") and hasattr(core, "layers"):
            break
        if hasattr(core, "model"):
            core = core.model
        else:
            break
    base = core

    activations = {}
    activations_raw = {}

    def hook_input(name):
        def _hook(module, inp, out):
            tensor = None
            if len(inp) > 0:
                tensor = inp[0]
            elif isinstance(out, tuple) and len(out) > 0 and out[0] is not None:
                # fallback: some modules may not get positional inputs due to kwargs-only call
                tensor = out[0]
            if tensor is not None:
                activations[name] = tensor.detach()
                activations_raw[name] = tensor
                if tensor.requires_grad:
                    tensor.retain_grad()
        return _hook

    def hook_output(name):
        def _hook(module, inp, out):
            tensor = out[0] if isinstance(out, tuple) else out
            activations[name] = tensor.detach()
            activations_raw[name] = tensor
            if tensor.requires_grad:
                tensor.retain_grad()
        return _hook

    def hook_pre_input(name):
        def _hook(module, inp):
            if len(inp) == 0:
                return
            tensor = inp[0]
            activations[name] = tensor.detach()
            activations_raw[name] = tensor
            if tensor.requires_grad:
                tensor.retain_grad()
        return _hook

    # embedding 输出
    if not hasattr(base, "embed_tokens"):
        raise AttributeError("Base model missing embed_tokens; adjust hook resolution.")
    base.embed_tokens.register_forward_hook(hook_output("hidden_states_emb"))

    # 选定层：0、1、最后一层
    layers = base.layers
    last_idx = len(layers) - 1

    # 解析需要对齐的层
    align_layers = []
    for token in args.align_layers:
        if token == "last":
            align_layers.append(last_idx)
        else:
            try:
                align_layers.append(int(token))
            except ValueError:
                continue
    align_layers = [i for i in align_layers if 0 <= i <= last_idx]

    # 如果 minimal_mode，则不注册任何中间 hook
    if not args.minimal_mode:
        for idx in align_layers:
            # 残差分支：捕获 input_layernorm 的输入，便于重建 residual + post-attn-norm 的加法点
            if hasattr(layers[idx], "input_layernorm"):
                layers[idx].input_layernorm.register_forward_pre_hook(hook_pre_input(f"hidden_before_attn_residual_l{idx}"))
            # pre-attn 输入：精确捕获 v_proj 的输入（等价于 self_attn 的输入，即 input_layernorm 输出）
            layers[idx].self_attn.v_proj.register_forward_pre_hook(hook_pre_input(f"hidden_before_attn_l{idx}"))
            # 重新捕获 o_proj 的输入作为 attn_context（带梯度），用于导出 dV 解析梯度
            if hasattr(layers[idx].self_attn, "o_proj"):
                layers[idx].self_attn.o_proj.register_forward_pre_hook(hook_pre_input(f"attn_context_l{idx}"))
            # 注意：不再用模块 hook 保存 attn_out_raw/hidden_after_attn，统一用手动重建
            layers[idx].mlp.register_forward_hook(hook_output(f"hidden_after_mlp_l{idx}"))
            # q/k/v 投影输出
            layers[idx].self_attn.q_proj.register_forward_hook(hook_output(f"q_proj_out_l{idx}"))
            layers[idx].self_attn.k_proj.register_forward_hook(hook_output(f"k_proj_out_l{idx}"))
            layers[idx].self_attn.v_proj.register_forward_hook(hook_output(f"v_proj_out_l{idx}"))
            # q/k norm（RoPE 前）
            if hasattr(layers[idx].self_attn, "q_norm"):
                layers[idx].self_attn.q_norm.register_forward_hook(hook_output(f"q_norm_out_l{idx}"))
            if hasattr(layers[idx].self_attn, "k_norm"):
                layers[idx].self_attn.k_norm.register_forward_hook(hook_output(f"k_norm_out_l{idx}"))
            # post-attn / post-mlp norm 输出
            layers[idx].post_attention_layernorm.register_forward_hook(hook_output(f"hidden_after_attn_norm_l{idx}"))
            layers[idx].post_feedforward_layernorm.register_forward_hook(hook_output(f"hidden_after_mlp_norm_l{idx}"))
            # pre-ffn norm 输出
            if hasattr(layers[idx], "pre_feedforward_layernorm"):
                layers[idx].pre_feedforward_layernorm.register_forward_hook(hook_output(f"hidden_before_mlp_norm_l{idx}"))
            # MLP 内部：gate/up/down，以及 gate*up（prod）输入 down_proj 的 pre-hook
            if hasattr(layers[idx], "mlp"):
                mlp = layers[idx].mlp
                if hasattr(mlp, "gate_proj"):
                    mlp.gate_proj.register_forward_hook(hook_output(f"gate_proj_out_l{idx}"))
                if hasattr(mlp, "up_proj"):
                    mlp.up_proj.register_forward_hook(hook_output(f"up_proj_out_l{idx}"))
                if hasattr(mlp, "down_proj"):
                    # 保存 down_proj 的输入（即 gate_act * up_proj 输出）
                    mlp.down_proj.register_forward_pre_hook(hook_pre_input(f"mlp_prod_l{idx}"))
                    mlp.down_proj.register_forward_hook(hook_output(f"down_proj_out_l{idx}"))

    # 额外导出：所选层的 MLP 权重，便于与 C++ 权重逐元素比对
    with torch.no_grad():
        for idx in align_layers:
            if hasattr(layers[idx], "mlp"):
                mlp = layers[idx].mlp
                if hasattr(mlp, "gate_proj"):
                    save_np(f"weights/gate_proj_weight_l{idx}", mlp.gate_proj.weight.detach().float(), dump_dir)
                if hasattr(mlp, "up_proj"):
                    save_np(f"weights/up_proj_weight_l{idx}", mlp.up_proj.weight.detach().float(), dump_dir)
                if hasattr(mlp, "down_proj"):
                    save_np(f"weights/down_proj_weight_l{idx}", mlp.down_proj.weight.detach().float(), dump_dir)
    # ===== 前向：不用模型自带 loss，手动算 reduction="sum" 的交叉熵 =====
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=None,                 # 不用内置 loss，自己算
        output_hidden_states=False,
        output_attentions=False,     # 手动用 q/k + mask 复现 attn_probs，避免实现差异
        use_cache=False,
    )
    logits = out.logits  # [1, seq_len, vocab]
    logits.retain_grad()

    # === 手动用 q/k 与 mask 复现 attn_probs/attn_scores（前向） ===
    def build_causal_mask(S: int) -> torch.Tensor:
        # 上三角（严格，j > i）位置为 -1e10，其余为 0
        m = torch.triu(torch.ones((S, S), dtype=torch.float32, device=device), diagonal=1)
        m = torch.where(m > 0.5, torch.full_like(m, -1e10), torch.zeros_like(m))
        return m.unsqueeze(0).unsqueeze(0)

    def build_padding_mask(attn_mask: torch.Tensor) -> torch.Tensor:
        S = attn_mask.shape[1]
        m = torch.zeros((1, 1, 1, S), dtype=torch.float32, device=device)
        m[0, 0, 0] = torch.where(attn_mask[0] < 0.5, torch.tensor(-1e10, device=device), torch.tensor(0.0, device=device))
        return m

    causal_mask = build_causal_mask(input_ids.shape[1])
    pad_mask = build_padding_mask(attention_mask)
    qk_scale = float(model.config.query_pre_attn_scalar) ** -0.5

    # 重用上面的 RoPE 实现
    def apply_rope_torch(x: torch.Tensor, rope_theta: float) -> torch.Tensor:
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

    with torch.no_grad():
        # 手写 RMSNorm（Gemma3 权重为 delta）：y = x_hat * (1 + w)
        # 与 C++ 侧 rms_norm 实现保持一致
        def rms_norm_manual(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
            x_f = x.float()
            inv_rms = torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + eps)
            y_f = x_f * inv_rms * (1.0 + w.float())
            return y_f.to(dtype=x.dtype)

        # 逐层顺序重建前向：用上一层的 manual 输出作为下一层输入，消除跨层语义偏差
        # 初始化 L0 的残差输入为真实前传捕获到的 block 输入
        if 0 in align_layers:
            resid_cur = activations_raw.get("hidden_before_attn_residual_l0")
        else:
            resid_cur = activations_raw.get("hidden_states_emb")  # 回退

        # 公用配置
        H = model.config.num_attention_heads
        H_kv = model.config.num_key_value_heads if hasattr(model.config, "num_key_value_heads") else 1
        D = model.config.head_dim if hasattr(model.config, "head_dim") else (model.config.hidden_size // H)
        eps = float(model.config.rms_norm_eps)
        qk_scale = float(model.config.query_pre_attn_scalar) ** -0.5

        def apply_rope_torch(x: torch.Tensor, rope_theta: float) -> torch.Tensor:
            # x: [B,H,S,D]
            b, h, S, Dd = x.shape
            out = x.clone()
            idx = torch.arange(Dd // 2, device=x.device, dtype=x.dtype)
            freq = 1.0 / (torch.tensor(rope_theta, dtype=x.dtype, device=x.device) ** (2.0 * idx / Dd))
            for pos in range(S):
                angle = pos * freq
                cosv = torch.cos(angle)
                sinv = torch.sin(angle)
                x1 = out[:, :, pos, 0::2].clone()
                x2 = out[:, :, pos, 1::2].clone()
                out[:, :, pos, 0::2] = x1 * cosv - x2 * sinv
                out[:, :, pos, 1::2] = x1 * sinv + x2 * cosv
            return out

        # 改為對齊請求的所有層逐層重建，便於中層（如 5、11）分析
        seq_layers = [i for i in align_layers]
        for idx in seq_layers:
            # 保存输入
            if resid_cur is None:
                resid_cur = activations_raw.get(f"hidden_before_attn_residual_l{idx}")
            save_np(f"hidden_states_in_l{idx}", resid_cur, dump_dir)

            # input layernorm
            in_w = layers[idx].input_layernorm.weight.detach()
            norm_in = rms_norm_manual(resid_cur, in_w, eps)
            save_np(f"hidden_before_attn_l{idx}", norm_in, dump_dir)

            # q/k/v 投影（包含 LoRA）
            q_lin = layers[idx].self_attn.q_proj(norm_in)
            k_lin = layers[idx].self_attn.k_proj(norm_in)
            v_lin = layers[idx].self_attn.v_proj(norm_in)
            save_np(f"q_proj_out_l{idx}", q_lin, dump_dir)
            save_np(f"k_proj_out_l{idx}", k_lin, dump_dir)
            save_np(f"v_proj_out_l{idx}", v_lin, dump_dir)

            # 变形为 multi-head 并做 q/k norm（手写 RMSNorm）
            Bsz, Slen, _ = q_lin.shape
            q_heads = q_lin.view(Bsz, Slen, H, D).permute(0, 2, 1, 3).contiguous()     # [B,H,S,D]
            k_heads = k_lin.view(Bsz, Slen, H_kv, D).permute(0, 2, 1, 3).contiguous()  # [B,Hk,S,D]
            v_heads = v_lin.view(Bsz, Slen, H_kv, D).permute(0, 2, 1, 3).contiguous()  # [B,Hk,S,D]
            qn_w = layers[idx].self_attn.q_norm.weight.detach()
            kn_w = layers[idx].self_attn.k_norm.weight.detach()
            qn = rms_norm_manual(q_heads, qn_w, eps)
            kn = rms_norm_manual(k_heads, kn_w, eps)
            save_np(f"q_norm_out_l{idx}", qn, dump_dir)
            save_np(f"k_norm_out_l{idx}", kn, dump_dir)
            # 覆蓋 hook 捕獲，避免最終 activations 保存時覆寫手工值
            activations[f"q_norm_out_l{idx}"] = qn.detach()
            activations[f"k_norm_out_l{idx}"] = kn.detach()

            # inv_rms（与 C++ 对齐：按最后一维 D 规约，keepdim=True）
            with torch.no_grad():
                q_inv_rms = torch.rsqrt(q_heads.float().pow(2).mean(dim=-1, keepdim=True) + eps)
                k_inv_rms = torch.rsqrt(k_heads.float().pow(2).mean(dim=-1, keepdim=True) + eps)
            save_np(f"q_inv_rms_l{idx}", q_inv_rms, dump_dir)
            save_np(f"k_inv_rms_l{idx}", k_inv_rms, dump_dir)

            # RoPE
            is_sliding = False
            if hasattr(model.config, "layer_types") and idx < len(model.config.layer_types):
                is_sliding = (model.config.layer_types[idx] == "sliding_attention")
            rope_theta = float(model.config.rope_local_base_freq) if is_sliding else float(model.config.rope_theta)
            q = apply_rope_torch(qn, rope_theta)
            k = apply_rope_torch(kn, rope_theta)

            # 扩展 KV 到 H
            if k.shape[1] == 1 and H > 1:
                k = k.repeat(1, H, 1, 1)
                v_full = v_heads.repeat(1, H, 1, 1)
            else:
                v_full = v_heads

            # scores + mask
            scores = torch.matmul(q, k.transpose(-1, -2)) * qk_scale
            scores = scores + causal_mask + pad_mask
            probs = torch.softmax(scores, dim=-1)
            save_np(f"attn_scores_l{idx}", scores, dump_dir)
            save_np(f"attn_probs_l{idx}", probs, dump_dir)

            # context -> [B,S,H*D]
            context = torch.matmul(probs, v_full)                      # [B,H,S,D]
            context = context.permute(0, 2, 1, 3).contiguous().view(Bsz, Slen, H * D)
            save_np(f"attn_context_l{idx}", context, dump_dir)

            # o_proj（包含 LoRA）
            attn_out = layers[idx].self_attn.o_proj(context)
            save_np(f"attn_out_raw_l{idx}", attn_out, dump_dir)
            save_np(f"hidden_after_attn_l{idx}", attn_out, dump_dir)

            # post-attn norm + 残差
            post_attn_w = layers[idx].post_attention_layernorm.weight.detach()
            attn_norm = rms_norm_manual(attn_out, post_attn_w, eps)
            save_np(f"hidden_after_attn_norm_l{idx}", attn_norm, dump_dir)
            attn_add = resid_cur + attn_norm
            save_np(f"hidden_after_attn_add_l{idx}", attn_add, dump_dir)

            # pre-FFN norm
            pre_ffn_w = layers[idx].pre_feedforward_layernorm.weight.detach()
            pre_ffn = rms_norm_manual(attn_add, pre_ffn_w, eps)
            save_np(f"hidden_before_mlp_norm_l{idx}", pre_ffn, dump_dir)

            # MLP 手写（无 LoRA 分支）
            X = pre_ffn.reshape(Bsz * Slen, -1).float()
            gate_w = layers[idx].mlp.gate_proj.weight.detach().float()
            up_w   = layers[idx].mlp.up_proj.weight.detach().float()
            down_w = layers[idx].mlp.down_proj.weight.detach().float()
            gate_lin = X @ gate_w.t()
            up_lin   = X @ up_w.t()
            tanh_in = 0.7978845608 * (gate_lin + 0.044715 * gate_lin.pow(3))
            gate_act = 0.5 * gate_lin * (1.0 + tanh_in.tanh())
            prod = gate_act * up_lin
            down_out = prod @ down_w.t()
            gate_lin = gate_lin.reshape(Bsz, Slen, -1).to(dtype=pre_ffn.dtype)
            up_lin   = up_lin.reshape(Bsz, Slen, -1).to(dtype=pre_ffn.dtype)
            prod     = prod.reshape(Bsz, Slen, -1).to(dtype=pre_ffn.dtype)
            down_out = down_out.reshape(Bsz, Slen, pre_ffn.shape[-1]).to(dtype=pre_ffn.dtype)
            save_np(f"gate_proj_out_l{idx}", gate_lin, dump_dir)
            save_np(f"up_proj_out_l{idx}", up_lin, dump_dir)
            save_np(f"mlp_prod_l{idx}",     prod, dump_dir)
            save_np(f"down_proj_out_l{idx}", down_out, dump_dir)
            save_np(f"hidden_after_mlp_l{idx}", down_out, dump_dir)

            post_ffn_w = layers[idx].post_feedforward_layernorm.weight.detach()
            mlp_norm = rms_norm_manual(down_out, post_ffn_w, eps)
            save_np(f"hidden_after_mlp_norm_l{idx}", mlp_norm, dump_dir)

            # 本层输出 -> 作为下一层输入
            final_out = attn_add + mlp_norm
            save_np(f"hidden_states_out_l{idx}", final_out, dump_dir)
            resid_cur = final_out

    # 主 loss：shift 一位后，reduction="sum"
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    vocab_size = shift_logits.size(-1)
    loss = F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="sum",             # ★ 与 C++ sum_debug 对齐
    )

    # 保存 forward 结果
    save_np("logits", logits, dump_dir)
    save_np("loss_scalar", torch.tensor([loss.item()]), dump_dir)
    for name, tensor in activations.items():
        save_np(name, tensor, dump_dir)

    # ===== per-token NLL：保持 reduction="none"，只用于对齐分析 =====
    shift_logits_nll = logits[:, :-1, :].contiguous()
    shift_labels_nll = labels[:, 1:].contiguous()
    vocab_size_nll = shift_logits_nll.size(-1)
    nll = F.cross_entropy(
        shift_logits_nll.view(-1, vocab_size_nll),
        shift_labels_nll.view(-1),
        reduction="none",            # 保持 none
        ignore_index=-100,
    ).view(shift_labels_nll.shape)
    save_np("per_token_nll", nll, dump_dir)

    # backward + 可选一步 AdamW（wd=0.0）
    loss.backward()

    # 补充：根据已dump的 q/k_norm 与 RoPE 参数，显式重建并保存 attn_scores（forward）
    def apply_rope_np(x: torch.Tensor, rope_theta: float) -> torch.Tensor:
        # x: [B,H,S,D]
        b, h, S, D = x.shape
        device = x.device
        dtype = x.dtype
        out = x.clone()
        idx = torch.arange(D // 2, device=device, dtype=dtype)
        freq = 1.0 / (rope_theta ** (2.0 * idx / D))
        for pos in range(S):
            angle = pos * freq
            cosv = torch.cos(angle)
            sinv = torch.sin(angle)
            x1 = out[:, :, pos, 0::2].clone()
            x2 = out[:, :, pos, 1::2].clone()
            out[:, :, pos, 0::2] = x1 * cosv - x2 * sinv
            out[:, :, pos, 1::2] = x1 * sinv + x2 * cosv
        return out
    def apply_rope_inv_np(x: torch.Tensor, rope_theta: float) -> torch.Tensor:
        # inverse rotation: use -sin so R^T = R(-theta)
        b, h, S, D = x.shape
        device = x.device
        dtype = x.dtype
        out = x.clone()
        idx = torch.arange(D // 2, device=device, dtype=dtype)
        freq = 1.0 / (rope_theta ** (2.0 * idx / D))
        for pos in range(S):
            angle = pos * freq
            cosv = torch.cos(angle)
            sinv = torch.sin(angle)
            x1 = out[:, :, pos, 0::2].clone()
            x2 = out[:, :, pos, 1::2].clone()
            # inverse: [x'; y'] = R(-theta)[x; y] = [ x*cos + y*sin; -x*sin + y*cos ]
            out[:, :, pos, 0::2] = x1 * cosv + x2 * sinv
            out[:, :, pos, 1::2] = -x1 * sinv + x2 * cosv
        return out

    def maybe_save_attn_scores(layer_idx: int, is_sliding: bool):
        qn_key = f"q_norm_out_l{layer_idx}"
        kn_key = f"k_norm_out_l{layer_idx}"
        if qn_key in activations_raw and kn_key in activations_raw:
            qn = activations_raw[qn_key]  # [B,Hq,S,D]
            kn = activations_raw[kn_key]  # [B,Hk,S,D]
            rope_theta = model.config.rope_local_base_freq if is_sliding else model.config.rope_theta
            q = apply_rope_np(qn, float(rope_theta))
            k = apply_rope_np(kn, float(rope_theta))
            if k.shape[1] == 1:
                k = k.repeat(1, q.shape[1], 1, 1)
            scores = torch.matmul(q, k.transpose(-1, -2))
            scores = scores * (float(model.config.query_pre_attn_scalar) ** -0.5)
            # 与 C++ 对齐：加上因果与 PAD mask
            scores = scores + build_causal_mask(q.shape[2]) + build_padding_mask(attention_mask)
            save_np(f"attn_scores_l{layer_idx}", scores, dump_dir)

    maybe_save_attn_scores(0, is_sliding=True)
    maybe_save_attn_scores(1, is_sliding=True)
    maybe_save_attn_scores(last_idx, is_sliding=False)

    grads_dir = dump_dir / "grads"
    ensure_dir(grads_dir)
    lora_params = {name: p for name, p in model.named_parameters() if "lora" in name}
    for name, param in lora_params.items():
        if param.grad is not None:
            save_np(f"grads/{name.replace('.', '_')}", param.grad, dump_dir)

    if args.do_adam_step:
        optimizer = torch.optim.AdamW(
            lora_params.values(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        updated_dir = dump_dir / "weights_after_step"
        ensure_dir(updated_dir)
        for name, param in lora_params.items():
            save_np(f"weights_after_step/{name.replace('.', '_')}", param, dump_dir)

    # dump logits.grad 与中间 hidden 的 grad
    if logits.grad is not None:
        save_np("dlogits", logits.grad, dump_dir)
    for key in [
        "hidden_states_emb",
        "hidden_before_attn_l0",
        "hidden_before_attn_l1",
        "hidden_before_attn_l17",
        "hidden_after_attn_l0",
        "hidden_after_attn_l1",
        "hidden_after_attn_l17",
        "hidden_after_mlp_l0",
        "hidden_after_mlp_l1",
        "hidden_after_mlp_l17",
        "hidden_after_attn_norm_l0",
        "hidden_after_attn_norm_l1",
        "hidden_after_attn_norm_l17",
        "hidden_after_mlp_norm_l0",
        "hidden_after_mlp_norm_l1",
        "hidden_after_mlp_norm_l17",
        "q_proj_out_l0",
        "q_proj_out_l1",
        "q_proj_out_l17",
        "k_proj_out_l0",
        "k_proj_out_l1",
        "k_proj_out_l17",
        "v_proj_out_l0",
        "v_proj_out_l1",
        "v_proj_out_l17",
        "hidden_before_mlp_norm_l0",
        "hidden_before_mlp_norm_l1",
        "hidden_before_mlp_norm_l17",
        "gate_proj_out_l0",
        "gate_proj_out_l1",
        "gate_proj_out_l17",
        "up_proj_out_l0",
        "up_proj_out_l1",
        "up_proj_out_l17",
        "mlp_prod_l0",
        "mlp_prod_l1",
        "mlp_prod_l17",
        "down_proj_out_l0",
        "down_proj_out_l1",
        "down_proj_out_l17",
        "q_norm_out_l0",
        "q_norm_out_l1",
        "q_norm_out_l17",
        "k_norm_out_l0",
        "k_norm_out_l1",
        "k_norm_out_l17",
        "attn_context_l0",
        "attn_context_l1",
        "attn_context_l17",
        "attn_out_raw_l0",
        "attn_out_raw_l1",
        "attn_out_raw_l17",
        # === 新增：attn_probs 的梯度 ===
        "attn_probs_l0",
        "attn_probs_l1",
        "attn_probs_l17",
    ]:
        if key in activations_raw and activations_raw[key].grad is not None:
            save_np(f"grads/{key}", activations_raw[key].grad, dump_dir)

    # 额外：基于 attn_probs 与其梯度，显式计算 softmax backward 得到 d(attn_scores)
    def softmax_backward_from_probs(grad_probs: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        # grad_scores = probs * (grad_probs - sum(grad_probs * probs, dim=-1, keepdim=True))
        prod = grad_probs * probs
        sum_prod = prod.sum(dim=-1, keepdim=True)
        return probs * (grad_probs - sum_prod)

    for idx in [0, 1, last_idx]:
        # 基于保存的前向与 G=grad(attn_context) 自行重建 probs、gprobs，并解析出 d_scores/d_mat/dq/dk/dv
        ctx_key = f"attn_context_l{idx}"
        qn_key = f"q_norm_out_l{idx}"
        kn_key = f"k_norm_out_l{idx}"
        vn_key = f"v_proj_out_l{idx}"
        if ctx_key in activations_raw and activations_raw[ctx_key].grad is not None \
           and qn_key in activations_raw and kn_key in activations_raw and vn_key in activations_raw:
            G = activations_raw[ctx_key].grad.detach().float()
            qn = activations_raw[qn_key].detach()
            kn = activations_raw[kn_key].detach()
            vn = activations_raw[vn_key].detach()                     # [B,S,D] (kvH=1)
            # 选择 RoPE 频率
            is_sliding = False
            if hasattr(base.config, "layer_types") and idx < len(base.config.layer_types):
                is_sliding = (base.config.layer_types[idx] == "sliding_attention")
            rope_theta = float(base.config.rope_local_base_freq) if is_sliding else float(base.config.rope_theta)
            # 应用 RoPE 到 q/k
            q = apply_rope_np(qn, rope_theta).float()                 # [B,H,S,D]
            k = apply_rope_np(kn, rope_theta).float()                 # [B,Hk,S,D]
            # 扩展 K/V 到 H
            H = base.config.num_attention_heads
            if k.shape[1] == 1 and H > 1:
                k = k.repeat(1, H, 1, 1)                              # [B,H,S,D]
            v_full = vn.view(vn.shape[0], 1, vn.shape[1], vn.shape[2]).repeat(1, H, 1, 1)  # [B,H,S,D]
            # 重建 scores/probs（与 C++ 一致）
            qk_scale = float(base.config.query_pre_attn_scalar) ** -0.5
            S = q.shape[2]
            scores = torch.matmul(q, k.transpose(-1, -2)) * qk_scale
            scores = scores + build_causal_mask(S) + build_padding_mask(attention_mask)
            probs = torch.softmax(scores, dim=-1).float()             # [B,H,S,S]
            # 还原 G 到 head 形态
            B, S, hidden = G.shape
            D = q.shape[-1]
            Hcfg = q.shape[1]
            G_heads = G.view(B, S, Hcfg, D).permute(0, 2, 1, 3).contiguous()  # [B,H,S,D]
            # gprobs = G_heads @ V^T
            gprobs = torch.matmul(G_heads, v_full.transpose(-1, -2))  # [B,H,S,S]
            # d_scores (softmax backward from probs, gprobs)
            d_scores = softmax_backward_from_probs(gprobs, probs)
            save_np(f"grads/attn_scores_l{idx}", d_scores, dump_dir)
            # d_mat = d_scores * qk_scale
            d_mat = d_scores * qk_scale
            save_np(f"grads/attn_dmat_l{idx}", d_mat, dump_dir)
            # dQ, dK, dV（head 形态）
            dQ = torch.matmul(d_mat, k)                               # [B,H,S,D]
            dK = torch.matmul(d_mat.transpose(-2, -1), q)             # [B,H,S,D]
            dV = torch.matmul(probs.transpose(-2, -1), G_heads)       # [B,H,S,D]
            save_np(f"grads/analytical_dq_heads_l{idx}", dQ, dump_dir)
            save_np(f"grads/analytical_dk_heads_l{idx}", dK, dump_dir)
            save_np(f"grads/analytical_dv_heads_l{idx}", dV, dump_dir)
            # 还原到 pre-RoPE 坐标系以便与 C++ 的 q_norm_out/k_norm_out.grad 对齐
            dQ_pre = apply_rope_inv_np(dQ, rope_theta)                 # [B,H,S,D]
            dK_pre = apply_rope_inv_np(dK, rope_theta)                 # [B,H,S,D]
            save_np(f"grads/analytical_dq_pre_l{idx}", dQ_pre, dump_dir)
            save_np(f"grads/analytical_dk_pre_l{idx}", dK_pre, dump_dir)

    print(f"[done] dumps saved to {dump_dir}")


if __name__ == "__main__":
    main()
