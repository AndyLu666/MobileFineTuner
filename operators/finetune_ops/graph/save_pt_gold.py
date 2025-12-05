#!/usr/bin/env python3
"""
生成 PyTorch/HuggingFace GPT-2 的金标 logits（用于 C++ 前向对齐）
"""

import torch
import json
from transformers import GPT2TokenizerFast, GPT2LMHeadModel

def main():
    # 加载预训练模型（禁用 dropout）
    tok = GPT2TokenizerFast.from_pretrained("/Users/tony/Documents/重新开始/gpt2_lora_finetune/pretrained/gpt2")
    model = GPT2LMHeadModel.from_pretrained("/Users/tony/Documents/重新开始/gpt2_lora_finetune/pretrained/gpt2")
    model.eval()
    
    # 固定输入
    text = "Hello, world!\n"
    x = tok(text, return_tensors="pt")
    
    print(f"Input text: {repr(text)}")
    print(f"Input IDs: {x['input_ids'].tolist()}")
    print(f"Attention mask: {x['attention_mask'].tolist()}")
    
    # 前向（禁用 dropout）
    with torch.no_grad():
        logits = model(**x).logits  # [1, S, V]
    
    # 保存末位 token 的 logits
    last_logits = logits[0, -1].float().cpu()  # [V]
    
    # Top-5
    topv, topi = torch.topk(last_logits, 5)
    print(f"\nPyTorch top-5 IDs:  {topi.tolist()}")
    print(f"PyTorch top-5 vals: {[f'{v:.6f}' for v in topv.tolist()]}")
    
    # 保存为 JSON（便于 C++ 读取）
    output_path = "/Users/tony/Documents/重新开始/operators/finetune_ops/graph/pt_last_logits.json"
    with open(output_path, "w") as f:
        json.dump(last_logits.tolist(), f)
    
    print(f"\nSaved PyTorch logits to: {output_path}")
    print(f"Logits shape: {last_logits.shape}, dtype: {last_logits.dtype}")
    print(f"Argmax: {last_logits.argmax().item()}")

if __name__ == "__main__":
    main()

