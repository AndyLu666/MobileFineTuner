#!/usr/bin/env python3
"""
Export Gemma 270M pretrained weights to C++ readable binary format
"""

import torch
import numpy as np
import struct
import os
import json
from transformers import GemmaTokenizer, GemmaForCausalLM

def export_real_gemma_weights(model, tokenizer):
    """Export real Gemma 3 270M weights"""
    print("Starting to export real Gemma 3 270M weights...")
    
    os.makedirs('models/gemma-270m/exported', exist_ok=True)
    
    # Get model configuration
    config = model.config
    print(f"Gemma 3 270M actual configuration:")
    print(f"  - Hidden size: {config.hidden_size}")
    print(f"  - Attention heads: {config.num_attention_heads}")
    print(f"  - KV heads: {config.num_key_value_heads}")
    print(f"  - Layers: {config.num_hidden_layers}")
    print(f"  - Intermediate size: {config.intermediate_size}")
    print(f"  - Vocabulary size: {config.vocab_size}")
    
    # Export embedding layer
    print("Exporting embedding layer weights...")
    wte = model.model.embed_tokens.weight.detach().cpu().numpy().astype(np.float32)
    print(f"Token embedding: {wte.shape}")
    with open('models/gemma-270m/exported/wte.bin', 'wb') as f:
        f.write(struct.pack('I', len(wte.shape)))
        f.write(struct.pack('II', *wte.shape))
        f.write(wte.tobytes())
    
    # Export language model head (Gemma usually shares weights with embedding layer)
    if hasattr(model, 'lm_head') and model.lm_head is not None:
        lm_head = model.lm_head.weight.detach().cpu().numpy().astype(np.float32)
    else:
        lm_head = wte  # Weight sharing
    print(f"Language model head: {lm_head.shape}")
    with open('models/gemma-270m/exported/lm_head.bin', 'wb') as f:
        f.write(struct.pack('I', len(lm_head.shape)))
        f.write(struct.pack('II', *lm_head.shape))
        f.write(lm_head.tobytes())
    
    # Export layer 0 Transformer weights (for single layer fine-tuning)
    layer_0 = model.model.layers[0]
    
    # Attention weights
    print("Exporting layer 0 Attention weights...")
    
    q_weight = layer_0.self_attn.q_proj.weight.detach().cpu().numpy().astype(np.float32)
    k_weight = layer_0.self_attn.k_proj.weight.detach().cpu().numpy().astype(np.float32)
    v_weight = layer_0.self_attn.v_proj.weight.detach().cpu().numpy().astype(np.float32)
    o_weight = layer_0.self_attn.o_proj.weight.detach().cpu().numpy().astype(np.float32)
    
    print(f"Q weight: {q_weight.shape}")
    print(f"K weight: {k_weight.shape}")
    print(f"V weight: {v_weight.shape}")
    print(f"O weight: {o_weight.shape}")
    
    for name, weight in [('q_weight', q_weight), ('k_weight', k_weight), 
                        ('v_weight', v_weight), ('o_weight', o_weight)]:
        with open(f'models/gemma-270m/exported/{name}.bin', 'wb') as f:
            f.write(struct.pack('I', len(weight.shape)))
            f.write(struct.pack('II', *weight.shape))
            f.write(weight.tobytes())
    
    # MLP weights (Gemma uses GeGLU)
    print("Exporting layer 0 MLP weights...")
    
    gate_weight = layer_0.mlp.gate_proj.weight.detach().cpu().numpy().astype(np.float32)
    up_weight = layer_0.mlp.up_proj.weight.detach().cpu().numpy().astype(np.float32)
    down_weight = layer_0.mlp.down_proj.weight.detach().cpu().numpy().astype(np.float32)
    
    print(f"Gate weight: {gate_weight.shape}")
    print(f"Up weight: {up_weight.shape}")
    print(f"Down weight: {down_weight.shape}")
    
    for name, weight in [('gate_weight', gate_weight), ('up_weight', up_weight), 
                        ('down_weight', down_weight)]:
        with open(f'models/gemma-270m/exported/{name}.bin', 'wb') as f:
            f.write(struct.pack('I', len(weight.shape)))
            f.write(struct.pack('II', *weight.shape))
            f.write(weight.tobytes())
    
    # RMSNorm weights
    print("Exporting RMSNorm weights...")
    
    rms_attn_weight = layer_0.input_layernorm.weight.detach().cpu().numpy().astype(np.float32)
    rms_ffn_weight = layer_0.post_attention_layernorm.weight.detach().cpu().numpy().astype(np.float32)
    
    print(f"Pre-attention RMSNorm: {rms_attn_weight.shape}")
    print(f"Pre-FFN RMSNorm: {rms_ffn_weight.shape}")
    
    for name, weight in [('rms_attn_weight', rms_attn_weight), 
                        ('rms_ffn_weight', rms_ffn_weight)]:
        with open(f'models/gemma-270m/exported/{name}.bin', 'wb') as f:
            f.write(struct.pack('I', len(weight.shape)))
            f.write(struct.pack('I', len(weight)))
            f.write(weight.tobytes())
    
    # Final RMSNorm
    rms_final_weight = model.model.norm.weight.detach().cpu().numpy().astype(np.float32)
    print(f"Final RMSNorm: {rms_final_weight.shape}")
    
    with open('models/gemma-270m/exported/rms_final_weight.bin', 'wb') as f:
        f.write(struct.pack('I', len(rms_final_weight.shape)))
        f.write(struct.pack('I', len(rms_final_weight)))
        f.write(rms_final_weight.tobytes())
    
    # Export tokenizer vocabulary
    print("Exporting tokenizer vocabulary...")
    vocab = tokenizer.get_vocab()
    vocab_json = {token: id for token, id in vocab.items()}
    
    with open('models/gemma-270m/vocab.json', 'w', encoding='utf-8') as f:
        json.dump(vocab_json, f, ensure_ascii=False, indent=2)
    
    print("Real Gemma 3 270M weight export complete!")
    print("Exported files:")
    for file in sorted(os.listdir('models/gemma-270m/exported')):
        size = os.path.getsize(f'models/gemma-270m/exported/{file}')
        print(f"  {file}: {size/1024/1024:.2f} MB")
    
    # Update C++ code configuration info
    print(f"\nPlease update configuration in C++ code:")
    print(f"  n_embd = {config.hidden_size}")
    print(f"  n_head = {config.num_attention_heads}")
    print(f"  n_kv_head = {config.num_key_value_heads}")
    print(f"  n_inner = {config.intermediate_size}")
    print(f"  vocab_size = {config.vocab_size}")

def export_weights():
    print("Loading Gemma 3 270M pretrained weights...")
    print("Using model: https://huggingface.co/google/gemma-3-270m")
    
    model_name = "google/gemma-3-270m"
    
    try:
        print("Downloading Gemma tokenizer...")
        tokenizer = GemmaTokenizer.from_pretrained(model_name)
        
        print("Downloading Gemma model weights (this may take a few minutes)...")
        try:
            # First try to use device_map (requires accelerate library)
            model = GemmaForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=torch.float32,
                device_map="cpu"  # Force CPU to avoid CUDA issues
            )
        except Exception as device_map_error:
            print(f"Warning: device_map method failed: {device_map_error}")
            print("Trying to load directly to CPU...")
            # Fallback to direct CPU loading
            model = GemmaForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=torch.float32
            )
            model = model.to('cpu')
        print(f"Successfully loaded Gemma model: {model_name}")
        
        # Export real Gemma weights
        export_real_gemma_weights(model, tokenizer)
        
    except Exception as e:
        print(f"Unable to load pretrained model: {e}")
        print("Possible reasons:")
        print("  1. Network connection issues")
        print("  2. HuggingFace token not set")
        print("  3. Access permission issues")
        print("Falling back to creating dummy weights...")
        create_dummy_weights()
        return

def create_dummy_weights():
    """Create dummy weight files for testing"""
    print("Creating Gemma 3 270M dummy weight files...")
    
    os.makedirs('models/gemma-270m/exported', exist_ok=True)
    
    # Gemma 3 270M actual configuration
    n_embd = 640
    n_head = 4
    n_kv_head = 1
    n_inner = 2048
    vocab_size = 262144
    head_dim = n_embd // n_head
    
    # Create dummy weights
    np.random.seed(42)  # Ensure reproducibility
    
    # Embedding layer
    wte = np.random.normal(0, 0.02, (vocab_size, n_embd)).astype(np.float32)
    with open('models/gemma-270m/exported/wte.bin', 'wb') as f:
        f.write(struct.pack('I', 2))
        f.write(struct.pack('II', vocab_size, n_embd))
        f.write(wte.tobytes())
    
    # Language model head (weight sharing)
    with open('models/gemma-270m/exported/lm_head.bin', 'wb') as f:
        f.write(struct.pack('I', 2))
        f.write(struct.pack('II', vocab_size, n_embd))
        f.write(wte.tobytes())
    
    # Attention weights
    q_weight = np.random.normal(0, 0.02, (n_embd, n_embd)).astype(np.float32)
    k_weight = np.random.normal(0, 0.02, (n_kv_head * head_dim, n_embd)).astype(np.float32)
    v_weight = np.random.normal(0, 0.02, (n_kv_head * head_dim, n_embd)).astype(np.float32)
    o_weight = np.random.normal(0, 0.02, (n_embd, n_embd)).astype(np.float32)
    
    for name, weight in [('q_weight', q_weight), ('k_weight', k_weight), 
                        ('v_weight', v_weight), ('o_weight', o_weight)]:
        with open(f'models/gemma-270m/exported/{name}.bin', 'wb') as f:
            f.write(struct.pack('I', 2))
            f.write(struct.pack('II', *weight.shape))
            f.write(weight.tobytes())
    
    # MLP weights
    gate_weight = np.random.normal(0, 0.02, (n_inner, n_embd)).astype(np.float32)
    up_weight = np.random.normal(0, 0.02, (n_inner, n_embd)).astype(np.float32)
    down_weight = np.random.normal(0, 0.02, (n_embd, n_inner)).astype(np.float32)
    
    for name, weight in [('gate_weight', gate_weight), ('up_weight', up_weight), 
                        ('down_weight', down_weight)]:
        with open(f'models/gemma-270m/exported/{name}.bin', 'wb') as f:
            f.write(struct.pack('I', 2))
            f.write(struct.pack('II', *weight.shape))
            f.write(weight.tobytes())
    
    # RMSNorm weights
    rms_attn_weight = np.ones(n_embd).astype(np.float32)
    rms_ffn_weight = np.ones(n_embd).astype(np.float32)
    rms_final_weight = np.ones(n_embd).astype(np.float32)
    
    for name, weight in [('rms_attn_weight', rms_attn_weight), 
                        ('rms_ffn_weight', rms_ffn_weight),
                        ('rms_final_weight', rms_final_weight)]:
        with open(f'models/gemma-270m/exported/{name}.bin', 'wb') as f:
            f.write(struct.pack('I', 1))
            f.write(struct.pack('I', len(weight)))
            f.write(weight.tobytes())
    
    print("Gemma 3 270M dummy weight files created!")
    print("Note: These are randomly initialized weights, only for testing code structure")
    print(f"Configuration parameters: n_embd={n_embd}, n_head={n_head}, n_inner={n_inner}, vocab_size={vocab_size}")

if __name__ == "__main__":
    export_weights()
