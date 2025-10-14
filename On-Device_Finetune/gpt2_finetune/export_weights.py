#!/usr/bin/env python3
"""
Export GPT-2 pretrained weights to C++ readable binary format (supports pytorch_model.bin or model.safetensors)
"""

import os
import struct
import numpy as np

try:
    import torch  # optional
except Exception:
    torch = None

try:
    from safetensors.numpy import load_file as st_load_np  # no torch required
except Exception:
    st_load_np = None

def _load_weights_dict():
    bin_path = 'models/gpt2/pytorch_model.bin'
    st_path = 'models/gpt2/model.safetensors'
    if os.path.exists(bin_path):
        if torch is None:
            raise RuntimeError("torch is required to read pytorch_model.bin, please install torch or use model.safetensors instead")
        print("Loading pytorch_model.bin ...")
        return torch.load(bin_path, map_location='cpu')
    if os.path.exists(st_path):
        if st_load_np is None:
            raise RuntimeError("safetensors is missing, please: pip install safetensors")
        print("Loading model.safetensors ...")
        return st_load_np(st_path)  # returns numpy array dict
    raise FileNotFoundError("Model weights not found: models/gpt2/pytorch_model.bin or models/gpt2/model.safetensors")


def _to_numpy(arr):
    if hasattr(arr, 'numpy'):
        return arr.numpy().astype(np.float32)
    if isinstance(arr, np.ndarray):
        return arr.astype(np.float32)
    # safetensors.numpy already returns numpy arrays; other cases force conversion
    return np.array(arr, dtype=np.float32)


def _get(weights, key):
    # Compatible with HuggingFace transformer.* prefix
    if key in weights:
        return weights[key]
    tkey = f"transformer.{key}"
    if tkey in weights:
        return weights[tkey]
    raise KeyError(f"Weight key not found: {key} / {tkey}")


def export_weights():
    print("Loading GPT-2 pretrained weights...")
    weights = _load_weights_dict()
    
    os.makedirs('models/gpt2/exported', exist_ok=True)
    
    wte = _to_numpy(_get(weights, 'wte.weight'))
    print(f"Token embedding: {wte.shape}")
    with open('models/gpt2/exported/wte.bin', 'wb') as f:
        f.write(struct.pack('I', len(wte.shape)))
        f.write(struct.pack('II', *wte.shape))
        f.write(wte.tobytes())
    
    wpe = _to_numpy(_get(weights, 'wpe.weight'))
    print(f"Position embedding: {wpe.shape}")
    with open('models/gpt2/exported/wpe.bin', 'wb') as f:
        f.write(struct.pack('I', len(wpe.shape)))
        f.write(struct.pack('II', *wpe.shape))
        f.write(wpe.tobytes())
    
    if 'lm_head.weight' in weights or 'transformer.lm_head.weight' in weights:
        lm_head = _to_numpy(_get(weights, 'lm_head.weight'))
    else:
        lm_head = _to_numpy(_get(weights, 'wte.weight'))
        print("Warning: Using wte weights as lm_head (weight sharing)")
    
    print(f"Language model head: {lm_head.shape}")
    with open('models/gpt2/exported/lm_head.bin', 'wb') as f:
        f.write(struct.pack('I', len(lm_head.shape)))
        f.write(struct.pack('II', *lm_head.shape))
        f.write(lm_head.tobytes())
    
    # Export weights by layer: supports standard multi-layer fine-tuning (e.g., only last 4-6 layers with LoRA)
    print("Counting Transformer layers and exporting by layer...")
    def _all_keys():
        return list(weights.keys())
    def _is_layer_key(k):
        return k.startswith('h.') or k.startswith('transformer.h.')
    def _layer_index(k):
        parts = k.split('.')
        # e.g., ['h','0','attn',...] or ['transformer','h','0','attn',...]
        if parts[0] == 'h':
            return int(parts[1])
        if parts[0] == 'transformer' and parts[1] == 'h':
            return int(parts[2])
        raise ValueError(k)

    layer_indices = sorted({ _layer_index(k) for k in _all_keys() if _is_layer_key(k) })
    print(f"Detected layers: {len(layer_indices)} layers -> {layer_indices}")

    for i in layer_indices:
        # Attention c_attn: (hidden, 3*hidden) and bias (3*hidden)
        attn_w = _to_numpy(_get(weights, f'h.{i}.attn.c_attn.weight'))
        attn_b = _to_numpy(_get(weights, f'h.{i}.attn.c_attn.bias'))
        proj_w = _to_numpy(_get(weights, f'h.{i}.attn.c_proj.weight'))
        proj_b = _to_numpy(_get(weights, f'h.{i}.attn.c_proj.bias'))

        # Split QKV
        hidden = attn_w.shape[0]
        qkv_w = attn_w.reshape(hidden, 3, hidden)
        q_w = qkv_w[:, 0, :].T
        k_w = qkv_w[:, 1, :].T
        v_w = qkv_w[:, 2, :].T

        # Bias not used for attention in current C++ path; keep only weights
        with open(f'models/gpt2/exported/h.{i}.q_weight.bin', 'wb') as f:
            f.write(struct.pack('I', len(q_w.shape)))
            f.write(struct.pack('II', *q_w.shape))
            f.write(q_w.tobytes())
        with open(f'models/gpt2/exported/h.{i}.k_weight.bin', 'wb') as f:
            f.write(struct.pack('I', len(k_w.shape)))
            f.write(struct.pack('II', *k_w.shape))
            f.write(k_w.tobytes())
        with open(f'models/gpt2/exported/h.{i}.v_weight.bin', 'wb') as f:
            f.write(struct.pack('I', len(v_w.shape)))
            f.write(struct.pack('II', *v_w.shape))
            f.write(v_w.tobytes())
        with open(f'models/gpt2/exported/h.{i}.o_weight.bin', 'wb') as f:
            pw_t = proj_w.T
            f.write(struct.pack('I', len(pw_t.shape)))
            f.write(struct.pack('II', *pw_t.shape))
            f.write(pw_t.tobytes())

        # LayerNorms
        ln1_w = _to_numpy(_get(weights, f'h.{i}.ln_1.weight'))
        ln1_b = _to_numpy(_get(weights, f'h.{i}.ln_1.bias'))
        with open(f'models/gpt2/exported/h.{i}.ln1_weight.bin', 'wb') as f:
            f.write(struct.pack('I', len(ln1_w.shape)))
            f.write(struct.pack('I', *ln1_w.shape))
            f.write(ln1_w.tobytes())
        with open(f'models/gpt2/exported/h.{i}.ln1_bias.bin', 'wb') as f:
            f.write(struct.pack('I', len(ln1_b.shape)))
            f.write(struct.pack('I', *ln1_b.shape))
            f.write(ln1_b.tobytes())

        ln2_w = _to_numpy(_get(weights, f'h.{i}.ln_2.weight'))
        ln2_b = _to_numpy(_get(weights, f'h.{i}.ln_2.bias'))
        with open(f'models/gpt2/exported/h.{i}.ln2_weight.bin', 'wb') as f:
            f.write(struct.pack('I', len(ln2_w.shape)))
            f.write(struct.pack('I', *ln2_w.shape))
            f.write(ln2_w.tobytes())
        with open(f'models/gpt2/exported/h.{i}.ln2_bias.bin', 'wb') as f:
            f.write(struct.pack('I', len(ln2_b.shape)))
            f.write(struct.pack('I', *ln2_b.shape))
            f.write(ln2_b.tobytes())

        # MLP
        mlp_fc_w = _to_numpy(_get(weights, f'h.{i}.mlp.c_fc.weight'))
        mlp_fc_b = _to_numpy(_get(weights, f'h.{i}.mlp.c_fc.bias'))
        with open(f'models/gpt2/exported/h.{i}.mlp_fc_weight.bin', 'wb') as f:
            f.write(struct.pack('I', len(mlp_fc_w.shape)))
            f.write(struct.pack('II', *mlp_fc_w.shape))
            f.write(mlp_fc_w.tobytes())
        with open(f'models/gpt2/exported/h.{i}.mlp_fc_bias.bin', 'wb') as f:
            f.write(struct.pack('I', len(mlp_fc_b.shape)))
            f.write(struct.pack('I', *mlp_fc_b.shape))
            f.write(mlp_fc_b.tobytes())

        mlp_proj_w = _to_numpy(_get(weights, f'h.{i}.mlp.c_proj.weight'))
        mlp_proj_b = _to_numpy(_get(weights, f'h.{i}.mlp.c_proj.bias'))
        with open(f'models/gpt2/exported/h.{i}.mlp_proj_weight.bin', 'wb') as f:
            f.write(struct.pack('I', len(mlp_proj_w.shape)))
            f.write(struct.pack('II', *mlp_proj_w.shape))
            f.write(mlp_proj_w.tobytes())
        with open(f'models/gpt2/exported/h.{i}.mlp_proj_bias.bin', 'wb') as f:
            f.write(struct.pack('I', len(mlp_proj_b.shape)))
            f.write(struct.pack('I', *mlp_proj_b.shape))
            f.write(mlp_proj_b.tobytes())

    # Backward compatibility: still write layer 0 files with old names for legacy C++ paths
    layer_0_attn_weight = _to_numpy(_get(weights, 'h.0.attn.c_attn.weight'))
    layer_0_proj_weight = _to_numpy(_get(weights, 'h.0.attn.c_proj.weight'))
    qkv_weight = layer_0_attn_weight.reshape(768, 3, 768)
    q_weight = qkv_weight[:, 0, :].T
    k_weight = qkv_weight[:, 1, :].T
    v_weight = qkv_weight[:, 2, :].T
    with open('models/gpt2/exported/q_weight.bin', 'wb') as f:
        f.write(struct.pack('I', len(q_weight.shape)))
        f.write(struct.pack('II', *q_weight.shape))
        f.write(q_weight.tobytes())
    with open('models/gpt2/exported/k_weight.bin', 'wb') as f:
        f.write(struct.pack('I', len(k_weight.shape)))
        f.write(struct.pack('II', *k_weight.shape))
        f.write(k_weight.tobytes())
    with open('models/gpt2/exported/v_weight.bin', 'wb') as f:
        f.write(struct.pack('I', len(v_weight.shape)))
        f.write(struct.pack('II', *v_weight.shape))
        f.write(v_weight.tobytes())
    with open('models/gpt2/exported/o_weight.bin', 'wb') as f:
        pw_t = layer_0_proj_weight.T
        f.write(struct.pack('I', len(pw_t.shape)))
        f.write(struct.pack('II', *pw_t.shape))
        f.write(pw_t.tobytes())
    
    print("Exporting LayerNorm weights...")
    
    ln1_weight = _to_numpy(_get(weights, 'h.0.ln_1.weight'))
    ln1_bias = _to_numpy(_get(weights, 'h.0.ln_1.bias'))
    print(f"Pre-attention LayerNorm: weight{ln1_weight.shape}, bias{ln1_bias.shape}")
    
    with open('models/gpt2/exported/ln1_weight.bin', 'wb') as f:
        f.write(struct.pack('I', len(ln1_weight.shape)))
        f.write(struct.pack('I', *ln1_weight.shape))
        f.write(ln1_weight.tobytes())
    
    with open('models/gpt2/exported/ln1_bias.bin', 'wb') as f:
        f.write(struct.pack('I', len(ln1_bias.shape)))
        f.write(struct.pack('I', *ln1_bias.shape))
        f.write(ln1_bias.tobytes())
    
    ln2_weight = _to_numpy(_get(weights, 'h.0.ln_2.weight'))
    ln2_bias = _to_numpy(_get(weights, 'h.0.ln_2.bias'))
    print(f"Pre-MLP LayerNorm: weight{ln2_weight.shape}, bias{ln2_bias.shape}")
    
    with open('models/gpt2/exported/ln2_weight.bin', 'wb') as f:
        f.write(struct.pack('I', len(ln2_weight.shape)))
        f.write(struct.pack('I', *ln2_weight.shape))
        f.write(ln2_weight.tobytes())
    
    with open('models/gpt2/exported/ln2_bias.bin', 'wb') as f:
        f.write(struct.pack('I', len(ln2_bias.shape)))
        f.write(struct.pack('I', *ln2_bias.shape))
        f.write(ln2_bias.tobytes())
    
    ln_f_weight = _to_numpy(_get(weights, 'ln_f.weight'))
    ln_f_bias = _to_numpy(_get(weights, 'ln_f.bias'))
    print(f"Final LayerNorm: weight{ln_f_weight.shape}, bias{ln_f_bias.shape}")
    
    with open('models/gpt2/exported/ln_f_weight.bin', 'wb') as f:
        f.write(struct.pack('I', len(ln_f_weight.shape)))
        f.write(struct.pack('I', *ln_f_weight.shape))
        f.write(ln_f_weight.tobytes())
    
    with open('models/gpt2/exported/ln_f_bias.bin', 'wb') as f:
        f.write(struct.pack('I', len(ln_f_bias.shape)))
        f.write(struct.pack('I', *ln_f_bias.shape))
        f.write(ln_f_bias.tobytes())
    
    print("Exporting MLP weights...")
    
    mlp_fc_weight = _to_numpy(_get(weights, 'h.0.mlp.c_fc.weight'))
    mlp_fc_bias = _to_numpy(_get(weights, 'h.0.mlp.c_fc.bias'))
    print(f"MLP first layer: weight{mlp_fc_weight.shape}, bias{mlp_fc_bias.shape}")
    
    with open('models/gpt2/exported/mlp_fc_weight.bin', 'wb') as f:
        f.write(struct.pack('I', len(mlp_fc_weight.shape)))
        f.write(struct.pack('II', *mlp_fc_weight.shape))
        f.write(mlp_fc_weight.tobytes())
    
    with open('models/gpt2/exported/mlp_fc_bias.bin', 'wb') as f:
        f.write(struct.pack('I', len(mlp_fc_bias.shape)))
        f.write(struct.pack('I', *mlp_fc_bias.shape))
        f.write(mlp_fc_bias.tobytes())
    
    mlp_proj_weight = _to_numpy(_get(weights, 'h.0.mlp.c_proj.weight'))
    mlp_proj_bias = _to_numpy(_get(weights, 'h.0.mlp.c_proj.bias'))
    print(f"MLP second layer: weight{mlp_proj_weight.shape}, bias{mlp_proj_bias.shape}")
    
    with open('models/gpt2/exported/mlp_proj_weight.bin', 'wb') as f:
        f.write(struct.pack('I', len(mlp_proj_weight.shape)))
        f.write(struct.pack('II', *mlp_proj_weight.shape))
        f.write(mlp_proj_weight.tobytes())
    
    with open('models/gpt2/exported/mlp_proj_bias.bin', 'wb') as f:
        f.write(struct.pack('I', len(mlp_proj_bias.shape)))
        f.write(struct.pack('I', *mlp_proj_bias.shape))
        f.write(mlp_proj_bias.tobytes())
    
    print("Complete weight export finished!")
    print("Exported files:")
    for file in sorted(os.listdir('models/gpt2/exported')):
        size = os.path.getsize(f'models/gpt2/exported/{file}')
        print(f"  {file}: {size/1024/1024:.2f} MB")

if __name__ == "__main__":
    export_weights()
