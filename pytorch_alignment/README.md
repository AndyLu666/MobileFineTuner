# PyTorch Alignment Baselines

This directory contains PyTorch reference implementations that mirror the C++ training pipelines for numerical validation and comparison.

## Available Scripts

### GPT-2 LoRA Fine-tuning
**File:** `gpt2_lora_finetune.py`  
**Mirrors:** `gpt2_lora_finetune/main.cpp`

PyTorch + PEFT implementation of GPT-2 LoRA training for validation against the C++ implementation.

**Usage:**
```bash
python pytorch_alignment/gpt2_lora_finetune.py \
  --data_dir data/wikitext2/wikitext-2-raw \
  --pretrained_dir gpt2_lora_finetune/pretrained/gpt2 \
  --lora_out pytorch_runs/gpt2_lora \
  --epochs 1 --batch_size 8 --grad_accum_steps 1 --seq_len 128 \
  --rank 8 --alpha 16 --lr 2e-4 --lora_dropout 0.0
```

**For GPT-2 Medium:**
```bash
python pytorch_alignment/gpt2_lora_finetune.py \
  --pretrained_dir gpt2_lora_finetune/pretrained/gpt2-medium \
  --lora_out pytorch_runs/gpt2_medium_lora \
  --batch_size 4 --epochs 1
```

---

### GPT-2 Full Fine-tuning
**File:** `gpt2_full_finetune.py`  
**Purpose:** Baseline for full parameter fine-tuning comparison

**Usage:**
```bash
python pytorch_alignment/gpt2_full_finetune.py \
  --data_dir data/wikitext2/wikitext-2-raw \
  --pretrained_dir gpt2_lora_finetune/pretrained/gpt2 \
  --output_dir pytorch_runs/gpt2_full \
  --epochs 1 --batch_size 4 --seq_len 128
```

---

### Gemma LoRA Fine-tuning
**File:** `gemma_lora_finetune.py`  
**Mirrors:** `operators/finetune_ops/optim/train_lora_gemma.cpp`

PyTorch + PEFT implementation of Gemma LoRA training.

**Usage:**
```bash
python pytorch_alignment/gemma_lora_finetune.py \
  --model_dir gemma-3-270m \
  --data_dir data/wikitext2/wikitext-2-raw \
  --output_dir pytorch_runs/gemma_270m_lora \
  --epochs 1 --batch 4 --grad_accum 1 --seq_len 256 \
  --learning_rate 2e-4 --warmup_ratio 0.03 \
  --target_mode full --lora_r 8 --lora_alpha 32 \
  --torch_dtype float32
```

**For Gemma 1B:**
```bash
python pytorch_alignment/gemma_lora_finetune.py \
  --model_dir gemma-3-1b \
  --batch 2 --grad_accum 2 \
  --output_dir pytorch_runs/gemma_1b_lora
```

---

### Qwen2.5-0.5B LoRA Fine-tuning
**File:** `qwen_lora_finetune.py`  
**Mirrors:** `qwen_lora_finetune/src/wikitext_main.cpp` and `src/mmlu_main.cpp`

Supports WikiText-2 (default) and MMLU JSONL (ids/mask) for masked causal LM training.

**Usage (WikiText-2):**
```bash
python pytorch_alignment/qwen_lora_finetune.py \
  --model_dir Qwen2.5-0.5B \
  --data_dir data/wikitext2/wikitext-2-raw \
  --output_dir pytorch_runs/qwen_wikitext_lora \
  --epochs 1 --batch 1 --seq_len 1024 \
  --lora_r 8 --lora_alpha 16 --lora_dropout 0.05
```

**Usage (MMLU JSONL, ids/mask):**
```bash
python pytorch_alignment/qwen_lora_finetune.py \
  --model_dir Qwen2.5-0.5B \
  --jsonl_train runs/mmlu_jsonl_qwen/train.jsonl \
  --jsonl_valid runs/mmlu_jsonl_qwen/valid.jsonl \
  --seq_len 128 --batch 8 --grad_accum 1 \
  --output_dir pytorch_runs/qwen_mmlu_lora \
  --target_mode qv  # q,v only (matches C++ default)
```

---

## Configuration Alignment

All scripts are configured to match C++ implementations:

| Parameter | GPT-2 | Gemma |
|-----------|-------|-------|
| **LoRA Rank** | 8 | 8 |
| **LoRA Alpha** | 16 | 32 |
| **LoRA Targets** | AttnQKV, AttnProj | q_proj, k_proj, v_proj, o_proj |
| **Optimizer** | AdamW | AdamW |
| **LR Schedule** | Warmup + Cosine | Warmup + Linear/Cosine |
| **Gradient Clip** | 1.0 | 1.0 |
| **Data Format** | Raw text files | Raw text files |

---

## Key Features

### Identical Training Flow
1. Load pretrained weights from local directory
2. Inject LoRA adapters using PEFT
3. Freeze base parameters
4. Train on WikiText-2 with identical tokenization
5. Save LoRA adapters in SafeTensors format

### Gradient Accumulation
Both implementations support gradient accumulation with proper scaling:
```
effective_batch_size = batch_size × grad_accum_steps
```

### Learning Rate Scheduling
- **Warmup:** Linear warmup for first N% of steps
- **Decay:** Cosine decay to 10% of base LR (GPT-2) or linear (Gemma)

### Loss Computation
Uses the same shifted cross-entropy:
- Logits: `[:, :-1, :]`
- Labels: `[:, 1:]`
- Matches C++ `lm_cross_entropy` implementation

---

## Validation Use Cases

### 1. Numerical Alignment Check
Compare loss curves and final adapters:
```bash
# Run C++ version
./build/gpt2_lora_finetune --steps 100 --lora_out cpp_lora.safetensors

# Run PyTorch version
python pytorch_alignment/gpt2_lora_finetune.py --steps 100 --lora_out pt_lora

# Compare outputs (losses should match within 1-2%)
```

### 2. Gradient Verification
Both implementations can output gradients for comparison during alignment debugging.

### 3. Performance Benchmarking
PyTorch version serves as speed and memory baseline for C++ optimization validation.

---

## Requirements

### Python Environment
```bash
python3 -m venv pt_env
source pt_env/bin/activate
pip install torch transformers peft datasets
```

### Data Preparation
WikiText-2 raw text files should be in:
```
data/wikitext2/wikitext-2-raw/
├── wiki.train.raw
├── wiki.valid.raw
└── wiki.test.raw
```

### Model Weights
Pretrained models should be downloaded locally:
- GPT-2: `gpt2_lora_finetune/pretrained/gpt2/`
- GPT-2 Medium: `gpt2_lora_finetune/pretrained/gpt2-medium/`
- Gemma 270M: `gemma-3-270m/`
- Gemma 1B: `gemma-3-1b/`