# MobileFineTuner

A lightweight C++ framework for finetuning Transformer language models (GPT-2, Gemma) on resource-constrained devices. Designed for CPU-only operation with minimal memory footprint, enabling on-device training without GPU requirements.

## Overview

MobileFineTuner provides a complete deep learning framework written in C++17, featuring:

- **Custom Autograd Engine**: Topological-sort based automatic differentiation supporting deep networks (12+ layers)
- **Memory-Efficient Operations**: Chunked softmax, blocked attention, and streaming computation patterns
- **LoRA Finetuning**: Low-rank adaptation for parameter-efficient training
- **PyTorch Alignment**: Reference implementations for numerical verification

### Supported Models

| Model | Parameters | Vocab Size | Layers | Hidden Dim |
|-------|------------|------------|--------|------------|
| GPT-2 Small | 124M | 50,257 | 12 | 768 |
| GPT-2 Medium | 355M | 50,257 | 24 | 1,024 |
| Gemma-3 270M | 270M | 262,144 | 18 | 1,536 |
| Gemma-3 1B | 1B | 262,144 | 26 | 1,152 |

## Project Structure

```
MobileFineTuner/
├── operators/                    # Core Deep Learning Framework
│   └── finetune_ops/
│       ├── core/                 # Tensor, Autograd, Tokenizers
│       │   ├── tensor.cpp/h      # Multi-dimensional tensor with autograd
│       │   ├── autograd_engine.cpp/h  # Backward pass computation
│       │   ├── ops.cpp/h         # Forward/backward operations
│       │   ├── tokenizer_bpe.cpp/h    # GPT-2 BPE tokenizer
│       │   └── tokenizer_gemma.cpp/h  # Gemma SentencePiece tokenizer
│       ├── graph/                # Model Architectures
│       │   ├── gpt2_model.cpp/h  # GPT-2 transformer implementation
│       │   ├── gemma_model.cpp/h # Gemma transformer implementation
│       │   ├── lora_injector.cpp/h    # LoRA weight injection
│       │   └── safetensors_loader.cpp/h  # Weight loading
│       ├── optim/                # Optimizers and Training
│       │   ├── adam.cpp/h        # Adam/AdamW optimizer
│       │   ├── trainer.cpp/h     # GPT-2 training loop
│       │   ├── gemma_trainer.cpp/h    # Gemma training loop
│       │   └── train_lora_*.cpp  # Training entry points
│       ├── data/                 # Data Loading
│       │   └── wikitext2_dataset.cpp/h  # WikiText-2 dataset
│       └── nn/                   # Neural Network Modules
│           └── modules.cpp/h     # Linear, LayerNorm, etc.
│
├── gpt2_lora_finetune/          # GPT-2 LoRA Training
│   ├── main.cpp                  # Training entry point
│   ├── eval_ppl.cpp              # Perplexity evaluation
│   └── eval_mmlu.cpp             # MMLU benchmark
│
├── pytorch_alignment/            # PyTorch Reference Implementations
│   ├── gpt2_lora_finetune.py    # GPT-2 LoRA (PyTorch)
│   └── gemma_lora_finetune.py   # Gemma LoRA (PyTorch)
│
├── scripts/                      # Utility Scripts
│   ├── pretokenize_wikitext2_gemma.py  # Data preprocessing
│   └── train_gemma3_lora_torch.py      # PyTorch training
│
└── docs/                         # Documentation
```

## Features

### Core Framework (`operators/finetune_ops/core/`)

- **Tensor**: N-dimensional array with autograd support, reference counting, and lazy evaluation
- **Autograd Engine**: Non-recursive topological sort backward pass, gradient accumulation, checkpointing
- **Operations**: 50+ differentiable operations including matmul, attention, layer normalization, GELU/SiLU activations

### Model Support (`operators/finetune_ops/graph/`)

- **GPT-2**: Full decoder-only transformer with multi-head attention
- **Gemma**: Grouped-query attention, RMSNorm, SwiGLU MLP, RoPE embeddings
- **LoRA Injection**: Automatic low-rank adapter insertion for q/k/v/o projections and MLP layers
- **SafeTensors**: Direct loading of HuggingFace model weights

### Training (`operators/finetune_ops/optim/`)

- **Adam/AdamW**: With bias correction, weight decay, and gradient clipping
- **Learning Rate Schedules**: Linear warmup with cosine/linear decay
- **Gradient Accumulation**: For larger effective batch sizes
- **Memory Optimization**: Streaming data loading, chunked loss computation

## Building

### Prerequisites

- C++17 compatible compiler (GCC 7+, Clang 5+, Apple Clang 10+)
- CMake 3.14+
- (Optional) Apple Accelerate or OpenBLAS for BLAS acceleration

### Build Commands

```bash
cd operators

# Standard build
mkdir build && cd build
cmake ..
make -j$(nproc)

# With BLAS acceleration (recommended)
mkdir build_fast && cd build_fast
cmake .. -DUSE_BLAS=ON -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Build Targets

| Target | Description |
|--------|-------------|
| `gpt2_lora_finetune` | GPT-2 LoRA finetuning executable |
| `train_lora_gemma` | Gemma LoRA finetuning executable |
| `eval_ppl` | Perplexity evaluation |

## Usage

### GPT-2 LoRA Finetuning

```bash
./gpt2_lora_finetune \
  --data_dir /path/to/wikitext2/wikitext-2-raw \
  --pretrained_dir /path/to/gpt2 \
  --lora_out ./output/gpt2_lora.safetensors \
  --epochs 1 \
  --batch_size 8 \
  --grad_accum_steps 1 \
  --seq_len 128 \
  --rank 8 \
  --alpha 16 \
  --lr 2e-4 \
  --warmup_steps 500 \
  --clip_grad_norm 1.0 \
  --data_fraction 0.5
```

### Gemma LoRA Finetuning

```bash
./train_lora_gemma \
  --model_dir /path/to/gemma-3-270m \
  --data_dir /path/to/wikitext2/wikitext-2-raw \
  --output_dir ./output \
  --targets attn \
  --seq_len 128 \
  --batch 8 \
  --grad_accum 1 \
  --epochs 1 \
  --data_fraction 0.5 \
  --lr 2e-4 \
  --warmup_ratio 0.05 \
  --max_grad_norm 1.0
```

### Using Pretokenized Data (Faster)

```bash
# Pretokenize dataset
python scripts/pretokenize_wikitext2_gemma.py \
  --model_dir /path/to/gemma-3-270m \
  --data_dir /path/to/wikitext2/wikitext-2-raw \
  --output_dir ./data/pretokenized

# Train with pretokenized data
./train_lora_gemma \
  --model_dir /path/to/gemma-3-270m \
  --pretokenized_path ./data/pretokenized/wt2_gemma_tokens.bin \
  --pretokenized_meta ./data/pretokenized/meta.json \
  --output_dir ./output \
  ...
```

## Command Line Arguments

### GPT-2 Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | required | Path to WikiText-2 raw data |
| `--pretrained_dir` | required | Path to pretrained GPT-2 weights |
| `--lora_out` | required | Output path for LoRA weights |
| `--epochs` | 1 | Number of training epochs |
| `--batch_size` | 8 | Training batch size |
| `--grad_accum_steps` | 1 | Gradient accumulation steps |
| `--seq_len` | 128 | Sequence length |
| `--rank` | 8 | LoRA rank |
| `--alpha` | 16 | LoRA alpha scaling |
| `--lr` | 2e-4 | Learning rate |
| `--warmup_steps` | 500 | Linear warmup steps |
| `--clip_grad_norm` | 1.0 | Gradient clipping norm |
| `--data_fraction` | 1.0 | Fraction of training data to use |
| `--seed` | 42 | Random seed |

### Gemma Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_dir` | required | Path to Gemma model directory |
| `--data_dir` | - | Path to raw text data |
| `--pretokenized_path` | - | Path to pretokenized binary |
| `--pretokenized_meta` | - | Path to pretokenized metadata |
| `--output_dir` | required | Output directory for checkpoints |
| `--targets` | attn | LoRA targets (attn, mlp, all) |
| `--batch` | 8 | Training batch size |
| `--grad_accum` | 1 | Gradient accumulation steps |
| `--epochs` | 1 | Number of training epochs |
| `--lr` | 2e-4 | Learning rate |
| `--warmup_ratio` | 0.05 | Warmup ratio of total steps |
| `--max_grad_norm` | 1.0 | Gradient clipping norm |
| `--data_fraction` | 1.0 | Fraction of training data |

## Performance

### Training Speed (Apple M1 Pro, CPU-only)

| Model | Batch Size | Seq Len | Time/Step |
|-------|------------|---------|-----------|
| GPT-2 Small | 8 | 128 | ~1.5s |
| GPT-2 Medium | 8 | 128 | ~4.5s |
| Gemma 270M | 8 | 128 | ~8s |
| Gemma 1B | 4 | 128 | ~25s |

### Memory Usage (RSS)

| Model | Batch Size | Peak RSS |
|-------|------------|----------|
| GPT-2 Small | 8 | ~2.5 GB |
| GPT-2 Medium | 8 | ~4.5 GB |
| Gemma 270M | 8 | ~6.0 GB |
| Gemma 1B | 4 | ~12 GB |

### Gradient Accumulation Memory Efficiency

Effective batch size = batch_size × grad_accum_steps

| Config | Batch | Accum | Effective | Peak RSS |
|--------|-------|-------|-----------|----------|
| High Memory | 8 | 1 | 8 | 6.0 GB |
| Balanced | 4 | 2 | 8 | 5.3 GB |
| Low Memory | 2 | 4 | 8 | 4.8 GB |
| Minimal | 1 | 8 | 8 | 4.4 GB |

## PyTorch Alignment

Reference PyTorch implementations are provided for numerical verification:

```bash
# GPT-2 PyTorch training
python pytorch_alignment/gpt2_lora_finetune.py \
  --data_dir /path/to/wikitext2/wikitext-2-raw \
  --pretrained_dir /path/to/gpt2 \
  --lora_out ./output/gpt2_lora_pt.safetensors \
  --epochs 1 \
  --batch_size 8 \
  --seq_len 128 \
  --rank 8 \
  --alpha 16.0 \
  --lr 2e-4

# Gemma PyTorch training
python pytorch_alignment/gemma_lora_finetune.py \
  --model_dir /path/to/gemma-3-270m \
  --data_dir /path/to/wikitext2/wikitext-2-raw \
  --output_dir ./output \
  --epochs 1 \
  --batch_size 8 \
  --seq_len 128 \
  --lora_rank 8 \
  --lora_alpha 32.0 \
  --lr 2e-4
```

## Model Weights

Download pretrained weights from HuggingFace:

```python
# GPT-2
from transformers import GPT2LMHeadModel, GPT2Tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")  # or "gpt2-medium"
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Gemma (requires authentication)
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-270m")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m")
```

Helper scripts are provided:
- `download_gpt2_medium.py`: Download GPT-2 Medium
- `download_gemma_1b.py`: Download Gemma 1B

## Technical Details

### LoRA Implementation

Low-Rank Adaptation injects trainable low-rank matrices into pretrained weights:

```
W' = W + BA
```

Where:
- W: Original frozen weight [d_out, d_in]
- B: Low-rank matrix [d_out, rank]
- A: Low-rank matrix [rank, d_in]
- rank << min(d_in, d_out)

The effective weight is scaled by alpha/rank.

### Memory Optimization Techniques

1. **Chunked Cross-Entropy**: Process vocabulary in chunks to reduce peak memory
2. **Gradient Checkpointing**: Trade compute for memory by recomputing activations
3. **Streaming Data Loading**: Load and tokenize data on-demand
4. **In-place Operations**: Minimize tensor allocations during backward pass

### Autograd Engine

The autograd engine uses iterative topological sort instead of recursion to handle deep networks:

1. Build computation graph during forward pass
2. Topological sort nodes by reverse order
3. Iterate through sorted nodes, computing gradients
4. Accumulate gradients for shared parameters

## Troubleshooting

### Out of Memory

- Reduce `batch_size` and increase `grad_accum_steps`
- Reduce `seq_len`
- Use pretokenized data to avoid tokenizer memory overhead

### Slow Training

- Enable BLAS acceleration: `cmake .. -DUSE_BLAS=ON`
- Use pretokenized data
- Build with Release mode: `cmake .. -DCMAKE_BUILD_TYPE=Release`

### NaN Loss

- Reduce learning rate
- Enable gradient clipping
- Check data for invalid tokens

## License

This project is for research and educational purposes.

## References

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [GPT-2: Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Gemma: Open Models Based on Gemini Research and Technology](https://arxiv.org/abs/2403.08295)

