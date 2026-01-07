# Scripts Directory

Organized collection of utility scripts for training, benchmarking, and debugging the mobile deep learning framework.

## Directory Structure

```
scripts/
├── benchmark/                    # Performance evaluation and benchmarking
├── Finetune/                     # Training and alignment scripts
├── pretokenize_wikitext2_gemma.py
├── plot_train200_results.py
├── train_gemma3_lora_torch.py
└── (Finetune/) run_qwen_lora_pt.sh
```

---

## Benchmark Scripts

Located in `benchmark/` directory.

### Parameter Sharding Tests

#### `test_all_models_sharding.sh`
Comprehensive sharding test across four models: GPT-2 Small, GPT-2 Medium, Gemma 270M, and Gemma 1B.

**Usage:**
```bash
bash scripts/benchmark/test_all_models_sharding.sh
```

**Output:** `runs/shard_benchmark_YYYYMMDD_HHMMSS/results.txt`

**Measured Metrics:**
- Peak memory usage (baseline vs sharding)
- Training time overhead
- Disk usage for offloaded parameters

---

#### `test_sharding_aggressive.sh`
Demonstrates sharding optimization with aggressive (minimal) budget settings to maximize memory savings.

**Usage:**
```bash
bash scripts/benchmark/test_sharding_aggressive.sh
```

**Budget Configuration:**
- GPT-2 Small: 200 MB (minimal viable budget)
- GPT-2 Medium: 250 MB
- Gemma 270M: 650 MB
- Gemma 1B: 1200 MB

**Note:** Budget values are calibrated to fit the largest single parameter while forcing frequent swapping for maximum memory reduction.

---

#### `test_shard_optimization.sh`
GPT-2 parameter sharding optimization test with multiple budget configurations (400MB, 200MB).

**Usage:**
```bash
bash scripts/benchmark/test_shard_optimization.sh
```

---

#### `test_gemma_shard.sh`
Isolated sharding test for Gemma 270M model.

**Usage:**
```bash
bash scripts/benchmark/test_gemma_shard.sh
```

---

### Energy-Aware Training Tests

#### `energy_benchmark.sh`
Benchmark test for energy-aware training scheduler with battery and temperature monitoring.

**Usage:**
```bash
bash scripts/benchmark/energy_benchmark.sh
```

**Features:**
- Dynamic sleep scheduling based on battery level
- Temperature-aware throttling
- Step-based scheduling support

---

#### `test_energy_function.sh`
Unit tests for energy scheduling functions.

**Usage:**
```bash
bash scripts/benchmark/test_energy_function.sh
```

---

## Finetune Scripts

Located in `Finetune/` directory. These scripts handle training runs, alignment verification, and performance measurement.

### Training Scripts

#### `run_gemma_1b_lora.sh`
Full training run for Gemma 1B with LoRA.

#### `run_gemma_1b_short.sh`
Short training run for quick testing.

#### `run_gpt2_medium_lora.sh`
GPT-2 Medium LoRA training.

#### `run_qwen_lora_pt.sh`
PyTorch Qwen2.5-0.5B LoRA training (WikiText-2 by default; supports JSONL MMLU if paths are provided via `--jsonl_*` in the script call).

#### `run_gemma_270m_grad_accum_test.sh` / `run_gemma_270m_grad_accum_test_v2.sh`
Gradient accumulation tests for Gemma 270M.

---

### Alignment and Verification

#### `run_all_alignment.sh`
Run C++ (GPT-2, Gemma) plus PyTorch (GPT-2, Gemma, Qwen) alignment tests.

#### `run_sequential_alignment.sh`
Sequential alignment verification for debugging.

#### `run_pytorch_alignment.sh`
PyTorch numerical alignment baseline (runs GPT-2, Gemma, and Qwen in parallel; override `MODEL_*`/`DATA_DIR`/`OUT_BASE` via env vars).

#### `run_pytorch_after_cpp.sh`
PyTorch verification after C++ implementation changes.

---

### Utilities

#### `measure_rss.sh`
Measure RSS (Resident Set Size) memory usage during training.

**Usage:**
```bash
bash scripts/Finetune/measure_rss.sh
```

#### `plot_loss_curve.py`
Plot training loss curves from log files.

**Usage:**
```bash
python3 scripts/Finetune/plot_loss_curve.py
```

---

## Root-Level Utility Scripts

### Data Processing

#### `pretokenize_wikitext2_gemma.py`
Preprocess WikiText-2 dataset for Gemma tokenizer.

**Usage:**
```bash
python3 scripts/pretokenize_wikitext2_gemma.py
```

**Output:**
- `data/wikitext2/pretokenized_gemma/wt2_gemma_tokens.bin`
- `data/wikitext2/pretokenized_gemma/meta.json`

---

### Visualization

#### `plot_train200_results.py`
Generate publication-quality plots and LaTeX tables from training results.

**Usage:**
```bash
python3 scripts/plot_train200_results.py
```

**Output:**
- PDF/PNG plots
- LaTeX tables
- CSV data files

**Location:** `results/train200/`

---

### PyTorch Baseline Training

#### `train_gemma3_lora_torch.py`
Reference implementation using PyTorch + PEFT for numerical alignment.

**Usage:**
```bash
python3 scripts/train_gemma3_lora_torch.py \
    --model_name_or_path gemma-3-270m \
    --output_dir ./pytorch_baseline \
    --max_steps 100
```

**Purpose:** 
- Provide ground truth for gradient/loss validation
- Benchmark comparison with C++ implementation

---

## Quick Start Guide

### 1. Before Running Any Script

Ensure the latest build is ready:

```bash
cd operators/build_shard_benchmark
cmake .. -DUSE_BLAS=ON -DCMAKE_BUILD_TYPE=Release
make -j8
```

### 2. Run Sharding Benchmark

```bash
bash scripts/benchmark/test_sharding_aggressive.sh
```

Expected runtime: ~15-20 minutes for all four models.

### 3. View Results

```bash
cat runs/shard_aggressive_*/results.txt
```

### 4. Clean Up Old Test Runs

```bash
rm -rf runs/shard_benchmark_*
rm -rf runs/shard_aggressive_*
```


