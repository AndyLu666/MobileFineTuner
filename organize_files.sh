#!/bin/bash
# 项目文件整理脚本
# 作用：整理训练结果、日志、图表、文档等，删除不需要的文件

set -e  # 遇到错误立即停止

cd /Users/yiyilu/Desktop/FT

echo "=========================================="
echo "开始整理项目文件..."
echo "=========================================="

# ============ 1. 创建新的目录结构 ============
echo ""
echo "[1/6] 创建目录结构..."
mkdir -p results/train200/{figures,tables,data}
mkdir -p logs/memory_tests
mkdir -p docs/archive
mkdir -p scripts

# ============ 2. 整理train200训练结果 ============
echo "[2/6] 整理train200训练结果..."

# 移动图表文件
echo "  - 移动图表到 results/train200/figures/"
mv -v train200_comprehensive.pdf train200_comprehensive.png results/train200/figures/ 2>/dev/null || true
mv -v train200_loss_curve.pdf train200_loss_curve.png results/train200/figures/ 2>/dev/null || true
mv -v train200_perplexity.pdf train200_perplexity.png results/train200/figures/ 2>/dev/null || true
mv -v train200_lr_schedule.pdf train200_lr_schedule.png results/train200/figures/ 2>/dev/null || true

# 移动LaTeX表格
echo "  - 移动表格到 results/train200/tables/"
mv -v train200_summary_table.tex results/train200/tables/ 2>/dev/null || true
mv -v train200_eval_table.tex results/train200/tables/ 2>/dev/null || true
mv -v train200_train_table.tex results/train200/tables/ 2>/dev/null || true
mv -v train200_performance_table.tex results/train200/tables/ 2>/dev/null || true

# 移动数据文件
echo "  - 移动数据到 results/train200/data/"
mv -v train200_detailed.csv results/train200/data/ 2>/dev/null || true
mv -v train200_data.json results/train200/data/ 2>/dev/null || true

# 移动脚本
echo "  - 移动脚本到 scripts/"
mv -v plot_train200_results.py scripts/ 2>/dev/null || true

# ============ 3. 整理训练日志 ============
echo "[3/6] 整理训练日志..."

# 保留最近的训练log（11月8日）- 这些都要保留
echo "  - 移动训练log到 logs/"
mv -v run_logs/train_acc1_steps*.log logs/ 2>/dev/null || true

# 保留内存测试相关的log和数据（11月8日）
echo "  - 移动内存测试log到 logs/memory_tests/"
mv -v run_logs/verify*.log logs/memory_tests/ 2>/dev/null || true
mv -v run_logs/verify*.pid logs/memory_tests/ 2>/dev/null || true
mv -v run_logs/mem_*.csv logs/memory_tests/ 2>/dev/null || true
mv -v run_logs/vmmap*.txt logs/memory_tests/ 2>/dev/null || true

# 删除空的run_logs目录
if [ -d "run_logs" ]; then
    rmdir run_logs 2>/dev/null && echo "  - 删除空目录 run_logs/" || echo "  - run_logs/目录非空，保留"
fi

# ============ 4. 整理文档 ============
echo "[4/6] 整理文档..."

# 保留最新的重要文档
echo "  - 保留 MEMORY_MANAGER_FIX.md (最新)"

# 归档documents/下的历史文档（全部移到archive）
echo "  - 归档旧文档到 docs/archive/"
mv -v documents/*.md docs/archive/ 2>/dev/null || true
mv -v documents/*.png docs/archive/ 2>/dev/null || true

# 删除空的documents目录
if [ -d "documents" ]; then
    rmdir documents 2>/dev/null && echo "  - 删除空目录 documents/" || echo "  - documents/目录非空，保留"
fi

# 移动当前重要文档
mv -v MEMORY_MANAGER_FIX.md docs/ 2>/dev/null || true

# ============ 5. 清理临时和测试文件 ============
echo "[5/6] 清理临时文件..."

# 清理tmp_wt测试目录（只用于内存leak测试的小数据集）
if [ -d "tmp_wt" ]; then
    echo "  - 删除测试目录 tmp_wt/"
    rm -rf tmp_wt
fi

# 清理旧的build目录（保留build_mac）
if [ -d "operators/build" ]; then
    echo "  - 删除旧build目录 operators/build/"
    rm -rf operators/build
fi

if [ -d "operators/build_modules" ]; then
    echo "  - 删除旧build目录 operators/build_modules/"
    rm -rf operators/build_modules
fi

# 清理Python虚拟环境（如果不需要可以删除）
# 如果你还需要plot_env，请注释掉下面这行
# echo "  - 删除Python虚拟环境 plot_env/"
# rm -rf plot_env

# ============ 6. 创建README ============
echo "[6/6] 创建目录说明文档..."

cat > results/README.md << 'EOF'
# Training Results

This directory contains all training results, including figures, tables, and raw data.

## Directory Structure

```
results/
├── train200/           # 200-step training results (WikiText-2)
│   ├── figures/        # PDF and PNG figures for papers
│   ├── tables/         # LaTeX tables
│   └── data/           # CSV and JSON data files
└── README.md          # This file
```

## Train200 Results Summary

- **Initial Validation PPL**: 44.54
- **Final Validation PPL**: 22.26
- **Improvement**: 50.0%
- **Training Steps**: 200
- **Dataset**: WikiText-2
- **Method**: GPT-2 LoRA (rank=8, alpha=16)

## Files

### Figures (results/train200/figures/)
- `train200_comprehensive.pdf/png` - Combined training curves
- `train200_loss_curve.pdf/png` - Loss convergence
- `train200_perplexity.pdf/png` - Perplexity comparison
- `train200_lr_schedule.pdf/png` - Learning rate schedule

### Tables (results/train200/tables/)
- `train200_summary_table.tex` - Configuration and results summary
- `train200_eval_table.tex` - Validation results
- `train200_train_table.tex` - Training loss details
- `train200_performance_table.tex` - Memory and performance stats

### Data (results/train200/data/)
- `train200_detailed.csv` - Complete training data (every 10 steps)
- `train200_data.json` - Raw data in JSON format
EOF

cat > logs/README.md << 'EOF'
# Training Logs

This directory contains all training and debugging logs.

## Directory Structure

```
logs/
├── memory_tests/       # Memory leak tests and verification logs
├── train_*.log        # Training logs
└── README.md          # This file
```

## Latest Training Logs

All logs are from November 8, 2024:

### Training Logs
- `train_acc1_steps500.log` - 500 steps training
- `train_acc1_steps3000.log` - 3000 steps training attempt
- `train_acc1_steps3000_step1.log` - 3000 steps training (step 1 config)
- `train_acc1_steps3000_step1_fast.log` - 3000 steps training (fast mode)

### Memory Tests
- `verify_pool_fix_500.log` - Memory pool fix verification (500 steps)
- `verify_system_alloc_500.log` - System allocation verification
- `mem_*.csv` - Memory usage tracking
- `vmmap*.txt` - Virtual memory map summaries
EOF

cat > docs/README.md << 'EOF'
# Documentation

## Current Documents

- `MEMORY_MANAGER_FIX.md` - Latest memory leak fix documentation (Nov 10, 2024)

## Archived Documents

See `archive/` directory for historical documentation:
- BLAS usage guides
- Implementation summaries
- Progress reports
- Status reports
EOF

cat > scripts/README.md << 'EOF'
# Scripts

This directory contains utility scripts for data processing and visualization.

## Available Scripts

- `plot_train200_results.py` - Generate figures and tables for 200-step training results
  - Usage: `python3 plot_train200_results.py`
  - Output: Generates PDF/PNG figures, LaTeX tables, and data files
EOF

echo ""
echo "=========================================="
echo "✅ 文件整理完成！"
echo "=========================================="
echo ""
echo "新的目录结构："
echo ""
echo "FT/"
echo "├── results/          # 训练结果（图表、表格、数据）"
echo "│   └── train200/     # 200步训练的完整结果"
echo "├── logs/             # 训练和测试日志"
echo "│   └── memory_tests/ # 内存测试log"
echo "├── docs/             # 文档"
echo "│   ├── archive/      # 历史文档"
echo "│   └── MEMORY_MANAGER_FIX.md"
echo "├── scripts/          # 脚本"
echo "├── data/             # 数据集"
echo "├── gpt2_lora_finetune/  # 主程序"
echo "└── operators/        # 算子库"
echo ""
echo "已删除："
echo "  - tmp_wt/          (测试用小数据集)"
echo "  - operators/build/ (旧build目录)"
echo "  - operators/build_modules/ (旧build目录)"
echo "  - documents/       (已归档到docs/archive/)"
echo ""
echo "保留但未移动："
echo "  - plot_env/        (Python虚拟环境，如需删除请手动执行)"
echo ""

