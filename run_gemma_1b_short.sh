#!/bin/bash
set -e

# =============================================================================
# Gemma 3 1B LoRA Finetuning - Short Run (300 steps for convergence analysis)
# =============================================================================

cd /Users/yiyilu/Desktop/FT_gemma_gpt2

echo "=== Gemma 1B LoRA 短训练 (300步收敛分析) ==="
echo "开始时间: $(date)"

# Output directory
OUT="/Users/yiyilu/Desktop/FT_gemma_gpt2/runs/gemma_1b_lora_short_300steps"
rm -rf "$OUT"
mkdir -p "$OUT"
echo "输出目录: $OUT"

# Binary with BLAS optimization
BINARY="/Users/yiyilu/Desktop/FT_gemma_gpt2/operators/build_fast/train_lora_gemma"

echo "开始后台训练..."

# 运行训练（后台运行）
nohup $BINARY \
  --model_dir "/Users/yiyilu/Desktop/FT_gemma_gpt2/gemma-3-1b" \
  --data_dir "/Users/yiyilu/Desktop/FT_gemma_gpt2/data/wikitext2/wikitext-2-raw" \
  --output_dir "$OUT" \
  --targets attn \
  --seq_len 128 \
  --batch 8 \
  --grad_accum 1 \
  --epochs 1 \
  --max_steps 300 \
  --data_fraction 0.5 \
  --lr 2e-4 \
  --warmup_ratio 0.05 \
  --max_grad_norm 1.0 \
  > "$OUT/train.log" 2>&1 &

PID=$!
echo "✅ 训练已在后台启动"
echo "  PID: $PID"
echo "  日志: $OUT/train.log"
echo ""
echo "监控命令:"
echo "  tail -f $OUT/train.log"
echo ""
echo "训练完成后画 loss 曲线:"
echo "  python plot_loss_curve.py $OUT/train.log"

