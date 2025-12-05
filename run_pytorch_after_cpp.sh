#!/bin/bash
# =============================================================================
# 自动化脚本：等待 C++ 训练完成后启动 PyTorch 训练
# =============================================================================

set -e
cd /Users/yiyilu/Desktop/FT_gemma_gpt2

PYTHON="/opt/anaconda3/envs/compsci371/bin/python"
LOG_FILE="/Users/yiyilu/Desktop/FT_gemma_gpt2/runs/pytorch_auto_run.log"

echo "========================================" | tee -a "$LOG_FILE"
echo "自动化 PyTorch 训练脚本启动" | tee -a "$LOG_FILE"
echo "启动时间: $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# 获取当前运行的 C++ 进程 PID
GPT2_CPP_PID=$(pgrep -f "gpt2_lora_finetune.*gpt2-medium" || echo "")
GEMMA_CPP_PID=$(pgrep -f "train_lora_gemma.*gemma-3-1b" || echo "")

echo "检测到的 C++ 进程:" | tee -a "$LOG_FILE"
echo "  GPT-2 Medium PID: ${GPT2_CPP_PID:-'未运行'}" | tee -a "$LOG_FILE"
echo "  Gemma 1B PID: ${GEMMA_CPP_PID:-'未运行'}" | tee -a "$LOG_FILE"

# 等待 GPT-2 C++ 完成
if [ -n "$GPT2_CPP_PID" ]; then
    echo "" | tee -a "$LOG_FILE"
    echo "[等待] GPT-2 Medium C++ 训练完成 (PID: $GPT2_CPP_PID)..." | tee -a "$LOG_FILE"
    while kill -0 "$GPT2_CPP_PID" 2>/dev/null; do
        sleep 60
        echo "  $(date '+%H:%M:%S') - GPT-2 C++ 仍在运行..." | tee -a "$LOG_FILE"
    done
    echo "[完成] GPT-2 Medium C++ 训练已结束 @ $(date)" | tee -a "$LOG_FILE"
fi

# 等待 Gemma C++ 完成
if [ -n "$GEMMA_CPP_PID" ]; then
    echo "" | tee -a "$LOG_FILE"
    echo "[等待] Gemma 1B C++ 训练完成 (PID: $GEMMA_CPP_PID)..." | tee -a "$LOG_FILE"
    while kill -0 "$GEMMA_CPP_PID" 2>/dev/null; do
        sleep 60
        echo "  $(date '+%H:%M:%S') - Gemma C++ 仍在运行..." | tee -a "$LOG_FILE"
    done
    echo "[完成] Gemma 1B C++ 训练已结束 @ $(date)" | tee -a "$LOG_FILE"
fi

echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "C++ 训练全部完成，开始 PyTorch 训练" | tee -a "$LOG_FILE"
echo "时间: $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# =============================================================================
# PyTorch GPT-2 Medium 训练
# =============================================================================
echo "" | tee -a "$LOG_FILE"
echo "[PyTorch] 启动 GPT-2 Medium 训练..." | tee -a "$LOG_FILE"

OUT_GPT2_PT="/Users/yiyilu/Desktop/FT_gemma_gpt2/runs/gpt2_medium_lora_pt_s128_b8_acc1_e1_lr2e-4"
mkdir -p "$OUT_GPT2_PT"

$PYTHON pytorch_alignment/gpt2_lora_finetune.py \
  --data_dir "data/wikitext2/wikitext-2-raw" \
  --pretrained_dir "gpt2_lora_finetune/pretrained/gpt2-medium" \
  --lora_out "$OUT_GPT2_PT/adapter" \
  --epochs 1 \
  --batch_size 8 \
  --grad_accum_steps 1 \
  --seq_len 128 \
  --rank 8 \
  --alpha 16 \
  --lr 2e-4 \
  --warmup_steps 500 \
  --clip_grad_norm 1.0 \
  --log_interval 1 \
  --eval_interval 500 \
  --eval_batches 200 \
  --eval_batch_size 2 \
  --save_every 1000 \
  --seed 42 \
  --data_fraction 0.5 \
  > "$OUT_GPT2_PT/train.log" 2>&1

echo "[PyTorch] GPT-2 Medium 训练完成 @ $(date)" | tee -a "$LOG_FILE"
echo "  日志: $OUT_GPT2_PT/train.log" | tee -a "$LOG_FILE"

# =============================================================================
# PyTorch Gemma 1B 训练
# =============================================================================
echo "" | tee -a "$LOG_FILE"
echo "[PyTorch] 启动 Gemma 1B 训练..." | tee -a "$LOG_FILE"

OUT_GEMMA_PT="/Users/yiyilu/Desktop/FT_gemma_gpt2/runs/gemma_1b_lora_pt_s128_b8_acc1_50p"
mkdir -p "$OUT_GEMMA_PT"

$PYTHON pytorch_alignment/gemma_lora_finetune.py \
  --model_dir "gemma-3-1b" \
  --data_dir "data/wikitext2/wikitext-2-raw" \
  --output_dir "$OUT_GEMMA_PT" \
  --target_mode attn \
  --seq_len 128 \
  --batch 8 \
  --grad_accum 1 \
  --epochs 1 \
  --data_fraction 0.5 \
  --learning_rate 2e-4 \
  --warmup_ratio 0.05 \
  --max_grad_norm 1.0 \
  --lora_r 8 \
  --lora_alpha 32.0 \
  --lora_dropout 0.1 \
  > "$OUT_GEMMA_PT/train.log" 2>&1

echo "[PyTorch] Gemma 1B 训练完成 @ $(date)" | tee -a "$LOG_FILE"
echo "  日志: $OUT_GEMMA_PT/train.log" | tee -a "$LOG_FILE"

# =============================================================================
# 完成
# =============================================================================
echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "✅ 全部训练完成!" | tee -a "$LOG_FILE"
echo "完成时间: $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "输出目录:" | tee -a "$LOG_FILE"
echo "  C++ GPT-2 Medium: runs/gpt2_medium_lora_s128_b8_acc1_e1_lr2e-4/" | tee -a "$LOG_FILE"
echo "  C++ Gemma 1B:     runs/gemma_1b_lora_s128_b4_acc1_50p/" | tee -a "$LOG_FILE"
echo "  PT  GPT-2 Medium: $OUT_GPT2_PT/" | tee -a "$LOG_FILE"
echo "  PT  Gemma 1B:     $OUT_GEMMA_PT/" | tee -a "$LOG_FILE"

