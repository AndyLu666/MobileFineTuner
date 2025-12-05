#!/bin/bash
# =============================================================================
# RSS 内存测量脚本
# 测量 GPT-2 Small, GPT-2 Medium, Gemma 270M, Gemma 1B 的 RSS
# 每个模型跑 10 步，每步记录 RSS
# =============================================================================

set -e
cd /Users/yiyilu/Desktop/FT_gemma_gpt2

PYTHON="/opt/anaconda3/envs/compsci371/bin/python"
RESULTS_DIR="runs/rss_measurements"
rm -rf "$RESULTS_DIR"
mkdir -p "$RESULTS_DIR"

echo "=========================================="
echo "RSS 内存测量实验"
echo "开始时间: $(date)"
echo "=========================================="

# 函数：监控进程 RSS
monitor_rss() {
    local PID=$1
    local OUTPUT_FILE=$2
    local MODEL_NAME=$3
    local STEPS=10
    local step=0
    
    echo "step,rss_kb,rss_mb,timestamp" > "$OUTPUT_FILE"
    
    while kill -0 "$PID" 2>/dev/null && [ $step -lt 60 ]; do
        RSS_KB=$(ps -o rss= -p "$PID" 2>/dev/null | tr -d ' ')
        if [ -n "$RSS_KB" ]; then
            RSS_MB=$((RSS_KB / 1024))
            TIMESTAMP=$(date '+%H:%M:%S')
            echo "$step,$RSS_KB,$RSS_MB,$TIMESTAMP" >> "$OUTPUT_FILE"
            echo "  [$MODEL_NAME] Step $step: RSS = ${RSS_MB} MB"
        fi
        step=$((step + 1))
        sleep 5
    done
}

# =============================================================================
# 1. GPT-2 Small (117M)
# =============================================================================
echo ""
echo "=========================================="
echo "[1/4] GPT-2 Small (117M)"
echo "=========================================="

GPT2_SMALL_OUT="$RESULTS_DIR/gpt2_small"
mkdir -p "$GPT2_SMALL_OUT"

./operators/build_fast/gpt2_lora_finetune \
  --data_dir "data/wikitext2/wikitext-2-raw" \
  --pretrained_dir "gpt2_lora_finetune/pretrained/gpt2" \
  --lora_out "$GPT2_SMALL_OUT/lora.safetensors" \
  --steps 10 \
  --batch_size 8 \
  --grad_accum_steps 1 \
  --seq_len 128 \
  --rank 8 \
  --alpha 16 \
  --lr 2e-4 \
  --warmup_steps 0 \
  --data_fraction 0.01 \
  --log_interval 1 \
  --eval_interval 0 \
  --save_every 0 \
  > "$GPT2_SMALL_OUT/train.log" 2>&1 &

GPT2_SMALL_PID=$!
echo "PID: $GPT2_SMALL_PID"

monitor_rss $GPT2_SMALL_PID "$GPT2_SMALL_OUT/rss.csv" "GPT2-Small"
wait $GPT2_SMALL_PID 2>/dev/null || true

# 获取最终 RSS
FINAL_RSS=$(tail -1 "$GPT2_SMALL_OUT/rss.csv" | cut -d',' -f3)
echo "✅ GPT-2 Small 完成，最终 RSS: ${FINAL_RSS} MB"

# =============================================================================
# 2. GPT-2 Medium (345M)
# =============================================================================
echo ""
echo "=========================================="
echo "[2/4] GPT-2 Medium (345M)"
echo "=========================================="

GPT2_MEDIUM_OUT="$RESULTS_DIR/gpt2_medium"
mkdir -p "$GPT2_MEDIUM_OUT"

./operators/build_fast/gpt2_lora_finetune \
  --data_dir "data/wikitext2/wikitext-2-raw" \
  --pretrained_dir "gpt2_lora_finetune/pretrained/gpt2-medium" \
  --lora_out "$GPT2_MEDIUM_OUT/lora.safetensors" \
  --steps 10 \
  --batch_size 8 \
  --grad_accum_steps 1 \
  --seq_len 128 \
  --rank 8 \
  --alpha 16 \
  --lr 2e-4 \
  --warmup_steps 0 \
  --data_fraction 0.01 \
  --log_interval 1 \
  --eval_interval 0 \
  --save_every 0 \
  > "$GPT2_MEDIUM_OUT/train.log" 2>&1 &

GPT2_MEDIUM_PID=$!
echo "PID: $GPT2_MEDIUM_PID"

monitor_rss $GPT2_MEDIUM_PID "$GPT2_MEDIUM_OUT/rss.csv" "GPT2-Medium"
wait $GPT2_MEDIUM_PID 2>/dev/null || true

FINAL_RSS=$(tail -1 "$GPT2_MEDIUM_OUT/rss.csv" | cut -d',' -f3)
echo "✅ GPT-2 Medium 完成，最终 RSS: ${FINAL_RSS} MB"

# =============================================================================
# 3. Gemma 270M
# =============================================================================
echo ""
echo "=========================================="
echo "[3/4] Gemma 270M"
echo "=========================================="

GEMMA_270M_OUT="$RESULTS_DIR/gemma_270m"
mkdir -p "$GEMMA_270M_OUT"

./operators/build_fast/train_lora_gemma \
  --model_dir "gemma-3-270m" \
  --data_dir "data/wikitext2/wikitext-2-raw" \
  --output_dir "$GEMMA_270M_OUT" \
  --targets attn \
  --seq_len 128 \
  --batch 8 \
  --grad_accum 1 \
  --epochs 1 \
  --max_steps 10 \
  --data_fraction 0.01 \
  --lr 2e-4 \
  --warmup_ratio 0 \
  --max_grad_norm 1.0 \
  > "$GEMMA_270M_OUT/train.log" 2>&1 &

GEMMA_270M_PID=$!
echo "PID: $GEMMA_270M_PID"

monitor_rss $GEMMA_270M_PID "$GEMMA_270M_OUT/rss.csv" "Gemma-270M"
wait $GEMMA_270M_PID 2>/dev/null || true

FINAL_RSS=$(tail -1 "$GEMMA_270M_OUT/rss.csv" | cut -d',' -f3)
echo "✅ Gemma 270M 完成，最终 RSS: ${FINAL_RSS} MB"

# =============================================================================
# 4. Gemma 1B
# =============================================================================
echo ""
echo "=========================================="
echo "[4/4] Gemma 1B"
echo "=========================================="

GEMMA_1B_OUT="$RESULTS_DIR/gemma_1b"
mkdir -p "$GEMMA_1B_OUT"

./operators/build_fast/train_lora_gemma \
  --model_dir "gemma-3-1b" \
  --data_dir "data/wikitext2/wikitext-2-raw" \
  --output_dir "$GEMMA_1B_OUT" \
  --targets attn \
  --seq_len 128 \
  --batch 8 \
  --grad_accum 1 \
  --epochs 1 \
  --max_steps 10 \
  --data_fraction 0.01 \
  --lr 2e-4 \
  --warmup_ratio 0 \
  --max_grad_norm 1.0 \
  > "$GEMMA_1B_OUT/train.log" 2>&1 &

GEMMA_1B_PID=$!
echo "PID: $GEMMA_1B_PID"

monitor_rss $GEMMA_1B_PID "$GEMMA_1B_OUT/rss.csv" "Gemma-1B"
wait $GEMMA_1B_PID 2>/dev/null || true

FINAL_RSS=$(tail -1 "$GEMMA_1B_OUT/rss.csv" | cut -d',' -f3)
echo "✅ Gemma 1B 完成，最终 RSS: ${FINAL_RSS} MB"

# =============================================================================
# 汇总结果
# =============================================================================
echo ""
echo "=========================================="
echo "RSS 测量结果汇总"
echo "=========================================="

SUMMARY_FILE="$RESULTS_DIR/summary.txt"

{
    echo "RSS 内存测量结果"
    echo "测量时间: $(date)"
    echo "配置: batch_size=8, seq_len=128, steps=10"
    echo ""
    echo "模型                参数量      峰值RSS(MB)"
    echo "================================================"
    
    # GPT-2 Small
    if [ -f "$GPT2_SMALL_OUT/rss.csv" ]; then
        PEAK=$(cut -d',' -f3 "$GPT2_SMALL_OUT/rss.csv" | tail -n +2 | sort -n | tail -1)
        echo "GPT-2 Small         117M        $PEAK"
    fi
    
    # GPT-2 Medium
    if [ -f "$GPT2_MEDIUM_OUT/rss.csv" ]; then
        PEAK=$(cut -d',' -f3 "$GPT2_MEDIUM_OUT/rss.csv" | tail -n +2 | sort -n | tail -1)
        echo "GPT-2 Medium        345M        $PEAK"
    fi
    
    # Gemma 270M
    if [ -f "$GEMMA_270M_OUT/rss.csv" ]; then
        PEAK=$(cut -d',' -f3 "$GEMMA_270M_OUT/rss.csv" | tail -n +2 | sort -n | tail -1)
        echo "Gemma 270M          270M        $PEAK"
    fi
    
    # Gemma 1B
    if [ -f "$GEMMA_1B_OUT/rss.csv" ]; then
        PEAK=$(cut -d',' -f3 "$GEMMA_1B_OUT/rss.csv" | tail -n +2 | sort -n | tail -1)
        echo "Gemma 1B            1B          $PEAK"
    fi
    
    echo "================================================"
} | tee "$SUMMARY_FILE"

echo ""
echo "详细 CSV 文件位置:"
echo "  $RESULTS_DIR/gpt2_small/rss.csv"
echo "  $RESULTS_DIR/gpt2_medium/rss.csv"
echo "  $RESULTS_DIR/gemma_270m/rss.csv"
echo "  $RESULTS_DIR/gemma_1b/rss.csv"
echo ""
echo "汇总文件: $SUMMARY_FILE"
echo ""
echo "✅ 测量完成！"
echo "结束时间: $(date)"

