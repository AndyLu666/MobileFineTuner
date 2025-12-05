#!/bin/bash
# =============================================================================
# Gemma 270M Gradient Accumulation 测试 V2
# 完全对齐 gemma_lora_s128_b8_acc1_50p 的参数
# 测试不同 batch_size 和 grad_accum 组合（总有效 batch = 8）
# =============================================================================

set -e
cd /Users/yiyilu/Desktop/FT_gemma_gpt2

BINARY="./operators/build_fast/train_lora_gemma"
RESULTS_BASE="runs/gemma_270m_grad_accum_test_v2"
rm -rf "$RESULTS_BASE"
mkdir -p "$RESULTS_BASE"

LOG_FILE="$RESULTS_BASE/experiment.log"

# ============================================
# 参数配置 (完全对齐 gemma_lora_s128_b8_acc1_50p)
# ============================================
MODEL_DIR="/Users/yiyilu/Desktop/FT_gemma_gpt2/gemma-3-270m"
PRETOK_PATH="/Users/yiyilu/Desktop/FT_gemma_gpt2/data/wikitext2/pretokenized_gemma/wt2_gemma_tokens.bin"
PRETOK_META="/Users/yiyilu/Desktop/FT_gemma_gpt2/data/wikitext2/pretokenized_gemma/meta.json"
TARGETS="attn"
SEQ_LEN=128
EPOCHS=1
DATA_FRACTION=0.5
LR="2e-4"
WARMUP_RATIO=0.05
MAX_GRAD_NORM=1.0

log() {
    echo "$1" | tee -a "$LOG_FILE"
}

# 函数：监控进程 RSS 并记录到文件
monitor_rss_background() {
    local PID=$1
    local OUTPUT_FILE=$2
    
    echo "step,rss_kb,rss_mb,timestamp" > "$OUTPUT_FILE"
    local step=0
    
    while kill -0 "$PID" 2>/dev/null; do
        RSS_KB=$(ps -o rss= -p "$PID" 2>/dev/null | tr -d ' ')
        if [ -n "$RSS_KB" ] && [ "$RSS_KB" -gt 0 ]; then
            RSS_MB=$((RSS_KB / 1024))
            TIMESTAMP=$(date '+%H:%M:%S')
            echo "$step,$RSS_KB,$RSS_MB,$TIMESTAMP" >> "$OUTPUT_FILE"
        fi
        step=$((step + 1))
        sleep 2
    done
}

# 函数：运行单个配置并监控
run_config() {
    local BATCH=$1
    local GRAD_ACCUM=$2
    local CONFIG_NAME="batch${BATCH}_accum${GRAD_ACCUM}"
    local OUT_DIR="$RESULTS_BASE/$CONFIG_NAME"
    
    mkdir -p "$OUT_DIR"
    
    log ""
    log "=========================================="
    log "配置: $CONFIG_NAME (有效batch = $((BATCH * GRAD_ACCUM)))"
    log "  batch_size: $BATCH"
    log "  grad_accum: $GRAD_ACCUM"
    log "开始时间: $(date)"
    log "=========================================="
    
    # 启动训练 (完全对齐参考配置)
    $BINARY \
      --model_dir "$MODEL_DIR" \
      --pretokenized_path "$PRETOK_PATH" \
      --pretokenized_meta "$PRETOK_META" \
      --output_dir "$OUT_DIR" \
      --targets $TARGETS \
      --seq_len $SEQ_LEN \
      --batch $BATCH \
      --grad_accum $GRAD_ACCUM \
      --epochs $EPOCHS \
      --data_fraction $DATA_FRACTION \
      --lr $LR \
      --warmup_ratio $WARMUP_RATIO \
      --max_grad_norm $MAX_GRAD_NORM \
      > "$OUT_DIR/train.log" 2>&1 &
    
    local TRAIN_PID=$!
    log "训练 PID: $TRAIN_PID"
    
    # 后台监控 RSS
    monitor_rss_background $TRAIN_PID "$OUT_DIR/rss.csv" &
    local MONITOR_PID=$!
    
    # 等待训练完成
    wait $TRAIN_PID 2>/dev/null || true
    
    # 停止监控
    kill $MONITOR_PID 2>/dev/null || true
    wait $MONITOR_PID 2>/dev/null || true
    
    # 提取结果
    log ""
    log "[$CONFIG_NAME] 训练完成"
    
    # 获取最终 loss 和 PPL
    FINAL_LINE=$(grep "^\[Step" "$OUT_DIR/train.log" | tail -1)
    if [ -n "$FINAL_LINE" ]; then
        log "最终训练状态: $FINAL_LINE"
    fi
    
    # 获取峰值 RSS
    if [ -f "$OUT_DIR/rss.csv" ]; then
        PEAK_RSS=$(cut -d',' -f3 "$OUT_DIR/rss.csv" | tail -n +2 | sort -n | tail -1)
        AVG_RSS=$(cut -d',' -f3 "$OUT_DIR/rss.csv" | tail -n +2 | awk '{sum+=$1; count++} END {if(count>0) printf "%.0f", sum/count; else print "N/A"}')
        log "峰值 RSS: ${PEAK_RSS} MB"
        log "平均 RSS: ${AVG_RSS} MB"
    fi
    
    # 获取总步数
    TOTAL_STEPS=$(grep "^\[Step" "$OUT_DIR/train.log" | wc -l | tr -d ' ')
    log "总步数: $TOTAL_STEPS"
    
    log "结束时间: $(date)"
}

# =============================================================================
# 主程序
# =============================================================================

log "=========================================="
log "Gemma 270M Gradient Accumulation 实验 V2"
log "实验开始时间: $(date)"
log "=========================================="
log ""
log "========== 参数配置 (对齐 gemma_lora_s128_b8_acc1_50p) =========="
log "  model_dir: $MODEL_DIR"
log "  pretokenized_path: $PRETOK_PATH"
log "  pretokenized_meta: $PRETOK_META"
log "  targets: $TARGETS"
log "  seq_len: $SEQ_LEN"
log "  epochs: $EPOCHS"
log "  data_fraction: $DATA_FRACTION"
log "  lr: $LR"
log "  warmup_ratio: $WARMUP_RATIO"
log "  max_grad_norm: $MAX_GRAD_NORM"
log "  LoRA: rank=8, alpha=32, dropout=0.1 (默认)"
log "  BLAS: ON (build_fast)"
log ""
log "========== 测试配置 =========="
log "  1. batch=4, grad_accum=2 (有效batch=8)"
log "  2. batch=2, grad_accum=4 (有效batch=8)"
log "  3. batch=1, grad_accum=8 (有效batch=8)"
log ""
log "预期: Train sequences=9750, 总步数≈1219 (9750/8)"
log ""

# 按顺序运行三个配置
run_config 4 2
run_config 2 4
run_config 1 8

# =============================================================================
# 汇总结果
# =============================================================================

log ""
log "=========================================="
log "实验结果汇总"
log "=========================================="

SUMMARY_FILE="$RESULTS_BASE/summary.csv"
echo "config,batch,grad_accum,effective_batch,total_steps,peak_rss_mb,avg_rss_mb,final_loss,final_ppl" > "$SUMMARY_FILE"

for CONFIG in "batch4_accum2:4:2" "batch2_accum4:2:4" "batch1_accum8:1:8"; do
    IFS=':' read -r CONFIG_NAME BATCH ACCUM <<< "$CONFIG"
    OUT_DIR="$RESULTS_BASE/$CONFIG_NAME"
    
    if [ -f "$OUT_DIR/train.log" ] && [ -f "$OUT_DIR/rss.csv" ]; then
        TOTAL_STEPS=$(grep "^\[Step" "$OUT_DIR/train.log" | wc -l | tr -d ' ')
        PEAK_RSS=$(cut -d',' -f3 "$OUT_DIR/rss.csv" | tail -n +2 | sort -n | tail -1)
        AVG_RSS=$(cut -d',' -f3 "$OUT_DIR/rss.csv" | tail -n +2 | awk '{sum+=$1; count++} END {if(count>0) printf "%.0f", sum/count; else print "N/A"}')
        
        # 提取最终 loss 和 PPL
        FINAL_LINE=$(grep "^\[Step" "$OUT_DIR/train.log" | tail -1)
        FINAL_LOSS=$(echo "$FINAL_LINE" | grep -oE "Loss=[0-9.]+" | cut -d'=' -f2)
        FINAL_PPL=$(echo "$FINAL_LINE" | grep -oE "PPL=[0-9.]+" | cut -d'=' -f2)
        
        EFFECTIVE_BATCH=$((BATCH * ACCUM))
        echo "$CONFIG_NAME,$BATCH,$ACCUM,$EFFECTIVE_BATCH,$TOTAL_STEPS,$PEAK_RSS,$AVG_RSS,$FINAL_LOSS,$FINAL_PPL" >> "$SUMMARY_FILE"
        
        log ""
        log "[$CONFIG_NAME]"
        log "  Batch: $BATCH, Grad Accum: $ACCUM, 有效 Batch: $EFFECTIVE_BATCH"
        log "  总步数: $TOTAL_STEPS"
        log "  峰值 RSS: ${PEAK_RSS} MB"
        log "  平均 RSS: ${AVG_RSS} MB"
        log "  最终 Loss: $FINAL_LOSS"
        log "  最终 PPL: $FINAL_PPL"
    fi
done

log ""
log "=========================================="
log "✅ 实验完成！"
log "结束时间: $(date)"
log "=========================================="
log ""
log "结果文件:"
log "  汇总: $SUMMARY_FILE"
log "  详细日志: $LOG_FILE"
log "  各配置目录: $RESULTS_BASE/batch*_accum*/"

echo ""
echo "✅ 全部完成！查看结果:"
echo "  cat $SUMMARY_FILE"
echo "  cat $LOG_FILE"

