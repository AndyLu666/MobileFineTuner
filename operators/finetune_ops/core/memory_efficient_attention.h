/**
 * @file memory_efficient_attention.h
 * @brief 内存高效的注意力实现（流式softmax，避免物化S×S矩阵）
 * 
 * 核心思想：
 * - 不显式构建完整的 scores/probs 矩阵 [B,H,S,S]
 * - 使用在线/流式 softmax 算法，分块计算，空间复杂度从 O(S²) 降到 O(S)
 * - 两遍扫描：第一遍计算 row-wise max/sumExp，第二遍归一化并累积到 context
 * 
 * 参考：
 * - FlashAttention (Dao et al., 2022)
 * - Online normalizer for softmax
 * - PyTorch SDPA memory-efficient kernel
 */

#pragma once

#include "tensor.h"
#include <vector>

namespace ops {

/**
 * @brief 内存高效注意力配置
 */
struct MemoryEfficientAttentionConfig {
    bool use_causal_mask = true;      // 是否使用因果掩码
    float scale = -1.0f;              // 缩放因子（-1表示自动：1/sqrt(head_dim)）
    int chunk_size = 512;             // 分块大小（用于极长序列，当前实现全序列）
    bool save_probs = false;          // 是否保存概率（调试用，默认不保存）
};

/**
 * @brief 内存高效的缩放点积注意力
 * 
 * @param q [batch, n_head, seq_len, head_dim]
 * @param k [batch, n_head, seq_len, head_dim]
 * @param v [batch, n_head, seq_len, head_dim]
 * @param causal_mask [seq_len, seq_len] 可选，上三角为-inf
 * @param config 配置选项
 * @return context [batch, n_head, seq_len, head_dim]
 * 
 * 特点：
 * - 不物化完整 scores/probs 矩阵
 * - 数值稳定（max-normalization）
 * - 内存占用 O(B·H·S·D) vs 原版 O(B·H·S² + B·H·S·D)
 * - CPU 实现，纯 C++，无外部依赖
 */
TensorPtr memory_efficient_attention(
    const TensorPtr& q,
    const TensorPtr& k,
    const TensorPtr& v,
    const TensorPtr& causal_mask = nullptr,
    const MemoryEfficientAttentionConfig& config = {}
);

/**
 * @brief 在线/流式 softmax（单行版本，用于注意力）
 * 
 * 给定一行 logits [S]，在不物化整行 exp 的情况下计算 softmax 权重并累积到输出。
 * 使用 Welford/Kahan 风格的在线算法，数值稳定。
 * 
 * @param logits 输入 logits（一行 S 个值）
 * @param values 对应的 values [S, D]
 * @param seq_len 序列长度 S
 * @param head_dim 头维度 D
 * @param output 输出累积缓冲 [D]
 * @param max_val 该行的最大 logit（数值稳定用）
 */
void online_softmax_weighted_sum(
    const float* logits,
    const float* values,
    int64_t seq_len,
    int64_t head_dim,
    float* output,
    float max_val
);

} // namespace ops

