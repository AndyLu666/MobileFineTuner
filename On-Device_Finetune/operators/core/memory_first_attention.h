/**
 * @file memory_first_attention.h
 * @brief Extreme memory-saving attention mechanism implementation (FlashAttention idea, pure C++)
 * 
 * Core innovations:
 * 1. No explicit construction of full O(S²) attention matrix
 * 2. Use row-column blocking + online softmax algorithm, memory complexity from O(S²) → O(S)
 * 3. Support causal mask (autoregressive models)
 * 4. Pure C++ implementation, no BLAS dependency
 * 
 * Memory comparison (sequence length S=512, hidden dimension d=64, batch B=2, heads H=12):
 * - Standard attention: scores [B,H,S,S] = 2*12*512*512*4B ≈ 25MB (scores only)
 * - This implementation: row_state [B,H,S,2] = 2*12*512*2*4B ≈ 98KB (row state only)
 * Peak memory reduced by ~256x
 */

#pragma once

#include "tensor.h"
#include "ops.h"
#include <cmath>
#include <algorithm>
#include <limits>

namespace ops {
namespace memory_first {

/**
 * @brief Online softmax state (for incremental updates)
 * For each row: maintain max, exp_sum, weighted output
 */
struct OnlineSoftmaxState {
    float max_val;      // Current row's maximum value (numerical stability)
    float exp_sum;      // Cumulative sum of exp(x - max)
    std::vector<float> weighted_output; // Weighted output accumulation
    
    OnlineSoftmaxState(int output_dim) 
        : max_val(-std::numeric_limits<float>::infinity()), 
          exp_sum(0.0f),
          weighted_output(output_dim, 0.0f) {}
    
    /**
     * @brief Update online softmax state
     * @param new_scores Newly computed score block [block_cols]
     * @param new_values Corresponding Value block [block_cols, head_dim]
     * @param block_cols Column block size
     * @param head_dim Head dimension
     */
    void update(const float* new_scores, const float* new_values, 
                int block_cols, int head_dim) {
        // Calculate maximum value of new block
        float new_max = max_val;
        for (int j = 0; j < block_cols; ++j) {
            new_max = std::max(new_max, new_scores[j]);
        }
        
        // If maximum changes, need to rescale previous accumulation
        if (new_max > max_val) {
            float scale = std::exp(max_val - new_max);
            exp_sum *= scale;
            for (auto& val : weighted_output) {
                val *= scale;
            }
            max_val = new_max;
        }
        
        // Accumulate contribution of new block
        for (int j = 0; j < block_cols; ++j) {
            float exp_score = std::exp(new_scores[j] - max_val);
            exp_sum += exp_score;
            
            // weighted_output += exp_score * new_values[j, :]
            for (int d = 0; d < head_dim; ++d) {
                weighted_output[d] += exp_score * new_values[j * head_dim + d];
            }
        }
    }
    
    /**
     * @brief Final normalization of output
     */
    void normalize() {
        if (exp_sum > 0.0f) {
            for (auto& val : weighted_output) {
                val /= exp_sum;
            }
        }
    }
};

/**
 * @brief Extreme memory-saving attention forward computation
 * 
 * @param query Query tensor [batch, seq_len, num_heads, head_dim]
 * @param key Key tensor [batch, seq_len, num_heads, head_dim]
 * @param value Value tensor [batch, seq_len, num_heads, head_dim]
 * @param causal Whether to use causal mask
 * @param row_block_size Row block size (default 32)
 * @param col_block_size Column block size (default 32)
 * @return Attention output [batch, seq_len, num_heads, head_dim]
 * 
 * Algorithm core:
 * 1. Iterate over Query row blocks Qi [row_block, head_dim]
 * 2. For each Qi, iterate over Key column blocks Kj [col_block, head_dim]
 * 3. Compute score block Sij = Qi @ Kj^T / sqrt(d) [row_block, col_block]
 * 4. Apply causal mask (if enabled)
 * 5. Use online softmax to update each row's state
 * 6. Accumulate weighted output using corresponding Value block
 * 7. Final normalization
 * 
 * Memory usage:
 * - Sij temporary block: row_block * col_block * 4B (default 32*32*4 = 4KB)
 * - Online state: batch * num_heads * row_block * (2 + head_dim) * 4B
 * - No O(S²) intermediate matrix
 */
TensorPtr memory_first_attention_forward(
    const TensorPtr& query,
    const TensorPtr& key,
    const TensorPtr& value,
    bool causal = false,
    int row_block_size = 32,
    int col_block_size = 32
);

/**
 * @brief Convenience interface: automatically convert from [batch, seq_len, n_embd] format and compute
 * 
 * @param x Input tensor [batch, seq_len, n_embd]
 * @param q_weight Query projection weight [n_embd, n_embd]
 * @param k_weight Key projection weight [n_embd, n_embd]
 * @param v_weight Value projection weight [n_embd, n_embd]
 * @param o_weight Output projection weight [n_embd, n_embd]
 * @param num_heads Number of attention heads
 * @param causal Whether to use causal mask
 * @return Attention output [batch, seq_len, n_embd]
 */
TensorPtr memory_first_multihead_attention(
    const TensorPtr& x,
    const TensorPtr& q_weight,
    const TensorPtr& k_weight,
    const TensorPtr& v_weight,
    const TensorPtr& o_weight,
    int num_heads,
    bool causal = false
);

} // namespace memory_first
} // namespace ops

