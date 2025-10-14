/**
 * @file chunked_softmax_ce.h
 * @brief Precise chunked Softmax + CrossEntropy (with backward propagation)
 * 
 * Core idea:
 * - Forward: Chunk lm_head along vocabulary dimension, use streaming logsumexp to accumulate normalization term
 * - Backward: Recompute softmax block by block, accumulate gradients to W.grad and X.grad
 * - Fully equivalent to standard implementation, but peak memory reduced from O(B*L*V) to O(B*L*C)
 * 
 * Use case: Language model output layer with very large vocabulary (V=262K)
 */

#pragma once

#include "tensor.h"
#include <cmath>
#include <limits>
#include <algorithm>

namespace ops {
namespace chunked_ce {

/**
 * @brief Streaming LogSumExp state
 */
struct StreamingLogSumExpState {
    float max_logit = -std::numeric_limits<float>::infinity();
    float sum_exp = 0.0f;
    
    void update(const float* logits, int64_t size) {
        // First pass: Find maximum value
        float local_max = max_logit;
        for (int64_t i = 0; i < size; ++i) {
            local_max = std::max(local_max, logits[i]);
        }
        
        // If new max is larger, need to rescale previous sum_exp
        if (local_max > max_logit) {
            sum_exp *= std::exp(max_logit - local_max);
            max_logit = local_max;
        }
        
        // Second pass: Accumulate exp
        for (int64_t i = 0; i < size; ++i) {
            sum_exp += std::exp(logits[i] - max_logit);
        }
    }
    
    float get_log_sum_exp() const {
        return max_logit + std::log(sum_exp);
    }
};

/**
 * @brief Precise chunked cross-entropy forward (without generating full logits)
 * 
 * @param X Input features [B, L, D]
 * @param W lm_head weights [V, D] (transposed) or [D, V]
 * @param targets Target classes [B, L] (int32)
 * @param chunk_size Chunk size (e.g., 2048/4096)
 * @param W_is_transposed Whether W is already transposed to [D, V]
 * @return loss scalar
 * 
 * Algorithm:
 * 1. For each (b, l), compute logits_chunk = X[b,l] @ W_chunk in chunks
 * 2. Use streaming logsumexp to accumulate normalization term
 * 3. Record logit for target class (only one scalar needed)
 * 4. Final loss = -mean(target_logit - logsumexp)
 */
TensorPtr chunked_cross_entropy_forward(
    const TensorPtr& X,           // [B, L, D]
    const TensorPtr& W,           // [V, D] or [D, V]
    const TensorPtr& targets,     // [B, L] int32
    int64_t chunk_size = 2048,
    bool W_is_transposed = false
);

/**
 * @brief Precise chunked cross-entropy backward (accumulate gradients block by block)
 * 
 * @param X Input features [B, L, D]
 * @param W lm_head weights [V, D] or [D, V]
 * @param targets Target classes [B, L]
 * @param grad_output Upstream gradient (scalar, usually 1/N)
 * @param chunk_size Chunk size
 * @param W_is_transposed Whether W is transposed
 * @return (grad_X, grad_W) gradient tensor pair
 * 
 * Algorithm:
 * 1. Recompute logits_chunk and softmax_chunk in blocks
 * 2. Calculate (p_chunk - y_one_hot_chunk) * grad_output
 * 3. Accumulate W.grad += X^T @ (p - y)
 * 4. Accumulate X.grad += (p - y) @ W^T
 */
std::pair<TensorPtr, TensorPtr> chunked_cross_entropy_backward(
    const TensorPtr& X,
    const TensorPtr& W,
    const TensorPtr& targets,
    float grad_output,
    int64_t chunk_size = 2048,
    bool W_is_transposed = false
);

/**
 * @brief Wrap complete forward+backward (auto-registered to computation graph)
 * 
 * @param X Input features [B, L, D]
 * @param W lm_head weights (supports [V,D] or [D,V])
 * @param targets Target classes [B, L]
 * @param chunk_size Chunk size
 * @return loss scalar tensor (supports backward)
 */
TensorPtr chunked_cross_entropy_loss(
    const TensorPtr& X,
    const TensorPtr& W,
    const TensorPtr& targets,
    int64_t chunk_size = 2048
);

} // namespace chunked_ce
} // namespace ops

