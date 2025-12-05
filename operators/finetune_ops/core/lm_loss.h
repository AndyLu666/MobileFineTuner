/**
 * @file lm_loss.h
 * @brief 语言模型专用损失函数（支持3D logits和ignore_index）
 */

#pragma once

#include "tensor.h"
#include <string>
#include <cmath>

namespace ops {

/**
 * @brief 语言模型Cross-Entropy Loss
 * 
 * @param logits [B, S, V] float32
 * @param labels [B, S] int32，PAD位置为ignore_index
 * @param ignore_index 忽略的标签值（默认-100）
 * @param reduction "mean" | "sum" | "none"
 * @return 
 *   - "mean": 标量loss（按有效token平均）
 *   - "sum": 标量loss（按有效token求和）
 *   - "none": [B,S] 逐token NLL
 * 
 * 特点：
 * - 数值稳定（logsumexp）
 * - 自动忽略ignore_index（不计入loss和梯度）
 * - 支持自动微分
 */
TensorPtr lm_cross_entropy(const TensorPtr& logits,
                          const TensorPtr& labels,
                          int ignore_index = -100,
                          const std::string& reduction = "mean");

/**
 * @brief 从mean NLL计算perplexity
 */
inline float perplexity_from_loss(float mean_nll) {
    return std::exp(mean_nll);
}

}  // namespace ops

