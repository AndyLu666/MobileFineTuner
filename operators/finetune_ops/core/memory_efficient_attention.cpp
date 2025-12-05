/**
 * @file memory_efficient_attention.cpp
 * @brief 内存高效注意力实现
 */

#include "memory_efficient_attention.h"
#include "ops.h"
#include <cmath>
#include <limits>
#include <algorithm>

namespace ops {

void online_softmax_weighted_sum(
    const float* logits,
    const float* values,
    int64_t seq_len,
    int64_t head_dim,
    float* output,
    float max_val
) {
    // 第一遍：计算归一化分母（使用 max_val 数值稳定）
    double sum_exp = 0.0;
    for (int64_t j = 0; j < seq_len; ++j) {
        sum_exp += std::exp(logits[j] - max_val);
    }
    
    // 第二遍：计算归一化权重并累积到 output
    float inv_sum = 1.0f / static_cast<float>(sum_exp);
    for (int64_t j = 0; j < seq_len; ++j) {
        float weight = std::exp(logits[j] - max_val) * inv_sum;
        const float* v_row = values + j * head_dim;
        
        for (int64_t d = 0; d < head_dim; ++d) {
            output[d] += weight * v_row[d];
        }
    }
}

TensorPtr memory_efficient_attention(
    const TensorPtr& q,
    const TensorPtr& k,
    const TensorPtr& v,
    const TensorPtr& causal_mask,
    const MemoryEfficientAttentionConfig& config
) {
    // 验证输入形状
    const auto& q_shape = q->shape();
    const auto& k_shape = k->shape();
    const auto& v_shape = v->shape();
    
    if (q_shape.size() != 4 || k_shape.size() != 4 || v_shape.size() != 4) {
        throw std::runtime_error("memory_efficient_attention: inputs must be 4D [B,H,S,D]");
    }
    
    int64_t batch = q_shape[0];
    int64_t n_head = q_shape[1];
    int64_t seq_len = q_shape[2];
    int64_t head_dim = q_shape[3];
    
    // 验证 k/v 形状匹配
    if (k_shape[0] != batch || k_shape[1] != n_head || k_shape[2] != seq_len || k_shape[3] != head_dim) {
        throw std::runtime_error("memory_efficient_attention: k shape mismatch");
    }
    if (v_shape[0] != batch || v_shape[1] != n_head || v_shape[2] != seq_len || v_shape[3] != head_dim) {
        throw std::runtime_error("memory_efficient_attention: v shape mismatch");
    }
    
    // 自动计算缩放因子
    float scale = (config.scale > 0) ? config.scale : (1.0f / std::sqrt(static_cast<float>(head_dim)));
    
    // 准备 causal mask（如果提供）
    const float* mask_data = nullptr;
    if (causal_mask) {
        if (causal_mask->shape().size() != 2 || 
            causal_mask->shape()[0] != seq_len || 
            causal_mask->shape()[1] != seq_len) {
            throw std::runtime_error("memory_efficient_attention: causal_mask must be [S,S]");
        }
        mask_data = causal_mask->data<float>();
    }
    
    // 创建输出张量
    auto context = zeros({batch, n_head, seq_len, head_dim}, kFloat32, kCPU);
    
    const float* q_data = q->data<float>();
    const float* k_data = k->data<float>();
    const float* v_data = v->data<float>();
    float* ctx_data = context->data<float>();
    
    // 主循环：逐 batch、逐 head、逐 query 行计算
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t h = 0; h < n_head; ++h) {
            // 当前 head 的基址
            int64_t head_offset = (b * n_head + h) * seq_len * head_dim;
            const float* q_head = q_data + head_offset;
            const float* k_head = k_data + head_offset;
            const float* v_head = v_data + head_offset;
            float* ctx_head = ctx_data + head_offset;
            
            // 逐行处理（每个 query position）
            for (int64_t i = 0; i < seq_len; ++i) {
                const float* q_row = q_head + i * head_dim;
                float* ctx_row = ctx_head + i * head_dim;
                
                // === 第一遍：计算 scores 并找最大值（数值稳定）===
                // 复用同一行缓冲，避免每行反复分配释放造成堆膨胀
                static thread_local std::vector<float> scores_buf;
                if (scores_buf.size() < static_cast<size_t>(seq_len)) {
                    scores_buf.resize(seq_len);
                }
                float* scores = scores_buf.data();
                float max_score = -std::numeric_limits<float>::infinity();
                
                for (int64_t j = 0; j < seq_len; ++j) {
                    // 计算点积：q[i] · k[j]
                    const float* k_row = k_head + j * head_dim;
                    float dot = 0.0f;
                    for (int64_t d = 0; d < head_dim; ++d) {
                        dot += q_row[d] * k_row[d];
                    }
                    
                    // 缩放
                    float score = dot * scale;
                    
                    // 应用因果掩码
                    if (config.use_causal_mask && j > i) {
                        score = -1e10f;  // 上三角屏蔽
                    }
                    
                    // 应用额外掩码（如果提供）
                    if (mask_data) {
                        score += mask_data[i * seq_len + j];
                    }
                    
                    scores[j] = score; // 写入可复用缓冲
                    max_score = std::max(max_score, score);
                }
                
                // === 第二遍：在线计算 softmax 并累积到 context ===
                // 初始化输出行
                std::fill(ctx_row, ctx_row + head_dim, 0.0f);
                
                // 计算归一化分母
                double sum_exp = 0.0;
                for (int64_t j = 0; j < seq_len; ++j) {
                    sum_exp += std::exp(scores[j] - max_score);
                }
                
                // 计算加权和
                float inv_sum = 1.0f / static_cast<float>(sum_exp);
                for (int64_t j = 0; j < seq_len; ++j) {
                    float weight = std::exp(scores[j] - max_score) * inv_sum;
                    const float* v_row = v_head + j * head_dim;
                    
                    for (int64_t d = 0; d < head_dim; ++d) {
                        ctx_row[d] += weight * v_row[d];
                    }
                }
            }
        }
    }
    
    // 设置梯度传播（如需要）
    if (q->requires_grad() || k->requires_grad() || v->requires_grad()) {
        context->set_requires_grad(true);
        
        // TODO: 实现内存高效注意力的反向传播
        // 当前策略：如需训练，暂时回退到标准注意力或手工实现梯度
        // 完整实现需要：保存 max_score/sum_exp 或在反向时重计算
        context->set_grad_fn([q, k, v, causal_mask, scale](const TensorPtr& grad_output) -> std::vector<TensorPtr> {
            // 占位实现：提示用户需要完整反向
            std::cerr << "[WARN] memory_efficient_attention backward not fully implemented yet" << std::endl;
            
            // 返回零梯度（需要时请实现完整反向或切回标准注意力）
            auto grad_q = zeros(q->shape(), q->dtype(), q->device());
            auto grad_k = zeros(k->shape(), k->dtype(), k->device());
            auto grad_v = zeros(v->shape(), v->dtype(), v->device());
            
            return {grad_q, grad_k, grad_v};
        });
    }
    
    return context;
}

} // namespace ops

