/**
 * @file lm_loss.cpp
 * @brief 语言模型损失函数实现
 */

#include "lm_loss.h"
#include "backward_functions.h"
#include "ops.h"
#ifdef USE_NEW_AUTOGRAD_ENGINE
#include "autograd_engine.h"
#endif
#include <cmath>
#include <limits>
#include <algorithm>

namespace ops {

// 语言模型Cross-Entropy的反向传播
class LMCrossEntropyBackward : public BackwardFunction {
public:
    LMCrossEntropyBackward(const TensorPtr& logits, const TensorPtr& labels,
                          int ignore_idx, size_t valid_cnt, const std::string& red)
        : logits_(logits), labels_(labels), ignore_index_(ignore_idx),
          valid_count_(valid_cnt), reduction_(red) {}
    
    std::vector<TensorPtr> apply(const TensorPtr& grad_output) override {
        // 形状（HF-style shift: logits[:, :-1] 对应 labels[:, 1:]）
        const auto& shape = logits_->shape();
        int64_t B = shape[0];
        int64_t S = shape[1];
        int64_t V = shape[2];
        int64_t S_eff = (S > 0) ? (S - 1) : 0;
        
        // 创建梯度张量
        auto grad_logits = zeros({B, S, V}, kFloat32, kCPU);
        float* grad_data = grad_logits->data<float>();
        const float* logits_data = logits_->data<float>();
        const int32_t* labels_data = labels_->data<int32_t>();
        
        // 缩放系数（mean 按有效token计数归一）
        float scale_base = 1.0f;
        if (reduction_ == "mean") {
            scale_base = (valid_count_ > 0) ? (1.0f / static_cast<float>(valid_count_)) : 0.0f;
        }
        
        // 逐token计算梯度（只覆盖 logits 的前 S-1 位）
        for (int64_t b = 0; b < B; ++b) {
            for (int64_t s = 0; s < S_eff; ++s) {
                int32_t y = labels_data[b * S + (s + 1)];
                if (y == ignore_index_ || y < 0 || y >= V) {
                    // 忽略项：梯度为0
                    continue;
                }
                
                // 计算该位置的softmax（数值稳定）
                const float* logit_row = logits_data + (b * S + s) * V;
                
                // 找最大值
                float max_val = -std::numeric_limits<float>::infinity();
                for (int64_t v = 0; v < V; ++v) {
                    max_val = std::max(max_val, logit_row[v]);
                }
                
                // 计算分母
                float denom = 0.0f;
                for (int64_t v = 0; v < V; ++v) {
                    denom += std::exp(logit_row[v] - max_val);
                }
                
                // 写入梯度：grad = (softmax - one_hot(y)) * local_coeff
                float inv_denom = 1.0f / denom;
                // grad_output：mean/sum 为标量；none 为逐token权重
                float outer = 1.0f;
                if (grad_output && grad_output->numel() > 0) {
                    if (reduction_ == "none") {
                        outer = grad_output->data<float>()[b * S + s];
                    } else {
                        outer = grad_output->data<float>()[0];
                    }
                }
                float coeff = outer * scale_base;
                
                float* grad_row = grad_data + (b * S + s) * V;
                for (int64_t v = 0; v < V; ++v) {
                    float p = std::exp(logit_row[v] - max_val) * inv_denom;
                    float g = p;
                    if (v == y) {
                        g -= 1.0f;
                    }
                    grad_row[v] = g * coeff;
                }
            }
        }
        
        return {grad_logits};
    }
    
private:
    TensorPtr logits_;
    TensorPtr labels_;
    int ignore_index_;
    size_t valid_count_;
    std::string reduction_;
};

TensorPtr lm_cross_entropy(const TensorPtr& logits,
                           const TensorPtr& labels,
                           int ignore_index,
                           const std::string& reduction) {
    // 支持 [B,S,V] × [B,S]，按有效token（label!=ignore_index）做归一
    if (logits->ndim() != 3) {
        throw std::runtime_error("lm_cross_entropy: logits must be [B,S,V]");
    }
    if (labels->ndim() != 2) {
        throw std::runtime_error("lm_cross_entropy: labels must be [B,S]");
    }

    const auto& shape = logits->shape();
    const int64_t B = shape[0];
    const int64_t S = shape[1];
    const int64_t V = shape[2];

    // 前向：数值稳定的 masked NLL（仅用于标量值；梯度由自定义Backward提供）
    const float* x = logits->data<float>();
    const int32_t* y = labels->data<int32_t>();

    double loss_sum = 0.0;
    int64_t valid_cnt = 0;

    for (int64_t b = 0; b < B; ++b) {
        for (int64_t s = 0; s + 1 < S; ++s) {  // shift: logits[:-1] vs labels[1:]
            int32_t cls = y[b * S + (s + 1)];
            if (cls == ignore_index) continue;
            if (cls < 0 || cls >= V) continue;

            const float* row = x + (b * S + s) * V;
            // max for stability
            float max_val = -std::numeric_limits<float>::infinity();
            for (int64_t v = 0; v < V; ++v) max_val = std::max(max_val, row[v]);
            // logsumexp
            double denom = 0.0;
            for (int64_t v = 0; v < V; ++v) denom += std::exp(row[v] - max_val);
            double log_sum_exp = max_val + std::log(denom);
            // NLL
            loss_sum += (log_sum_exp - row[cls]);
            valid_cnt++;
        }
    }

    float out_val = 0.0f;
    if (reduction == "none") {
        // 返回 [B,S] 逐token NLL（无效位=0），仅计算 logits[:-1] × labels[1:]
        auto out = zeros({B, S}, kFloat32, kCPU);
        float* out_data = out->data<float>();
        for (int64_t b = 0; b < B; ++b) {
            for (int64_t s = 0; s + 1 < S; ++s) {
                int32_t cls = y[b * S + (s + 1)];
                if (cls == ignore_index || cls < 0 || cls >= V) { out_data[b * S + s] = 0.0f; continue; }
                const float* row = x + (b * S + s) * V;
                float max_val = -std::numeric_limits<float>::infinity();
                for (int64_t v = 0; v < V; ++v) max_val = std::max(max_val, row[v]);
                double denom = 0.0;
                for (int64_t v = 0; v < V; ++v) denom += std::exp(row[v] - max_val);
                double log_sum_exp = max_val + std::log(denom);
                out_data[b * S + s] = static_cast<float>(log_sum_exp - row[cls]);
            }
        }
        // 挂接Backward（仅对logits回传；labels不需要梯度）
        if (logits->requires_grad()) {
            out->set_requires_grad(true);
            auto backward_fn = std::make_shared<LMCrossEntropyBackward>(logits, labels, ignore_index,
                                                                       static_cast<size_t>(valid_cnt), reduction);
            #ifdef USE_NEW_AUTOGRAD_ENGINE
            autograd::Engine::instance().register_node(out, {logits}, backward_fn);
            #else
            out->set_grad_fn([backward_fn, logits](const TensorPtr& grad_output) -> std::vector<TensorPtr> {
                auto grads = backward_fn->apply(grad_output);
                if (logits->requires_grad()) accumulate_gradient(logits, grads[0]);
                return grads;
            });
            #endif
        }
        return out;
    } else if (reduction == "sum" || reduction == "sum_debug") {
        // sum_debug: 用于对齐调试，始终返回标量 sum，并在 backward 中不做 valid_count 归一
        out_val = static_cast<float>(loss_sum);
        if (reduction == "sum_debug") {
            valid_cnt = 1;  // backward 时 scale_base = 1
        }
    } else { // mean（默认）
        out_val = (valid_cnt > 0) ? static_cast<float>(loss_sum / static_cast<double>(valid_cnt)) : 0.0f;
    }

    auto out = full({1}, out_val, kFloat32, kCPU);
    if (logits->requires_grad()) {
        out->set_requires_grad(true);
        auto backward_fn = std::make_shared<LMCrossEntropyBackward>(logits, labels, ignore_index,
                                                                   static_cast<size_t>(valid_cnt), reduction);
        #ifdef USE_NEW_AUTOGRAD_ENGINE
        autograd::Engine::instance().register_node(out, {logits}, backward_fn);
        #else
        out->set_grad_fn([backward_fn, logits](const TensorPtr& grad_output) -> std::vector<TensorPtr> {
            auto grads = backward_fn->apply(grad_output);
            if (logits->requires_grad()) accumulate_gradient(logits, grads[0]);
            return grads;
        });
        #endif
    }
    return out;
}

}  // namespace ops
