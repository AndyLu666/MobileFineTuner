#pragma once

#include "../core/tensor.h"
#include <vector>
#include <memory>
#include <string>

namespace ops {

struct EmbeddingConfig {
    int vocab_size{50257};
    int embedding_dim{768};
    bool trainable{true};
    float dropout{0.0f};   // 目前未使用，占位以兼容旧配置
    int max_norm{-1};      // 未实现裁剪逻辑，占位
    bool sparse{false};    // 未实现稀疏更新，占位
};

class EmbeddingLayer {
public:
    explicit EmbeddingLayer(const EmbeddingConfig& config);
    ~EmbeddingLayer() = default;

    // 输入为 token id 列表
    TensorPtr forward(const std::vector<int>& input_ids, bool training = false);
    // 输入为整型张量 [S] 或 [B,S]（int32），返回 [S,E] 或 [B,S,E]
    TensorPtr forward(const TensorPtr& input_ids, bool training = false);

    // 获取/设置参数与梯度
    std::vector<TensorPtr> get_parameters() const;
    std::vector<TensorPtr> get_gradients() const;
    void zero_grad();
    // 简单的参数更新（直接加到权重上，用于外部优化器已计算好的更新量）
    void update_parameters(const std::vector<TensorPtr>& updates);

    void save_weights(const std::string& path) const;
    void load_weights(const std::string& path);

    size_t get_param_count() const;
    void set_weights(const TensorPtr& weights);
    TensorPtr get_embedding_weights() const { return weight_; }

private:
    EmbeddingConfig config_;
    TensorPtr weight_;       // [vocab_size, embedding_dim]
    TensorPtr grad_weight_;  // 与 weight_ 同形状

    // 为简化，与旧实现保持少量状态（可选）
    std::vector<int> last_input_ids_;
    TensorPtr last_output_;
};

class PositionalEmbeddingLayer {
public:
    PositionalEmbeddingLayer(int max_position, int embedding_dim, bool trainable = true);
    ~PositionalEmbeddingLayer() = default;

    // 从输入推断序列长度
    TensorPtr forward(const TensorPtr& input, bool training = false);
    // 或者直接指定序列长度
    TensorPtr forward(int sequence_length, bool training = false);

    std::vector<TensorPtr> get_parameters() const;
    std::vector<TensorPtr> get_gradients() const;
    void zero_grad();
    void update_parameters(const std::vector<TensorPtr>& updates);

    void save_weights(const std::string& path) const;
    void load_weights(const std::string& path);

    size_t get_param_count() const;

private:
    int max_position_;
    int embedding_dim_;
    bool trainable_;

    TensorPtr position_embeddings_;        // [max_position, embedding_dim]
    TensorPtr grad_position_embeddings_;   // 同形状

    int last_seq_len_{0};
    TensorPtr last_output_;
};

}
