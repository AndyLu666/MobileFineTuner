#pragma once

#include "../core/grad_tensor.h"
#include "base_layer.h"
#include <vector>
#include <memory>

namespace ops {

struct EmbeddingConfig : public LayerConfig {
    int vocab_size;
    int embedding_dim;
    int max_norm;
    float scale_grad_by_freq;
    bool sparse;

    EmbeddingConfig() : LayerConfig(),
        vocab_size(50257), embedding_dim(768), max_norm(-1),
        scale_grad_by_freq(0.0f), sparse(false) {}

    EmbeddingConfig(
        int vocab_size_,
        int embedding_dim_,
        bool trainable_ = true,
        float dropout_ = 0.0f
    ) : LayerConfig(embedding_dim_, embedding_dim_, trainable_, dropout_),
        vocab_size(vocab_size_),
        embedding_dim(embedding_dim_),
        max_norm(-1),
        scale_grad_by_freq(0.0f),
        sparse(false) {}
};

class EmbeddingLayer : public BaseLayer {
public:
    EmbeddingLayer(const EmbeddingConfig& config);
    ~EmbeddingLayer() override = default;

    TensorPtr forward(const Tensor& input, bool training = false) override;
    TensorPtr forward(const std::vector<int>& input_ids, bool training = false);

    TensorPtr backward(const Tensor& grad_output) override;

    std::vector<TensorPtr> get_parameters() override;
    std::vector<TensorPtr> get_gradients() override;

    void update_parameters(const std::vector<TensorPtr>& updates) override;

    void save_weights(const std::string& path) const override;
    void load_weights(const std::string& path) override;

    size_t get_param_count() const override;

    void set_weights(const TensorPtr& weights);
    TensorPtr get_embedding_weights() const { return weight_; }

protected:
    void init_parameters() override;

private:
    EmbeddingConfig embedding_config_;

    TensorPtr weight_;

    TensorPtr grad_weight_;

    std::vector<int> last_input_ids_;
    TensorPtr last_output_;

    void embed_tokens(const std::vector<int>& input_ids, TensorPtr& output);
    void update_embedding_gradients(const std::vector<int>& input_ids,
                                   const TensorPtr& grad_output);
};

class PositionalEmbeddingLayer : public BaseLayer {
public:
    PositionalEmbeddingLayer(int max_position, int embedding_dim, bool trainable = true);
    ~PositionalEmbeddingLayer() override = default;

    TensorPtr forward(const Tensor& input, bool training = false) override;
    TensorPtr forward(int sequence_length, bool training = false);

    TensorPtr backward(const Tensor& grad_output) override;

    std::vector<TensorPtr> get_parameters() override;
    std::vector<TensorPtr> get_gradients() override;

    void update_parameters(const std::vector<TensorPtr>& updates) override;

    void save_weights(const std::string& path) const override;
    void load_weights(const std::string& path) override;

    size_t get_param_count() const override;

protected:
    void init_parameters() override;

private:
    int max_position_;
    int embedding_dim_;

    TensorPtr position_embeddings_;

    TensorPtr grad_position_embeddings_;

    int last_seq_len_;
    TensorPtr last_output_;
};

}
