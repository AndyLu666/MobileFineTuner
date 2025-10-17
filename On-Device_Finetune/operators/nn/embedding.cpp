#include "embedding_layer.h"
#include "../core/tensor_ops.h"
#include <random>
#include <fstream>
#include <iostream>

namespace ops {

EmbeddingLayer::EmbeddingLayer(const EmbeddingConfig& config)
    : BaseLayer(config), embedding_config_(config) {
    init_parameters();
}

void EmbeddingLayer::init_parameters() {

    weight_ = std::make_shared<Tensor>(
        std::vector<int>{embedding_config_.vocab_size, embedding_config_.embedding_dim});

    std::random_device rd;
    std::mt19937 gen(rd());
    float std_dev = std::sqrt(2.0f / (embedding_config_.vocab_size + embedding_config_.embedding_dim));
    std::normal_distribution<float> dist(0.0f, std_dev);

    for (auto& val : weight_->data()) {
        val = dist(gen);
    }

    if (config_.trainable) {
        grad_weight_ = std::make_shared<Tensor>(weight_->shape());
        std::fill(grad_weight_->data().begin(), grad_weight_->data().end(), 0.0f);
    }

    std::cout << "EmbeddingLayer: initialize [" << embedding_config_.vocab_size
              << ", " << embedding_config_.embedding_dim << "] weight" << std::endl;
}

TensorPtr EmbeddingLayer::forward(const Tensor& input, bool training) {

    std::vector<int> input_ids;
    const auto& input_data = input.data();

    for (float val : input_data) {
        input_ids.push_back(static_cast<int>(val));
    }

    return forward(input_ids, training);
}

TensorPtr EmbeddingLayer::forward(const std::vector<int>& input_ids, bool training) {
    int seq_len = input_ids.size();

    auto output = std::make_shared<Tensor>(
        std::vector<int>{seq_len, embedding_config_.embedding_dim});

    embed_tokens(input_ids, output);

    if (training && config_.trainable) {
        last_input_ids_ = input_ids;
        last_output_ = output;
    }

    return output;
}

void EmbeddingLayer::embed_tokens(const std::vector<int>& input_ids, TensorPtr& output) {
    int seq_len = input_ids.size();
    int emb_dim = embedding_config_.embedding_dim;

    auto& output_data = output->data();
    const auto& weight_data = weight_->data();

    for (int i = 0; i < seq_len; ++i) {
        int token_id = input_ids[i];

        if (token_id >= 0 && token_id < embedding_config_.vocab_size) {
            for (int j = 0; j < emb_dim; ++j) {
                output_data[i * emb_dim + j] = weight_data[token_id * emb_dim + j];
            }
        } else {

            for (int j = 0; j < emb_dim; ++j) {
                output_data[i * emb_dim + j] = 0.0f;
            }
        }
    }
}

TensorPtr EmbeddingLayer::backward(const Tensor& grad_output) {
    if (!config_.trainable || last_input_ids_.empty()) {
        return nullptr;
    }

    update_embedding_gradients(last_input_ids_, std::make_shared<Tensor>(grad_output));

    return nullptr;
}

void EmbeddingLayer::update_embedding_gradients(const std::vector<int>& input_ids,
                                               const TensorPtr& grad_output) {
    int seq_len = input_ids.size();
    int emb_dim = embedding_config_.embedding_dim;

    auto& grad_weight_data = grad_weight_->data();
    const auto& grad_output_data = grad_output->data();

    for (int i = 0; i < seq_len; ++i) {
        int token_id = input_ids[i];

        if (token_id >= 0 && token_id < embedding_config_.vocab_size) {
            for (int j = 0; j < emb_dim; ++j) {
                grad_weight_data[token_id * emb_dim + j] += grad_output_data[i * emb_dim + j];
            }
        }
    }
}

std::vector<TensorPtr> EmbeddingLayer::get_parameters() {
    return {weight_};
}

std::vector<TensorPtr> EmbeddingLayer::get_gradients() {
    if (config_.trainable) {
        return {grad_weight_};
    }
    return {};
}

void EmbeddingLayer::update_parameters(const std::vector<TensorPtr>& updates) {
    if (config_.trainable && !updates.empty()) {
        auto& weight_data = weight_->data();
        const auto& update_data = updates[0]->data();

        for (size_t i = 0; i < weight_data.size(); ++i) {
            weight_data[i] += update_data[i];
        }
    }
}

void EmbeddingLayer::save_weights(const std::string& path) const {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file for writing: " + path);
    }

    file.write(reinterpret_cast<const char*>(&embedding_config_.vocab_size), sizeof(int));
    file.write(reinterpret_cast<const char*>(&embedding_config_.embedding_dim), sizeof(int));

    const auto& weight_data = weight_->data();
    file.write(reinterpret_cast<const char*>(weight_data.data()),
               weight_data.size() * sizeof(float));
}

void EmbeddingLayer::load_weights(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file for reading: " + path);
    }

    int vocab_size, embedding_dim;
    file.read(reinterpret_cast<char*>(&vocab_size), sizeof(int));
    file.read(reinterpret_cast<char*>(&embedding_dim), sizeof(int));

    if (vocab_size != embedding_config_.vocab_size ||
        embedding_dim != embedding_config_.embedding_dim) {
        throw std::runtime_error("Configuration mismatch when loading embedding weights");
    }

    auto& weight_data = weight_->data();
    file.read(reinterpret_cast<char*>(weight_data.data()),
              weight_data.size() * sizeof(float));
}

size_t EmbeddingLayer::get_param_count() const {
    return weight_->size();
}

void EmbeddingLayer::set_weights(const TensorPtr& weights) {
    if (weights->shape() != weight_->shape()) {
        throw std::runtime_error("Weight shape mismatch");
    }

    weight_ = weights;
}

PositionalEmbeddingLayer::PositionalEmbeddingLayer(int max_position, int embedding_dim, bool trainable)
    : BaseLayer(LayerConfig(embedding_dim, embedding_dim, trainable, 0.0f)),
      max_position_(max_position), embedding_dim_(embedding_dim), last_seq_len_(0) {
    init_parameters();
}

void PositionalEmbeddingLayer::init_parameters() {

    position_embeddings_ = std::make_shared<Tensor>(
        std::vector<int>{max_position_, embedding_dim_});

    auto& pos_data = position_embeddings_->data();

    for (int pos = 0; pos < max_position_; ++pos) {
        for (int i = 0; i < embedding_dim_; ++i) {
            float angle = pos / std::pow(10000.0f, 2.0f * (i / 2) / embedding_dim_);

            if (i % 2 == 0) {
                pos_data[pos * embedding_dim_ + i] = std::sin(angle);
            } else {
                pos_data[pos * embedding_dim_ + i] = std::cos(angle);
            }
        }
    }

    if (config_.trainable) {
        grad_position_embeddings_ = std::make_shared<Tensor>(position_embeddings_->shape());
        std::fill(grad_position_embeddings_->data().begin(),
                  grad_position_embeddings_->data().end(), 0.0f);
    }

    std::cout << "PositionalEmbeddingLayer: initialize [" << max_position_
              << ", " << embedding_dim_ << "] positionembedding" << std::endl;
}

TensorPtr PositionalEmbeddingLayer::forward(const Tensor& input, bool training) {

    int seq_len = input.shape()[0];
    return forward(seq_len, training);
}

TensorPtr PositionalEmbeddingLayer::forward(int sequence_length, bool training) {
    if (sequence_length > max_position_) {
        throw std::runtime_error("Sequence length exceeds maximum position");
    }

    auto output = std::make_shared<Tensor>(
        std::vector<int>{sequence_length, embedding_dim_});

    auto& output_data = output->data();
    const auto& pos_data = position_embeddings_->data();

    for (int i = 0; i < sequence_length; ++i) {
        for (int j = 0; j < embedding_dim_; ++j) {
            output_data[i * embedding_dim_ + j] = pos_data[i * embedding_dim_ + j];
        }
    }

    if (training && config_.trainable) {
        last_seq_len_ = sequence_length;
        last_output_ = output;
    }

    return output;
}

TensorPtr PositionalEmbeddingLayer::backward(const Tensor& grad_output) {
    if (!config_.trainable || last_seq_len_ == 0) {
        return nullptr;
    }

    auto& grad_pos_data = grad_position_embeddings_->data();
    const auto& grad_output_data = grad_output.data();

    for (int i = 0; i < last_seq_len_; ++i) {
        for (int j = 0; j < embedding_dim_; ++j) {
            grad_pos_data[i * embedding_dim_ + j] += grad_output_data[i * embedding_dim_ + j];
        }
    }

    return nullptr;
}

std::vector<TensorPtr> PositionalEmbeddingLayer::get_parameters() {
    return {position_embeddings_};
}

std::vector<TensorPtr> PositionalEmbeddingLayer::get_gradients() {
    if (config_.trainable) {
        return {grad_position_embeddings_};
    }
    return {};
}

void PositionalEmbeddingLayer::update_parameters(const std::vector<TensorPtr>& updates) {
    if (config_.trainable && !updates.empty()) {
        auto& pos_data = position_embeddings_->data();
        const auto& update_data = updates[0]->data();

        for (size_t i = 0; i < pos_data.size(); ++i) {
            pos_data[i] += update_data[i];
        }
    }
}

void PositionalEmbeddingLayer::save_weights(const std::string& path) const {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file for writing: " + path);
    }

    file.write(reinterpret_cast<const char*>(&max_position_), sizeof(int));
    file.write(reinterpret_cast<const char*>(&embedding_dim_), sizeof(int));

    const auto& pos_data = position_embeddings_->data();
    file.write(reinterpret_cast<const char*>(pos_data.data()),
               pos_data.size() * sizeof(float));
}

void PositionalEmbeddingLayer::load_weights(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file for reading: " + path);
    }

    int max_pos, emb_dim;
    file.read(reinterpret_cast<char*>(&max_pos), sizeof(int));
    file.read(reinterpret_cast<char*>(&emb_dim), sizeof(int));

    if (max_pos != max_position_ || emb_dim != embedding_dim_) {
        throw std::runtime_error("Configuration mismatch when loading positional embeddings");
    }

    auto& pos_data = position_embeddings_->data();
    file.read(reinterpret_cast<char*>(pos_data.data()),
              pos_data.size() * sizeof(float));
}

size_t PositionalEmbeddingLayer::get_param_count() const {
    return position_embeddings_->size();
}

}
