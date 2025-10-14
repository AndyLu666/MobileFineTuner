#include "operators.h"
#include "optim/adam.h"
#include "optim/mobile_optimizer_extensions.h"
#include "optim/optimizer_utils.h"
#ifdef USE_NEW_AUTOGRAD_ENGINE
#include "core/autograd_engine.h"
#endif
// Pure C++ implementation: no dependency on BLAS/Accelerate
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <chrono>
#include <random>
#include <algorithm>
#include <map>
#include <iomanip>
#include <unordered_map>
#include <numeric>
#include <cmath>
#include <cstring>
#ifdef __APPLE__
#include <sys/resource.h>
#include <sys/sysctl.h>
#include <unistd.h>
#include <mach/mach.h>
#include <mach/task.h>
#include <malloc/malloc.h>
#elif defined(_WIN32)
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#include <unistd.h>
#endif

using namespace ops;

// Simplified Memory Monitor (native GPT2 compatible)
class SimpleMemoryMonitor {
public:
    static size_t getCurrentRSS() {
        #ifdef __APPLE__
        struct task_basic_info info;
        mach_msg_type_number_t size = TASK_BASIC_INFO_COUNT;
        kern_return_t kerr = task_info(mach_task_self(), TASK_BASIC_INFO, 
                                       (task_info_t)&info, &size);
        if (kerr == KERN_SUCCESS) {
            return info.resident_size;
        }
        #endif
        return 0;
    }
    
    static std::string formatMemorySize(size_t bytes) {
        if (bytes >= 1024 * 1024 * 1024) {
            return std::to_string(bytes / (1024.0 * 1024.0 * 1024.0)) + " GB";
        } else if (bytes >= 1024 * 1024) {
            return std::to_string(bytes / (1024.0 * 1024.0)) + " MB";
        } else {
            return std::to_string(bytes / 1024.0) + " KB";
        }
    }
};

// Root fix: removed ineffective "healthy memory management" heavy logic
// Now Tensor uses MemoryManager pool, only need low-frequency RSS monitoring

TensorPtr mul_scalar(const TensorPtr& tensor, float scalar);
TensorPtr load_binary_weights(const std::string& filepath);
TensorPtr load_binary_tensor_1d(const std::string& filepath);

struct LoRAFinetuneConfig {
    int n_embd = 768;
    int n_head = 12;
    int block_size = 1024;
    int vocab_size = 50257;
    int n_layer = 12;
    
    int lora_rank = 8;
    float lora_alpha = 16.0f;
    int lora_layers = 6; // apply LoRA to last N layers (industrial default: 4-6)
    bool lora_q = true;
    bool lora_k = false;
    bool lora_v = true;
    bool lora_o = false;
    
    int batch_size = 2;
    int grad_accum_steps = 1;  // Gradient accumulation steps (effective batch = batch_size * grad_accum_steps)
    float lr = 3e-4f;
    int max_epochs = 3;
    int max_train_steps = -1;  // Max training steps (-1=unlimited, for early stopping)

    void print() const {
        std::cout << "GPT-2 LoRA Fine-tuning Configuration:" << std::endl;
        std::cout << "  Embedding Dimension: " << n_embd << std::endl;
        std::cout << "  Attention Heads: " << n_head << std::endl;
        std::cout << "  Sequence Length: " << block_size << std::endl;
        std::cout << "  Vocabulary Size: " << vocab_size << std::endl;
        std::cout << "  Num Layers: " << n_layer << std::endl;
        std::cout << "  LoRA Rank: " << lora_rank << std::endl;
        std::cout << "  LoRA Alpha: " << lora_alpha << std::endl;
        std::cout << "  LoRA Target Layers (last N): " << lora_layers << std::endl;
        std::cout << "  LoRA Targets: q=" << (lora_q?"1":"0")
                  << ", k=" << (lora_k?"1":"0")
                  << ", v=" << (lora_v?"1":"0")
                  << ", o=" << (lora_o?"1":"0") << std::endl;
        std::cout << "  Batch Size: " << batch_size << std::endl;
        std::cout << "  Gradient Accumulation Steps: " << grad_accum_steps << std::endl;
        std::cout << "  Effective Batch Size: " << (batch_size * grad_accum_steps) << std::endl;
        std::cout << "  Learning Rate: " << lr << std::endl;
        std::cout << "  Training Epochs: " << max_epochs << std::endl;
        if (max_train_steps > 0) {
            std::cout << "  Max Training Steps: " << max_train_steps << " (early stopping)" << std::endl;
        }
    }
};

class GPT2Tokenizer {
private:
    std::unordered_map<std::string, int> vocab_;
    std::vector<std::pair<std::string, std::string>> merges_;
    int pad_token_id_;

public:
    GPT2Tokenizer(const std::string& vocab_file) : pad_token_id_(50256) {
        std::ifstream file(vocab_file);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open vocabulary file: " + vocab_file);
        }

        std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        file.close();
        
        // Robust JSON parsing: handle {"token": id, ...} format
        size_t pos = 0;
        while (pos < content.length()) {
            // Find "key"
            size_t quote1 = content.find('"', pos);
            if (quote1 == std::string::npos) break;
            
            size_t quote2 = quote1 + 1;
            while (quote2 < content.length() && content[quote2] != '"') {
                if (content[quote2] == '\\') quote2++;  // Skip escape
                quote2++;
            }
            if (quote2 >= content.length()) break;
            
            std::string key = content.substr(quote1 + 1, quote2 - quote1 - 1);
            
            // Find number after :
            size_t colon = content.find(':', quote2);
            if (colon == std::string::npos) break;
            
            size_t num_start = colon + 1;
            while (num_start < content.length() && std::isspace(content[num_start])) num_start++;
            
            size_t num_end = num_start;
            bool is_negative = (num_start < content.length() && content[num_start] == '-');
            if (is_negative) num_end++;
            
            while (num_end < content.length() && std::isdigit(static_cast<unsigned char>(content[num_end]))) {
                num_end++;
            }
            
            if (num_end > num_start) {
                try {
                    int token_id = std::stoi(content.substr(num_start, num_end - num_start));
                    vocab_[key] = token_id;
                } catch (...) {
                    // Skip failed parsing entries
                }
            }
            
            pos = num_end;
        }

        std::string merges_file = vocab_file.substr(0, vocab_file.find_last_of('/')) + "/merges.txt";
        std::ifstream merge_stream(merges_file);
        if (merge_stream.is_open()) {
            std::string line;
            std::getline(merge_stream, line);
            while (std::getline(merge_stream, line)) {
                if (line.empty()) continue;
                size_t space_pos = line.find(' ');
                if (space_pos != std::string::npos) {
                    std::string first = line.substr(0, space_pos);
                    std::string second = line.substr(space_pos + 1);
                    merges_.push_back({first, second});
                }
            }
        }

        std::cout << "Loaded GPT-2 vocabulary: " << vocab_.size() << " tokens" << std::endl;
        std::cout << "Loaded BPE merge rules: " << merges_.size() << " rules" << std::endl;
        
        // Validate vocabulary size (GPT-2 standard is 50257)
        if (vocab_.size() < 50000) {
            std::cerr << "Warning: vocabulary size " << vocab_.size() << " is much smaller than standard GPT-2's 50257" << std::endl;
            std::cerr << "   This may cause training anomalies. Please check if vocab.json file is complete." << std::endl;
        }
    }

    std::string preprocess_text(const std::string& text) const {
        std::string result = text;
        for (size_t i = 0; i < result.length(); ++i) {
            if (result[i] == ' ') {
                result = result.substr(0, i) + "Ġ" + result.substr(i + 1);
            }
        }
        return result;
    }

    std::vector<int> encode(const std::string& text) const {
        std::string processed_text = preprocess_text(text);
        std::vector<int> tokens;

        for (size_t i = 0; i < processed_text.length(); ) {
            std::string best_match;
            int best_id = pad_token_id_;
            
            for (size_t len = std::min(processed_text.length() - i, size_t(20)); len > 0; --len) {
                std::string candidate = processed_text.substr(i, len);
                auto it = vocab_.find(candidate);
                if (it != vocab_.end()) {
                    best_match = candidate;
                    best_id = it->second;
                    break;
                }
            }
            
            tokens.push_back(best_id);
            i += std::max(size_t(1), best_match.length());
        }

        return tokens;
    }

    int get_pad_token_id() const { return pad_token_id_; }
    int vocab_size() const { return static_cast<int>(vocab_.size()); }
};

class WikiTextDataLoader {
private:
    std::vector<std::vector<int>> sequences_;
    std::vector<size_t> indices_;
    int block_size_;
    size_t current_idx_;
    std::unique_ptr<GPT2Tokenizer> tokenizer_;
    std::mt19937 rng_;

public:
    WikiTextDataLoader(const std::string& file_path, const std::string& vocab_file, int block_size, int max_sequences = -1, float data_fraction = 1.0f)
        : block_size_(block_size), current_idx_(0), rng_(std::random_device{}()) {

        tokenizer_ = std::make_unique<GPT2Tokenizer>(vocab_file);

        std::ifstream file(file_path);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + file_path);
        }

        std::string line;
        int count = 0;

        while (std::getline(file, line) && (max_sequences == -1 || count < max_sequences)) {
            if (line.empty()) continue;

            size_t text_start = line.find("\"text\":\"") + 8;
            size_t text_end = line.find_last_of('"');

            if (text_start != std::string::npos && text_end != std::string::npos && text_end > text_start) {
                std::string text = line.substr(text_start, text_end - text_start);
                
                auto tokens = tokenizer_->encode(text);
                std::vector<int> padded_tokens = tokens;
                if (padded_tokens.size() < static_cast<size_t>(block_size)) {
                    padded_tokens.resize(block_size, tokenizer_->get_pad_token_id());
                }
                
                for (size_t i = 0; i <= padded_tokens.size() - block_size; i += block_size) {
                    std::vector<int> seq(padded_tokens.begin() + i, padded_tokens.begin() + i + block_size);
                    sequences_.push_back(seq);
                    count++;
                    if (max_sequences != -1 && count >= max_sequences) break;
                }
            }
        }

        // Limit sequence count based on data_fraction
        if (data_fraction < 1.0f) {
            size_t sequences_to_keep = static_cast<size_t>(sequences_.size() * data_fraction);
            if (sequences_to_keep == 0) sequences_to_keep = 1; // Keep at least 1 sequence
            sequences_.resize(sequences_to_keep);
        }

        indices_.resize(sequences_.size());
        std::iota(indices_.begin(), indices_.end(), 0);
        shuffle_data();

        std::cout << "Loaded " << sequences_.size() << " WikiText sequences using GPT-2 tokenizer";
        if (data_fraction < 1.0f) {
            std::cout << " (using " << std::fixed << std::setprecision(1) << (data_fraction * 100) << "% data)";
        }
        std::cout << std::endl;
    }
    
    void shuffle_data() {
        std::shuffle(indices_.begin(), indices_.end(), rng_);
        current_idx_ = 0;
        std::cout << "Data shuffled, starting new epoch" << std::endl;
    }
    
    bool is_epoch_complete() const {
        return current_idx_ >= sequences_.size();
    }

    int vocab_size() const { return tokenizer_->vocab_size(); }

    std::pair<TensorPtr, TensorPtr> get_batch(int batch_size) {
        auto input_ids = zeros({batch_size, block_size_ - 1}, DType::kInt32, DeviceManager::cpu());
        auto targets = zeros({batch_size, block_size_ - 1}, DType::kInt32, DeviceManager::cpu());

        int32_t* input_data = input_ids->data<int32_t>();
        int32_t* target_data = targets->data<int32_t>();

        for (int b = 0; b < batch_size; ++b) {
            if (current_idx_ >= sequences_.size()) {
                current_idx_ = sequences_.size() - 1;
            }

            const auto& seq = sequences_[indices_[current_idx_++]];

            for (int i = 0; i < block_size_ - 1; ++i) {
                input_data[b * (block_size_ - 1) + i] = seq[i];
                target_data[b * (block_size_ - 1) + i] = seq[i + 1];  // Fix: should be int32 not float
            }
        }

        return {input_ids, targets};
    }

    size_t size() const { return sequences_.size(); }
};

class LoRALinear {
private:
    TensorPtr weight_;
    TensorPtr lora_A_;
    TensorPtr lora_B_;
    TensorPtr bias_;
    float alpha_;
    int rank_;

public:
    LoRALinear(int input_dim, int output_dim, int rank = 8, float alpha = 16.0f, bool has_bias = false) 
        : alpha_(alpha), rank_(rank) {
        
        float std_dev = 1.0f / std::sqrt(static_cast<float>(rank));
        lora_A_ = mul_scalar(randn({input_dim, rank}), std_dev);
        lora_B_ = zeros({rank, output_dim});
        
        lora_A_->set_requires_grad(true);
        lora_B_->set_requires_grad(true);
        
        if (has_bias) {
            bias_ = zeros({output_dim});
            bias_->set_requires_grad(true);
        }
    }
    
    void load_pretrained_weight(const TensorPtr& pretrained_weight, const TensorPtr& pretrained_bias = nullptr) {
        weight_ = pretrained_weight;
        weight_->set_requires_grad(false);
        
        if (pretrained_bias) {
            bias_ = pretrained_bias;
            bias_->set_requires_grad(false);
        }
    }
    
    TensorPtr forward(const TensorPtr& input) {
        // Safety checks
        if (!input) throw std::runtime_error("LoRALinear::forward: input is null");
        if (!weight_) throw std::runtime_error("LoRALinear::forward: weight is null (not loaded?)");
        if (!lora_A_) throw std::runtime_error("LoRALinear::forward: lora_A is null");
        if (!lora_B_) throw std::runtime_error("LoRALinear::forward: lora_B is null");
        
        float scaling = alpha_ / static_cast<float>(rank_);
        return lora_linear(input, weight_, lora_A_, lora_B_, scaling, bias_);
    }
    
    std::vector<TensorPtr> parameters() {
        std::vector<TensorPtr> params = {lora_A_, lora_B_};
        if (bias_ && bias_->requires_grad()) {
            params.push_back(bias_);
        }
        return params;
    }
};

class GPT2LoRAAttention {
private:
    std::unique_ptr<LoRALinear> q_proj_;
    std::unique_ptr<LoRALinear> k_proj_;
    std::unique_ptr<LoRALinear> v_proj_;
    std::unique_ptr<LoRALinear> o_proj_;
    int n_embd_, n_head_, head_dim_;
    bool enable_q_ = true, enable_k_ = false, enable_v_ = true, enable_o_ = false;
    TensorPtr base_q_weight_, base_k_weight_, base_v_weight_, base_o_weight_;

public:
    GPT2LoRAAttention(int n_embd, int n_head, int lora_rank = 8,
                      bool enable_q = true, bool enable_k = false,
                      bool enable_v = true, bool enable_o = false)
        : n_embd_(n_embd), n_head_(n_head), head_dim_(n_embd / n_head),
          enable_q_(enable_q), enable_k_(enable_k), enable_v_(enable_v), enable_o_(enable_o) {
        
        std::cout << "Creating LoRA Multi-Head Attention: " << n_head << " heads, " << n_embd << " dim, rank=" << lora_rank << std::endl;
        
        // Only create LoRA adapters when enabled; base weights loaded per layer later
        if (enable_q_) q_proj_ = std::make_unique<LoRALinear>(n_embd, n_embd, lora_rank);
        if (enable_k_) k_proj_ = std::make_unique<LoRALinear>(n_embd, n_embd, lora_rank);
        if (enable_v_) v_proj_ = std::make_unique<LoRALinear>(n_embd, n_embd, lora_rank);
        if (enable_o_) o_proj_ = std::make_unique<LoRALinear>(n_embd, n_embd, lora_rank);
        
        std::cout << "LoRA Attention initialized (q=" << enable_q_ << " k=" << enable_k_ 
                  << " v=" << enable_v_ << " o=" << enable_o_ << ")" << std::endl;
    }

    TensorPtr forward(const TensorPtr& x) {
        if (!x) throw std::runtime_error("GPT2LoRAAttention::forward: input is null");
        
        auto batch_size = x->shape()[0];
        auto seq_len = x->shape()[1];

        // Unified implementation: B*H 3D path, ensuring matmul batch dimension compatibility
        TensorPtr q, k, v;
        if (q_proj_) {
            q = q_proj_->forward(x);
        } else {
            if (!base_q_weight_) throw std::runtime_error("base_q_weight is null (layer weights not loaded?)");
            q = matmul(x, base_q_weight_);
        }
        
        if (k_proj_) {
            k = k_proj_->forward(x);
        } else {
            if (!base_k_weight_) throw std::runtime_error("base_k_weight is null");
            k = matmul(x, base_k_weight_);
        }
        
        if (v_proj_) {
            v = v_proj_->forward(x);
        } else {
            if (!base_v_weight_) throw std::runtime_error("base_v_weight is null");
            v = matmul(x, base_v_weight_);
        }

        #ifdef USE_NEW_AUTOGRAD_ENGINE
        if (q) q->set_requires_grad(true);
        if (k) k->set_requires_grad(true);
        if (v) v->set_requires_grad(true);
        #endif

        // Reshape to multi-head format: [batch, seq_len, num_heads, head_dim]
        auto q_reshaped = reshape(q, {batch_size, seq_len, n_head_, head_dim_});
        auto k_reshaped = reshape(k, {batch_size, seq_len, n_head_, head_dim_});
        auto v_reshaped = reshape(v, {batch_size, seq_len, n_head_, head_dim_});

        // Adjust to [batch, num_heads, seq_len, head_dim]
        auto q_bnhd = transpose(q_reshaped, 1, 2);
        auto k_bnhd = transpose(k_reshaped, 1, 2);
        auto v_bnhd = transpose(v_reshaped, 1, 2);

        // Merge B and H dimensions: get [B*H, S, D]
        auto q_bh = reshape(q_bnhd, {batch_size * n_head_, seq_len, head_dim_});
        auto k_bh = reshape(k_bnhd, {batch_size * n_head_, seq_len, head_dim_});
        auto v_bh = reshape(v_bnhd, {batch_size * n_head_, seq_len, head_dim_});

        // scores: [BH, S, S]
        auto k_bh_T = transpose(k_bh, -2, -1);
        auto scores = matmul(q_bh, k_bh_T);
        auto scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));
        auto scaled_scores = mul_scalar(scores, scale);

        // Causal mask [S,S] broadcast to [BH,S,S]
        auto causal_mask = create_causal_mask(seq_len);
        auto masked_scores = apply_mask(scaled_scores, causal_mask);

        // softmax & multiply V
        auto attn_weights = softmax(masked_scores, -1);
        auto attn_output_bh = matmul(attn_weights, v_bh); // [BH, S, D]

        // Restore to [B, H, S, D] -> [B, S, H, D] -> [B, S, n_embd]
        auto attn_bhsd = reshape(attn_output_bh, {batch_size, n_head_, seq_len, head_dim_});
        auto attn_bshd = transpose(attn_bhsd, 1, 2);
        auto output_reshaped = reshape(attn_bshd, {batch_size, seq_len, n_embd_});

        if (o_proj_) {
            auto out = o_proj_->forward(output_reshaped);
            #ifdef USE_NEW_AUTOGRAD_ENGINE
            if (out) out->set_requires_grad(true);
            #endif
            return out;
        } else {
            if (!base_o_weight_) throw std::runtime_error("base_o_weight is null");
            auto out = matmul(output_reshaped, base_o_weight_);
            #ifdef USE_NEW_AUTOGRAD_ENGINE
            if (out) out->set_requires_grad(true);
            #endif
            return out;
        }
    }

    std::vector<TensorPtr> parameters() {
        std::vector<TensorPtr> params;
        if (q_proj_) { auto q_params = q_proj_->parameters(); params.insert(params.end(), q_params.begin(), q_params.end()); }
        if (k_proj_) { auto k_params = k_proj_->parameters(); params.insert(params.end(), k_params.begin(), k_params.end()); }
        if (v_proj_) { auto v_params = v_proj_->parameters(); params.insert(params.end(), v_params.begin(), v_params.end()); }
        if (o_proj_) { auto o_params = o_proj_->parameters(); params.insert(params.end(), o_params.begin(), o_params.end()); }
        
        return params;
    }

    // weight loaders: if LoRA enabled, load to LoRA's base; otherwise store directly as base weight
    void q_proj_load(const TensorPtr& w) {
        if (q_proj_) {
            q_proj_->load_pretrained_weight(w);
        } else {
            base_q_weight_ = w;
            base_q_weight_->set_requires_grad(false);
        }
    }
    void k_proj_load(const TensorPtr& w) {
        if (k_proj_) {
            k_proj_->load_pretrained_weight(w);
        } else {
            base_k_weight_ = w;
            base_k_weight_->set_requires_grad(false);
        }
    }
    void v_proj_load(const TensorPtr& w) {
        if (v_proj_) {
            v_proj_->load_pretrained_weight(w);
        } else {
            base_v_weight_ = w;
            base_v_weight_->set_requires_grad(false);
        }
    }
    void o_proj_load(const TensorPtr& w) {
        if (o_proj_) {
            o_proj_->load_pretrained_weight(w);
        } else {
            base_o_weight_ = w;
            base_o_weight_->set_requires_grad(false);
        }
    }
};

class GPT2MLP {
private:
    TensorPtr fc_weight_, fc_bias_;
    TensorPtr proj_weight_, proj_bias_;
    int n_embd_, n_inner_;
    int layer_index_ = -1;

public:
    GPT2MLP(int n_embd, int n_inner = -1, int layer_index = -1) : n_embd_(n_embd), layer_index_(layer_index) {
        n_inner_ = (n_inner == -1) ? 4 * n_embd : n_inner;
        
        std::cout << "Creating MLP layer: " << n_embd << " -> " << n_inner_ << " -> " << n_embd << std::endl;
        
        try {
            std::cout << "Loading pretrained MLP weights..." << std::endl;
            std::string prefix = "models/gpt2/exported/";
            if (layer_index_ >= 0) {
                prefix += "h." + std::to_string(layer_index_) + ".";
            }
            fc_weight_ = load_binary_weights(prefix + "mlp_fc_weight.bin");
            fc_bias_ = load_binary_tensor_1d(prefix + "mlp_fc_bias.bin");
            proj_weight_ = load_binary_weights(prefix + "mlp_proj_weight.bin");
            proj_bias_ = load_binary_tensor_1d(prefix + "mlp_proj_bias.bin");
            
            fc_weight_->set_requires_grad(false);
            fc_bias_->set_requires_grad(false);
            proj_weight_->set_requires_grad(false);
            proj_bias_->set_requires_grad(false);
            
            std::cout << "Pretrained MLP weights loaded successfully" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "Cannot load pretrained MLP weights: " << e.what() << std::endl;
            
            float scale = 0.02f / std::sqrt(static_cast<float>(n_embd_));
            fc_weight_ = mul_scalar(randn({n_inner_, n_embd_}), scale);
            fc_bias_ = zeros({n_inner_});
            proj_weight_ = mul_scalar(randn({n_embd_, n_inner_}), scale);
            proj_bias_ = zeros({n_embd_});
            
            fc_weight_->set_requires_grad(false);
            fc_bias_->set_requires_grad(false);
            proj_weight_->set_requires_grad(false);
            proj_bias_->set_requires_grad(false);
        }
    }

    TensorPtr forward(const TensorPtr& x) {
#ifdef DISABLE_BLAS_COMPLETELY
        // Extreme memory saving mode: blocked MLP (no complete intermediate layer)
        // Reshape input to [batch*seq, n_embd] format
        auto x_shape = x->shape();
        int64_t batch_seq = x_shape[0] * x_shape[1];
        int64_t n_embd = x_shape[2];
        
        auto x_flat = reshape(x, {batch_seq, n_embd});
        
        // Call blocked MLP (chunk_size = 256, intermediate layer processed in chunks)
        auto output_flat = ops::memory_first::memory_first_mlp_forward(
            x_flat, fc_weight_, fc_bias_, proj_weight_, proj_bias_, 128
        );
        
        // Reshape back to original shape
        return reshape(output_flat, x_shape);
#else
        // Standard mode: original implementation
        auto hidden = add(matmul(x, fc_weight_), fc_bias_);
        auto activated = gelu(hidden);
        auto output = add(matmul(activated, proj_weight_), proj_bias_);
        return output;
#endif
    }

    std::vector<TensorPtr> parameters() {
        return {};
    }
};

class GPT2LoRATransformerBlock {
private:
    std::unique_ptr<GPT2LoRAAttention> attention_;
    std::unique_ptr<GPT2MLP> mlp_;
    TensorPtr ln1_weight_, ln1_bias_;
    TensorPtr ln2_weight_, ln2_bias_;
    LoRAFinetuneConfig config_;
    int layer_index_ = 0;

public:
    GPT2LoRATransformerBlock(const LoRAFinetuneConfig& config, int layer_index) : config_(config), layer_index_(layer_index) {
        bool apply_lora = layer_index_ >= (config.n_layer - config.lora_layers);
        attention_ = std::make_unique<GPT2LoRAAttention>(
            config.n_embd, config.n_head, config.lora_rank,
            apply_lora && config.lora_q,
            apply_lora && config.lora_k,
            apply_lora && config.lora_v,
            apply_lora && config.lora_o
        );
        mlp_ = std::make_unique<GPT2MLP>(config.n_embd, -1, layer_index_);
        
        try {
            std::cout << "Loading LayerNorm weights..." << std::endl;
            ln1_weight_ = load_binary_tensor_1d("models/gpt2/exported/h." + std::to_string(layer_index_) + ".ln1_weight.bin");
            ln1_bias_ = load_binary_tensor_1d("models/gpt2/exported/h." + std::to_string(layer_index_) + ".ln1_bias.bin");
            ln2_weight_ = load_binary_tensor_1d("models/gpt2/exported/h." + std::to_string(layer_index_) + ".ln2_weight.bin");
            ln2_bias_ = load_binary_tensor_1d("models/gpt2/exported/h." + std::to_string(layer_index_) + ".ln2_bias.bin");
            
            ln1_weight_->set_requires_grad(false);
            ln1_bias_->set_requires_grad(false);
            ln2_weight_->set_requires_grad(false);
            ln2_bias_->set_requires_grad(false);
            
            std::cout << "LayerNorm weights loaded successfully" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "Cannot load LayerNorm weights: " << e.what() << std::endl;
            ln1_weight_ = ones({config.n_embd});
            ln1_bias_ = zeros({config.n_embd});
            ln2_weight_ = ones({config.n_embd});
            ln2_bias_ = zeros({config.n_embd});
            
            ln1_weight_->set_requires_grad(false);
            ln1_bias_->set_requires_grad(false);
            ln2_weight_->set_requires_grad(false);
            ln2_bias_->set_requires_grad(false);
        }
        
        std::cout << "Complete GPT-2 LoRA Transformer Block created" << std::endl;
    }

    // Load attention weights per layer
    void load_attention_weights_per_layer() {
        try {
            auto q_path = "models/gpt2/exported/h." + std::to_string(layer_index_) + ".q_weight.bin";
            auto k_path = "models/gpt2/exported/h." + std::to_string(layer_index_) + ".k_weight.bin";
            auto v_path = "models/gpt2/exported/h." + std::to_string(layer_index_) + ".v_weight.bin";
            auto o_path = "models/gpt2/exported/h." + std::to_string(layer_index_) + ".o_weight.bin";

            if (attention_) {
                attention_->q_proj_load(load_binary_weights(q_path));
                attention_->k_proj_load(load_binary_weights(k_path));
                attention_->v_proj_load(load_binary_weights(v_path));
                attention_->o_proj_load(load_binary_weights(o_path));
            }
        } catch (const std::exception& e) {
            std::cout << "Warning: failed to load per-layer attention weights for layer " << layer_index_ << ": " << e.what() << std::endl;
        }
    }

    bool q_proj_exists() const { return true; }
    bool k_proj_exists() const { return true; }
    bool v_proj_exists() const { return true; }
    bool o_proj_exists() const { return true; }

    TensorPtr forward(const TensorPtr& x) {
        auto ln1_out = layer_norm(x, ln1_weight_, ln1_bias_);
        auto attn_out = attention_->forward(ln1_out);
        auto residual1 = add(x, attn_out);
        
        auto ln2_out = layer_norm(residual1, ln2_weight_, ln2_bias_);
        auto mlp_out = mlp_->forward(ln2_out);
        auto residual2 = add(residual1, mlp_out);
        
        return residual2;
    }

    std::vector<TensorPtr> parameters() {
        auto attn_params = attention_->parameters();
        auto mlp_params = mlp_->parameters();
        
        std::vector<TensorPtr> params;
        params.insert(params.end(), attn_params.begin(), attn_params.end());
        params.insert(params.end(), mlp_params.begin(), mlp_params.end());
        
        return params;
    }
};

class GPT2LoRAFinetune {
private:
    std::vector<std::unique_ptr<GPT2LoRATransformerBlock>> layers_;
    TensorPtr wte_, wpe_;
    // Use wte<->lm_head tying (weight sharing), no longer storing separate lm_head
    TensorPtr ln_f_weight_, ln_f_bias_;
    LoRAFinetuneConfig config_;

public:
    GPT2LoRAFinetune(const LoRAFinetuneConfig& config) : config_(config) {
        try {
            std::cout << "Loading pretrained embedding weights..." << std::endl;
            wte_ = load_binary_weights("models/gpt2/exported/wte.bin");
            wpe_ = load_binary_weights("models/gpt2/exported/wpe.bin");
            // Use wte<->lm_head tying (weight sharing): don't load separate lm_head
            // GPT-2 standard practice: lm_head = wte.T, saves ~150MB resident memory
            
            wte_->set_requires_grad(false);
            wpe_->set_requires_grad(false);
            
            std::cout << "Pretrained embedding weights loaded successfully!" << std::endl;
            std::cout << "Using wte<->lm_head tying (weight sharing), saves ~150MB resident memory" << std::endl;
            
            ops::MemoryManager::instance().force_cleanup();
        } catch (const std::exception& e) {
            std::cout << "Warning: Cannot load pretrained embedding weights: " << e.what() << std::endl;
            wte_ = randn({config.vocab_size, config.n_embd});
            wpe_ = randn({config.block_size, config.n_embd});
            
            wte_->set_requires_grad(false);
            wpe_->set_requires_grad(false);
        }

        // Build complete stack
        layers_.reserve(config.n_layer);
        for (int i = 0; i < config.n_layer; ++i) {
            layers_.push_back(std::make_unique<GPT2LoRATransformerBlock>(config, i));
        }
        // Load attention weights per layer (Q/K/V/O)
        for (int i = 0; i < config.n_layer; ++i) {
            layers_[i]->load_attention_weights_per_layer();
        }

        try {
            std::cout << "Loading final LayerNorm weights..." << std::endl;
            ln_f_weight_ = load_binary_tensor_1d("models/gpt2/exported/ln_f_weight.bin");
            ln_f_bias_ = load_binary_tensor_1d("models/gpt2/exported/ln_f_bias.bin");
            
            ln_f_weight_->set_requires_grad(false);
            ln_f_bias_->set_requires_grad(false);
            
            std::cout << "Final LayerNorm weights loaded successfully" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "Cannot load final LayerNorm weights: " << e.what() << std::endl;
            ln_f_weight_ = ones({config.n_embd});
            ln_f_bias_ = zeros({config.n_embd});
            
            ln_f_weight_->set_requires_grad(false);
            ln_f_bias_->set_requires_grad(false);
        }

        std::cout << "Complete GPT-2 LoRA fine-tuning model created" << std::endl;
        std::cout << "  Frozen parameters: Embedding + MLP + LayerNorm + LM head (pretrained weights)" << std::endl;
        std::cout << "  Trainable parameters: LoRA adapters in Attention layers" << std::endl;
    }

    TensorPtr forward(const TensorPtr& input_ids) {
        auto batch_size = input_ids->shape()[0];
        auto seq_len = input_ids->shape()[1];

        auto input_ids_int = input_ids->data<int32_t>();
        
        auto token_embeds = zeros({batch_size, seq_len, config_.n_embd});
        auto pos_embeds = zeros({batch_size, seq_len, config_.n_embd});

        float* token_data = token_embeds->data<float>();
        float* pos_data = pos_embeds->data<float>();
        const float* wte_data = wte_->data<float>();
        const float* wpe_data = wpe_->data<float>();

        // Standard embedding lookup implementation - triple loop
        for (int b = 0; b < batch_size; ++b) {
            for (int s = 0; s < seq_len; ++s) {
                int token_id = input_ids_int[b * seq_len + s];
                
                // Boundary check: prevent token_id from exceeding vocabulary range
                if (token_id < 0 || token_id >= config_.vocab_size) {
                    token_id = 0;  // Use UNK token
                }
                
                // Prevent position encoding overflow
                int pos_id = std::min(s, config_.block_size - 1);
                
                // Copy token embedding element by element
                for (int d = 0; d < config_.n_embd; ++d) {
                    token_data[(b * seq_len + s) * config_.n_embd + d] = 
                        wte_data[token_id * config_.n_embd + d];
                }
                
                // Copy position embedding element by element
                for (int d = 0; d < config_.n_embd; ++d) {
                    pos_data[(b * seq_len + s) * config_.n_embd + d] = 
                        wpe_data[pos_id * config_.n_embd + d];
                }
            }
        }

        // Before building x, explicitly enable computation graph tracking to ensure subsequent operators register with new engine
        token_embeds->set_requires_grad(true);
        pos_embeds->set_requires_grad(true);
        auto x = add(token_embeds, pos_embeds);
        // Key: enable computation graph from input activation to ensure backward reaches LoRA parameters
        #ifdef USE_NEW_AUTOGRAD_ENGINE
        x->set_requires_grad(true);
        #endif
        // Forward through all layers
        for (int i = 0; i < static_cast<int>(layers_.size()); ++i) {
            x = layers_[i]->forward(x);
        }
        auto transformer_output = x;
        auto ln_f_output = layer_norm(transformer_output, ln_f_weight_, ln_f_bias_);
        
        // Use wte as lm_head (tying): logits = ln_f_output @ wte^T
        // Call zero-copy right matrix transpose matmul
        auto logits = matmul_rhs_T(ln_f_output, wte_);
        
        return logits;
    }

    TensorPtr compute_loss(const TensorPtr& logits, const TensorPtr& targets) {
        auto batch_size = logits->shape()[0];
        auto seq_len = logits->shape()[1];
        auto vocab_size = logits->shape()[2];
        auto flattened_logits = reshape(logits, {batch_size * seq_len, vocab_size});
        auto flattened_targets = reshape(targets, {batch_size * seq_len});
        return cross_entropy_loss(flattened_logits, flattened_targets);
    }

    std::vector<TensorPtr> trainable_parameters() {
        std::vector<TensorPtr> params;
        for (auto &blk : layers_) {
            auto p = blk->parameters();
            params.insert(params.end(), p.begin(), p.end());
        }
        return params;
    }

    void print_parameter_info() {
        auto trainable_params = trainable_parameters();
        int64_t total_trainable = 0;
        for (auto& param : trainable_params) {
            total_trainable += param->numel();
        }
        
        int64_t total_params = config_.vocab_size * config_.n_embd * 2 + 
                              config_.block_size * config_.n_embd +
                              config_.n_embd * config_.n_embd * 4 + 
                              config_.n_embd * 4 * config_.n_embd * 2 +
                              config_.n_embd * 6;

        std::cout << "LoRA Model Parameter Information:" << std::endl;
        std::cout << "  Total parameters: " << total_params << std::endl;
        std::cout << "  LoRA parameters: " << total_trainable << " (Attention layer LoRA adapters)" << std::endl;
        std::cout << "  Frozen parameters: " << (total_params - total_trainable) << std::endl;
        std::cout << "  Trainable ratio: " << std::fixed << std::setprecision(5) 
                  << (float)total_trainable/total_params*100 << "%" << std::endl;
    }
};

// Use Adam optimizer from operators library, removed custom optimizer

class LoRAFinetunerer {
private:
    std::unique_ptr<GPT2LoRAFinetune> model_;
    std::unique_ptr<ops::Adam> optimizer_;
    std::unique_ptr<MetricsLogger> metrics_logger_;
    LoRAFinetuneConfig config_;
    std::vector<TensorPtr> trainable_params_;  // Save parameter references
    
    // Mobile optimizer extensions
    std::unique_ptr<ops::optim::MobileGradientClipper> gradient_clipper_;
    std::unique_ptr<ops::optim::MobileLRScheduler> lr_scheduler_;
    int total_training_steps_;

public:
    const std::vector<TensorPtr>& get_trainable_params() const { return trainable_params_; }
    LoRAFinetunerer(const LoRAFinetuneConfig& config) : config_(config) {
        // Initialize logging system
        LogManager::init_logger("gpt2_finetune/logs", "gpt2_lora_training.log", LogLevel::INFO, true);
        metrics_logger_ = std::make_unique<MetricsLogger>("gpt2_finetune/logs", "gpt2_lora_metrics.csv");
        
        OPS_LOG_INFO("=== GPT2 LoRA Fine-tuning Started ===");
        OPS_LOG_INFO_F("Configuration: batch_size=%d, lr=%.1e, epochs=%d, lora_rank=%d", 
                       config.batch_size, config.lr, config.max_epochs, config.lora_rank);
        
        model_ = std::make_unique<GPT2LoRAFinetune>(config);
        model_->print_parameter_info();

        trainable_params_ = model_->trainable_parameters();
        #ifdef USE_NEW_AUTOGRAD_ENGINE
        // Ensure new engine writes gradients back to param->grad() after backward
        for (auto &p : trainable_params_) {
            if (p) p->retain_grad();
        }
        #endif
        
        // Validate trainable parameters
        std::cout << "Trainable parameter validation:" << std::endl;
        for (size_t i = 0; i < trainable_params_.size(); ++i) {
            auto& param = trainable_params_[i];
            std::cout << "  Parameter " << i << ": shape=[" << param->shape()[0] << ", " << param->shape()[1] 
                      << "], requires_grad=" << (param->requires_grad() ? "true" : "false") << std::endl;
        }
        
        // Use AdamW optimizer (with weight decay)
        ops::AdamConfig adam_config(
            config.lr,         // learning rate
            0.9f,              // beta1
            0.999f,            // beta2
            1e-8f,             // epsilon
            0.01f,             // weight_decay (AdamW style regularization)
            1.0f,              // clip_grad_norm
            false              // amsgrad
        );
        optimizer_ = std::make_unique<ops::Adam>(adam_config);
        
        // Initialize mobile optimizer extensions (gradient clipping + LR scheduling)
        ops::optim::GradientClippingConfig clip_config;
        clip_config.enabled = true;
        clip_config.max_grad_norm = 1.0f;
        clip_config.use_global_norm = true;
        clip_config.adaptive_clipping = true;
        clip_config.adaptive_factor = 0.01f;
        gradient_clipper_ = std::make_unique<ops::optim::MobileGradientClipper>(clip_config, nullptr);
        
        std::cout << "Mobile optimizer extensions enabled (gradient clipping max_norm=1.0)" << std::endl;

        OPS_LOG_INFO("LoRA fine-tuner initialization completed with Adam optimizer");
        std::cout << "LoRA fine-tuner initialization completed with Adam optimizer" << std::endl;
    }

    void train(WikiTextDataLoader& data_loader) {
        OPS_LOG_INFO("Starting LoRA fine-tuning training");
        std::cout << "\nStarting LoRA fine-tuning..." << std::endl;
        
        // Start training
        
        size_t total_sequences = data_loader.size();
        int steps_per_epoch = (total_sequences + config_.batch_size - 1) / config_.batch_size;
        total_training_steps_ = steps_per_epoch * config_.max_epochs;
        
        OPS_LOG_INFO_F("Training dataset info: total_sequences=%zu, steps_per_epoch=%d", total_sequences, steps_per_epoch);
        std::cout << "Training data: " << total_sequences << " sequences, " << steps_per_epoch << " steps/epoch" << std::endl;
        std::cout << "Effective batch size: " << (config_.batch_size * config_.grad_accum_steps) << " (batch=" 
                  << config_.batch_size << " x accum=" << config_.grad_accum_steps << ")" << std::endl;
        
        // Initialize learning rate scheduler (fixed warmup steps + cosine decay)
        ops::optim::LRSchedulerConfig lr_config;
        lr_config.type = ops::optim::LRSchedulerType::WARM_UP_COSINE;
        lr_config.base_lr = config_.lr;
        lr_config.min_lr = config_.lr * 0.1f;
        lr_config.warmup_steps = 1500;  // Fixed 1500 step warmup (industry common)
        lr_config.decay_steps = total_training_steps_;
        lr_scheduler_ = std::make_unique<ops::optim::MobileLRScheduler>(lr_config, nullptr);
        
        std::cout << "LR scheduler: Warmup " << lr_config.warmup_steps 
                  << " steps -> Cosine decay to " << lr_config.min_lr 
                  << " (total steps=" << total_training_steps_ << ")" << std::endl;

        auto training_start = std::chrono::high_resolution_clock::now();

        for (int epoch = 0; epoch < config_.max_epochs; ++epoch) {
            OPS_LOG_INFO_F("Starting Epoch %d/%d", epoch + 1, config_.max_epochs);
            std::cout << "\nEpoch " << (epoch + 1) << "/" << config_.max_epochs << std::endl;
            
            auto epoch_start = std::chrono::high_resolution_clock::now();
            
            if (epoch > 0) {
                data_loader.shuffle_data();
                OPS_LOG_INFO("Data reshuffled");
            }

            float total_loss = 0.0f;
            int step = 0;
            int global_step = epoch * steps_per_epoch;  // Global step count (across epochs)
            int accum_step = 0;  // Current accumulation step (0 to grad_accum_steps-1)
            auto batch_start_time = std::chrono::high_resolution_clock::now(); // For recording start time of 10 steps

            while (!data_loader.is_epoch_complete()) {
                // Early stopping check: if max_train_steps is set and reached, end early
                if (config_.max_train_steps > 0 && global_step >= config_.max_train_steps) {
                    OPS_LOG_INFO_F("Reached max_train_steps=%d, stopping training", config_.max_train_steps);
                    std::cout << "\nReached max training steps " << config_.max_train_steps << ", training ended" << std::endl;
                    break;
                }
                // Memory guard for automatic cleanup
                ops::MemoryGuard memory_guard;
                
                // Detailed timing monitoring
                auto step_start = std::chrono::high_resolution_clock::now();
                
                // Native memory monitoring
                size_t memory_before = SimpleMemoryMonitor::getCurrentRSS();
                
                auto data_start = std::chrono::high_resolution_clock::now();
                auto [input_ids, targets] = data_loader.get_batch(config_.batch_size);
                auto data_end = std::chrono::high_resolution_clock::now();
                auto data_time = std::chrono::duration_cast<std::chrono::milliseconds>(data_end - data_start).count();

                auto forward_start = std::chrono::high_resolution_clock::now();
                auto logits = model_->forward(input_ids);
                auto forward_end = std::chrono::high_resolution_clock::now();
                auto forward_time = std::chrono::duration_cast<std::chrono::milliseconds>(forward_end - forward_start).count();
                
                auto loss_start = std::chrono::high_resolution_clock::now();
                auto loss = model_->compute_loss(logits, targets);
                float loss_val = loss->item<float>();
                
                // Record original loss for statistics (before scaling)
                float metrics_loss = loss_val;
                
                // Gradient accumulation: loss divided by accumulation steps
                if (config_.grad_accum_steps > 1) {
                    auto scaled_loss_data = loss->data<float>();
                    scaled_loss_data[0] /= static_cast<float>(config_.grad_accum_steps);
                    loss_val /= static_cast<float>(config_.grad_accum_steps);
                }
                
                total_loss += metrics_loss;  // Accumulate original loss
                auto loss_end = std::chrono::high_resolution_clock::now();
                auto loss_time = std::chrono::duration_cast<std::chrono::milliseconds>(loss_end - loss_start).count();

                // Only clear gradients on first accumulation
                auto zero_grad_start = std::chrono::high_resolution_clock::now();
                if (accum_step == 0) {
                    optimizer_->zero_grad(trainable_params_);
                }
                auto zero_grad_end = std::chrono::high_resolution_clock::now();
                auto zero_grad_time = std::chrono::duration_cast<std::chrono::milliseconds>(zero_grad_end - zero_grad_start).count();
                
                // Backward propagation
                auto backward_start = std::chrono::high_resolution_clock::now();
                loss->backward();
                auto backward_end = std::chrono::high_resolution_clock::now();
                auto backward_time = std::chrono::duration_cast<std::chrono::milliseconds>(backward_end - backward_start).count();
                
                // Only update parameters when accumulation is full
                bool should_update = (accum_step == config_.grad_accum_steps - 1);
                auto optim_start = std::chrono::high_resolution_clock::now();
                if (should_update) {
                    // Collect gradients (only collect when updating parameters, avoid holding extra references)
                    auto collect_start = std::chrono::high_resolution_clock::now();
                    std::vector<TensorPtr> gradients;
                    gradients.reserve(trainable_params_.size());
                    for (auto& param : trainable_params_) {
                        gradients.push_back(param->grad()); // Allow nullptr
                    }
                    auto collect_end = std::chrono::high_resolution_clock::now();
                    auto collect_time = std::chrono::duration_cast<std::chrono::milliseconds>(collect_end - collect_start).count();

                    // Apply gradient clipping (mobile optimizer extension)
                    auto clip_start = std::chrono::high_resolution_clock::now();
                    float grad_norm_before_clip = gradient_clipper_->clip_gradients(gradients);
                    auto clip_end = std::chrono::high_resolution_clock::now();
                    auto clip_time = std::chrono::duration_cast<std::chrono::milliseconds>(clip_end - clip_start).count();
                    
                    // Check gradient validity (only when updating parameters)
                    int valid_gradients = 0;
                    for (auto& grad : gradients) {
                        if (grad) valid_gradients++;
                    }
                    
                    // Print gradient info every 10 steps (more frequent for diagnosis)
                    if ((step % 10) == 0) {
                        float sample = 0.0f;
                        for (auto& g : gradients) { if (g) { sample = g->numel() > 0 ? g->data<float>()[0] : 0.0f; break; } }
                        std::cout << "    Gradient check: " << valid_gradients << "/" << trainable_params_.size()
                                  << " valid, norm before clip=" << std::fixed << std::setprecision(6) << grad_norm_before_clip
                                  << " | sample gradient[0]=" << std::setprecision(6) << sample
                                  << std::endl;
                    }
                    
                    // Update learning rate (scheduler)
                    float current_lr = lr_scheduler_->step();
                    optimizer_->set_learning_rate(current_lr);
                    
                    // Execute parameter update
                    optimizer_->step(trainable_params_, gradients);
                    
                    // Record current learning rate
                    if (step == 0 || (step / config_.grad_accum_steps) % 25 == 0) {
                        std::cout << "    Current learning rate: " << std::scientific << std::setprecision(3) << current_lr << std::endl;
                    }

                    // Clear gradient references
                    for (auto& grad : gradients) { if (grad) grad.reset(); }
                }
                auto optim_end = std::chrono::high_resolution_clock::now();
                auto optim_time = std::chrono::duration_cast<std::chrono::milliseconds>(optim_end - optim_start).count();
                
                accum_step = (accum_step + 1) % config_.grad_accum_steps;
                
                // Thorough memory cleanup (fix root cause: computation graph accumulation)
                
                // 1. Immediately release all forward/backward local variables (order matters)
                loss.reset();
                logits.reset();
                input_ids.reset();
                targets.reset();
                
                // 2. Clean autograd computation graph (core: must clean every step, even during gradient accumulation)
                #ifdef USE_NEW_AUTOGRAD_ENGINE
                ops::autograd::Engine::instance().clear_graph();
                #endif
                
                // 3. Clean parameter gradient computation graph references (but keep gradient values for accumulation)
                // Key: gradient values already accumulated in param->grad_ data, cleaning grad_fn doesn't affect values
                if (!should_update) {
                    // During accumulation: clean gradient tensor's grad_fn and graph, but keep data
                    for (auto& param : trainable_params_) {
                        if (param->grad()) {
                            param->grad()->set_grad_fn(nullptr);
                        }
                    }
                } else {
                    // After update: completely clear gradients
                    for (auto& param : trainable_params_) {
                        if (param->grad()) {
                            param->grad().reset();
                            param->set_grad(nullptr);
                        }
                    }
                }
                
                // 4. Clear all weak references held by MemoryManager
                ops::MemoryManager::instance().clear_computation_graph();
                
                // 5. Force reclaim unused memory
                ops::MemoryManager::instance().clear_unused_memory();
                ops::MemoryManager::instance().cleanup_dead_references();
                ops::MemoryManager::instance().force_cleanup();
                
                // 6. macOS specific: notify system to release memory pressure
                #ifdef __APPLE__
                malloc_zone_pressure_relief(nullptr, 0);
                #endif
                
                // 7. Clean cache every step to avoid RSS peak accumulation
                ops::MemoryManager::instance().clear_cache();
                
                // 8. Reset step-level arena (extreme memory saving mode)
                #ifdef DISABLE_BLAS_COMPLETELY
                ops::get_step_arena().reset();
                #endif
                
                auto step_end = std::chrono::high_resolution_clock::now();
                auto total_step_time = std::chrono::duration_cast<std::chrono::milliseconds>(step_end - step_start).count();
                
                // Only show detailed timing on first step
                if (step == 0) {
                    std::cout << "First step timing: forward=" << forward_time << "ms backward=" << backward_time << "ms total=" << total_step_time << "ms" << std::endl;
                    std::cout << "  Step 1/" << steps_per_epoch << " | Loss: " << std::fixed << std::setprecision(4) << metrics_loss << " | first step complete" << std::endl;
                }

                auto step_duration = std::chrono::duration_cast<std::chrono::milliseconds>(step_end - step_start);

                step++;
                global_step++;
                float avg_loss = total_loss / step;
                
                // Log training metrics
                metrics_logger_->log_training_step(epoch + 1, step, metrics_loss, avg_loss, config_.lr, step_duration.count());
                
                // Print every 10 steps (reduce output volume)
                if (step > 0 && step % 10 == 0) {
                    size_t current_rss = SimpleMemoryMonitor::getCurrentRSS();
                    std::cout << "  Step " << step << "/" << steps_per_epoch
                              << " | Loss: " << std::fixed << std::setprecision(4) << metrics_loss
                              << " | Avg: " << std::setprecision(4) << avg_loss
                              << " | " << step_duration.count() << "ms"
                              << " | RSS: " << SimpleMemoryMonitor::formatMemorySize(current_rss) << std::endl;
                    std::cout.flush();
                }

                // Training health check (keep anomaly detection)
                if (std::isnan(loss_val) || std::isinf(loss_val)) {
                    OPS_LOG_ERROR_F("Training unstable, Loss anomaly: %f", loss_val);
                    std::cout << "Warning: detected anomalous Loss: " << loss_val << ", skipping this step" << std::endl;
                    continue;
                }
            }

            auto epoch_end = std::chrono::high_resolution_clock::now();
            auto epoch_duration = std::chrono::duration_cast<std::chrono::seconds>(epoch_end - epoch_start);
            
            // Root fix: simplify epoch end report
            size_t final_rss = SimpleMemoryMonitor::getCurrentRSS();
            std::cout << "\nEpoch " << (epoch + 1) << " summary:" << std::endl;
            std::cout << "  - Training steps: " << step << " steps" << std::endl;
            std::cout << "  - Average Loss: " << std::fixed << std::setprecision(4) << (total_loss / step) << std::endl;
            std::cout << "  - Time: " << epoch_duration.count() << " seconds" << std::endl;
            std::cout << "  - Final RSS: " << SimpleMemoryMonitor::formatMemorySize(final_rss) << std::endl;
            
            // Epoch end
            
            float epoch_avg_loss = total_loss / step;
            
            // Log epoch summary
            metrics_logger_->log_epoch_summary(epoch + 1, epoch_avg_loss, step, epoch_duration.count());
            
            OPS_LOG_INFO_F("Epoch %d completed: avg_loss=%.4f, total_steps=%d, time=%ds", 
                      epoch + 1, epoch_avg_loss, step, (int)epoch_duration.count());
            std::cout << "Epoch " << (epoch + 1) << " completed - Average Loss: "
                      << std::fixed << std::setprecision(4) << epoch_avg_loss 
                      << " (" << step << " steps)" << std::endl;
            
            // Cleanup after each epoch
            ops::MemoryManager::instance().force_cleanup();
        }

        auto training_end = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::minutes>(training_end - training_start);
        
        OPS_LOG_INFO_F("LoRA fine-tuning training completed, total time: %d minutes", (int)total_duration.count());
        std::cout << "\nLoRA fine-tuning completed" << std::endl;
        
        // Final cleanup
        ops::MemoryManager::instance().force_cleanup();
    }
};

TensorPtr mul_scalar(const TensorPtr& tensor, float scalar) {
    // Directly use mul function from operators library
    return mul(tensor, scalar);
}

TensorPtr load_binary_weights(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open weight file: " + filepath);
    }

    uint32_t ndim;
    file.read(reinterpret_cast<char*>(&ndim), sizeof(uint32_t));

    if (ndim != 2) {
        throw std::runtime_error("Weight file should be 2D tensor, but actual dimensions: " + std::to_string(ndim));
    }

    uint32_t dim0, dim1;
    file.read(reinterpret_cast<char*>(&dim0), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&dim1), sizeof(uint32_t));

    auto tensor = zeros({static_cast<int64_t>(dim0), static_cast<int64_t>(dim1)});
    size_t data_size = dim0 * dim1 * sizeof(float);
    file.read(reinterpret_cast<char*>(tensor->data<float>()), data_size);
    file.close();
    std::cout << "Loaded weights: " << filepath << " [" << dim0 << ", " << dim1 << "]" << std::endl;
    return tensor;
}

TensorPtr load_binary_tensor_1d(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open weight file: " + filepath);
    }
    
    uint32_t ndim;
    file.read(reinterpret_cast<char*>(&ndim), sizeof(uint32_t));
    
    if (ndim != 1) {
        throw std::runtime_error("Expected 1D tensor, but actual dimensions: " + std::to_string(ndim));
    }
    
    uint32_t size;
    file.read(reinterpret_cast<char*>(&size), sizeof(uint32_t));
    
    auto tensor = zeros({static_cast<int64_t>(size)});
    file.read(reinterpret_cast<char*>(tensor->data<float>()), size * sizeof(float));
    file.close();
    std::cout << "Loaded 1D weights: " << filepath << " [" << size << "]" << std::endl;
    return tensor;
}

int main() {
    // Key: macOS memory safety settings (must be set at startup)
    setenv("VECLIB_MAXIMUM_THREADS", "1", 1);
    setenv("OMP_NUM_THREADS", "1", 1);
    setenv("OPENBLAS_NUM_THREADS", "1", 1);
    setenv("MKL_NUM_THREADS", "1", 1);
    setenv("MALLOC_PER_THREAD", "0", 1);      // Reduce per-thread malloc cache
    setenv("MALLOC_ZONE_CONSCIOUS", "1", 1);  // Enable zone pressure relief
    
    std::cout << "GPT-2 LoRA Fine-tuning - Industrial Optimized Version" << std::endl;
    std::cout << "================================================" << std::endl;
    std::cout << "Task: Fine-tune GPT-2 Attention layers with LoRA adapters" << std::endl;
    std::cout << "Using WikiText dataset for supervised fine-tuning" << std::endl;
    std::cout << "Features: Gradient accumulation + Gradient clipping + LR scheduling + Topological sort engine" << std::endl;
    std::cout << "Training: Multi-Head Attention LoRA adapters (last 6 layers Q/V)" << std::endl;

    try {
        LoRAFinetuneConfig config;
        
        // Optimized training configuration (industrial effective parameters)
        config.block_size = 64;          // Sequence length: 64 (better training effect)
        config.vocab_size = 50257;
        config.batch_size = 1;           // Single sample training (short test)
        config.grad_accum_steps = 1;     // Turn off accumulation for short test, easier to locate gradients
        config.lr = 5e-4f;               // Learning rate: increased to 5e-4 to accelerate convergence
        config.max_epochs = 1;           // Single epoch training
        
        // LoRA configuration (industrial effective settings)
        config.lora_rank = 8;            // LoRA rank: 8 (standard value)
        config.lora_alpha = 32.0f;       // LoRA alpha: 32 (matching rank=8)
        config.n_layer = 12;             // GPT-2 small default 12 layers
        config.lora_layers = 6;          // Apply LoRA to last 6 layers
        config.lora_q = true;            // Industry common: Q/V
        config.lora_k = false;
        config.lora_v = true;
        config.lora_o = false;           // Optional: enabling O may have additional benefits
        
        config.print();
        
        std::cout << "\nPreparing WikiText data and GPT-2 tokenizer..." << std::endl;
        std::cout << "Using WikiText-2 training set (optimized configuration)" << std::endl;
        
        // Experiment: 10% data
        WikiTextDataLoader data_loader("data/wikitext2_train.jsonl", "models/gpt2/vocab.json", 
                                      config.block_size, -1, 0.10f);
        
        // Short test: fixed 200 steps
        config.max_train_steps = 200;
        
        std::cout << "\nCreating GPT-2 LoRA fine-tuning model..." << std::endl;
        LoRAFinetunerer trainer(config);
        // Show retain_grad flag status
        {
            auto& params = trainer.get_trainable_params();
            std::cout << "retain_grad flags: ";
            int cnt = 0;
            for (auto &p : params) {
                std::cout << (p && p->retains_grad() ? 1 : 0);
                if (++cnt < (int)params.size()) std::cout << ",";
            }
            std::cout << std::endl;
        }
        
        trainer.train(data_loader);
        
    } catch (const std::exception& e) {
            std::cout << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
