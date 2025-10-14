#include "operators.h"
#include "optim/adam_amp.h"
#include "utils/grad_scaler.h"
#include "utils/memory_ledger.h"
#include "gemma_tokenizer.h"
#include "memory/mobile_param_manager.h"
#include "memory/mobile_param_optimizations.h"
#include "memory/mobile_specific_optimizations.h"
#include "core/chunked_softmax_ce.h"
#include <iostream>
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
#include <cmath>
#include <cstring>
#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/task.h>
#endif

using namespace ops;
using namespace ops::memory;
#include "activations/deepspeed_checkpoint_integration.h"

TensorPtr mul_scalar(const TensorPtr& tensor, float scalar);
TensorPtr load_binary_weights(const std::string& filepath);
TensorPtr load_binary_tensor_1d(const std::string& filepath);

struct GemmaLoRAConfig {
    // Use real Gemma 3 270M configuration parameters
    int n_embd = 640;           // Gemma 3 270M actual hidden size
    int n_head = 4;             // Gemma 3 270M actual attention heads
    int n_kv_head = 1;          // Single KV head for GQA
    int n_layer = 18;           // Phase 3: Restore full 18 layers (with chunked CE + Tied support)
    int n_inner = 2048;         // Gemma 3 270M actual intermediate size
    int block_size = 64;        // Phase 3: Restore to 64 (chunked CE solved memory issues)
    int vocab_size = 262144;    // Gemma 3 270M actual vocabulary size

    // LoRA parameters
    int lora_rank = 8;          // Phase 3: Restore to 8 (standard LoRA configuration)
    float lora_alpha = 8.0f;    // Lower LoRA scaling factor
    float lora_dropout = 0.1f;  // LoRA dropout

    int batch_size = 1;         // Edge device memory limit
    float lr = 5e-5f;           // Fix: Lower learning rate for shorter sequences
    int max_epochs = 3;         // Training epochs
    int steps_per_epoch = 200;  // Steps per epoch
    float rms_norm_eps = 1e-6f; // Gemma RMSNorm epsilon
    float rope_theta = 10000.0f;// Gemma RoPE theta parameter
    bool quiet_mode = false;    // Simplified output mode
    
    // Memory optimization configuration
    MobileParamConfig memory_config;
    
    GemmaLoRAConfig() {
        // Configure memory optimization settings
        memory_config.max_gpu_memory_mb = 128;         // Lower to 128MB (chunked CE greatly reduced peak)
        memory_config.max_cpu_memory_mb = 512;         // Lower to 512MB
        // Phase 2: Enable frozen weight FP16 storage (training params remain FP32)
        memory_config.enable_quantization = true;      // Enable frozen weight quantization
        memory_config.param_persistence_threshold = 2; // Keep 2 layers in memory
        memory_config.enable_pin_memory = false;       // Disable pinned memory to save overhead
        memory_config.max_pinned_memory_mb = 0;        // 0MB pinned memory
        memory_config.enable_prefetch = true;          // Enable prefetch
        memory_config.enable_storage_offload = true;   // Enable storage offload
        memory_config.eviction_threshold = 0.7;        // 70% triggers offload (more aggressive)
        memory_config.enable_thermal_monitoring = false; // Temporarily disable thermal management to avoid auto-exit
    }

    void print() const {
        std::cout << "Configuration: " << n_layer << " layers, " << n_embd << " dims, " << n_head << " heads, sequence length " << block_size << std::endl;
        std::cout << "LoRA: rank " << lora_rank << ", alpha " << lora_alpha << ", learning rate " << lr << std::endl;
        std::cout << "Training: " << max_epochs << " epochs, batch size " << batch_size << std::endl;
        std::cout << std::endl;
    }
};
// Simple memory monitor
class RtMem {
public:
    static size_t rss_bytes() {
#ifdef __APPLE__
        struct task_basic_info info;
        mach_msg_type_number_t size = TASK_BASIC_INFO_COUNT;
        if (task_info(mach_task_self(), TASK_BASIC_INFO, (task_info_t)&info, &size) == KERN_SUCCESS) {
            return static_cast<size_t>(info.resident_size);
        }
#endif
        return 0;
    }
    static std::string fmt(size_t bytes) {
        const double KB = 1024.0;
        const double MB = KB * 1024.0;
        const double GB = MB * 1024.0;
        std::ostringstream oss;
        oss.setf(std::ios::fixed); oss.precision(2);
        if (bytes >= (size_t)GB) { oss << (bytes / GB) << " GB"; }
        else if (bytes >= (size_t)MB) { oss << (bytes / MB) << " MB"; }
        else if (bytes >= (size_t)KB) { oss << (bytes / KB) << " KB"; }
        else { oss << bytes << " B"; }
        return oss.str();
    }
};

// Utility function declarations
TensorPtr mul_scalar(const TensorPtr& tensor, float scalar) {
    return mul(tensor, scalar);
}

TensorPtr load_binary_weights(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open weight file: " + filepath);
    }

    uint32_t ndim;
    file.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));

    std::vector<int64_t> shape(ndim);
    for (uint32_t i = 0; i < ndim; ++i) {
        uint32_t dim;
        file.read(reinterpret_cast<char*>(&dim), sizeof(dim));
        shape[i] = static_cast<int64_t>(dim);
    }

    auto tensor = zeros(shape);
    int64_t total_elements = tensor->numel();
    file.read(reinterpret_cast<char*>(tensor->data<float>()), total_elements * sizeof(float));
    file.close();

    std::cout << "Loaded weight: " << filepath << " ";
    std::cout << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i];
        if (i < shape.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    return tensor;
}

TensorPtr load_binary_tensor_1d(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open 1D tensor file: " + filepath);
    }

    uint32_t ndim;
    file.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));

    if (ndim != 1) {
        throw std::runtime_error("Expected 1D tensor but got " + std::to_string(ndim) + "D tensor");
    }

    uint32_t size;
    file.read(reinterpret_cast<char*>(&size), sizeof(size));

    auto tensor = zeros({static_cast<int64_t>(size)});
    file.read(reinterpret_cast<char*>(tensor->data<float>()), size * sizeof(float));
    file.close();

    std::cout << "Loaded 1D weight: " << filepath << " [" << size << "]" << std::endl;

    return tensor;
}

// Data loader (simplified version, reusing Gemma tokenizer logic)
class GemmaTokenizedDataLoader {
private:
    std::vector<std::vector<int32_t>> sequences_;
    size_t current_idx_;
    int seq_len_;

public:
    GemmaTokenizedDataLoader(int seq_len = 2048) : current_idx_(0), seq_len_(seq_len) {
        std::cout << "DataLoader initialized, sequence length: " << seq_len_ << std::endl;
        
        // Load real Gemma token sequences
        if (!load_real_tokens("real_gemma_tokens.json")) {
            std::cout << "Unable to load real token file, using fallback generation" << std::endl;
            generate_fallback_tokens();
        }
    }

private:
    bool load_real_tokens(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            return false;
        }
        
        std::string content((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());
        
        // Simple JSON parsing (only parse the sequences array we need)
        size_t sequences_pos = content.find("\"sequences\":");
        if (sequences_pos == std::string::npos) {
            return false;
        }
        
        size_t array_start = content.find("[", sequences_pos);
        if (array_start == std::string::npos) {
            return false;
        }
        
        // Find array end position
        int bracket_count = 0;
        size_t array_end = array_start;
        for (size_t i = array_start; i < content.length(); ++i) {
            if (content[i] == '[') bracket_count++;
            if (content[i] == ']') bracket_count--;
            if (bracket_count == 0) {
                array_end = i;
                break;
            }
        }
        
        // Parse sequences
        std::string sequences_str = content.substr(array_start + 1, array_end - array_start - 1);
        
        // Simple parsing of each sequence (assuming well-formatted)
        size_t pos = 0;
        while (pos < sequences_str.length()) {
            size_t seq_start = sequences_str.find("[", pos);
            if (seq_start == std::string::npos) break;
            
            size_t seq_end = sequences_str.find("]", seq_start);
            if (seq_end == std::string::npos) break;
            
            std::string seq_str = sequences_str.substr(seq_start + 1, seq_end - seq_start - 1);
            
            // Parse numbers
            std::vector<int32_t> seq;
            std::stringstream ss(seq_str);
            std::string token_str;
            
            while (std::getline(ss, token_str, ',')) {
                // Remove spaces
                token_str.erase(0, token_str.find_first_not_of(" \t"));
                token_str.erase(token_str.find_last_not_of(" \t") + 1);
                
                if (!token_str.empty()) {
                    seq.push_back(std::stoi(token_str));
                }
            }
            
            if (seq.size() == seq_len_) {
                sequences_.push_back(seq);
            }
            
            pos = seq_end + 1;
        }
        
        if (sequences_.empty()) {
            return false;
        }
        
        // Calculate token range
        int max_token = 0, min_token = 262144;
        for (const auto& seq : sequences_) {
            for (int token : seq) {
                max_token = std::max(max_token, token);
                min_token = std::min(min_token, token);
            }
        }
        
        std::cout << "Loaded " << sequences_.size() << " token sequences (range: " << min_token << " ~ " << max_token << ")" << std::endl;
        return true;
    }
    
    void generate_fallback_tokens() {
        // Fallback random generation (keep original logic as backup)
        for (int i = 0; i < 200; ++i) {
            std::vector<int32_t> seq(seq_len_);
            for (int j = 0; j < seq_len_; ++j) {
                if (j < seq_len_ / 4) {
                    seq[j] = (j * 13 + i * 7) % 2000 + 1;
                } else if (j < seq_len_ / 2) {
                    seq[j] = ((i * seq_len_ + j * 17) % 48000) + 2000;
                } else if (j < 3 * seq_len_ / 4) {
                    seq[j] = ((i * j * 23) % 100000) + 50000;
                } else {
                    seq[j] = ((i * j * 31) % 50000) + 150000;
                }
                
                if (seq[j] >= 262144) {
                    seq[j] = (seq[j] % 100000) + 1;
                }
            }
            sequences_.push_back(seq);
        }
        
        std::cout << "Generated " << sequences_.size() << " fallback token sequences" << std::endl;
    }

public:

    std::pair<TensorPtr, TensorPtr> get_batch(int batch_size) {
        auto input_ids = zeros({batch_size, seq_len_ - 1}, DType::kInt32, kCPU);
        // Cross-entropy needs integer class indices
        auto targets = zeros({batch_size, seq_len_ - 1}, DType::kInt32, kCPU);

        int32_t* input_data = input_ids->data<int32_t>();
        float* target_data = targets->data<float>();

        for (int b = 0; b < batch_size; ++b) {
            if (current_idx_ >= sequences_.size()) {
                current_idx_ = 0;
            }

            const auto& seq = sequences_[current_idx_++];

            for (int i = 0; i < seq_len_ - 1; ++i) {
                input_data[b * (seq_len_ - 1) + i] = seq[i];
            reinterpret_cast<int32_t*>(targets->data_ptr())[b * (seq_len_ - 1) + i] = static_cast<int32_t>(seq[i + 1]);
            }
        }

        return {input_ids, targets};
    }

    size_t size() const { return sequences_.size(); }
    bool has_more() const { return true; }  // Always return true, as data can be reused in a loop
    void reset() { current_idx_ = 0; }
};

// LoRA linear layer
class GemmaLoRALinear {
private:
    TensorPtr weight_;          // Frozen pretrained weight
    TensorPtr bias_;            // Frozen bias
    TensorPtr lora_A_;          // LoRA A matrix (trainable)
    TensorPtr lora_B_;          // LoRA B matrix (trainable)
    float alpha_;
    int rank_;
    
    // Memory management
    std::shared_ptr<MobileParameterManager> param_manager_;
    std::string param_id_prefix_;

public:
    GemmaLoRALinear(int input_dim, int output_dim, int rank, float alpha, 
                   std::shared_ptr<MobileParameterManager> param_manager = nullptr,
                   const std::string& param_prefix = "", bool has_bias = false) 
        : alpha_(alpha), rank_(rank), param_manager_(param_manager), param_id_prefix_(param_prefix) {
        
        // Standard LoRA initialization: A uses Kaiming initialization, B initialized to small random values
        // This way LoRA starts with small random contributions and can begin learning
        float std_a = std::sqrt(2.0f / static_cast<float>(input_dim)); // Kaiming initialization
        float std_b = 0.01f; // Small random initialization instead of 0
        lora_A_ = mul_scalar(randn({input_dim, rank}), std_a);
        lora_B_ = mul_scalar(randn({rank, output_dim}), std_b);
        
        lora_A_->set_requires_grad(true);
        lora_B_->set_requires_grad(true);
        
        // Register parameters to memory manager
        if (param_manager_) {
            param_manager_->register_parameter(param_id_prefix_ + "_lora_A", lora_A_, true);   // Trainable
            param_manager_->register_parameter(param_id_prefix_ + "_lora_B", lora_B_, true);   // Trainable
        }
        
        if (has_bias) {
            bias_ = zeros({output_dim});
            if (param_manager_) {
                param_manager_->register_parameter(param_id_prefix_ + "_bias", bias_, true);
            }
        }
    }
    
    void load_pretrained_weight(const TensorPtr& pretrained_weight, const TensorPtr& pretrained_bias = nullptr) {
        weight_ = pretrained_weight;
        weight_->set_requires_grad(false);
        
        // Register pretrained weights to memory manager
        if (param_manager_) {
            param_manager_->register_parameter(param_id_prefix_ + "_weight", weight_, false);  // Not trainable
        }
        
        if (pretrained_bias) {
            bias_ = pretrained_bias;
            bias_->set_requires_grad(false);
            if (param_manager_) {
                param_manager_->register_parameter(param_id_prefix_ + "_bias", bias_, false);
            }
        }
    }
    
    TensorPtr forward(const TensorPtr& input) {
        // Re-enable LoRA
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

// Gemma LoRA Attention layer
class GemmaLoRAAttention {
private:
    std::unique_ptr<GemmaLoRALinear> wq_, wk_, wv_, wo_;
    int n_embd_, n_head_, n_kv_head_, head_dim_;
    float rope_theta_;
    int layer_idx_;
    std::shared_ptr<MobileParameterManager> param_manager_;

public:
    GemmaLoRAAttention(int n_embd, int n_head, int n_kv_head, float rope_theta, int layer_idx, 
                      int lora_rank, float lora_alpha,
                      std::shared_ptr<MobileParameterManager> param_manager = nullptr)
        : n_embd_(n_embd), n_head_(n_head), n_kv_head_(n_kv_head), 
          head_dim_(256), rope_theta_(rope_theta), layer_idx_(layer_idx), param_manager_(param_manager) {

        std::cout << "Creating Gemma LoRA GQA Attention layer " << layer_idx << ": " << n_head << " query heads, " 
                  << n_kv_head << " kv heads, " << n_embd << " dim, LoRA rank=" << lora_rank << std::endl;

        // Create LoRA linear layers - fix dimension settings, add memory management
        std::string layer_prefix = "layer_" + std::to_string(layer_idx) + "_attn";
        wq_ = std::make_unique<GemmaLoRALinear>(n_embd, n_head * head_dim_, lora_rank, lora_alpha, 
                                              param_manager_, layer_prefix + "_wq");    // 640 -> 1024
        wk_ = std::make_unique<GemmaLoRALinear>(n_embd, n_kv_head * head_dim_, lora_rank, lora_alpha,
                                              param_manager_, layer_prefix + "_wk"); // 640 -> 256
        wv_ = std::make_unique<GemmaLoRALinear>(n_embd, n_kv_head * head_dim_, lora_rank, lora_alpha,
                                              param_manager_, layer_prefix + "_wv"); // 640 -> 256
        wo_ = std::make_unique<GemmaLoRALinear>(n_head * head_dim_, n_embd, lora_rank, lora_alpha,
                                              param_manager_, layer_prefix + "_wo");    // 1024 -> 640

        try {
            auto q_weight = load_binary_weights("models/gemma-270m/exported/q_weight_" + std::to_string(layer_idx) + ".bin");
            auto k_weight_raw = load_binary_weights("models/gemma-270m/exported/k_weight_" + std::to_string(layer_idx) + ".bin");
            auto v_weight_raw = load_binary_weights("models/gemma-270m/exported/v_weight_" + std::to_string(layer_idx) + ".bin");
            auto o_weight = load_binary_weights("models/gemma-270m/exported/o_weight_" + std::to_string(layer_idx) + ".bin");
            
            // Transpose all weights to match expected shape [input_dim, output_dim]
            auto q_weight_t = transpose(q_weight, 0, 1);     // [1024, 640] -> [640, 1024]
            auto k_weight = transpose(k_weight_raw, 0, 1);   // [256, 640] -> [640, 256]
            auto v_weight = transpose(v_weight_raw, 0, 1);   // [256, 640] -> [640, 256]
            auto o_weight_t = transpose(o_weight, 0, 1);     // [640, 1024] -> [1024, 640]
            
            wq_->load_pretrained_weight(q_weight_t);
            wk_->load_pretrained_weight(k_weight);
            wv_->load_pretrained_weight(v_weight);
            wo_->load_pretrained_weight(o_weight_t);
            
        } catch (const std::exception& e) {
            std::cout << "Warning: Unable to load pretrained weights, will use random initialization: " << e.what() << std::endl;
        }

        std::cout << "Gemma LoRA Attention layer weights set (pretrained weights frozen, LoRA weights trainable)" << std::endl;
    }

    TensorPtr forward(const TensorPtr& x) {
        auto batch_size = x->shape()[0];
        auto seq_len = x->shape()[1];

        auto q = wq_->forward(x);
        auto k = wk_->forward(x);
        auto v = wv_->forward(x);
        
        try {
            auto q_reshaped = reshape(q, {batch_size, seq_len, n_head_, head_dim_});
            auto k_reshaped = reshape(k, {batch_size, seq_len, n_kv_head_, head_dim_});
            auto v_reshaped = reshape(v, {batch_size, seq_len, n_kv_head_, head_dim_});

            // Transpose to (batch, heads, seq, head_dim)
            auto q_t = transpose(q_reshaped, 1, 2);
            auto k_t = transpose(k_reshaped, 1, 2);
            auto v_t = transpose(v_reshaped, 1, 2);

            // Expand K and V heads for GQA
            int heads_per_kv = n_head_ / n_kv_head_;
            
            // Apply RoPE position encoding
            auto q_rope = apply_rope(q_t, seq_len, head_dim_, rope_theta_);
            auto k_rope = apply_rope(k_t, seq_len, head_dim_, rope_theta_);

            // GQA implementation: Repeat KV heads to match Q head count
            auto k_expanded = repeat_kv_heads(k_rope, heads_per_kv);  
            auto v_expanded = repeat_kv_heads(v_t, heads_per_kv);

            // Standard attention computation
            auto k_transposed = transpose(k_expanded, -2, -1);
            auto scores = matmul(q_rope, k_transposed);
            auto scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));
            auto scaled_scores = mul_scalar(scores, scale);

            // Causal mask
            auto causal_mask = create_causal_mask(seq_len);
            auto masked_scores = apply_mask(scaled_scores, causal_mask, -1e9f);

            auto attn_weights = softmax(masked_scores, -1);
            auto attn_output = matmul(attn_weights, v_expanded);

            // Transpose back and reshape
            auto output_transposed = transpose(attn_output, 1, 2);
            auto concat_output = reshape(output_transposed, {batch_size, seq_len, n_head_ * head_dim_});

            return wo_->forward(concat_output);
            
        } catch (const std::exception& e) {
            std::cout << "Attention forward failed: " << e.what() << std::endl;
            throw;
        }
    }

    std::vector<TensorPtr> parameters() {
        std::vector<TensorPtr> params;
        auto wq_params = wq_->parameters();
        auto wk_params = wk_->parameters();
        auto wv_params = wv_->parameters();
        auto wo_params = wo_->parameters();
        
        params.insert(params.end(), wq_params.begin(), wq_params.end());
        params.insert(params.end(), wk_params.begin(), wk_params.end());
        params.insert(params.end(), wv_params.begin(), wv_params.end());
        params.insert(params.end(), wo_params.begin(), wo_params.end());
        
        return params;
    }
};

// GeGLU activation function
TensorPtr geglu(const TensorPtr& gate, const TensorPtr& up) {
    // GeGLU = GELU(gate) * up
    if (!same_shape(gate, up)) {
        throw std::runtime_error("gate and up tensors must have the same shape for GeGLU");
    }
    
    auto result = zeros(gate->shape(), gate->dtype(), gate->device());
    const float* gate_data = gate->data<float>();
    const float* up_data = up->data<float>();
    float* result_data = result->data<float>();
    
    for (int64_t i = 0; i < gate->numel(); ++i) {
        float gate_val = gate_data[i];
        float up_val = up_data[i];
        
        // GELU approximation
        float tanh_input = 0.7978845608f * (gate_val + 0.044715f * gate_val * gate_val * gate_val);
        float gelu_gate = 0.5f * gate_val * (1.0f + std::tanh(tanh_input));
        
        result_data[i] = gelu_gate * up_val;
    }
    
    if (gate->requires_grad() || up->requires_grad()) {
        result->set_requires_grad(true);
    }
    
    return result;
}

// Gemma MLP layer (keep frozen)
class GemmaMLP {
private:
    TensorPtr gate_weight_, up_weight_, down_weight_;
    int n_embd_, n_inner_;
    int layer_idx_;

public:
    GemmaMLP(int n_embd, int n_inner, int layer_idx) : n_embd_(n_embd), n_inner_(n_inner), layer_idx_(layer_idx) {
        std::cout << "Creating Gemma MLP layer " << layer_idx << " (GeGLU, frozen): " << n_embd_ << " -> " << n_inner_ << " -> " << n_embd_ << std::endl;

        try {
            std::cout << "Loading layer " << layer_idx << " pretrained MLP weights..." << std::endl;
            auto gate_weight_raw = load_binary_weights("models/gemma-270m/exported/gate_weight_" + std::to_string(layer_idx) + ".bin");
            auto up_weight_raw = load_binary_weights("models/gemma-270m/exported/up_weight_" + std::to_string(layer_idx) + ".bin");
            auto down_weight_raw = load_binary_weights("models/gemma-270m/exported/down_weight_" + std::to_string(layer_idx) + ".bin");
            
            // Transpose weights to match expected shape [input_dim, output_dim]
            gate_weight_ = transpose(gate_weight_raw, 0, 1);
            up_weight_ = transpose(up_weight_raw, 0, 1);
            down_weight_ = transpose(down_weight_raw, 0, 1);
            
            std::cout << "Layer " << layer_idx << " pretrained MLP weights loaded successfully!" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "Warning: Unable to load pretrained MLP weights, using random initialization: " << e.what() << std::endl;

            float scale = 0.02f / std::sqrt(static_cast<float>(n_embd_));
            gate_weight_ = mul_scalar(randn({n_inner_, n_embd_}), scale);
            up_weight_ = mul_scalar(randn({n_inner_, n_embd_}), scale);
            down_weight_ = mul_scalar(randn({n_embd_, n_inner_}), scale);
        }

        // Keep MLP weights frozen
        gate_weight_->set_requires_grad(false);
        up_weight_->set_requires_grad(false);
        down_weight_->set_requires_grad(false);
        
        std::cout << "MLP weights frozen (not participating in training)" << std::endl;
    }

    TensorPtr forward(const TensorPtr& x) {
        // GeGLU: down(GELU(gate(x)) * up(x))
        auto gate_proj = matmul(x, gate_weight_);
        auto up_proj = matmul(x, up_weight_);
        
        auto geglu_output = geglu(gate_proj, up_proj);
        auto output = matmul(geglu_output, down_weight_);

        return output;
    }

    // MLP does not return trainable parameters
    std::vector<TensorPtr> parameters() {
        return {};
    }
};

// Gemma LoRA Transformer Block
class GemmaLoRATransformerBlock {
private:
    std::unique_ptr<GemmaLoRAAttention> attention_;
    std::unique_ptr<GemmaMLP> mlp_;
    TensorPtr rms_attn_weight_;
    TensorPtr rms_ffn_weight_;
    GemmaLoRAConfig config_;
    int layer_idx_;

public:
    GemmaLoRATransformerBlock(const GemmaLoRAConfig& config, int layer_idx, 
                             std::shared_ptr<MobileParameterManager> param_manager = nullptr) 
        : config_(config), layer_idx_(layer_idx) {
        std::cout << "Creating Gemma LoRA Transformer Block " << layer_idx << "..." << std::endl;

        attention_ = std::make_unique<GemmaLoRAAttention>(config_.n_embd, config_.n_head, config_.n_kv_head, 
                                                        config_.rope_theta, layer_idx, 
                                                        config_.lora_rank, config_.lora_alpha, param_manager);
        mlp_ = std::make_unique<GemmaMLP>(config_.n_embd, config_.n_inner, layer_idx);

        try {
            std::cout << "Loading layer " << layer_idx << " RMSNorm weights..." << std::endl;
            auto rms_attn_raw = load_binary_weights("models/gemma-270m/exported/rms_attn_weight_" + std::to_string(layer_idx) + ".bin");
            auto rms_ffn_raw = load_binary_weights("models/gemma-270m/exported/rms_ffn_weight_" + std::to_string(layer_idx) + ".bin");
            // If shape is [640, 1], reshape to [640]
            rms_attn_weight_ = reshape(rms_attn_raw, {config_.n_embd});
            rms_ffn_weight_ = reshape(rms_ffn_raw, {config_.n_embd});
            std::cout << "Layer " << layer_idx << " RMSNorm weights loaded successfully!" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "Warning: Unable to load RMSNorm weights, using random initialization: " << e.what() << std::endl;
            rms_attn_weight_ = ones({config_.n_embd});
            rms_ffn_weight_ = ones({config_.n_embd});
        }

        // Keep RMSNorm weights frozen
        rms_attn_weight_->set_requires_grad(false);
        rms_ffn_weight_->set_requires_grad(false);

        std::cout << "Complete Gemma LoRA Transformer Block created!" << std::endl;
    }

    TensorPtr forward(const TensorPtr& x) {
        // Activation checkpointing: save activations at alternate layers (even layers checkpointed)
        // Recomputing during backward pass reduces activation memory
        auto ln1_out = rms_norm(x, rms_attn_weight_, config_.rms_norm_eps);
        auto attn_out = attention_->forward(ln1_out);
        auto residual1 = add(x, attn_out);
        
        auto ln2_out = rms_norm(residual1, rms_ffn_weight_, config_.rms_norm_eps);
        auto mlp_out = mlp_->forward(ln2_out);
        auto residual2 = add(residual1, mlp_out);
        
        return residual2;
    }

    std::vector<TensorPtr> parameters() {
        // Only return Attention layer's LoRA parameters
        return attention_->parameters();
    }
};

// Main model
class GemmaLoRAFinetune {
private:
    TensorPtr wte_, lm_head_;
    std::vector<std::unique_ptr<GemmaLoRATransformerBlock>> transformer_blocks_;
    TensorPtr ln_f_weight_;
    GemmaLoRAConfig config_;
    
    // Memory manager
    std::shared_ptr<MobileParameterManager> param_manager_;

public:
    GemmaLoRAFinetune(const GemmaLoRAConfig& config) : config_(config) {
        if (!config_.quiet_mode) {
            std::cout << "Creating Gemma LoRA fine-tuning model (memory optimized version)..." << std::endl;
        }
        
        // Initialize memory manager
        param_manager_ = std::make_shared<MobileParameterManager>(config_.memory_config);
        
        if (!config_.quiet_mode) {
            auto stats = param_manager_->get_memory_stats();
            std::cout << "Memory manager initialized:" << std::endl;
            std::cout << "  - GPU limit: " << config_.memory_config.max_gpu_memory_mb << "MB" << std::endl;
            std::cout << "  - CPU limit: " << config_.memory_config.max_cpu_memory_mb << "MB" << std::endl;
            std::cout << "  - Quantization mode: FP16" << std::endl;
            std::cout << "  - Parameter persistence threshold: " << config_.memory_config.param_persistence_threshold << " layers" << std::endl;
        }
        
        try {
            wte_ = load_binary_weights("models/gemma-270m/exported/wte.bin");
            // Memory optimization: Tied Embeddings
            // lm_head shares weights with wte, directly saves ~640MB of resident memory
            lm_head_ = wte_;  // Share the same tensor
            std::cout << "Enabled Tied Embeddings: saves ~640MB memory" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "Warning: Unable to load pretrained embedding weights: " << e.what() << std::endl;
            
            float scale = 0.02f / std::sqrt(static_cast<float>(config_.n_embd));
            wte_ = mul_scalar(randn({config_.vocab_size, config_.n_embd}), scale);
            lm_head_ = wte_;  // Share
        }

        // Freeze embedding weights and register to memory manager (register only once)
        wte_->set_requires_grad(false);
        param_manager_->register_parameter("wte_lm_head_tied", wte_, false);

        // Create 18 transformer layers (using memory management)
        transformer_blocks_.reserve(config_.n_layer);
        for (int i = 0; i < config_.n_layer; ++i) {
            if (!config_.quiet_mode) {
                std::cout << "Creating transformer block " << i << "..." << std::endl;
            }
            transformer_blocks_.push_back(std::make_unique<GemmaLoRATransformerBlock>(config_, i, param_manager_));
        }

        try {
            std::cout << "Loading final RMSNorm weights..." << std::endl;
            auto ln_f_raw = load_binary_weights("models/gemma-270m/exported/rms_final_weight.bin");
            ln_f_weight_ = reshape(ln_f_raw, {config_.n_embd});
            std::cout << "Final RMSNorm weights loaded successfully!" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "Warning: Unable to load final RMSNorm weights: " << e.what() << std::endl;
            ln_f_weight_ = ones({config_.n_embd});
        }

        ln_f_weight_->set_requires_grad(false);
        
        std::cout << "Complete Gemma LoRA fine-tuning model created!" << std::endl;
        std::cout << "  - Frozen parameters: Embedding layer + MLP + RMSNorm + LM head (pretrained weights)" << std::endl;
        std::cout << "  - Trainable parameters: LoRA adapters in Attention layers" << std::endl;
    }

    TensorPtr forward(const TensorPtr& input_ids) {
        auto batch_size = input_ids->shape()[0];
        auto seq_len = input_ids->shape()[1];

        auto input_ids_int = input_ids->data<int32_t>();

        // Token embedding
        auto token_emb = zeros({batch_size, seq_len, config_.n_embd});
        const float* wte_data = wte_->data<float>();
        float* token_emb_data = token_emb->data<float>();

        for (int64_t b = 0; b < batch_size; ++b) {
            for (int64_t s = 0; s < seq_len; ++s) {
                int32_t token_id = input_ids_int[b * seq_len + s];
                
                if (token_id < 0 || token_id >= config_.vocab_size) {
                    token_id = 0;
                }
                
                for (int64_t d = 0; d < config_.n_embd; ++d) {
                    token_emb_data[b * seq_len * config_.n_embd + s * config_.n_embd + d] = 
                        wte_data[token_id * config_.n_embd + d];
                }
            }
        }

        // Pass through 18 transformer layers
        // Temporarily disable checkpoint to debug Step 30 crash issue
        auto transformer_output = token_emb;
        for (int i = 0; i < config_.n_layer; ++i) {
            transformer_output = transformer_blocks_[i]->forward(transformer_output);
        }
        
        auto ln_f_output = rms_norm(transformer_output, ln_f_weight_, config_.rms_norm_eps);
        
        auto lm_head_t = transpose(lm_head_, 0, 1);
        auto logits_raw = matmul(ln_f_output, lm_head_t);
        
        // Fix: Scale logits to reasonable range
        // Gemma may need special scaling factor
        float logits_scale = 1.0f / std::sqrt(static_cast<float>(config_.n_embd));  // Similar to attention scaling
        auto logits = mul_scalar(logits_raw, logits_scale);
        
        // Logits debug info removed
        
        return logits;
    }

    TensorPtr compute_loss(const TensorPtr& logits, const TensorPtr& targets) {
        auto batch_size = logits->shape()[0];
        auto seq_len = logits->shape()[1];
        auto vocab_size = logits->shape()[2];
        // Direct cross-entropy (internally uses log_softmax), ensure target is int32
        auto flattened_logits = reshape(logits, {batch_size * seq_len, vocab_size});
        auto flattened_targets = reshape(targets, {batch_size * seq_len});
        return cross_entropy_loss(flattened_logits, flattened_targets);
    }
    
    /**
     * @brief Compute loss using chunked CE (directly from transformer output, without generating full logits)
     * Peak memory reduced from O(B*L*V) to O(B*L*chunk)
     */
    TensorPtr compute_loss_chunked(const TensorPtr& input_ids, const TensorPtr& targets, int64_t chunk_size = 4096) {
        auto batch_size = input_ids->shape()[0];
        auto seq_len = input_ids->shape()[1];
        
        // Forward to ln_f_output (without generating logits)
        auto input_ids_int = input_ids->data<int32_t>();
        
        // Token embedding
        auto token_emb = zeros({batch_size, seq_len, config_.n_embd});
        const float* wte_data = wte_->data<float>();
        float* token_emb_data = token_emb->data<float>();
        
        for (int64_t b = 0; b < batch_size; ++b) {
            for (int64_t s = 0; s < seq_len; ++s) {
                int32_t token_id = input_ids_int[b * seq_len + s];
                
                if (token_id < 0 || token_id >= config_.vocab_size) {
                    token_id = 0;
                }
                
                for (int64_t d = 0; d < config_.n_embd; ++d) {
                    token_emb_data[b * seq_len * config_.n_embd + s * config_.n_embd + d] = 
                        wte_data[token_id * config_.n_embd + d];
                }
            }
        }
        
        // Pass through transformer
        auto transformer_output = token_emb;
        for (int i = 0; i < config_.n_layer; ++i) {
            transformer_output = transformer_blocks_[i]->forward(transformer_output);
        }
        
        auto ln_f_output = rms_norm(transformer_output, ln_f_weight_, config_.rms_norm_eps);
        
        // Apply logits scaling to ln_f_output
        float logits_scale = 1.0f / std::sqrt(static_cast<float>(config_.n_embd));
        auto ln_f_scaled = mul_scalar(ln_f_output, logits_scale);
        
        // Use chunked CE (lm_head shape [V, D], needs automatic detection)
        auto loss = chunked_ce::chunked_cross_entropy_loss(ln_f_scaled, lm_head_, targets, chunk_size);
        
        return loss;
    }

    std::vector<TensorPtr> trainable_parameters() {
        std::vector<TensorPtr> params;
        for (int i = 0; i < config_.n_layer; ++i) {
            auto layer_params = transformer_blocks_[i]->parameters();
            params.insert(params.end(), layer_params.begin(), layer_params.end());
        }
        return params;
    }

    void print_parameter_info() {
        auto trainable_params = trainable_parameters();
        int64_t total_trainable = 0;
        for (auto& param : trainable_params) {
            total_trainable += param->numel();
        }
        
        // Estimate total parameters (Gemma 270M)
        int64_t total_params = config_.vocab_size * config_.n_embd * 2 + 
                              1024 * config_.n_embd +  // Position embedding
                              config_.n_embd * config_.n_embd * 4 +  // attention
                              config_.n_embd * config_.n_inner * 3 + // MLP
                              config_.n_embd * 3;  // RMSNorms

        std::cout << "Gemma LoRA model parameter info:" << std::endl;
        std::cout << "  - Total parameters: " << total_params << std::endl;
        std::cout << "  - LoRA parameters: " << total_trainable << " (Attention layer LoRA adapters)" << std::endl;
        std::cout << "  - Frozen parameters: " << (total_params - total_trainable) << std::endl;
        std::cout << "  - Trainable ratio: " << std::fixed << std::setprecision(5) 
                  << (float)total_trainable/total_params*100 << "%" << std::endl;
    }
};

// Simple optimizer
class SimpleOptimizer {
private:
    std::vector<TensorPtr> parameters_;
    float lr_;

public:
    SimpleOptimizer(const std::vector<TensorPtr>& parameters, float lr) : parameters_(parameters), lr_(lr) {}

    void step() {
        for (auto& param : parameters_) {
            if (param->grad()) {
                auto grad = param->grad();
                auto scaled_grad = mul_scalar(grad, lr_);
                auto new_param = sub(param, scaled_grad);
                
                std::memcpy(param->data<float>(), new_param->data<float>(), 
                           param->numel() * sizeof(float));
                
                param->set_grad(nullptr);
            }
        }
    }

    void zero_grad() {
        for (auto& param : parameters_) {
            param->set_grad(nullptr);
        }
    }
};

// LoRA fine-tuner (18-layer standard version, relies on memory optimization)
class GemmaLoRAFinetuner {
private:
    std::unique_ptr<GemmaLoRAFinetune> model_;
    std::unique_ptr<SimpleOptimizer> optimizer_;
    GemmaLoRAConfig config_;

public:
    GemmaLoRAFinetuner(const GemmaLoRAConfig& config) : config_(config) {
        model_ = std::make_unique<GemmaLoRAFinetune>(config_);
        
        auto trainable_params = model_->trainable_parameters();
        optimizer_ = std::make_unique<SimpleOptimizer>(trainable_params, config_.lr);
        
        std::cout << "Gemma LoRA fine-tuner initialized (18-layer standard version)!" << std::endl;
        std::cout << "  - Optimizer: SimpleOptimizer (SGD style)" << std::endl;
        std::cout << "  - Memory optimization: MobileParamManager (FP16 quantization + offload)" << std::endl;
        std::cout << "  - Trainable parameters: " << trainable_params.size() << " LoRA adapters" << std::endl;
    }

    void train(GemmaTokenizedDataLoader& dataloader) {
        std::cout << "\nStarting Gemma LoRA fine-tuning..." << std::endl;
        
        // Fixed 200 steps per epoch
        size_t total_sequences = dataloader.size();
        int steps_per_epoch = 200;  // Fixed 200 steps, not dependent on dynamic calculation
        
        std::cout << "DataLoader status:" << std::endl;
        std::cout << "  - Total sequences: " << total_sequences << std::endl;
        std::cout << "  - Steps per epoch: " << steps_per_epoch << std::endl;
        std::cout << "  - Batch size: " << config_.batch_size << std::endl;
        std::cout << "  - Sequence length: " << config_.block_size << std::endl;

        for (int epoch = 0; epoch < config_.max_epochs; ++epoch) {
            std::cout << "\nEpoch " << (epoch + 1) << "/" << config_.max_epochs << std::endl;
            
            dataloader.reset();
            float epoch_loss = 0.0f;
            int steps = 0;

            auto epoch_start = std::chrono::high_resolution_clock::now();

            // Fixed 200-step training loop, not dependent on has_more()
            for (int step = 0; step < steps_per_epoch; ++step) {
                try {
                    auto step_start = std::chrono::high_resolution_clock::now();

                    auto [input_ids, targets] = dataloader.get_batch(config_.batch_size);

                    optimizer_->zero_grad();

                    // Memory optimization: Use chunked CE, avoid generating full logits [B,L,V]
                    // Peak memory reduced from O(B*L*V) to O(B*L*chunk_size)
                    auto loss = model_->compute_loss_chunked(input_ids, targets, 4096);
                    
                    float loss_val = loss->item<float>();
                    
                    // Check if loss is abnormal
                    if (std::isnan(loss_val) || std::isinf(loss_val)) {
                        std::cout << "Warning: Loss abnormal = " << loss_val << ", skipping this step" << std::endl;
                        
                        // Release and clean up
                        loss.reset();
                        input_ids.reset();
                        targets.reset();
                        ops::MemoryManager::instance().force_cleanup();
                        continue;
                    }
                    
                    epoch_loss += loss_val;
                    steps++;

                    // Backward propagation
                    loss->backward();
                    
                    // Key fix: Immediately release large tensors
                    loss.reset();
                    input_ids.reset();
                    targets.reset();
                    
                    // Key fix: Force memory cleanup
                    ops::MemoryManager::instance().clear_unused_memory();
                    ops::MemoryManager::instance().cleanup_dead_references();
                    
                    // Optimizer update
                    optimizer_->step();
                    
                    // Key fix: Clean up again after update
                    ops::MemoryManager::instance().force_cleanup();

                auto step_end = std::chrono::high_resolution_clock::now();
                auto step_duration = std::chrono::duration_cast<std::chrono::milliseconds>(step_end - step_start);

                // Fix: Additional cleanup every 5 steps (more frequent)
                if ((step + 1) % 5 == 0) {
                    ops::MemoryManager::instance().clear_unused_memory();
                    ops::MemoryManager::instance().cleanup_dead_references();
                    ops::MemoryManager::instance().force_cleanup();
                }

                if ((step + 1) % 10 == 0 || step == 0) {
                    float avg_loss = epoch_loss / steps;
                    size_t current_rss = RtMem::rss_bytes();
                    std::cout << "  Step " << (step + 1) << "/" << steps_per_epoch 
                              << " - Loss: " << std::fixed << std::setprecision(4) << loss_val
                              << " - Avg: " << avg_loss
                              << " - Time: " << step_duration.count() << "ms"
                              << " - Memory: " << RtMem::fmt(current_rss)
                              << std::endl;
                    std::cout.flush();  // Force flush output
                }
                
                // Also print brief progress every 5 steps
                if ((step + 1) % 5 == 0 && (step + 1) % 10 != 0) {
                    std::cout << "  ." << std::flush;
                }
                
                // Print detailed memory ledger every 50 steps
                if ((step + 1) % 50 == 0) {
                    std::cout << "  Debug: Completed " << (step + 1) << " steps, continuing training..." << std::endl;
                    
                    // Get all parameters for memory ledger
                    auto trainable_params = model_->trainable_parameters();
                    auto ledger = MemoryLedger::compute(trainable_params, trainable_params, 
                                                       RtMem::rss_bytes());
                    std::cout << ledger.to_string() << std::endl;
                }
                    
                } catch (const std::exception& e) {
                    std::cout << "Step " << (step + 1) << " exception: " << e.what() << std::endl;
                    std::cout << "   Current memory: " << RtMem::fmt(RtMem::rss_bytes()) << std::endl;
                    // Continue to next step instead of crashing
                    continue;
                } catch (...) {
                    std::cout << "Step " << (step + 1) << " unknown exception" << std::endl;
                    std::cout << "   Current memory: " << RtMem::fmt(RtMem::rss_bytes()) << std::endl;
                    continue;
                }
            }

            auto epoch_end = std::chrono::high_resolution_clock::now();
            auto epoch_duration = std::chrono::duration_cast<std::chrono::seconds>(epoch_end - epoch_start);
            
            float avg_epoch_loss = epoch_loss / steps;
            std::cout << "Epoch " << (epoch + 1) << " completed (actually completed " << steps << " steps)" << std::endl;
            std::cout << "  Average loss: " << std::fixed << std::setprecision(4) << avg_epoch_loss << std::endl;
            std::cout << "  Time: " << epoch_duration.count() << " seconds" << std::endl;
        }

        std::cout << "\nGemma LoRA fine-tuning complete!" << std::endl;
    }
};

int main() {
        std::cout << "Gemma 270M LoRA Fine-tuning - Operators Framework Version" << std::endl;
        std::cout << "-------------------------------------------------------" << std::endl;

    try {
        // Initialize global checkpoint manager (default configuration is fine)
        ops::memory::GlobalCheckpointManager::initialize();
        GemmaLoRAConfig config;
        config.print();

        std::cout << "\nPreparing training data..." << std::endl;
        GemmaTokenizedDataLoader dataloader(config.block_size);

        GemmaLoRAFinetuner trainer(config);

        trainer.train(dataloader);

    } catch (const std::exception& e) {
        std::cout << "Training failed: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
