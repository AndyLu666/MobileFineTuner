/**
 * @file gpt2_model.h
 * @brief GPT-2 模型轻量外壳（组装 gpt2_components，支持 tie-weights 与 LoRA 注入）
 */

#pragma once

#include "../core/tensor.h"
#include "../nn/lora_linear.h"
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>

namespace ops {

/**
 * @brief GPT-2 Transformer Block权重（包含LoRALinear模块）
 */
struct GPT2BlockWeights {
    // LayerNorm
    TensorPtr ln_1_weight, ln_1_bias;
    TensorPtr ln_2_weight, ln_2_bias;
    
    // Attention融合权重（用于初始化）
    TensorPtr attn_qkv_weight;   // [C, 3C]
    TensorPtr attn_qkv_bias;     // [3C]
    TensorPtr attn_proj_weight;  // [C, C]
    TensorPtr attn_proj_bias;    // [C]
    
    // MLP权重
    TensorPtr mlp_fc_in_weight;   // [C, 4C]
    TensorPtr mlp_fc_in_bias;     // [4C]
    TensorPtr mlp_fc_out_weight;  // [4C, C]
    TensorPtr mlp_fc_out_bias;    // [C]
    
    // LoRA增强的线性层
    std::unique_ptr<LoRALinear> qkv_lin;
    std::unique_ptr<LoRALinear> proj_lin;
    std::unique_ptr<LoRALinear> fc_in_lin;
    std::unique_ptr<LoRALinear> fc_out_lin;
    
    bool lora_initialized = false;
};

/**
 * @brief GPT-2 配置（与 HuggingFace 对齐）
 */
struct GPT2Config {
    int vocab_size = 50257;
    int n_positions = 1024;
    int n_embd = 768;
    int n_layer = 12;
    int n_head = 12;
    float layernorm_eps = 1e-5f;      // 与 HF 一致
    bool tie_word_embeddings = true;  // lm_head ↔ wte
    bool use_cache = false;           // 训练时关闭 KV cache
    
    // 内存优化选项
    bool use_memory_efficient_attention = true;   // 使用流式softmax（推荐）
    bool use_bf16_activations = false;            // 激活降精到BF16（需要时开启）
    
    // 从 config.json 加载（后续实现）
    static GPT2Config from_pretrained(const std::string& config_path);
};

/**
 * @brief GPT-2 模型（轻量外壳）
 * 
 * 组装 Embedding + TransformerBlock + LayerNorm + lm_head
 * 支持：
 * - tie_weights（lm_head ↔ wte）
 * - assign_weight（供 safetensors_loader 填充）
 * - 前向推理（input_ids + attention_mask → logits）
 */
class GPT2Model {
public:
    explicit GPT2Model(const GPT2Config& config);
    ~GPT2Model() = default;
    
    /**
     * @brief 前向传播
     * @param input_ids [batch, seq_len] int32
     * @param attention_mask [batch, seq_len] int32/float (1=有效, 0=pad)
     * @return logits [batch, seq_len, vocab_size]
     */
    TensorPtr forward(const TensorPtr& input_ids,
                     const TensorPtr& attention_mask = nullptr);
    
    /**
     * @brief 绑定 lm_head.weight ↔ wte.weight（同一内存）
     */
    void tie_weights();
    
    /**
     * @brief 供 safetensors_loader 填充权重
     * @param key 内部键名（如 "wte.weight", "blocks.0.ln_1.weight"）
     * @param tensor 权重张量
     */
    void assign_weight(const std::string& key, const TensorPtr& tensor);
    
    /**
     * @brief 获取所有参数（用于优化器）
     */
    std::vector<TensorPtr> parameters();
    
    /**
     * @brief 获取可训练参数（LoRA 注入后，只返回 LoRA 参数）
     */
    std::vector<TensorPtr> trainable_parameters();
    
    /**
     * @brief 冻结除 LoRA 外的所有参数
     */
    void freeze_base_parameters();
    
    /**
     * @brief 打印模型信息
     */
    void print_model_info() const;
    
    const GPT2Config& config() const { return config_; }

    // ========================= LoRA 绑定所需 Getter =========================
    // 返回可写指针（TensorPtr*），供 LoRA 注入器绑定到实际层参数
    // i ∈ [0, config_.n_layer)
    std::pair<TensorPtr*, TensorPtr*> attn_qkv_params(int i);
    std::pair<TensorPtr*, TensorPtr*> attn_proj_params(int i);
    std::pair<TensorPtr*, TensorPtr*> mlp_fc_in_params(int i);
    std::pair<TensorPtr*, TensorPtr*> mlp_fc_out_params(int i);
    
    /**
     * @brief 访问block（供LoRA注入器使用）
     */
    GPT2BlockWeights& get_block(int i) { return blocks_[i]; }
    const GPT2BlockWeights& get_block(int i) const { return blocks_[i]; }
    
    /**
     * @brief 收集所有LoRA可训练参数
     */
    std::vector<TensorPtr> get_lora_parameters();
    
    /**
     * @brief 初始化LoRA模块（在加载权重后调用）
     */
    void init_lora_modules();

private:
    GPT2Config config_;
    
    // Embeddings
    TensorPtr wte_weight_;  // [vocab_size, n_embd]
    TensorPtr wpe_weight_;  // [n_positions, n_embd]
    
    // Transformer blocks（使用外部定义的GPT2BlockWeights）
    std::vector<GPT2BlockWeights> blocks_;
    
    // Final LayerNorm
    TensorPtr ln_f_weight_;
    TensorPtr ln_f_bias_;
    
    // lm_head（tie 到 wte_weight_，不单独存储）
    bool weights_tied_ = false;
    
    // 内部工具
    TensorPtr build_causal_mask(int seq_len);
    TensorPtr build_padding_mask(const TensorPtr& attention_mask);
    TensorPtr layer_norm(const TensorPtr& x, const TensorPtr& weight, const TensorPtr& bias);
    TensorPtr gelu_new(const TensorPtr& x);
    TensorPtr embedding_lookup(const TensorPtr& weight, const TensorPtr& indices);
    TensorPtr forward_block(const TensorPtr& x, int block_idx,
                           const TensorPtr& causal_mask,
                           const TensorPtr& pad_mask);
};

}  // namespace ops

