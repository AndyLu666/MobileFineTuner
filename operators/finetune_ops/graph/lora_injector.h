/**
 * @file lora_injector.h
 * @brief LoRA 注入器（支持 split_qkv、merge/unmerge、save/load）
 */

#pragma once

#include "../core/tensor.h"
#include "gpt2_model.h"
#include <vector>
#include <string>
#include <unordered_map>

namespace ops {

/**
 * @brief LoRA 目标层类型
 */
enum class LoraTarget {
    AttnQKV,    // Attention QKV（支持拆分为 q/k/v 三组）
    AttnProj,   // Attention 输出投影
    MlpFcIn,    // MLP fc_in (C → 4C)
    MlpFcOut    // MLP fc_out (4C → C)
};

/**
 * @brief LoRA 配置
 */
struct LoraSpec {
    int rank = 8;
    float alpha = 16.0f;
    float dropout = 0.05f;
    bool split_qkv = true;  // QKV 是否拆分为 q/k/v 三组（推荐）
    
    // 默认只对注意力层应用 LoRA（与 PyTorch PEFT 传统做法一致）
    // 如需包含 MLP，可手动添加 LoraTarget::MlpFcIn 和 LoraTarget::MlpFcOut
    std::vector<LoraTarget> targets = {
        LoraTarget::AttnQKV,   // Attention Q/K/V 投影
        LoraTarget::AttnProj   // Attention 输出投影
    };
    
    std::vector<int> layers;  // 空=全层
    
    LoraSpec() = default;
    
    // 默认配置（开箱即用，仅注意力层）
    static LoraSpec default_config() {
        LoraSpec spec;
        spec.rank = 8;
        spec.alpha = 16.0f;
        spec.dropout = 0.05f;
        spec.split_qkv = true;
        return spec;
    }
    
    // 包含 MLP 的配置（更广覆盖）
    static LoraSpec full_config() {
        LoraSpec spec;
        spec.rank = 8;
        spec.alpha = 16.0f;
        spec.dropout = 0.05f;
        spec.split_qkv = true;
        spec.targets = {
            LoraTarget::AttnQKV,
            LoraTarget::AttnProj,
            LoraTarget::MlpFcIn,
            LoraTarget::MlpFcOut
        };
        return spec;
    }
};

/**
 * @brief 单个 LoRA 状态（A/B 矩阵 + 元信息）
 */
struct LoraState {
    TensorPtr A;  // [in, r]
    TensorPtr B;  // [r, out]
    float scale;  // alpha / r
    float dropout_p;
    bool enabled = true;
    
    // 初始化（A ~ N(0, 1/r), B = 0）
    void init(int64_t in_features, int64_t out_features, int rank, float alpha, float dropout);
};

/**
 * @brief LoRA 注入器
 */
class LoraInjector {
public:
    LoraInjector() = default;
    ~LoraInjector() = default;
    
    /**
     * @brief 在模型中注入 LoRA 并冻结 base 权重
     * @param model GPT2Model 实例
     * @param spec LoRA 配置
     */
    void inject(GPT2Model& model, const LoraSpec& spec);
    
    /**
     * @brief 合并 LoRA 到 base 权重（推理前）
     * W' = W + B @ A * scale
     */
    void merge();
    
    /**
     * @brief 还原 LoRA（训练继续）
     * W = W' - B @ A * scale
     */
    void unmerge();
    
    /**
     * @brief 合并模型中所有 LoRALinear 的 LoRA 到 base
     */
    void merge_all(GPT2Model& model);
    
    /**
     * @brief 还原模型中所有 LoRALinear 的 LoRA
     */
    void unmerge_all(GPT2Model& model);
    
    /**
     * @brief 收集可训练参数（仅 LoRA A/B）
     * @return LoRA 参数列表
     */
    std::vector<TensorPtr> collect_lora_parameters() const;
    
    /**
     * @brief 保存 LoRA 权重到 safetensors
     * @param path 输出路径
     */
    void save_lora_safetensors(const std::string& path) const;
    
    /**
     * @brief 从 safetensors 加载 LoRA 权重
     * @param path 输入路径
     */
    void load_lora_safetensors(const std::string& path);
    
    /**
     * @brief 打印 LoRA 注入信息
     */
    void print_info() const;
    
    /**
     * @brief 获取所有LoRA可训练参数（A和B矩阵）
     * @return 所有LoRA参数的列表
     */
    std::vector<TensorPtr> get_trainable_params();
    
    /**
     * @brief LoRA 增强的线性前向（包装函数）
     * @param x 输入 [*, in]
     * @param W base 权重 [in, out]
     * @param bias base bias [out]（可选）
     * @param lora LoRA 状态（可选）
     * @param training 是否训练模式（影响 dropout）
     * @return 输出 [*, out]
     */
    static TensorPtr lora_linear_forward(const TensorPtr& x,
                                        const TensorPtr& W,
                                        const TensorPtr& bias,
                                        const LoraState* lora,
                                        bool training = false);

private:
    struct Hook {
        std::string name;
        TensorPtr* W_ptr;
        TensorPtr* bias_ptr;
        LoraState state;
        // 当仅作用于权重的部分列时（如 Q/K/V 拆分），指定列范围
        // 列范围为 [col_offset, col_offset + col_size)
        int64_t col_offset = 0;
        int64_t col_size = -1;  // -1 表示覆盖整个 out 维度
    };
    
    std::vector<Hook> hooks_;
    LoraSpec spec_;
    bool merged_ = false;
    int num_layers_ = 0;
    
    // 内部辅助
    void inject_qkv_split(GPT2Model& model, int layer_idx, int rank, float alpha, float dropout);
    void inject_qkv_fused(GPT2Model& model, int layer_idx, int rank, float alpha, float dropout);
    void inject_layer(GPT2Model& model, int layer_idx, const std::string& layer_name,
                     int64_t in_features, int64_t out_features,
                     int rank, float alpha, float dropout);
};

}  // namespace ops

