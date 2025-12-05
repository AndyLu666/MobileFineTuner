/**
 * @file lora_saver.h
 * @brief LoRA safetensors 存取（PEFT兼容）
 */

#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include "../core/tensor.h"
#include "gpt2_model.h"
#include "lora_injector.h"

namespace ops {

struct LoRAState {
    int rank = 8;
    float alpha = 16.0f;
    float dropout = 0.0f;
    bool split_qkv = true;
    std::vector<LoraTarget> targets;
    
    // 每层每目标的A/B
    // key: "layer.{i}.{target}.lora_A" / "lora_B"
    std::unordered_map<std::string, TensorPtr> tensors;
    
    bool compatible_with(const GPT2Model& model) const;
};

class LoraSaver {
public:
    // 保存：model必须已注入LoRA
    static void save_safetensors(const std::string& path, 
                                  const GPT2Model& model,
                                  const LoraSpec& spec);
    
    // 加载：返回state，由外部attach到model
    static LoRAState load_safetensors(const std::string& path);
    
    // 将加载的state attach到model
    static void attach_from_state(GPT2Model& model, const LoRAState& state);
    
    // 辅助方法（公开，供外部使用）
    static std::string make_peft_key(int layer, const std::string& target, const std::string& ab);
    static bool parse_peft_key(const std::string& key, int& layer, std::string& target, std::string& ab);
};

} // namespace ops
