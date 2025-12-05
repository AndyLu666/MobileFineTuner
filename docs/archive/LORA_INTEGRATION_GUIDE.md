# LoRA前向集成实施指南

## 🎯 目标
在GPT2Model中集成LoRALinear，实现训练时动态LoRA增量计算

---

## 📋 Step 1: 修改BlockWeights结构

### 当前结构（gpt2_model.h:106-126）
```cpp
struct BlockWeights {
    TensorPtr ln_1_weight, ln_1_bias;
    TensorPtr attn_qkv_weight;  // [C, 3C]
    TensorPtr attn_qkv_bias;    // [3C]
    TensorPtr attn_proj_weight; // [C, C]
    TensorPtr attn_proj_bias;
    TensorPtr ln_2_weight, ln_2_bias;
    TensorPtr mlp_fc_in_weight;  // [C, 4C]
    TensorPtr mlp_fc_in_bias;
    TensorPtr mlp_fc_out_weight; // [4C, C]
    TensorPtr mlp_fc_out_bias;
};
```

### 目标结构
```cpp
#include "../nn/lora_linear.h"

struct BlockWeights {
    // LayerNorm权重保持不变
    TensorPtr ln_1_weight, ln_1_bias;
    TensorPtr ln_2_weight, ln_2_bias;
    
    // 原始融合权重（用于初始化和存储）
    TensorPtr attn_qkv_weight;  // [C, 3C]
    TensorPtr attn_qkv_bias;    // [3C]
    TensorPtr attn_proj_weight;
    TensorPtr attn_proj_bias;
    TensorPtr mlp_fc_in_weight;
    TensorPtr mlp_fc_in_bias;
    TensorPtr mlp_fc_out_weight;
    TensorPtr mlp_fc_out_bias;
    
    // LoRA增强的线性层（6个模块）
    std::unique_ptr<LoRALinear> q_lin;
    std::unique_ptr<LoRALinear> k_lin;
    std::unique_ptr<LoRALinear> v_lin;
    std::unique_ptr<LoRALinear> proj_lin;
    std::unique_ptr<LoRALinear> fc_in_lin;
    std::unique_ptr<LoRALinear> fc_out_lin;
    
    // 初始化标志
    bool lora_ready = false;
};
```

---

## 📋 Step 2: 修改构造函数初始化

### 位置：gpt2_model.cpp:19-71（构造函数）

**当前代码**：直接初始化TensorPtr  
**修改方案**：权重仍然初始化，但LoRALinear延迟到assign_weight完成后

**新增方法**：
```cpp
// gpt2_model.cpp（在assign_weight之后调用）
void GPT2Model::init_lora_modules() {
    int C = config_.n_embd;
    
    for (auto& block : blocks_) {
        if (block.lora_ready) continue;  // 已初始化
        
        // 创建q/k/v的slice视图
        // 注意：需要实现slice操作，或手动创建子矩阵引用
        
        // 简化版：先用整个权重（待优化）
        block.q_lin = std::make_unique<LoRALinear>(
            block.attn_qkv_weight, block.attn_qkv_bias);
        block.k_lin = std::make_unique<LoRALinear>(
            block.attn_qkv_weight, block.attn_qkv_bias);
        block.v_lin = std::make_unique<LoRALinear>(
            block.attn_qkv_weight, block.attn_qkv_bias);
        
        block.proj_lin = std::make_unique<LoRALinear>(
            block.attn_proj_weight, block.attn_proj_bias);
        block.fc_in_lin = std::make_unique<LoRALinear>(
            block.mlp_fc_in_weight, block.mlp_fc_in_bias);
        block.fc_out_lin = std::make_unique<LoRALinear>(
            block.mlp_fc_out_weight, block.mlp_fc_out_bias);
        
        block.lora_ready = true;
    }
}
```

**调用时机**：
```cpp
// train_lora_gpt2.cpp主程序
model.assign_weight(...);  // 加载所有权重
model.init_lora_modules(); // 初始化LoRA模块
lora.inject(model, spec);  // 注入LoRA参数
```

---

## 📋 Step 3: 修改forward_block使用LoRALinear

### 位置：gpt2_model.cpp:298-447（forward_block函数）

### 当前QKV处理（line 344-370）
```cpp
TensorPtr qkv = matmul(x, w.attn_qkv_weight);
qkv = add(qkv, w.attn_qkv_bias);

// 手工切分Q/K/V
const float* qkv_data = qkv->data<float>();
auto q = zeros({B, S, C}, kFloat32, kCPU);
auto k = zeros({B, S, C}, kFloat32, kCPU);
auto v = zeros({B, S, C}, kFloat32, kCPU);
// ... memcpy切分 ...
```

### 目标代码
```cpp
// 直接用LoRALinear的forward（已包含LoRA增量）
auto q = w.q_lin->forward(x);  // [B, S, C]
auto k = w.k_lin->forward(x);
auto v = w.v_lin->forward(x);

// 注意：如果q/k/v共享权重，需要实现slice操作
// 临时方案：让每个都用完整qkv_weight，但只取特定列
```

### Attention Proj（line 449-451）
```cpp
// 当前
auto attn_out = matmul(context, w.attn_proj_weight);
attn_out = add(attn_out, w.attn_proj_bias);

// 目标
auto attn_out = w.proj_lin->forward(context);
```

### MLP（line 472-479）
```cpp
// 当前
auto h = matmul(x, w.mlp_fc_in_weight);
h = add(h, w.mlp_fc_in_bias);
h = gelu_new(h);
auto mlp_out = matmul(h, w.mlp_fc_out_weight);
mlp_out = add(mlp_out, w.mlp_fc_out_bias);

// 目标
auto h = w.fc_in_lin->forward(x);
h = gelu_new(h);
auto mlp_out = w.fc_out_lin->forward(h);
```

---

## 📋 Step 4: 修改LoraInjector的inject方法

### 位置：lora_injector.cpp:41-90（inject函数）

### 当前逻辑
创建Hook，保存W指针和LoRA状态

### 目标逻辑
直接调用block的LoRALinear模块的attach_lora()

### 伪代码
```cpp
void LoraInjector::inject(GPT2Model& model, const LoraSpec& spec) {
    int n_layer = model.config().n_layer;
    int C = model.config().n_embd;
    float scale = spec.alpha / spec.rank;
    
    for (int i = 0; i < n_layer; ++i) {
        // 获取block（需要GPT2Model暴露接口）
        auto& block = model.get_block(i);
        
        // 为每个目标创建A/B并attach
        for (auto target : spec.targets) {
            if (target == LoraTarget::AttnQKV && spec.split_qkv) {
                // Q/K/V分别注入
                auto [A_q, B_q] = create_lora_params(C, spec.rank, C);
                auto [A_k, B_k] = create_lora_params(C, spec.rank, C);
                auto [A_v, B_v] = create_lora_params(C, spec.rank, C);
                
                block.q_lin->attach_lora(A_q, B_q, scale, 0, C);
                block.k_lin->attach_lora(A_k, B_k, scale, 0, C);
                block.v_lin->attach_lora(A_v, B_v, scale, 0, C);
            }
            
            if (target == LoraTarget::AttnProj) {
                auto [A, B] = create_lora_params(C, spec.rank, C);
                block.proj_lin->attach_lora(A, B, scale);
            }
            
            // mlp_fc_in, mlp_fc_out类似
        }
    }
    
    // 冻结base权重
    freeze_base_weights(model);
}
```

---

## 📋 Step 5: 收集可训练参数

### 新增方法（gpt2_model.h）
```cpp
class GPT2Model {
public:
    /**
     * @brief 收集所有LoRA可训练参数
     */
    std::vector<TensorPtr> get_lora_parameters();
};
```

### 实现（gpt2_model.cpp）
```cpp
std::vector<TensorPtr> GPT2Model::get_lora_parameters() {
    std::vector<TensorPtr> params;
    for (const auto& block : blocks_) {
        if (!block.lora_ready) continue;
        
        auto add_params = [&](const std::unique_ptr<LoRALinear>& lin) {
            if (lin) {
                auto ps = lin->trainable_parameters();
                params.insert(params.end(), ps.begin(), ps.end());
            }
        };
        
        add_params(block.q_lin);
        add_params(block.k_lin);
        add_params(block.v_lin);
        add_params(block.proj_lin);
        add_params(block.fc_in_lin);
        add_params(block.fc_out_lin);
    }
    return params;
}
```

---

## 🧪 验证测试更新

### test_backward_sanity.cpp修改
```cpp
// 替换
auto lora_params = lora.get_trainable_params();

// 为
auto lora_params = model.get_lora_parameters();
```

---

## ⚠️ 临时简化方案（快速验证）

由于QKV需要slice操作，而Tensor可能没有slice接口，**临时方案**：

```cpp
// BlockWeights中暂时不分q/k/v，只用一个qkv_lin
struct BlockWeights {
    ...
    std::unique_ptr<LoRALinear> qkv_lin;   // 统一处理QKV
    std::unique_ptr<LoRALinear> proj_lin;
    std::unique_ptr<LoRALinear> fc_in_lin;
    std::unique_ptr<LoRALinear> fc_out_lin;
};

// forward_block
auto qkv = w.qkv_lin->forward(x);  // [B, S, 3C]，包含LoRA增量
// 然后手工切分Q/K/V（与原代码相同）
```

这样避免实现slice，快速验证梯度流。

---

## 📊 预计工作量

| 任务 | 预计时间 | 优先级 |
|------|---------|--------|
| 修改BlockWeights | 30分钟 | P0 |
| 修改forward_block | 30分钟 | P0 |
| 修改inject方法 | 30分钟 | P0 |
| 测试梯度流 | 20分钟 | P0 |
| 10步训练验证 | 20分钟 | P0 |
| LoRA save/load | 60分钟 | P1 |

**总计**：约3小时完成B阶段

---

**当前会话状态**：
- Token使用：约25万/100万
- 剩余：75万（足够）
- 建议：继续本会话，一鼓作气完成

**我立即开始实施修改吗？**
