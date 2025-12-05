# GPT-2 LoRA微调 - 最后待办清单

## 🎯 唯一剩余任务：LoRA前向集成（预计2小时）

---

## ✅ 已完成（本次会话）

### A阶段
- ✅ Batched matmul bug修复（ops.cpp:490-526）
- ✅ 前向对齐4.2e-4
- ✅ LoRA测试全绿

### B阶段组件
- ✅ WikiText-2 Dataset完整实现
- ✅ LM Cross-Entropy Loss
- ✅ LoRALinear模块（nn/lora_linear.{h,cpp}）
- ✅ Trainer骨架
- ✅ GPT2Model已添加：
  - `init_lora_modules()`方法
  - `get_lora_parameters()`方法
  - `get_block(i)`访问器
  - BlockWeights结构已添加LoRALinear成员

---

## ⏳ 待完成（下次会话2小时）

### 任务1：修改forward_block使用LoRALinear（1小时）

**文件**：`operators/finetune_ops/graph/gpt2_model.cpp`

**需要修改的4个位置**：

#### 位置1：QKV线性（已部分完成）
当前已改为：
```cpp
if (w.lora_initialized && w.qkv_lin) {
    qkv = w.qkv_lin->forward(x);
} else {
    qkv = matmul(x, w.attn_qkv_weight);
    qkv = add(qkv, w.attn_qkv_bias);
}
```
✅ 已完成

#### 位置2：Attention Projection
需要找到并修改（搜索关键词：`context.*attn_proj`）：
```cpp
// 当前代码
auto attn_out = matmul(context, w.attn_proj_weight);
attn_out = add(attn_out, w.attn_proj_bias);

// 修改为
TensorPtr attn_out;
if (w.lora_initialized && w.proj_lin) {
    attn_out = w.proj_lin->forward(context);
} else {
    attn_out = matmul(context, w.attn_proj_weight);
    attn_out = add(attn_out, w.attn_proj_bias);
}
```

#### 位置3：MLP fc_in
需要找到并修改（搜索关键词：`mlp_fc_in_weight`）：
```cpp
// 当前代码
auto h = matmul(x, w.mlp_fc_in_weight);
h = add(h, w.mlp_fc_in_bias);

// 修改为
TensorPtr h;
if (w.lora_initialized && w.fc_in_lin) {
    h = w.fc_in_lin->forward(x);
} else {
    h = matmul(x, w.mlp_fc_in_weight);
    h = add(h, w.mlp_fc_in_bias);
}
```

#### 位置4：MLP fc_out
需要找到并修改（搜索关键词：`mlp_fc_out_weight`）：
```cpp
// 当前代码  
auto mlp_out = matmul(h, w.mlp_fc_out_weight);
mlp_out = add(mlp_out, w.mlp_fc_out_bias);

// 修改为
TensorPtr mlp_out;
if (w.lora_initialized && w.fc_out_lin) {
    mlp_out = w.fc_out_lin->forward(h);
} else {
    mlp_out = matmul(h, w.mlp_fc_out_weight);
    mlp_out = add(mlp_out, w.mlp_fc_out_bias);
}
```

---

### 任务2：修改LoraInjector为配置器模式（30分钟）

**文件**：`operators/finetune_ops/graph/lora_injector.cpp`

**修改inject方法**：
```cpp
void LoraInjector::inject(GPT2Model& model, const LoraSpec& spec) {
    int n_layer = 12;  // 或从model获取
    int C = 768;       // config_.n_embd
    float scale = spec.alpha / float(spec.rank);
    
    // 确保LoRA模块已初始化
    model.init_lora_modules();
    
    // 为每层attach LoRA
    for (int i = 0; i < n_layer; ++i) {
        auto& block = model.get_block(i);
        
        // 创建LoRA参数（Kaiming初始化）
        auto create_AB = [&](int in_dim, int out_dim) {
            auto A = zeros({in_dim, spec.rank}, kFloat32, kCPU);
            auto B = zeros({spec.rank, out_dim}, kFloat32, kCPU);
            // Kaiming初始化A，B初始化为0
            float* A_data = A->data<float>();
            float bound = std::sqrt(6.0f / (in_dim + spec.rank));
            std::mt19937 gen(42 + i);
            std::uniform_real_distribution<float> dist(-bound, bound);
            for (int64_t j = 0; j < A->numel(); ++j) {
                A_data[j] = dist(gen);
            }
            return std::make_pair(A, B);
        };
        
        // QKV（统一处理，暂不split）
        if (std::find(spec.targets.begin(), spec.targets.end(), LoraTarget::AttnQKV) != spec.targets.end()) {
            auto [A, B] = create_AB(C, 3*C);
            block.qkv_lin->attach_lora(A, B, scale, 0, 3*C);
        }
        
        // Attention Proj
        if (std::find(spec.targets.begin(), spec.targets.end(), LoraTarget::AttnProj) != spec.targets.end()) {
            auto [A, B] = create_AB(C, C);
            block.proj_lin->attach_lora(A, B, scale, 0, C);
        }
        
        // MLP fc_in
        if (std::find(spec.targets.begin(), spec.targets.end(), LoraTarget::MlpFcIn) != spec.targets.end()) {
            auto [A, B] = create_AB(C, 4*C);
            block.fc_in_lin->attach_lora(A, B, scale, 0, 4*C);
        }
        
        // MLP fc_out
        if (std::find(spec.targets.begin(), spec.targets.end(), LoraTarget::MlpFcOut) != spec.targets.end()) {
            auto [A, B] = create_AB(4*C, C);
            block.fc_out_lin->attach_lora(A, B, scale, 0, C);
        }
    }
    
    std::cout << "[LoraInjector] Injected LoRA to " << n_layer << " layers" << std::endl;
}
```

**修改get_trainable_params**：
```cpp
std::vector<TensorPtr> LoraInjector::get_trainable_params() {
    // 这个方法已经过时，使用model.get_lora_parameters()代替
    return {};  // 或删除此方法
}
```

---

### 任务3：更新测试程序（20分钟）

**文件**：`operators/finetune_ops/optim/test_backward_sanity.cpp`

**关键修改**：
```cpp
// 加载权重后
model.init_lora_modules();  // ← 新增：初始化LoRA模块

// 注入LoRA
lora.inject(model, lora_spec);

// 收集参数
auto lora_params = model.get_lora_parameters();  // ← 改用model方法
```

---

### 任务4：10步训练验证（10分钟）

创建简化的训练脚本：
```cpp
// test_10step_lora_train.cpp
for (int step = 0; step < 10; ++step) {
    auto batch = dataset.next_batch(1);
    
    // Forward
    auto logits = model.forward(batch.input_ids, batch.attention_mask);
    auto loss = lm_cross_entropy(logits, batch.labels, -100, "mean");
    
    // Backward
    loss->backward();
    
    // Optimizer step
    optimizer->step(lora_params, grads);
    
    // Zero grad
    for (auto& p : lora_params) p->zero_grad();
    
    printf("Step %d: Loss=%.4f\n", step, loss->data<float>()[0]);
}
```

**期望输出**：
```
Step 0: Loss=3.9-4.2
Step 5: Loss=3.7-3.9
Step 10: Loss=3.6-3.8  (下降0.1-0.3)
```

---

## 🧪 验收测试（4件套）

### 测试1：零影响性
```cpp
// 将所有A/B置零
for (auto& p : lora_params) {
    float* data = p->data<float>();
    for (int64_t i = 0; i < p->numel(); ++i) data[i] = 0.0f;
}

// Forward两次对比
auto logits1 = model.forward(...);  // 有LoRA（但A/B=0）
// 临时清空LoRA
// auto logits2 = model.forward(...);  // 无LoRA
// 期望：max_abs_diff(logits1, logits2) < 1e-6
```

### 测试2：Merge等价性
```cpp
auto logits_dynamic = model.forward(...);  // 动态LoRA

// Merge
for (int i = 0; i < n_layer; ++i) {
    auto& b = model.get_block(i);
    if (b.qkv_lin) b.qkv_lin->merge_to_base();
    if (b.proj_lin) b.proj_lin->merge_to_base();
    if (b.fc_in_lin) b.fc_in_lin->merge_to_base();
    if (b.fc_out_lin) b.fc_out_lin->merge_to_base();
}

auto logits_merged = model.forward(...);  // 使用merged权重

// 期望：max_abs_diff < 1e-6
```

### 测试3：梯度非零
```cpp
// test_backward_sanity已实现
// 期望：所有LoRA参数grad_norm > 0且有限
```

### 测试4：10步下降
```cpp
// 已在任务4中描述
// 期望：loss连续下降
```

---

## 🚨 注意事项

1. **init_lora_modules调用时机**：
   ```cpp
   model.assign_weight(...);  // 加载所有权重
   model.init_lora_modules(); // ← 必须在权重加载后
   lora.inject(model, spec);  // 才能attach LoRA
   ```

2. **Tensor接口差异**：
   - 我们使用`TensorPtr`（shared_ptr）
   - 用户示例中的`Tensor`需要改为`TensorPtr`
   - `.contiguous()`可能不存在，可以先不调用

3. **namespace**：
   - 我们使用`namespace ops`
   - 用户示例中的`finetune`需要改为`ops`

---

## 📋 下次会话启动清单

1. ✅ 读取本文档（FINAL_TODO.md）
2. ✅ 读取SESSION_HANDOVER.md了解完整背景
3. ⏳ 完成任务1-4（按顺序）
4. ⏳ 运行4个验收测试
5. ✅ B阶段完成！

---

## 🏁 B阶段完成标准

```
✅ test_backward_sanity通过（梯度非零）
✅ 10步训练loss下降0.1-0.3
✅ 零影响性测试通过（<1e-6）
✅ Merge等价性测试通过（<1e-6）
```

完成后B阶段100%闭环，可进入C阶段（MMLU评测）。

---

**预计剩余时间**：
- B阶段收尾：2小时
- C阶段（MMLU）：2-3小时
- **项目总剩余**：4-5小时

🚀 加油，胜利在望！

