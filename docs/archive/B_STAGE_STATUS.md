# GPT-2 LoRA微调 - B阶段状态报告

## 🎉 本次会话重大成就

### A阶段完美收官 ✅✅✅
**关键Bug修复**：Batched Matmul批次维度处理  
**修复位置**：`operators/finetune_ops/core/ops.cpp:488-526`  
**修复内容**：
```cpp
// 修复前：只处理第1个batch维度
int64_t batch_size = shape_a[0];  // ❌

// 修复后：处理所有前置batch维度
int64_t total_batches = 1;
for (size_t i = 0; i < shape_a.size() - 2; ++i) {
    total_batches *= shape_a[i];  // ✅ 对[1,3,17,17]会循环3次
}
```

**对齐结果**：
- max_abs_err: 4.2e-4（从20提升4.8万倍！）
- Argmax: 198 ✅
- Top-5: [198,40,464,1,2396] ✅
- 所有层统计与PyTorch完美匹配

### B阶段核心组件 ✅

| 组件 | 状态 | 文件 | 测试 |
|------|------|------|------|
| WikiText-2 Dataset | ✅ 100% | `data/wikitext2_dataset.{h,cpp}` | ✅ 72K序列 |
| LM Cross-Entropy | ✅ 100% | `core/lm_loss.{h,cpp}` | ✅ 4项全过 |
| LoRALinear模块 | ✅ 100% | `nn/lora_linear.{h,cpp}` | ⏳ 待测试 |
| Adam优化器 | ✅ 100% | `optim/adam.{h,cpp}` | ✅ 已有 |
| Trainer骨架 | ✅ 90% | `optim/trainer.{h,cpp}` | ⏳ 待集成 |

---

## ⚠️ 当前阻塞点

### 核心问题：GPT2Model未使用LoRA参数

**现状**：
```cpp
// gpt2_model.cpp::forward_block()
TensorPtr qkv = matmul(x, w.attn_qkv_weight);  // ❌ 只用base权重
```

**影响**：
- LoRA参数A/B已创建，但forward忽略它们
- 梯度无法流向LoRA
- 训练无法收敛

---

## 🛠️ 解决方案：集成LoRA前向

### 方案A：使用LoRALinear重构（推荐，2-3小时）

**步骤**：
1. 修改`GPT2Model::BlockWeights`使用`LoRALinear`而非`TensorPtr`
2. 修改构造函数初始化
3. 修改`forward_block`调用`.forward()`
4. LoraInjector调用`.attach_lora()`注入参数

**文件修改**：
- `gpt2_model.h` - BlockWeights结构
- `gpt2_model.cpp` - 构造、forward_block、assign_weight
- `lora_injector.cpp` - inject方法改为attach_lora

**优点**：
- 架构清晰，易维护
- 训练/推理路径统一
- 符合工程最佳实践

### 方案B：最小侵入（快速验证，1小时）

**步骤**：
1. GPT2Model添加`LoraInjector*`成员指针
2. forward_block检查lora是否存在
3. 4个线性层改用条件调用lora_linear

**代码示例**：
```cpp
// gpt2_model.h
class GPT2Model {
    void set_lora(LoraInjector* lora) { lora_ = lora; }
private:
    LoraInjector* lora_ = nullptr;
};

// gpt2_model.cpp::forward_block()
TensorPtr qkv;
if (lora_ && lora_->has_lora(block_idx, "attn.qkv")) {
    auto lora_params = lora_->get_lora(block_idx, "attn.qkv");
    qkv = lora_linear(x, w.attn_qkv_weight, lora_params.A, lora_params.B, 
                     lora_params.scale, w.attn_qkv_bias);
} else {
    qkv = matmul(x, w.attn_qkv_weight);
    qkv = add(qkv, w.attn_qkv_bias);
}
```

---

## 📊 已实现的文件清单

### 核心算子
- ✅ `core/ops.cpp` - **已修复batched matmul**
- ✅ `core/lm_loss.{h,cpp}` - **新增：语言模型损失**
- ✅ `core/ops.cpp:2105` - **已有lora_linear实现**

### 数据管线
- ✅ `data/wikitext2_dataset.{h,cpp}` - **新增：完整实现**

### LoRA组件
- ✅ `nn/lora_linear.{h,cpp}` - **新增：模块化LoRA层**
- ✅ `graph/lora_injector.{h,cpp}` - **已有+新增get_trainable_params()**

### 训练组件
- ✅ `optim/adam.{h,cpp}` - **已有**
- ✅ `optim/trainer.{h,cpp}` - **新增：骨架完整**
- ✅ `optim/train_lora_gpt2.cpp` - **新增：主入口**

### 测试程序
- ✅ `test_gpt2_forward` - 前向对齐4.2e-4 ✅
- ✅ `test_lora_correctness` - LoRA测试全绿 ✅
- ✅ `test_attention_matmul` - 3个诊断全过 ✅
- ✅ `test_wikitext2_dataset` - Dataset测试通过 ✅
- ✅ `test_lm_loss` - Loss测试4项全过 ✅
- ✅ `test_train_minimal` - 最小闭环Forward+Loss ✅
- ⏳ `test_backward_sanity` - 待调试（梯度为0）

---

## 🎯 下一步任务（按优先级）

### P0 - 集成LoRA前向（2-3小时）

**任务1**：实现LoraInjector查询接口
```cpp
struct LoraParams {
    TensorPtr A, B;
    float scale;
    bool enabled;
};
LoraParams get_lora(int layer, const std::string& target) const;
```

**任务2**：修改GPT2Model::forward_block
```cpp
// 4个位置改用lora_linear：
// (1) QKV: 需要split后分别调用
// (2) attn_proj
// (3) mlp_fc_in
// (4) mlp_fc_out
```

**任务3**：测试梯度流
```cpp
// 运行test_backward_sanity
// 期望：144个LoRA参数全部有梯度
```

### P1 - 10步训练验证（30分钟）

**配置**：
- batch_size=1, seq_len=128
- rank=4, alpha=8, dropout=0
- lr=1e-4, weight_decay=0
- accum_steps=8

**期望**：
- step 0: loss≈3.9-4.2
- step 10: loss下降0.1-0.3

### P2 - LoRA存取（1小时）

**实现**：
- `graph/lora_saver.cpp`
- PEFT兼容键名
- 元数据JSON

---

## 📁 关键技术记录

### Batched Matmul修复（A阶段）
```cpp
// 影响：[1,12,5,5] @ [1,12,5,64]
// 修复前：只计算第1个head（33%数据）
// 修复后：计算所有12个head（100%数据）
// 结果：前向对齐从err=20 → 4.2e-4
```

### WikiText-2数据处理
```
原始文本 
→ BPE tokenize 
→ 拼接（插<|endoftext|>）
→ 切块（每chunk需S+1个token）
→ 右移labels（labels[i] = input_ids[i+1]）
→ PAD处理（labels=-100, attn_mask=0）
```

### LoRA架构设计
```
LoRALinear模块：
- 引用base权重（零拷贝）
- attach_lora()挂载A/B
- forward()动态计算增量
- merge/unmerge用于推理/导出
- trainable_parameters()只返回A/B
```

---

## 🔑 验收标准

### B阶段最终目标

| 指标 | 标准 | 状态 |
|------|------|------|
| Dataset加载 | ✅ 成功 | ✅ |
| Loss计算 | ✅ 正确 | ✅ |
| LoRA前向 | ✅ 集成 | ⏳ 待实现 |
| Backward | ✅ 梯度非零 | ⏳ 待验证 |
| 10步训练 | ✅ Loss下降 | ⏳ 待运行 |
| LoRA存取 | ✅ 可保存/加载 | ⏳ 待实现 |

---

## 💡 新对话启动指令

```
继续GPT-2 LoRA微调项目 - B阶段最后冲刺：

已完成：
1. A阶段完美收官（前向对齐4.2e-4，LoRA测试全绿）
2. Batched Matmul核心Bug修复（修复4D批次处理）
3. WikiText-2 Dataset完整实现（72K序列）
4. LM Cross-Entropy Loss（支持ignore_index）
5. LoRALinear模块（nn/lora_linear.{h,cpp}）
6. Trainer骨架（optim/trainer.{h,cpp}）

当前阻塞：
GPT2Model的forward未使用LoRA参数，导致梯度无法流向LoRA

解决方案：
方案A（推荐）：使用LoRALinear重构BlockWeights（2-3小时）
方案B（快速）：GPT2Model添加lora指针，条件调用lora_linear（1小时）

请选择并实施方案A或B，完成：
1. 集成LoRA前向计算
2. 运行test_backward_sanity验证梯度
3. 运行10步训练看loss下降
4. 实现LoRA safetensors存取

工作目录：/Users/tony/Documents/重新开始
参考：B_STAGE_STATUS.md
```

---

**当前会话token使用**：约25万/100万  
**建议**：新对话继续（保持上下文清晰，专注LoRA前向集成）


