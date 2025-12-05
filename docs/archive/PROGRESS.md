# GPT-2 LoRA 微调项目进度

## ✅ 已完成模块

### P0: BPE Tokenizer（与 HuggingFace 完全对齐）
- **文件**：`operators/finetune_ops/core/tokenizer_bpe.{h,cpp}`
- **功能**：
  - Byte-Level BPE（256 字节→Unicode 映射）
  - 加载 vocab.json (50257词) + merges.txt (49992对)
  - 支持 `\uXXXX` JSON 转义
  - encode/decode 与 HF 完全一致
  - batch_encode（padding/truncation/attention_mask）
- **测试**：✅ 所有单测 PASS
  - 中英混合+emoji："今天天气不错🙂, GPT-2 rocks!"
  - 空格/换行/标点：" a b\n\n--==??"
  - 往返一致性验证

### P1: SafeTensors Loader
- **文件**：`operators/finetune_ops/graph/safetensors_loader.{h,cpp}`
- **功能**：
  - 纯 C++ 解析 safetensors 格式
  - 支持 FP32/FP16（自动升格）
  - 自动转置线性层权重 [out,in]→[in,out]
  - GPT2KeyMapper（148 条键名映射）
- **测试**：✅ 成功解析 GPT-2 model.safetensors（161 个张量）
  - wte=[50257,768], wpe=[1024,768]
  - 所有层权重形状正确

### P1+: GPT2Model 外壳
- **文件**：`operators/finetune_ops/graph/gpt2_model.{h,cpp}`
- **功能**：
  - 轻量封装（基于 gpt2_components）
  - tie_weights（lm_head ↔ wte）
  - assign_weight（供 loader 填充）
  - forward（完整 12层 transformer）
  - forward_block（严格按 HF 逻辑）：
    - QKV 切分（q,k,v 顺序）
    - Multi-head attention（缩放 1/√Hd）
    - Causal + padding mask 合成
    - GELU new（tanh 近似）
    - LayerNorm eps=1e-5
- **状态**：✅ 核心代码已完成

### P1.5: LoRA 注入器
- **文件**：`operators/finetune_ops/graph/lora_injector.{h,cpp}`
- **功能**：
  - 支持 split_qkv（q/k/v 各一组 LoRA）
  - 目标层：attn.qkv, attn.proj, mlp.fc_in, mlp.fc_out
  - 初始化：A ~ N(0, 1/r), B = 0
  - merge/unmerge（幂等性保证）
  - collect_lora_parameters（仅返回 A/B）
  - lora_linear_forward（包装函数，支持 dropout）
- **状态**：✅ 框架已完成，待补充权重指针获取

## 📋 待完成模块

### P0.5: Data Pipeline
- WikiText-2 切块（拼接/滑窗）
- MMLU prompt 构造（评测/训练两套样式）
- labels 中 PAD→-100

### P2: Trainer
- AdamW(decoupled)
- 梯度裁剪（clip_norm=1.0）
- 梯度累积
- LR scheduler（cosine + warmup）
- 训练循环（forward→CE→backward→step→log）

### P2.5: MMLU Evaluator
- 多选评分（条件对数似然、长度归一）
- Subject/overall 准确率

## 🎯 下一步立即要做

1. **补充 GPT2Model 的权重指针 getter**（供 LoraInjector 使用）
2. **编译并测试 GPT2Model 前向**（加载权重→固定输入→打印 logits）
3. **LoRA 零影响性测试**（A=0,B=0 时输出一致）
4. **LoRA merge 幂等性测试**
5. **编写训练入口**（train_lora_gpt2.cpp）

## 📂 目录结构

```
operators/
  finetune_ops/
    core/
      tokenizer_bpe.{h,cpp}      ✅
      tensor, ops, autograd等     ✅
    graph/
      safetensors_loader.{h,cpp} ✅
      gpt2_model.{h,cpp}         ✅
      lora_injector.{h,cpp}      ✅
    nn/, transformer/, optim/    ✅
  opt_ops/
    (优化项模块)
```

## 🔑 关键约定

- **权重布局**：Linear 统一 [in,out]，HF 加载时转置
- **LoRA 形式**：A[in,r], B[r,out], ΔW=A@B, scale=alpha/r
- **tie_weights**：lm_head.weight ↔ wte.weight（同一内存）
- **QKV 切分**：split_qkv=true，q/k/v 各一组 LoRA
- **GELU**：gelu_new（tanh 近似）
- **LayerNorm**：eps=1e-5

## 📊 验收标准

### Tokenizer
- ✅ encode/decode 往返一致
- ✅ 与 HF 完全对齐

### SafeTensors
- ✅ 解析 161 个张量
- ✅ 形状正确

### 前向对齐（待验证）
- max_abs_err(logits_cpp, logits_pt) ≤ 1e-4
- argmax 一致
- top-5 基本一致

### LoRA（待验证）
- 零影响性（A=0,B=0）
- merge 幂等性
- 参数收集正确

## 🚀 最终目标

完整训练环：
```
tokenizer → dataset → model(+LoRA) → forward → CE → backward → 
AdamW(仅 LoRA) → clip → scheduler → step → log → save LoRA → 
eval(WikiText-2 ppl + MMLU acc)
```

