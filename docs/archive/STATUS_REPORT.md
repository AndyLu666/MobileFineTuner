# GPT-2 LoRA 微调项目状态汇报

## 当前进度：A 阶段（前向对齐 + LoRA 测试）

### ✅ 已完成
1. **核心组件**
   - BPE Tokenizer（与 HF 完全对齐）
   - SafeTensors Loader（权重加载+转置）
   - GPT2Model 外壳（12 层前向+tie-weights）
   - LoRA 注入器（split_qkv+merge/unmerge）
   - 测试代码（test_gpt2_forward.cpp + test_lora_correctness.cpp）

2. **数据资源**
   - GPT-2 预训练权重（model.safetensors，已下载）
   - WikiText-2 raw（train/valid/test，已下载）
   - MMLU 完整 57 科目（已下载）

### 🚧 正在解决（编译问题）
- **问题**：ops.cpp 中引用了 mobile_matmul 和 fp16（属于 opt_ops 优化模块）
- **临时方案**：用纯 C++ naive matmul 替换，fp16 转换暂时注释（不影响前向对齐测试）
- **下一步**：完成编译 → 生成 PyTorch 金标 → 运行前向对齐测试

### 📋 待验证（A 阶段剩余）
1. 前向对齐：max_abs_err ≤ 1e-4，argmax 一致
2. LoRA 零影响：B=0 时输出与 base 一致
3. LoRA 幂等性：merge/unmerge 可逆

## 项目结构确认

### 目标
- **任务**：GPT-2 LoRA 微调（纯 C++）
- **模型**：GPT-2 small（12 层全注入 LoRA）
- **数据**：WikiText-2（训练）+ MMLU（评测）
- **对齐**：与 PyTorch/HuggingFace + PEFT 对齐

### 主运行代码（待创建）
```bash
./build/train_lora_gpt2 \
  --model_path pretrained/gpt2 \
  --train_data data/wikitext2/wikitext-2-raw/wiki.train.raw \
  --lora_rank 8 \
  --learning_rate 2e-4
```

##  下一步计划（按优先级）
1. **立即**：完成编译 → 运行前向对齐测试（A 阶段）
2. **今天**：实现 Trainer 最小闭环（AdamW+clip+scheduler）
3. **今天**：实现 LoRA safetensors save/load（PEFT 兼容）
4. **明天**：WikiText-2 数据管线 + 完整训练流程

## 已知限制
- 当前使用 naive matmul（O(n³)），性能较慢但数值正确
- FP16 转换暂时禁用（不影响 FP32 训练）
- 后续可接入 BLAS/opt_ops 优化模块提速

## 预计时间
- A 阶段（前向对齐测试）：2 小时内完成
- B 阶段（Trainer+数据管线）：今明两天完成
- 完整训练流程：本周内可开始实验

