# GPT-2 LoRA 微调：头号三件事落地进度

## ✅ 已完成（可立即验证）

### 1. 权重 Getter + LoRA 注入绑定

**文件**：`operators/finetune_ops/graph/gpt2_model.{h,cpp}`, `lora_injector.{h,cpp}`

**功能**：
- GPT2Model 新增 4 个 getter：`attn_qkv_params(i)`, `attn_proj_params(i)`, `mlp_fc_in_params(i)`, `mlp_fc_out_params(i)`
- LoraInjector 注入时绑定实际权重指针（支持 split_qkv 列偏移）
- merge/unmerge 仅对目标列范围操作（子矩阵加减 ΔW）
- 注入后自动冻结 base 权重（仅 LoRA A/B 参与训练）

**验证**：
```bash
# 编译后运行 test_lora_correctness 即可验证
```

### 2. 前向对齐 + LoRA 正确性测试（CI 级别）

**文件**：
- `operators/finetune_ops/graph/save_pt_gold.py`：生成 PyTorch 金标 logits
- `operators/finetune_ops/graph/test_gpt2_forward.cpp`：C++ 前向对齐测试
- `operators/finetune_ops/graph/test_lora_correctness.cpp`：LoRA 零影响 + merge/unmerge 幂等性测试

**验收标准**：
- 前向对齐：`max_abs_err ≤ 1e-4`，`argmax(cpp) == argmax(pt)`
- LoRA 零影响：注入后 B=0 → 输出与未注入一致（`err < 1e-6`）
- merge 幂等：连续 merge 两次 → 权重不变（`err < 1e-8`）
- unmerge 幂等：连续 unmerge 两次 → 权重不变（`err < 1e-8`）
- unmerge 还原：`unmerge()` 后回到 W0（`err < 1e-6`）

**运行方式**：
```bash
# 1. 生成 PyTorch 金标
python3 operators/finetune_ops/graph/save_pt_gold.py

# 2. 编译并运行测试
cd operators/build
make test_gpt2_forward
./test_gpt2_forward

make test_lora_correctness
./test_lora_correctness
```

## 📋 待完成（头号三件事剩余部分）

### 3. Trainer 最小闭环

**目标**：能在 WikiText-2 上跑 10 步，loss 有效下降

**组件**：
- AdamW(decoupled)：betas=(0.9,0.999), eps=1e-8, weight_decay=0.01
- 梯度裁剪：`clip_grad_norm=1.0`
- LR Scheduler：cosine 或 linear + warmup_ratio=0.03
- 梯度累积：实现"有效 batch"
- 训练循环：`zero_grad → forward → CE(ignore_index=-100) → backward → clip → (accumulate) → optimizer.step → scheduler.step`
- 日志：每步打印 loss、lr、grad_norm、吞吐

**文件**（待创建）：
- `operators/finetune_ops/optim/trainer.{h,cpp}`
- `operators/finetune_ops/optim/train_lora_gpt2.cpp`（训练入口）

### 4. LoRA safetensors save/load（PEFT 兼容）

**目标**：存取 LoRA 权重，并与 PyTorch PEFT 互通

**键名规范**（PEFT 对齐）：
```
lora.blocks.{i}.attn.q.A   [C, r]
lora.blocks.{i}.attn.q.B   [r, C]
lora.blocks.{i}.attn.k.A
lora.blocks.{i}.attn.k.B
lora.blocks.{i}.attn.v.A
lora.blocks.{i}.attn.v.B
lora.blocks.{i}.attn.proj.A
lora.blocks.{i}.attn.proj.B
lora.blocks.{i}.mlp.fc_in.A
lora.blocks.{i}.mlp.fc_in.B
lora.blocks.{i}.mlp.fc_out.A
lora.blocks.{i}.mlp.fc_out.B
meta.rank (scalar)
meta.alpha
meta.dropout
meta.split_qkv (0/1)
```

**验证**：
- C++ 保存 LoRA → Python 加载 → HF+PEFT 前向 → logits 一致（`max_abs_err < 1e-4`）

**文件**（待补充实现）：
- `operators/finetune_ops/graph/lora_injector.cpp`：补充 `save_lora_safetensors` 与 `load_lora_safetensors`
- `operators/finetune_ops/graph/test_lora_peft_compat.py`：Python 端验证脚本

## 🎯 第二梯队（稳态训练 & 评测质量）

### 5. WikiText-2 数据管线

**目标**：拼接切块，labels 的 PAD=-100

**功能**：
- 样本间插入 `<|endoftext|>`
- 拼接切块（按 `max_seq_length`）或滑窗切块（`stride=seq_len-overlap`）
- attention_mask 与 causal mask 对齐
- labels 中 PAD 位置设为 -100

**文件**（待创建）：
- `operators/finetune_ops/data/wikitext2_dataset.{h,cpp}`

### 6. MMLU 评测器

**目标**：长度归一的条件对数似然评分

**功能**：
- 每个选项算 `log p(option | prompt)`，按 token 数归一
- argmax 计分
- 固化模板与随机种子
- 输出 per-subject 与 overall 准确率

**文件**（待创建）：
- `operators/finetune_ops/data/mmlu_dataset.{h,cpp}`
- `operators/finetune_ops/eval/mmlu_evaluator.{h,cpp}`

## 📊 当前状态

| 组件 | 状态 | 文件 | 验证方式 |
|------|------|------|----------|
| Tokenizer (BPE) | ✅ 完成 | `tokenizer_bpe.{h,cpp}` | 单测 PASS |
| SafeTensors Loader | ✅ 完成 | `safetensors_loader.{h,cpp}` | 解析 161 张量 |
| GPT2Model 外壳 | ✅ 完成 | `gpt2_model.{h,cpp}` | 前向对齐测试 |
| LoRA 注入器 | ✅ 完成 | `lora_injector.{h,cpp}` | 零影响+幂等性测试 |
| 前向对齐测试 | ✅ 完成 | `test_gpt2_forward.cpp` | 待运行 |
| LoRA 正确性测试 | ✅ 完成 | `test_lora_correctness.cpp` | 待运行 |
| Trainer 最小闭环 | 🚧 待做 | `trainer.{h,cpp}` | - |
| LoRA save/load | 🚧 待做 | `lora_injector.cpp` | - |
| WikiText-2 管线 | 🚧 待做 | `wikitext2_dataset.{h,cpp}` | - |
| MMLU 评测器 | 🚧 待做 | `mmlu_evaluator.{h,cpp}` | - |

## 🚀 下一步立即行动

1. **编译并运行前向对齐测试**（生成 PyTorch 金标 → 运行 C++ 测试）
2. **编译并运行 LoRA 正确性测试**（零影响 + 幂等性）
3. **实现 Trainer 最小闭环**（AdamW + clip + scheduler + 梯度累积）
4. **实现 LoRA safetensors save/load**（PEFT 兼容键名）

完成这 4 步后，即可开始有意义的实验并与社区工具链互通。

