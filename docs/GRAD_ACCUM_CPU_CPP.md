### CPU 纯 C++ 路线：梯度累积（grad_accum）使用说明

本文件聚焦“在 CPU 上、使用本仓库纯 C++ 训练管线”进行 LoRA 微调时的梯度累积（gradient accumulation）实践与指引。适用于以下可执行程序：
- `build/train_lora_gemma`（Gemma 3 文本模型 LoRA 训练）
- `build/gpt2_lora_finetune`（GPT‑2 LoRA 训练）


## 1. 梯度累积的作用与原理

- **目的**：在 CPU/内存受限场景下，无法使用很大的 batch。梯度累积通过把一个“有效大 batch”的一次更新拆成多次“微批（micro-batches）”前/反传，累计梯度后再执行一次优化器更新，从而实现与大 batch 近似的训练稳定性。
- **有效批大小（单机单进程）**：
  - `effective_batch_size = micro_batch_size × grad_accum`
- **实现方式（本仓库已统一为方案 A）**：
  - 方案 A：每个 micro-step 将 `loss` 先除以 `grad_accum` 再反传；累计到 `grad_accum` 次后，执行一次 `clip_grad_norm → optimizer.step → zero_grad`。
  - 说明：方案 B（先直接累计梯度，最后统一除以 `accum`）在数值上等价，但本仓库现已统一到方案 A，便于与主流生态一致，且实现更简洁。


## 2. 本仓库 C++ 实现细节（CPU 路线）

- `operators/finetune_ops/optim/gemma_trainer.cpp`（供 `train_lora_gemma` 使用）
  - 每个 micro-step：计算 loss，按 `1/grad_accum_steps` 缩放后 backward。
  - 当累计次数达到 `grad_accum_steps`：
    - 只在这一刻进行一次梯度剪裁（clip）
    - 根据学习率调度设置 `lr`，执行 `optimizer.step()`
    - 清梯度（`zero_grad`），并将累计计数与累计 loss 复位

- `gpt2_lora_finetune/main.cpp`
  - 在一个有效步内循环 `accum` 次 micro-step：每次将 `loss` 按 `1/accum` 缩放后 backward；累计到整除时执行梯度剪裁、`optimizer.step()` 与 `zero_grad`（与 Gemma 路线一致）。

- 两条路线均保证：
  - 只在“有效更新步”上调用一次 `optimizer.step()` 与一次梯度剪裁
  - 学习率调度随“有效更新步”推进


## 3. 参数与接口（CPU）

### 3.1 Gemma 3 C++ 训练（`build/train_lora_gemma`）
- 常用参数（摘选）：
  - `--model_dir`：权重与 tokenizer 目录（例如 `gemma-3-270m/`）
  - `--data_dir`：WikiText-2 原始数据目录（包含 `wiki.train.raw`、`wiki.valid.raw`）
  - `--output_dir`：输出目录（保存 LoRA 权重等）
  - `--seq_len`：序列长度
  - `--batch`：micro-batch 大小（单次前向/反向的样本数）
  - `--grad_accum`：梯度累积步数
  - `--epochs` / `--max_steps`：训练轮数 / 最多有效更新步数
  - `--lr`、`--warmup_ratio`、`--max_grad_norm`、`--weight_decay`：优化相关
  - `--targets`：LoRA 注入范围（`attn`/`full`/`light`）

- 示例（CPU 上运行，一次有效更新对应 4×8 的有效 batch）：
  ```bash
  /Users/tony/Documents/FT（gemma完成版）/build/train_lora_gemma \
    --model_dir /Users/tony/Documents/FT（gemma完成版）/gemma-3-270m \
    --data_dir  /Users/tony/Documents/FT（gemma完成版）/data/wikitext2/wikitext-2-raw \
    --output_dir /Users/tony/Documents/FT（gemma完成版）/runs/gemma_lora_cpu_cpp \
    --seq_len 128 \
    --batch 4 \
    --grad_accum 8 \
    --epochs 1 \
    --lr 2e-4 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1.0 \
    --targets attn
  # 有效批大小（单进程）= 4 × 8 = 32
  ```

### 3.2 GPT‑2 C++ 训练（`build/gpt2_lora_finetune`）
- 常用参数（摘选）：
  - `--data_dir`、`--pretrained_dir`、`--lora_out`、`--epochs`
  - `--batch_size`、`--grad_accum_steps`、`--seq_len`
  - `--lr`、`--weight_decay`、`--warmup_steps`、`--clip_grad_norm`
  - `--log_interval`、`--eval_interval`、`--save_every`

- 示例（CPU 上运行，一次有效更新对应 1×8 的有效 batch）：
  ```bash
  /Users/tony/Documents/FT（gemma完成版）/build/gpt2_lora_finetune \
    --data_dir /Users/tony/Documents/FT（gemma完成版）/data/wikitext2/wikitext-2-raw \
    --pretrained_dir /Users/tony/Documents/FT（gemma完成版）/gpt2_lora_finetune/pretrained/gpt2 \
    --lora_out /Users/tony/Documents/FT（gemma完成版）/runs/gpt2_lora_cpu_cpp/gpt2_lora.safetensors \
    --epochs 1 \
    --batch_size 1 \
    --grad_accum_steps 8 \
    --seq_len 128 \
    --lr 1e-4 \
    --clip_grad_norm 1.0 \
    --log_interval 1
  # 有效批大小（单进程）= 1 × 8 = 8
  ```


## 4. 学习率、剪裁与日志

- **学习率调度**：两条路线均只在“有效更新步”推进调度。提高 `grad_accum` 会减少每个 epoch 的有效更新次数，进而影响 warmup 与衰减的相对位置；如需“保持总更新步数不变”，请相应调整 `epochs` 或 `max_steps`。
- **梯度剪裁**：只在有效更新步进行一次剪裁，位置在 `optimizer.step()` 之前；这能稳定训练且避免对每个 micro-step 重复剪裁的开销。
- **日志**：
  - `train_lora_gemma`：每 `logging_steps` 个有效步打印一次（默认 1）。
  - `gpt2_lora_finetune`：每 `--log_interval` 个有效步打印一次，并可开启周期评估/保存检查点。


## 5. 在 CPU 上的实用建议

- **吞吐与稳定性折中**：在 CPU 上，`batch` 建议尽量小以避免内存紧张，再用较大的 `grad_accum` 达到目标“有效批”。
- **总步数与时间**：`grad_accum` 越大，每个有效更新需要的 micro-step 越多，单步时间越长属正常。若你希望“总有效步数”不变但缩短墙钟时间，可降低 `seq_len` 或减少 `epochs` 进行快速验证。
- **学习率调参**：经验上可近似随“有效批大小”线性放缩学习率，但最佳值依任务/数据而定。建议以保守学习率启动，观察 loss/PPL 后再微调。


## 6. 参考入口与源码位置

- 可执行程序：
  - `build/train_lora_gemma`
  - `build/gpt2_lora_finetune`
- 关键源码：
  - Gemma 训练配置与循环：`operators/finetune_ops/optim/train_lora_gemma.cpp`
  - Gemma 梯度累积细节：`operators/finetune_ops/optim/gemma_trainer.cpp`
  - GPT‑2 梯度累积细节：`gpt2_lora_finetune/main.cpp`




