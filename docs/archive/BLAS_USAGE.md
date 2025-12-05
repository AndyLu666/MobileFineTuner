# BLAS 加速使用指南

## 功能说明

BLAS（Basic Linear Algebra Subprograms）用于加速矩阵乘法运算，通常可以提升 2-5 倍性能。

## 开关控制

### 1. 启用 BLAS（推荐）

**方法 A：CMake 参数（推荐）**
```bash
cd /path/to/project
rm -rf operators/build
cmake -S operators -B operators/build -DBUILD_TESTS=OFF -DUSE_BLAS=ON
cmake --build operators/build --target gpt2_lora_finetune
```

**方法 B：环境变量**
```bash
export OPS_USE_BLAS=ON
cd /path/to/project
rm -rf operators/build
cmake -S operators -B operators/build -DBUILD_TESTS=OFF
cmake --build operators/build --target gpt2_lora_finetune
```

### 2. 禁用 BLAS（纯 C++）

**方法 A：CMake 参数**
```bash
cd /path/to/project
rm -rf operators/build
cmake -S operators -B operators/build -DBUILD_TESTS=OFF -DUSE_BLAS=OFF
cmake --build operators/build --target gpt2_lora_finetune
```

**方法 B：环境变量**
```bash
export OPS_USE_BLAS=OFF
cd /path/to/project
rm -rf operators/build
cmake -S operators -B operators/build -DBUILD_TESTS=OFF
cmake --build operators/build --target gpt2_lora_finetune
```

## 平台差异

### macOS（推荐使用 BLAS）
- 使用系统自带的 **Accelerate Framework**
- 在 Apple Silicon (M1/M2/M3) 上性能优化最佳
- 无需额外安装依赖

### Linux
- 需要安装 OpenBLAS 或 MKL：
  ```bash
  # Ubuntu/Debian
  sudo apt-get install libopenblas-dev
  
  # CentOS/RHEL
  sudo yum install openblas-devel
  ```
- 安装后重新编译即可

## 验证 BLAS 是否启用

### 编译时检查
```bash
cmake -S operators -B operators/build -DBUILD_TESTS=OFF -DUSE_BLAS=ON
```
输出应包含：
- `🚀 BLAS已启用`
- macOS: `✅ 已链接 Accelerate Framework`
- Linux: `✅ 已链接 BLAS: /path/to/blas`

### 运行时检查
训练时观察每步耗时：
- **启用 BLAS**：每步约 1-3 秒
- **纯 C++**：每步约 5-10 秒

## 性能对比（预期）

| 配置 | seq_len | grad_accum | 每步耗时 | 总训练时间（2 epochs） |
|------|---------|------------|----------|----------------------|
| 纯 C++ | 64 | 8 | 5-10秒 | 25-50小时 |
| BLAS | 64 | 8 | 1-3秒 | 5-15小时 |
| 纯 C++ | 128 | 8 | 20-50秒 | 50-125小时 |
| BLAS | 128 | 8 | 5-15秒 | 12-35小时 |

## 常见问题

### Q: 为什么启用 BLAS 后没有加速？
A: 检查：
1. 编译输出是否显示 "BLAS已启用"
2. 是否真的重新编译了（`rm -rf operators/build`）
3. macOS 是否有 Xcode Command Line Tools
4. Linux 是否安装了 OpenBLAS

### Q: BLAS 会影响训练结果吗？
A: 不会。BLAS 只是优化的矩阵乘法实现，数学结果与纯 C++ 相同（浮点误差在 1e-6 以内）。

### Q: 可以在训练中途切换吗？
A: 不可以。需要重新编译可执行文件。建议：
- 开发/调试：禁用 BLAS（编译更快）
- 正式训练：启用 BLAS（运行更快）

## 推荐配置

### 本地 Mac 开发
```bash
# 启用 BLAS + 小参数快速测试
cmake -S operators -B operators/build -DBUILD_TESTS=OFF -DUSE_BLAS=ON
cmake --build operators/build --target gpt2_lora_finetune

cd operators/build
./gpt2_lora_finetune \
  --epochs 2 \
  --batch_size 1 \
  --grad_accum_steps 8 \
  --seq_len 64 \
  --rank 8 \
  --alpha 16 \
  --lr 5e-5 \
  --weight_decay 0.0 \
  --warmup_steps 60 \
  --clip_grad_norm 1.0 \
  --lora_dropout 0.0 \
  --log_interval 10 \
  --eval_interval 500 \
  --eval_batches 50 \
  --eval_batch_size 2 \
  --save_every 1000 \
  --ema_beta 0.9 \
  --seed 42 \
  --eval_out eval_log.jsonl \
  --lora_out checkpoints/wt2_r8a16_e2_blas.safetensors
```

### 服务器训练
```bash
# 1. 先检查是否有 OpenBLAS
ldconfig -p | grep blas

# 2. 如果没有，安装
sudo apt-get install libopenblas-dev

# 3. 启用 BLAS 编译
cd /home/LYT0830-dku/重新开始
rm -rf operators/build
cmake -S operators -B operators/build -DBUILD_TESTS=OFF -DUSE_BLAS=ON
cmake --build operators/build --target gpt2_lora_finetune -- -j$(nproc)

# 4. 运行训练
cd operators/build
./gpt2_lora_finetune --epochs 2 ... # 使用正式参数
```

## 总结

- **推荐启用 BLAS**：显著加速训练，无副作用
- **切换方法**：通过 `-DUSE_BLAS=ON/OFF` 控制
- **验证方法**：观察编译输出和训练速度

