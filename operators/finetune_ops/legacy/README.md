# Legacy NN Module Files

本目录下保留了若干早期 NN 实现（如 `lora.h/cpp`、`lora_ops.h/cpp`、`attention.h/cpp`、`mlp.h/cpp`、`module.h`、`layers.h` 等），这些文件仍依赖旧版 Tensor 接口以及不存在的 `BaseLayer`/`LayerConfig`/`GradTensorPtr` 等类型，仅供参考。若直接参与构建会因为接口不匹配而编译失败。

## 当前状态
- ✅ `nn/lora_linear.h/cpp`: 已迁移到现代 TensorPtr/ops 接口，正常使用（仍位于 `finetune_ops/nn/` 内）
- ✅ `nn/embedding.h/cpp`: 已迁移到现代接口，继续保留在 `finetune_ops/nn/`
- ⚠️ `legacy/nn/lora.h/cpp`: 依赖已移除的 BaseLayer/LayerConfig
- ⚠️ `legacy/nn/lora_ops.h/cpp`: 使用旧式 `.data/.shape` Tensor API
- ⚠️ `legacy/nn/attention.h/cpp`、`legacy/nn/mlp.h/cpp`、`legacy/nn/module.h`、`legacy/nn/layers.h`: 皆为旧接口示例
- ⚠️ `legacy/gpt2_finetune_model.h`: 依赖 GradTensorPtr/Tokenizer 等旧接口

## 构建配置
这些遗留文件已从 CMakeLists.txt 的 FINETUNE_SOURCES 中排除，不会参与编译。

## 现行替代方案
- LoRA 功能：使用 `nn/lora_linear.h` + `graph/lora_injector.h`
- Attention：在 `graph/gpt2_model.cpp` 中内联实现
- Embedding：使用 `nn/embedding.h`（已迁移）或 gpt2_model 的 embedding_lookup

## 如需启用旧文件
需要先迁移到现代 Tensor/ops 接口或实现缺失的基类框架。
