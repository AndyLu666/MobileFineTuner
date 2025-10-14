/**
 * @file nn.h
 * @brief Neural Network Layers Unified Entry Point
 * 
 * Provides unified access interface for all neural network layers,
 * similar to PyTorch's torch.nn
 */

#pragma once

// Base module
#include "nn/module.h"

// Basic layers
#include "nn/layers.h"
#include "nn/embedding.h"
#include "nn/attention.h"
#include "nn/mlp.h"

// LoRA support
#include "nn/lora.h"
#include "nn/lora_ops.h"

namespace ops {
namespace nn {

// Provide convenient aliases (optional)
// using Linear = ops::nn::Linear;
// using Embedding = ops::nn::Embedding;
// using Attention = ops::nn::Attention;

} // namespace nn
} // namespace ops

