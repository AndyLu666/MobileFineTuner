/**
 * @file transformer.h
 * @brief Transformer Architecture Unified Entry Point
 * 
 * Provides core components for Transformer models such as GPT-2/Gemma
 */

#pragma once

// Transformer core components
#include "transformer/transformer_block.h"
#include "transformer/kv_cache.h"
#include "transformer/gpt2_components.h"
#include "transformer/autoregressive_ops.h"

// GPT-2 model definition (if available)
#ifdef GPT2_FINETUNE_MODEL_H
#include "transformer/gpt2_finetune_model.h"
#endif

namespace ops {
namespace transformer {

// Provide convenient aliases (optional)
// using TransformerBlock = ops::transformer::TransformerBlock;
// using KVCache = ops::transformer::KVCache;

} // namespace transformer
} // namespace ops

