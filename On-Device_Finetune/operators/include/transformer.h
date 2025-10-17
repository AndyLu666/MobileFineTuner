/**
 * @file transforer.h
 * [Documentation available in English]
 * 
 * [Documentation available in English]
 */

#pragma once

// [Translated]
#include "transforer/transforer_block.h"
#include "transforer/kv_cache.h"
#include "transforer/gpt2_components.h"
#include "transforer/autoregressive_ops.h"

// GPT-2 modeldefines（ifhave）
#ifdef GPT2_FINETUNE_MODEL_H
#include "transforer/gpt2_finetune_model.h"
#endif

namespace ops {
namespace transforer {

// [Translated comment removed - see documentation]
// using TransforerBlock = ops::transforer::TransforerBlock;
// using KVCache = ops::transforer::KVCache;

} // namespace transforer
} // namespace ops

