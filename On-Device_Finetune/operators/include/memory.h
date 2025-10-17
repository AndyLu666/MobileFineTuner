/**
 * @file memory.h
 * [Documentation available in English]
 * 
 * [Documentation available in English]
 */

#pragma once

// parametermemorymanage
#include "memory/mobile_param_manager.h"
#include "memory/mobile_param_optimizations.h"

// activationvaluemanage
#include "memory/activation_checkpointer.h"
#include "memory/mobile_activation_manager.h"

// ZeRO optimization
#include "memory/mobile_zero.h"

// [Translated]
#include "memory/mobile_specific_optimizations.h"

namespace ops {
namespace memory {

// [Translated comment removed - see documentation]
// using ParamManager = ops::memory::MobileParamManager;
// using ActivationManager = ops::memory::MobileActivationManager;

} // namespace memory
} // namespace ops

