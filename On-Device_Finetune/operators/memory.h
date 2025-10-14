/**
 * @file memory.h
 * @brief High-Level Memory Management Unified Entry Point
 * 
 * Provides high-level memory management features including parameter management,
 * activation management, and ZeRO optimization
 */

#pragma once

// Parameter memory management
#include "memory/mobile_param_manager.h"
#include "memory/mobile_param_optimizations.h"

// Activation management
#include "memory/activation_checkpointer.h"
#include "memory/mobile_activation_manager.h"

// ZeRO optimization
#include "memory/mobile_zero.h"

// Mobile-specific optimizations
#include "memory/mobile_specific_optimizations.h"

namespace ops {
namespace memory {

// Provide convenient aliases (optional)
// using ParamManager = ops::memory::MobileParamManager;
// using ActivationManager = ops::memory::MobileActivationManager;

} // namespace memory
} // namespace ops

