/**
 * @file operators.h
 * @brief Main header file for the operators deep learning framework
 * 
 * This is the primary include file that providess access to all functionality
 * of the operators framework. It includes all core components and providess
 * convenient namespace aliases for common operations.
 * 
 * @version 1.0.0
 * @author Operators Framework Team
 */

#pragma once

#include "core/dtype.h"
#include "core/device.h"
#include "core/tensor.h"
#include "core/ops.h"
#include "core/logger.h"
#include "core/memory_manager.h"

// [Translated comment removed - see documentation]
#include <iostream>

// Memory management system (industrial-grade parameter/activation management)
#ifdef ENABLE_MEMORY_MODULE
#include "memory/mobile_param_manager.h"
#include "memory/mobile_param_optimizations.h"
#include "memory/mobile_specific_optimizations.h"
#include "memory/mobile_zero.h"
#endif

// Activation management system (gradient checkpointing/activation compression)
#ifdef ENABLE_ACTIVATIONS_MODULE
#include "activations/mobile_activation_manager.h"
#include "activations/deepspeed_checkpoint_integration.h"
#endif

namespace ops {

    /**
     * @brief Functional API namespace
     * 
     * This namespace providess a functional interface to common operations,
     * similar to to PyTorch's functional API. It includes frequently used
     * operations like activations, normalizations, and loss functions.
     */
    namespace F {

        // Activation functions
        using ops::relu;
        using ops::gelu;
        using ops::sigmoid;
        using ops::softmax;
        
        // Normalization functions
        using ops::layer_norm;
        
        // Regularization functions
        using ops::dropout;
        
        // Linear algebra operations
        using ops::linear;
        using ops::matmul;
        using ops::transpose;
        using ops::reshape;
        
        // Reduction operations
        using ops::sum;
        using ops::mean;
        
        // Loss functions
        using ops::cross_entropy_loss;
        using ops::mse_loss;
    }
}

#define OPERATORS_VERSION_MAJOR 2
#define OPERATORS_VERSION_MINOR 0
#define OPERATORS_VERSION_PATCH 0

#define OPERATORS_VERSION "2.0.0"

namespace ops {

    // [Translated comment removed - see documentation]
    inline void init() {
        std::cout << "Operators " << OPERATORS_VERSION << " initialized" << std::endl;
        std::cout << "Available devices: ";
        if (DeviceManager::cuda_available()) {
            std::cout << "CUDA ";
        }
        if (DeviceManager::metal_available()) {
            std::cout << "Metal ";
        }
        std::cout << "CPU" << std::endl;
    }

    inline void cleanup() {
                // [Translated]
        LogManager::shutdown();
    }
}

