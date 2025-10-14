#!/bin/bash

# Gemma 270M LoRA fine-tuning build script
echo "Compiling Gemma 270M LoRA fine-tuning program..."

# Check dependencies
if [ ! -d "../operators" ]; then
    echo "Error: operators directory not found, please ensure this script runs in llama.cpp/gemma_finetune/ directory"
    exit 1
fi

# Compile LoRA version - using pure C++ memory-first implementation (BLAS/Accelerate disabled)
g++ -std=c++17 -DUSE_NEW_AUTOGRAD_ENGINE \
    -O3 \
    -DDISABLE_BLAS=ON -DDISABLE_BLAS_COMPLETELY \
    -I../operators \
    -I../operators/core \
    -I../operators/optim \
    -I../operators/utils \
    -I../operators/activations \
    -I../operators/memory \
    gemma_lora_finetune.cpp \
    ../operators/core/tensor.cpp \
    ../operators/core/autograd_engine.cpp \
    ../operators/core/step_arena.cpp \
    ../operators/core/ops.cpp \
    ../operators/core/backward_functions.cpp \
    ../operators/core/memory_manager.cpp \
    ../operators/core/memory_first_attention.cpp \
    ../operators/core/memory_first_mlp.cpp \
    ../operators/core/logger.cpp \
    ../operators/core/utils.cpp \
    ../operators/optim/optimizer.cpp \
    ../operators/optim/adam.cpp \
    ../operators/optim/adam_amp.cpp \
    ../operators/core/tokenizer.cpp \
    ../operators/core/mobile_safe_matmul.cpp \
    ../operators/core/chunked_softmax_ce.cpp \
    ../operators/activations/deepspeed_checkpoint_integration.cpp \
    ../operators/memory/activation_checkpointer.cpp \
    ../operators/memory/mobile_param_manager.cpp \
    ../operators/memory/mobile_param_optimizations.cpp \
    ../operators/memory/mobile_specific_optimizations.cpp \
    ../operators/memory/mobile_zero.cpp \
    -o gemma_lora_finetune

if [ $? -eq 0 ]; then
    echo "Gemma LoRA version compiled successfully!"
    echo "Run with: ./gemma_lora_finetune"
else
    echo "Compilation failed!"
    exit 1
fi
