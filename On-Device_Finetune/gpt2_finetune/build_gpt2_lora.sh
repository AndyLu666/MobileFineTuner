#!/bin/bash

# GPT2 LoRA Fine-tuning Build Script
# Resolves dependencies and interface issues

set -e  # Exit immediately on error

echo "Starting to compile GPT2 LoRA fine-tuning program (BLAS disabled / memory first)..."

# Check current directory
if [ ! -f "gpt2_lora_finetune.cpp" ]; then
    echo "Error: Please run this script in the gpt2_finetune directory"
    exit 1
fi

# First compile operators (BLAS/Accelerate disabled)
echo "Building operators (BLAS disabled)..."
OPERATORS_DIR="$(cd .. && pwd)/operators"
if [ -d "$OPERATORS_DIR" ]; then
    mkdir -p "$OPERATORS_DIR/build"
    pushd "$OPERATORS_DIR/build" >/dev/null
    cmake .. -DDISABLE_BLAS=ON -DCMAKE_BUILD_TYPE=Release
    make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
    OPERATORS_LIB="$OPERATORS_DIR/build/lib/liboperators.a"
    if [ ! -f "$OPERATORS_LIB" ]; then
        echo "Operators static library not found: $OPERATORS_LIB"; exit 1
    fi
    popd >/dev/null
else
    echo "Operators directory not found: $OPERATORS_DIR"; exit 1
fi

# Create build directory
BUILD_DIR="build"
if [ -d "$BUILD_DIR" ]; then
    echo "Cleaning old build directory..."
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "Configuring CMake (BLAS disabled, linking pre-built operators)..."

# Configure CMake, enable all optimization options
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DDISABLE_BLAS=ON \
    -DUSE_NEW_AUTOGRAD_ENGINE=ON \
    -DUSE_MOBILE_OPTIMIZER=ON \
    -DAUTOGRAD_DEBUG=OFF \
    -DOPERATORS_LIBRARY="$OPERATORS_LIB" \
    -DCMAKE_CXX_STANDARD=17

echo "Starting to compile gpt2_finetune..."

# Compile with multi-core acceleration
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

if [ $? -eq 0 ]; then
    echo "Compilation successful!"
    echo "Executable file location: $(pwd)/bin/gpt2_lora_finetune"
    
    # Check if executable file exists
    if [ -f "bin/gpt2_lora_finetune" ]; then
        echo "Test run..."
        ./bin/gpt2_lora_finetune --help 2>/dev/null || echo "Program has been generated, ready to use"
    else
        echo "Warning: Executable file not found in expected location, please check compilation output"
    fi
else
    echo "Compilation failed, please check error messages"
    exit 1
fi

echo "Build complete!"
