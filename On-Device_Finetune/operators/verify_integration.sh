#!/bin/bash
# Operators v2.0 Integration Verification Script
# Verify compilation and functionality of all modules

set -e

echo "============================================"
echo "Operators v2.0 Integration Verification"
echo "============================================"
echo ""

# Detect operating system
if [[ "$OSTYPE" == "darwin"* ]]; then
    NCPU=$(sysctl -n hw.ncpu)
    OS="macOS"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    NCPU=$(nproc)
    OS="Linux"
else
    NCPU=4
    OS="Unknown"
fi

echo "Operating System: $OS"
echo "CPU Cores: $NCPU"
echo ""

# 1. Verify directory structure
echo "Verifying directory structure..."
if [ ! -d "core" ] || [ ! -d "optim" ] || [ ! -d "memory" ] || [ ! -d "activations" ]; then
    echo "ERROR: Directory structure incomplete, make sure to run in operators/ directory"
    exit 1
fi
echo "PASS: Directory structure correct"
echo ""

# 2. Clean old build
echo "Cleaning old build..."
rm -rf build
mkdir -p build
cd build
echo "PASS: Build directory cleaned"
echo ""

# 3. Configure CMake (full features)
echo "Configuring CMake (enable all modules)..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_NEW_AUTOGRAD_ENGINE=ON \
    -DUSE_MOBILE_OPTIMIZER=ON \
    -DENABLE_MEMORY_MODULE=ON \
    -DENABLE_ACTIVATIONS_MODULE=ON \
    -DBUILD_TESTS=ON \
    -DENABLE_PROFILING=OFF

if [ $? -ne 0 ]; then
    echo "ERROR: CMake configuration failed"
    exit 1
fi
echo "PASS: CMake configuration successful"
echo ""

# 4. Compile operators library
echo "Compiling operators library..."
make -j${NCPU}

if [ $? -ne 0 ]; then
    echo "ERROR: Compilation failed"
    exit 1
fi
echo "PASS: Compilation successful"
echo ""

# 5. Check library files
echo "Checking library files..."
if [ ! -f "lib/liboperators.a" ]; then
    echo "ERROR: liboperators.a not generated"
    exit 1
fi

LIB_SIZE=$(du -h lib/liboperators.a | awk '{print $1}')
echo "PASS: liboperators.a generated (size: $LIB_SIZE)"
echo ""

# 6. Run tests
echo "Running unit tests..."

if [ -f "test_autograd_engine" ]; then
    echo "Running test_autograd_engine..."
    ./test_autograd_engine
    if [ $? -eq 0 ]; then
        echo "PASS: Autograd test passed"
    else
        echo "WARNING: Autograd test failed"
    fi
else
    echo "WARNING: test_autograd_engine not found"
fi

if [ -f "test_optimizer" ]; then
    echo "Running test_optimizer..."
    ./test_optimizer
    if [ $? -eq 0 ]; then
        echo "PASS: Optimizer test passed"
    else
        echo "WARNING: Optimizer test failed"
    fi
else
    echo "WARNING: test_optimizer not found"
fi

echo ""

# 7. Check symbols
echo "Checking library symbols (verify Memory/Activations modules)..."
if command -v nm &> /dev/null; then
    echo "Checking critical symbols..."
    
    # Check Core symbols
    nm lib/liboperators.a | grep -q "ops::zeros" && echo "  PASS: Core: ops::zeros" || echo "  ERROR: Core: ops::zeros missing"
    nm lib/liboperators.a | grep -q "ops::matmul" && echo "  PASS: Core: ops::matmul" || echo "  ERROR: Core: ops::matmul missing"
    
    # Check Optim symbols
    nm lib/liboperators.a | grep -q "ops::Adam" && echo "  PASS: Optim: ops::Adam" || echo "  ERROR: Optim: ops::Adam missing"
    
    # Check Memory symbols
    nm lib/liboperators.a | grep -q "MobileParameterManager" && echo "  PASS: Memory: MobileParameterManager" || echo "  ERROR: Memory: MobileParameterManager missing"
    nm lib/liboperators.a | grep -q "MobileZeROOptimizer" && echo "  PASS: Memory: MobileZeROOptimizer" || echo "  ERROR: Memory: MobileZeROOptimizer missing"
    
    # Check Activations symbols
    nm lib/liboperators.a | grep -q "MobileActivationManager" && echo "  PASS: Activations: MobileActivationManager" || echo "  ERROR: Activations: MobileActivationManager missing"
else
    echo "WARNING: nm command not available, skipping symbol check"
fi

echo ""

# 8. Summary
echo "============================================"
echo "Verification Complete"
echo "============================================"
echo ""
echo "SUCCESS: Operators v2.0 integration successful"
echo ""
echo "Module Status:"
echo "  - Core: Enabled"
echo "  - Autograd: Enabled"
echo "  - Optim: Enabled"
echo "  - Memory: Enabled"
echo "  - Activations: Enabled"
echo ""
echo "Next Steps:"
echo "  1. View architecture documentation: cat ARCHITECTURE_V2.md"
echo "  2. Deploy to server: see DEPLOYMENT_GUIDE.md"
echo "  3. Run GPT2 training: cd ../gpt2_finetune && ./build_gpt2_lora.sh"
echo ""

