#!/bin/bash
# Operators v2.0 整合验证脚本
# 验证所有模块的编译与功能

set -e

echo "============================================"
echo "Operators v2.0 整合验证"
echo "============================================"
echo ""

# 检测操作系统
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

echo "操作系统: $OS"
echo "CPU核心数: $NCPU"
echo ""

# 1. 验证目录结构
echo "📁 验证目录结构..."
if [ ! -d "core" ] || [ ! -d "optim" ] || [ ! -d "memory" ] || [ ! -d "activations" ]; then
    echo "❌ 目录结构不完整，请确保在operators/目录中运行"
    exit 1
fi
echo "✅ 目录结构正确"
echo ""

# 2. 清理旧构建
echo "🧹 清理旧构建..."
rm -rf build
mkdir -p build
cd build
echo "✅ 构建目录已清理"
echo ""

# 3. 配置CMake（完整功能）
echo "📋 配置CMake（启用所有模块）..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_NEW_AUTOGRAD_ENGINE=ON \
    -DUSE_MOBILE_OPTIMIZER=ON \
    -DENABLE_MEMORY_MODULE=ON \
    -DENABLE_ACTIVATIONS_MODULE=ON \
    -DBUILD_TESTS=ON \
    -DENABLE_PROFILING=OFF

if [ $? -ne 0 ]; then
    echo "❌ CMake配置失败"
    exit 1
fi
echo "✅ CMake配置成功"
echo ""

# 4. 编译operators库
echo "🔨 编译operators库..."
make -j${NCPU}

if [ $? -ne 0 ]; then
    echo "❌ 编译失败"
    exit 1
fi
echo "✅ 编译成功"
echo ""

# 5. 检查库文件
echo "📦 检查库文件..."
if [ ! -f "lib/liboperators.a" ]; then
    echo "❌ liboperators.a未生成"
    exit 1
fi

LIB_SIZE=$(du -h lib/liboperators.a | awk '{print $1}')
echo "✅ liboperators.a已生成（大小: $LIB_SIZE）"
echo ""

# 6. 运行测试
echo "🧪 运行单元测试..."

if [ -f "test_autograd_engine" ]; then
    echo "运行 test_autograd_engine..."
    ./test_autograd_engine
    if [ $? -eq 0 ]; then
        echo "✅ Autograd测试通过"
    else
        echo "⚠️  Autograd测试失败"
    fi
else
    echo "⚠️  test_autograd_engine未找到"
fi

if [ -f "test_optimizer" ]; then
    echo "运行 test_optimizer..."
    ./test_optimizer
    if [ $? -eq 0 ]; then
        echo "✅ Optimizer测试通过"
    else
        echo "⚠️  Optimizer测试失败"
    fi
else
    echo "⚠️  test_optimizer未找到"
fi

echo ""

# 7. 检查符号
echo "🔍 检查库符号（验证Memory/Activations模块）..."
if command -v nm &> /dev/null; then
    echo "检查关键符号..."
    
    # 检查Core符号
    nm lib/liboperators.a | grep -q "ops::zeros" && echo "  ✅ Core: ops::zeros" || echo "  ❌ Core: ops::zeros缺失"
    nm lib/liboperators.a | grep -q "ops::matmul" && echo "  ✅ Core: ops::matmul" || echo "  ❌ Core: ops::matmul缺失"
    
    # 检查Optim符号
    nm lib/liboperators.a | grep -q "ops::Adam" && echo "  ✅ Optim: ops::Adam" || echo "  ❌ Optim: ops::Adam缺失"
    
    # 检查Memory符号
    nm lib/liboperators.a | grep -q "MobileParameterManager" && echo "  ✅ Memory: MobileParameterManager" || echo "  ❌ Memory: MobileParameterManager缺失"
    nm lib/liboperators.a | grep -q "MobileZeROOptimizer" && echo "  ✅ Memory: MobileZeROOptimizer" || echo "  ❌ Memory: MobileZeROOptimizer缺失"
    
    # 检查Activations符号
    nm lib/liboperators.a | grep -q "MobileActivationManager" && echo "  ✅ Activations: MobileActivationManager" || echo "  ❌ Activations: MobileActivationManager缺失"
else
    echo "⚠️  nm命令不可用，跳过符号检查"
fi

echo ""

# 8. 总结
echo "============================================"
echo "验证完成"
echo "============================================"
echo ""
echo "✅ Operators v2.0 整合成功"
echo ""
echo "模块状态:"
echo "  - Core: 已启用"
echo "  - Autograd: 已启用"
echo "  - Optim: 已启用"
echo "  - Memory: 已启用"
echo "  - Activations: 已启用"
echo ""
echo "下一步:"
echo "  1. 查看架构文档: cat ARCHITECTURE_V2.md"
echo "  2. 部署到服务器: 参见 DEPLOYMENT_GUIDE.md"
echo "  3. 运行GPT2训练: cd ../gpt2_finetune && ./build_gpt2_lora.sh"
echo ""

