cmake_minimum_required(VERSION 3.10)
project(Operators VERSION 2.0.0 LANGUAGES CXX)

# ============================================================================
# 工业级配置选项
# ============================================================================
option(USE_NEW_AUTOGRAD_ENGINE "使用拓扑排序自动微分引擎" ON)
option(AUTOGRAD_DEBUG "启用自动微分调试输出" OFF)
option(USE_MOBILE_OPTIMIZER "启用移动端优化器扩展" ON)
option(ENABLE_MEMORY_MODULE "启用Memory模块（参数管理/Zero优化）" ON)
option(ENABLE_ACTIVATIONS_MODULE "启用Activations模块（梯度检查点/激活管理）" ON)
option(USE_ARENA_ALLOCATOR "使用Arena分区内存管理（根治物理足迹增长）" OFF)
option(USE_BLAS "启用BLAS加速（ON=Accelerate/OpenBLAS，OFF=纯C++）" OFF)
option(DISABLE_MEMORY_POOL "禁用自定义内存池，Tensor直接使用系统malloc/free" OFF)
# 允许通过环境变量覆盖（例如: OPS_USE_BLAS=ON/OFF）
if(DEFINED ENV{OPS_USE_BLAS})
    string(TOUPPER "$ENV{OPS_USE_BLAS}" _OPS_USE_BLAS)
    if(_OPS_USE_BLAS STREQUAL "ON" OR _OPS_USE_BLAS STREQUAL "1" OR _OPS_USE_BLAS STREQUAL "TRUE")
        set(USE_BLAS ON CACHE BOOL "Enable BLAS via env" FORCE)
    elseif(_OPS_USE_BLAS STREQUAL "OFF" OR _OPS_USE_BLAS STREQUAL "0" OR _OPS_USE_BLAS STREQUAL "FALSE")
        set(USE_BLAS OFF CACHE BOOL "Disable BLAS via env" FORCE)
    endif()
endif()
option(BUILD_TESTS "构建测试程序" ON)
option(BUILD_EXAMPLES "构建示例程序" OFF)
option(ENABLE_PROFILING "启用性能分析" OFF)
option(ENABLE_GPT2_DEBUG_PRINT "启用GPT-2前向详细调试打印" OFF)

# ============================================================================
# C++ 标准
# ============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============================================================================
# 编译选项
# ============================================================================
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O2")

if(AUTOGRAD_DEBUG)
    add_definitions(-DAUTOGRAD_DEBUG)
    message(STATUS "自动微分调试输出已启用")
endif()

if(USE_NEW_AUTOGRAD_ENGINE)
    add_definitions(-DUSE_NEW_AUTOGRAD_ENGINE)
    message(STATUS "新自动微分引擎已启用（拓扑排序）")
endif()

if(USE_ARENA_ALLOCATOR)
    add_definitions(-DUSE_ARENA_ALLOCATOR)
    message(STATUS "🔧 Arena分区内存管理已启用（根治物理足迹增长）")
endif()

if(USE_BLAS)
    message(STATUS "🚀 BLAS已启用")
else()
    add_definitions(-DDISABLE_BLAS_COMPLETELY)
    message(STATUS "✅ BLAS已禁用（使用纯C++ memory-first内核）")
endif()

if(DISABLE_MEMORY_POOL)
    add_definitions(-DDISABLE_MEMORY_POOL)
    message(STATUS "⚠️ 已禁用自定义内存池（Tensor使用系统malloc/free，用于诊断RSS增长）")
endif()

# 控制 GPT-2 调试打印
if(ENABLE_GPT2_DEBUG_PRINT)
    add_definitions(-DENABLE_GPT2_DEBUG_PRINT)
    message(STATUS "GPT-2 前向调试打印: ON")
else()
    message(STATUS "GPT-2 前向调试打印: OFF")
endif()

if(ENABLE_PROFILING)
    add_definitions(-DENABLE_PROFILING)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
    message(STATUS "性能分析已启用")
endif()

# ============================================================================
# 包含目录（新增 finetune_ops 与 opt_ops 根目录）
# ============================================================================
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/finetune_ops
    ${CMAKE_CURRENT_SOURCE_DIR}/opt_ops
    ${CMAKE_CURRENT_SOURCE_DIR}/opt_ops/core_ext
    ${CMAKE_CURRENT_SOURCE_DIR}/opt_ops/parameter
)

# =========================================================================
# 源文件列表
# =========================================================================

set(FINETUNE_SOURCES
    # 核心系统文件
    finetune_ops/core/tensor.cpp
    finetune_ops/core/ops.cpp
    finetune_ops/core/autograd_engine.cpp
    finetune_ops/core/backward_functions.cpp
    finetune_ops/core/logger.cpp
    finetune_ops/core/memory_manager.cpp
    finetune_ops/core/step_arena.cpp
    finetune_ops/core/utils.cpp
    finetune_ops/core/lm_loss.cpp
    finetune_ops/core/memory_efficient_attention.cpp
    finetune_ops/core/performance_monitor.cpp
    
    # GPT-2 微调核心文件
    finetune_ops/core/tokenizer_bpe.cpp
    finetune_ops/graph/gemma_model.cpp
    finetune_ops/graph/gemma_lora_injector.cpp
    finetune_ops/core/tokenizer_gemma.cpp
    finetune_ops/graph/safetensors_loader.cpp
    finetune_ops/graph/gpt2_model.cpp
    finetune_ops/graph/lora_injector.cpp
    
    # B 阶段：数据加载与训练
    finetune_ops/data/wikitext2_dataset.cpp
    finetune_ops/nn/lora_linear.cpp
    finetune_ops/optim/optimizer.cpp
    finetune_ops/optim/adam.cpp
    finetune_ops/optim/trainer.cpp
    
    # C 阶段：LoRA存取
    finetune_ops/graph/lora_saver.cpp
    finetune_ops/optim/gemma_trainer.cpp
)

# OPT_SOURCES暂时完全禁用（A/B阶段不需要，避免依赖问题）
set(OPT_SOURCES
    # 所有opt_ops文件暂时禁用
)

# =========================================================================
# 创建静态库
# =========================================================================
add_library(operators STATIC
    ${FINETUNE_SOURCES}
    ${OPT_SOURCES}
)

# =============================================================================
# 模块化目标（finetune_ops / opt_ops）
# =============================================================================
# 采用新目录结构（不再引用 modules 子目录）

# ============================================================================
# 链接库检测（Accelerate/BLAS）
# ============================================================================
if(USE_BLAS)
    if(APPLE)
        # macOS: 使用 Accelerate Framework
        find_library(ACCELERATE_LIBRARY Accelerate)
        if(ACCELERATE_LIBRARY)
            target_link_libraries(operators PUBLIC ${ACCELERATE_LIBRARY})
            add_definitions(-DUSE_BLAS)
            message(STATUS "✅ 已链接 Accelerate Framework: ${ACCELERATE_LIBRARY}")
        else()
            message(WARNING "未找到 Accelerate，回退到纯C++实现")
            add_definitions(-DDISABLE_BLAS_COMPLETELY)
        endif()
    else()
        # Linux/Windows: 尝试 OpenBLAS 或 MKL
        find_package(BLAS)
        if(BLAS_FOUND)
            target_link_libraries(operators PUBLIC ${BLAS_LIBRARIES})
            add_definitions(-DUSE_BLAS)
            message(STATUS "✅ 已链接 BLAS: ${BLAS_LIBRARIES}")
        else()
            message(WARNING "未找到 BLAS，回退到纯C++实现")
            add_definitions(-DDISABLE_BLAS_COMPLETELY)
        endif()
    endif()
else()
    message(STATUS "🔧 BLAS已禁用，使用纯C++实现")
endif()

# ============================================================================
# 线程支持
# ============================================================================
find_package(Threads REQUIRED)
target_link_libraries(operators PUBLIC Threads::Threads)

# ============================================================================
# 输出配置
# ============================================================================
set_target_properties(operators PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    VERSION ${PROJECT_VERSION}
)

# ============================================================================
# 安装规则
# ============================================================================
install(TARGETS operators
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
    DESTINATION include/operators
    FILES_MATCHING PATTERN "*.h"
)

# =============================================================================
# 模块化目标（finetune_ops / opt_ops）
# =============================================================================
# 采用新目录结构（不再引用 modules 子目录）

# ============================================================================
# 测试目标
# ============================================================================
if(BUILD_TESTS)
    enable_testing()
    
    # 自动微分引擎测试
    if(USE_NEW_AUTOGRAD_ENGINE)
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../tests/test_autograd_engine.cpp)
            add_executable(test_autograd_engine
                ../tests/test_autograd_engine.cpp
            )
            target_link_libraries(test_autograd_engine operators)
            add_test(NAME AutogradEngine COMMAND test_autograd_engine)
        endif()
    endif()
    
    # 优化器测试
    if(USE_MOBILE_OPTIMIZER)
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/optim/comprehensive_optimizer_test.cpp)
            add_executable(test_optimizer
                optim/comprehensive_optimizer_test.cpp
            )
            target_link_libraries(test_optimizer operators)
            add_test(NAME Optimizer COMMAND test_optimizer)
        endif()
    endif()
    
    # Memory模块测试
    if(ENABLE_MEMORY_MODULE)
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_memory_module.cpp)
            add_executable(test_memory_module
                tests/test_memory_module.cpp
            )
            target_link_libraries(test_memory_module operators)
            add_test(NAME MemoryModule COMMAND test_memory_module)
        endif()
    endif()
    
    # Activations模块测试
    if(ENABLE_ACTIVATIONS_MODULE)
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_activations_module.cpp)
            add_executable(test_activations_module
                tests/test_activations_module.cpp
            )
            target_link_libraries(test_activations_module operators)
            add_test(NAME ActivationsModule COMMAND test_activations_module)
        endif()
    endif()
    
    # === B阶段测试程序 ===
    
    # 最小训练闭环
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/finetune_ops/optim/test_train_minimal.cpp)
        add_executable(test_train_minimal
            finetune_ops/optim/test_train_minimal.cpp
        )
        target_link_libraries(test_train_minimal operators)
    endif()
    
    # Backward梯度测试
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/finetune_ops/optim/test_backward_sanity.cpp)
        add_executable(test_backward_sanity
            finetune_ops/optim/test_backward_sanity.cpp
        )
        target_link_libraries(test_backward_sanity operators)
    endif()
    
    # 简单梯度测试
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/finetune_ops/optim/test_simple_grad.cpp)
        add_executable(test_simple_grad
            finetune_ops/optim/test_simple_grad.cpp
        )
        target_link_libraries(test_simple_grad operators)
    endif()
    
    # QKT * softmax backward unit test
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/finetune_ops/optim/test_qkt_softmax_grad.cpp)
        add_executable(test_qkt_softmax_grad
            finetune_ops/optim/test_qkt_softmax_grad.cpp
        )
        target_link_libraries(test_qkt_softmax_grad operators)
    endif()
    
    # RMSNorm forward+backward small-dimension unit test (T4)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/finetune_ops/optim/test_rmsnorm_unit.cpp)
        add_executable(test_rmsnorm_unit
            finetune_ops/optim/test_rmsnorm_unit.cpp
        )
        target_link_libraries(test_rmsnorm_unit operators)
        add_test(NAME RMSNormUnit COMMAND test_rmsnorm_unit)
    endif()
    
    # repeat_kv + softmax backward unit test
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/finetune_ops/optim/test_repeat_kv_softmax_grad.cpp)
        add_executable(test_repeat_kv_softmax_grad
            finetune_ops/optim/test_repeat_kv_softmax_grad.cpp
        )
        target_link_libraries(test_repeat_kv_softmax_grad operators)
    endif()
    
    # T3: single-layer attention-only backward microtest
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/finetune_ops/optim/test_attention_single_layer_backward.cpp)
        add_executable(test_attention_single_layer_backward
            finetune_ops/optim/test_attention_single_layer_backward.cpp
        )
        target_link_libraries(test_attention_single_layer_backward operators)
    endif()
    
    # LoRALinear梯度测试
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/finetune_ops/optim/test_lora_grad.cpp)
        add_executable(test_lora_grad
            finetune_ops/optim/test_lora_grad.cpp
        )
        target_link_libraries(test_lora_grad operators)
    endif()

    # 独立 LoRALinear 前向/反向数值对齐测试
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/finetune_ops/nn/test_lora_linear.cpp)
        add_executable(test_lora_linear
            finetune_ops/nn/test_lora_linear.cpp
        )
        target_link_libraries(test_lora_linear operators)
        add_test(NAME LoRALinearUnit COMMAND test_lora_linear)
    endif()
    
    # Cross-Entropy梯度测试
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/finetune_ops/optim/test_ce_grad.cpp)
        add_executable(test_ce_grad
            finetune_ops/optim/test_ce_grad.cpp
        )
        target_link_libraries(test_ce_grad operators)
    endif()
    
    # 10步训练收敛验证
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/finetune_ops/optim/test_10step_convergence.cpp)
        add_executable(test_10step_convergence
            finetune_ops/optim/test_10step_convergence.cpp
        )
        target_link_libraries(test_10step_convergence operators)
    endif()
    
    # LoRA Round-Trip测试
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/finetune_ops/graph/test_lora_roundtrip.cpp)
        add_executable(test_lora_roundtrip
            finetune_ops/graph/test_lora_roundtrip.cpp
        )
        target_link_libraries(test_lora_roundtrip operators)
    endif()

    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/finetune_ops/graph/test_gemma_config.cpp)
        add_executable(test_gemma_config
            finetune_ops/graph/test_gemma_config.cpp
        )
        target_link_libraries(test_gemma_config operators)
    endif()

    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/finetune_ops/graph/test_gemma_forward.cpp)
        add_executable(test_gemma_forward
            finetune_ops/graph/test_gemma_forward.cpp
        )
        target_link_libraries(test_gemma_forward operators)
    endif()
    
    # Baseline GPT-2训练（全参数微调对比）
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/finetune_ops/optim/test_baseline_10step.cpp)
        add_executable(test_baseline_10step
            finetune_ops/optim/test_baseline_10step.cpp
        )
        target_link_libraries(test_baseline_10step operators)
    endif()
    
    # 固定batch对比测试
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/finetune_ops/optim/test_fixed_batch_compare.cpp)
        add_executable(test_fixed_batch_compare
            finetune_ops/optim/test_fixed_batch_compare.cpp
        )
        target_link_libraries(test_fixed_batch_compare operators)
    endif()
    
    message(STATUS "✅ 测试目标已启用")
endif()

# ============================================================================
# 主程序：gpt2_lora_finetune
# ============================================================================
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../gpt2_lora_finetune/main.cpp)
    add_executable(gpt2_lora_finetune
        ../gpt2_lora_finetune/main.cpp
    )
    target_link_libraries(gpt2_lora_finetune operators)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/finetune_ops/optim/train_lora_gemma.cpp)
    add_executable(train_lora_gemma
        finetune_ops/optim/train_lora_gemma.cpp
    )
    target_link_libraries(train_lora_gemma operators)
endif()

# MMLU eval
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../gpt2_lora_finetune/eval_mmlu.cpp)
    add_executable(eval_mmlu
        ../gpt2_lora_finetune/eval_mmlu.cpp
        ../gpt2_lora_finetune/mmlu/mmlu_runner.cpp
    )
    target_include_directories(eval_mmlu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../gpt2_lora_finetune)
    target_link_libraries(eval_mmlu operators)
endif()

# WikiText-2 PPL eval
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../gpt2_lora_finetune/eval_ppl.cpp)
    add_executable(eval_ppl
        ../gpt2_lora_finetune/eval_ppl.cpp
    )
    target_include_directories(eval_ppl PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../gpt2_lora_finetune)
    target_link_libraries(eval_ppl operators)
endif()

# Quick PPL eval
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../gpt2_lora_finetune/quick_eval_ppl.cpp)
    add_executable(quick_eval_ppl
        ../gpt2_lora_finetune/quick_eval_ppl.cpp
    )
    target_include_directories(quick_eval_ppl PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../gpt2_lora_finetune)
    target_link_libraries(quick_eval_ppl operators)
endif()

# Quick PPL eval with LoRA support
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../gpt2_lora_finetune/quick_eval_lora.cpp)
    add_executable(quick_eval_lora
        ../gpt2_lora_finetune/quick_eval_lora.cpp
    )
    target_include_directories(quick_eval_lora PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../gpt2_lora_finetune)
    target_link_libraries(quick_eval_lora operators)
endif()

# ============================================================================
# 配置摘要
# ============================================================================
message(STATUS "")
message(STATUS "============================================")
message(STATUS "Operators 框架配置摘要 v${PROJECT_VERSION}")
message(STATUS "============================================")
message(STATUS "C++ 标准: ${CMAKE_CXX_STANDARD}")
message(STATUS "核心模块:")
message(STATUS "  - 自动微分引擎: ${USE_NEW_AUTOGRAD_ENGINE}")
message(STATUS "  - 移动端优化器: ${USE_MOBILE_OPTIMIZER}")
message(STATUS "高级模块:")
message(STATUS "  - Memory模块: ${ENABLE_MEMORY_MODULE}")
message(STATUS "  - Activations模块: ${ENABLE_ACTIVATIONS_MODULE}")
message(STATUS "开发工具:")
message(STATUS "  - 构建测试: ${BUILD_TESTS}")
message(STATUS "  - 性能分析: ${ENABLE_PROFILING}")
message(STATUS "============================================")
message(STATUS "")
