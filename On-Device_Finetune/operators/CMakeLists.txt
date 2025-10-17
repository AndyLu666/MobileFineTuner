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
option(BUILD_TESTS "构建测试程序" ON)
option(BUILD_EXAMPLES "构建示例程序" OFF)
option(ENABLE_PROFILING "启用性能分析" OFF)

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

if(ENABLE_PROFILING)
    add_definitions(-DENABLE_PROFILING)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
    message(STATUS "性能分析已启用")
endif()

# ============================================================================
# 包含目录
# ============================================================================
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# ============================================================================
# 源文件列表
# ============================================================================

# Core 模块（核心张量/算子/自动微分）
set(CORE_SOURCES
    core/tensor.cpp
    core/ops.cpp
    core/chunked_softmax_ce.cpp
    core/utils.cpp
    core/logger.cpp
    core/memory_manager.cpp
    core/mobile_safe_matmul.cpp
    core/memory_first_attention.cpp
    core/memory_first_mlp.cpp
    core/step_arena.cpp
    core/tokenizer.cpp
)

# Autograd 模块（自动微分）
# backward_functions 在兩種模式下都需要；新引擎僅額外加入 autograd_engine
list(APPEND CORE_SOURCES
    core/backward_functions.cpp
)
if(USE_NEW_AUTOGRAD_ENGINE)
    list(APPEND CORE_SOURCES
        core/autograd_engine.cpp
    )
endif()

# Optimizer 模块（优化器）
set(OPTIMIZER_SOURCES
    optim/optimizer.cpp
    optim/adam.cpp
)

if(USE_MOBILE_OPTIMIZER)
    list(APPEND OPTIMIZER_SOURCES
        optim/mobile_optimizer_extensions.cpp
        optim/mobile_optimizer_state_manager.cpp
        optim/mobile_optimizer_advanced.cpp
    )
    message(STATUS "移动端优化器扩展已启用")
endif()

# Memory 模块（参数管理/ZeRO优化）
set(MEMORY_SOURCES "")
if(ENABLE_MEMORY_MODULE)
    list(APPEND MEMORY_SOURCES
        memory/mobile_param_manager.cpp
        memory/mobile_param_optimizations.cpp
        memory/mobile_specific_optimizations.cpp
        memory/mobile_zero.cpp
    )
    add_definitions(-DENABLE_MEMORY_MODULE)
    message(STATUS "✅ Memory模块已启用（参数管理/ZeRO优化）")
endif()

# Activations 模块（激活管理/梯度检查点）
set(ACTIVATIONS_SOURCES "")
if(ENABLE_ACTIVATIONS_MODULE)
    list(APPEND ACTIVATIONS_SOURCES
        activations/mobile_activation_manager.cpp
        activations/deepspeed_checkpoint_integration.cpp
        memory/activation_checkpointer.cpp
    )
    add_definitions(-DENABLE_ACTIVATIONS_MODULE)
    message(STATUS "✅ Activations模块已启用（激活管理/检查点）")
endif()

# ============================================================================
# 创建静态库
# ============================================================================
add_library(operators STATIC
    ${CORE_SOURCES}
    ${OPTIMIZER_SOURCES}
    ${MEMORY_SOURCES}
    ${ACTIVATIONS_SOURCES}
)

# ============================================================================
# 链接库检测（Accelerate/BLAS）
# 移除所有 BLAS/Accelerate 相关逻辑，统一使用纯 C++ 实现
message(STATUS "🔧 BLAS 已移除，统一使用纯C++实现")

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
    
    message(STATUS "✅ 测试目标已启用")
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
