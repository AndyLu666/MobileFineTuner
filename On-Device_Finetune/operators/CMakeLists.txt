cmake_minimum_required(VERSION 3.10)
project(Operators VERSION 2.0.0 LANGUAGES CXX)

# ============================================================================
# Industrial-Grade Configuration Options
# ============================================================================
option(USE_NEW_AUTOGRAD_ENGINE "Use topological sorting autograd engine" ON)
option(AUTOGRAD_DEBUG "Enable autograd debug output" OFF)
option(USE_MOBILE_OPTIMIZER "Enable mobile optimizer extensions" ON)
option(ENABLE_MEMORY_MODULE "Enable Memory module (parameter management/Zero optimization)" ON)
option(ENABLE_ACTIVATIONS_MODULE "Enable Activations module (gradient checkpointing/activation management)" ON)
option(BUILD_TESTS "Build test programs" ON)
option(BUILD_EXAMPLES "Build example programs" OFF)
option(ENABLE_PROFILING "Enable performance profiling" OFF)

# ============================================================================
# C++ Standard
# ============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============================================================================
# Compilation Options
# ============================================================================
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O2")

if(AUTOGRAD_DEBUG)
    add_definitions(-DAUTOGRAD_DEBUG)
    message(STATUS "Autograd debug output enabled")
endif()

if(USE_NEW_AUTOGRAD_ENGINE)
    add_definitions(-DUSE_NEW_AUTOGRAD_ENGINE)
    message(STATUS "New autograd engine enabled (topological sorting)")
endif()

if(ENABLE_PROFILING)
    add_definitions(-DENABLE_PROFILING)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
    message(STATUS "Performance profiling enabled")
endif()

# ============================================================================
# Include Directories
# ============================================================================
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# ============================================================================
# Source File List
# ============================================================================

# Core module (core tensor/operators/autograd)
set(CORE_SOURCES
    core/tensor.cpp
    core/ops.cpp
    core/utils.cpp
    core/logger.cpp
    core/memory_manager.cpp
    core/mobile_safe_matmul.cpp
    core/memory_first_attention.cpp
    core/memory_first_mlp.cpp
    core/step_arena.cpp
    core/tokenizer.cpp
)

# Autograd module (automatic differentiation)
# backward_functions is required in both modes; new engine only adds autograd_engine
list(APPEND CORE_SOURCES
    core/backward_functions.cpp
)
if(USE_NEW_AUTOGRAD_ENGINE)
    list(APPEND CORE_SOURCES
        core/autograd_engine.cpp
    )
endif()

# Optimizer module (optimizers)
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
    message(STATUS "Mobile optimizer extensions enabled")
endif()

# Memory module (parameter management/ZeRO optimization)
set(MEMORY_SOURCES "")
if(ENABLE_MEMORY_MODULE)
    list(APPEND MEMORY_SOURCES
        memory/mobile_param_manager.cpp
        memory/mobile_param_optimizations.cpp
        memory/mobile_specific_optimizations.cpp
        memory/mobile_zero.cpp
    )
    add_definitions(-DENABLE_MEMORY_MODULE)
    message(STATUS "Memory module enabled (parameter management/ZeRO optimization)")
endif()

# Activations module (activation management/gradient checkpointing)
set(ACTIVATIONS_SOURCES "")
if(ENABLE_ACTIVATIONS_MODULE)
    list(APPEND ACTIVATIONS_SOURCES
        activations/mobile_activation_manager.cpp
        activations/deepspeed_checkpoint_integration.cpp
    )
    add_definitions(-DENABLE_ACTIVATIONS_MODULE)
    message(STATUS "Activations module enabled (activation management/checkpointing)")
endif()

# ============================================================================
# Create Static Library
# ============================================================================
add_library(operators STATIC
    ${CORE_SOURCES}
    ${OPTIMIZER_SOURCES}
    ${MEMORY_SOURCES}
    ${ACTIVATIONS_SOURCES}
)

# ============================================================================
# Link Library Detection (Accelerate/BLAS)
# Remove all BLAS/Accelerate related logic, use pure C++ implementation uniformly
message(STATUS "BLAS removed, using pure C++ implementation uniformly")

# ============================================================================
# Thread Support
# ============================================================================
find_package(Threads REQUIRED)
target_link_libraries(operators PUBLIC Threads::Threads)

# ============================================================================
# Output Configuration
# ============================================================================
set_target_properties(operators PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    VERSION ${PROJECT_VERSION}
)

# ============================================================================
# Installation Rules
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
# Test Targets
# ============================================================================
if(BUILD_TESTS)
    enable_testing()
    
    # Autograd engine tests
    if(USE_NEW_AUTOGRAD_ENGINE)
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../tests/test_autograd_engine.cpp)
            add_executable(test_autograd_engine
                ../tests/test_autograd_engine.cpp
            )
            target_link_libraries(test_autograd_engine operators)
            add_test(NAME AutogradEngine COMMAND test_autograd_engine)
        endif()
    endif()
    
    # Optimizer tests
    if(USE_MOBILE_OPTIMIZER)
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/optim/comprehensive_optimizer_test.cpp)
            add_executable(test_optimizer
                optim/comprehensive_optimizer_test.cpp
            )
            target_link_libraries(test_optimizer operators)
            add_test(NAME Optimizer COMMAND test_optimizer)
        endif()
    endif()
    
    # Memory module tests
    if(ENABLE_MEMORY_MODULE)
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_memory_module.cpp)
            add_executable(test_memory_module
                tests/test_memory_module.cpp
            )
            target_link_libraries(test_memory_module operators)
            add_test(NAME MemoryModule COMMAND test_memory_module)
        endif()
    endif()
    
    # Activations module tests
    if(ENABLE_ACTIVATIONS_MODULE)
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_activations_module.cpp)
            add_executable(test_activations_module
                tests/test_activations_module.cpp
            )
            target_link_libraries(test_activations_module operators)
            add_test(NAME ActivationsModule COMMAND test_activations_module)
        endif()
    endif()
    
    message(STATUS "Test targets enabled")
endif()

# ============================================================================
# Configuration Summary
# ============================================================================
message(STATUS "")
message(STATUS "============================================")
message(STATUS "Operators Framework Configuration Summary v${PROJECT_VERSION}")
message(STATUS "============================================")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Core Modules:")
message(STATUS "  - Autograd Engine: ${USE_NEW_AUTOGRAD_ENGINE}")
message(STATUS "  - Mobile Optimizer: ${USE_MOBILE_OPTIMIZER}")
message(STATUS "Advanced Modules:")
message(STATUS "  - Memory Module: ${ENABLE_MEMORY_MODULE}")
message(STATUS "  - Activations Module: ${ENABLE_ACTIVATIONS_MODULE}")
message(STATUS "Development Tools:")
message(STATUS "  - Build Tests: ${BUILD_TESTS}")
message(STATUS "  - Performance Profiling: ${ENABLE_PROFILING}")
message(STATUS "============================================")
message(STATUS "")
