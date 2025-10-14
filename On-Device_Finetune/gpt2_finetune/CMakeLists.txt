cmake_minimum_required(VERSION 3.10)
project(GPT2_LoRA_Finetune VERSION 1.0.0 LANGUAGES CXX)

# Configuration Options
option(USE_NEW_AUTOGRAD_ENGINE "Use topological sort autograd engine" ON)
option(USE_MOBILE_OPTIMIZER "Enable mobile optimizer extensions" ON)
option(AUTOGRAD_DEBUG "Enable autograd debug output" OFF)
option(FORCE_BUILD_OPERATORS_FROM_SOURCE "Ignore existing operators library, build from source" ON)
# BLAS/Accelerate options removed, using pure C++ implementation

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compilation Options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O2")

# Add preprocessor definitions
if(USE_NEW_AUTOGRAD_ENGINE)
    add_definitions(-DUSE_NEW_AUTOGRAD_ENGINE)
    message(STATUS "New autograd engine enabled")
endif()

if(USE_MOBILE_OPTIMIZER)
    add_definitions(-DUSE_MOBILE_OPTIMIZER)
    message(STATUS "Mobile optimizer enabled")
endif()

if(AUTOGRAD_DEBUG)
    add_definitions(-DAUTOGRAD_DEBUG)
    message(STATUS "Autograd debug output enabled")
endif()

# Include Directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../operators)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../common)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../ggml/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)

# Find Dependencies

# Find GGML library
find_library(GGML_LIBRARY 
    NAMES ggml libggml
    PATHS ${CMAKE_CURRENT_SOURCE_DIR}/../build
          ${CMAKE_CURRENT_SOURCE_DIR}/../ggml
          ${CMAKE_CURRENT_SOURCE_DIR}/../build/ggml/src
    PATH_SUFFIXES lib
)

# Find Common library
find_library(COMMON_LIBRARY 
    NAMES common libcommon
    PATHS ${CMAKE_CURRENT_SOURCE_DIR}/../build
          ${CMAKE_CURRENT_SOURCE_DIR}/../build/common
    PATH_SUFFIXES lib
)

# Find Operators library
find_library(OPERATORS_LIBRARY 
    NAMES operators liboperators
    PATHS ${CMAKE_CURRENT_SOURCE_DIR}/../operators/build
          ${CMAKE_CURRENT_SOURCE_DIR}/../build/operators
    PATH_SUFFIXES lib
)

# Source Files

# Operators core source files (if library not found, compile sources directly)
set(OPERATORS_CORE_SOURCES
    ../operators/core/tensor.cpp
    ../operators/core/ops.cpp
    ../operators/core/utils.cpp
    ../operators/core/logger.cpp
    ../operators/core/memory_manager.cpp
    ../operators/core/mobile_safe_matmul.cpp
    ../operators/core/tokenizer.cpp
)

# Autograd source files
if(USE_NEW_AUTOGRAD_ENGINE)
    list(APPEND OPERATORS_CORE_SOURCES
        ../operators/core/backward_functions.cpp
        ../operators/core/autograd_engine.cpp
    )
endif()

# Optimizer source files
set(OPERATORS_OPTIM_SOURCES
    ../operators/optim/optimizer.cpp
    ../operators/optim/adam.cpp
)

if(USE_MOBILE_OPTIMIZER)
    list(APPEND OPERATORS_OPTIM_SOURCES
        ../operators/optim/mobile_optimizer_extensions.cpp
        ../operators/optim/mobile_optimizer_state_manager.cpp
        ../operators/optim/mobile_optimizer_advanced.cpp
    )
endif()

# Main program source files
set(MAIN_SOURCES
    gpt2_lora_finetune.cpp
)

# Create Executable
add_executable(gpt2_lora_finetune ${MAIN_SOURCES})

# Library Linking Configuration

# Use pre-compiled operators library (do not rebuild from source)
if(OPERATORS_LIBRARY)
    target_link_libraries(gpt2_lora_finetune ${OPERATORS_LIBRARY})
    message(STATUS "Found Operators library: ${OPERATORS_LIBRARY}")
else()
    message(FATAL_ERROR "Operators library not found, please build operators directory first")
endif()

# Common library not needed for now (GPT2 training does not depend on it)
# if(COMMON_LIBRARY)
#     target_link_libraries(gpt2_lora_finetune ${COMMON_LIBRARY})
# endif()

if(GGML_LIBRARY)
    target_link_libraries(gpt2_lora_finetune ${GGML_LIBRARY})
    message(STATUS "Found GGML library: ${GGML_LIBRARY}")
else()
    message(WARNING "GGML library not found, may need to build GGML first")
endif()

# System-Specific Library Linking

# Declaration: Not using BLAS/Accelerate, pure C++ implementation
message(STATUS "BLAS removed, using pure C++ implementation")

# Apple systems: other system libraries
if(APPLE)
    target_link_libraries(gpt2_lora_finetune "-framework Foundation")
endif()

# Linux systems: link math and thread libraries
if(UNIX AND NOT APPLE)
    target_link_libraries(gpt2_lora_finetune m pthread)
endif()

# Windows systems
if(WIN32)
    target_link_libraries(gpt2_lora_finetune ws2_32)
endif()

# Compiler-Specific Settings

# GCC/Clang specific settings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(gpt2_lora_finetune PRIVATE
        -march=native
        -ffast-math
        -funroll-loops
    )
endif()

# Output Settings
set_target_properties(gpt2_lora_finetune PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Installation Rules
install(TARGETS gpt2_lora_finetune
    RUNTIME DESTINATION bin
)

# Configuration Summary
message(STATUS "")
message(STATUS "GPT2 LoRA Finetune Configuration Summary")
message(STATUS "Version: ${PROJECT_VERSION}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "New Autograd Engine: ${USE_NEW_AUTOGRAD_ENGINE}")
message(STATUS "Mobile Optimizer: ${USE_MOBILE_OPTIMIZER}")
message(STATUS "Debug Mode: ${AUTOGRAD_DEBUG}")
if(OPERATORS_LIBRARY)
    message(STATUS "Operators Library: ${OPERATORS_LIBRARY}")
else()
    message(STATUS "Operators Library: Built-in source files")
endif()
if(COMMON_LIBRARY)
    message(STATUS "Common Library: ${COMMON_LIBRARY}")
else()
    message(STATUS "Common Library: Built-in source files")
endif()
if(GGML_LIBRARY)
    message(STATUS "GGML Library: ${GGML_LIBRARY}")
else()
    message(STATUS "GGML Library: Not found")
endif()
message(STATUS "")
