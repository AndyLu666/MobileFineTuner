# CMakeLists.txt for Mobile Optimizer State Manager

cmake_minimum_required(VERSION 3.10)

# Add optimizer state manager library
add_library(mobile_optimizer_state STATIC
    mobile_optimizer_state_manager.cpp
)

# Include directories
target_include_directories(mobile_optimizer_state PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../core
    ${CMAKE_CURRENT_SOURCE_DIR}/../parameter
)

# Link core and parameter libraries
target_link_libraries(mobile_optimizer_state
    tensor_core
    mobile_param_manager
)

# Compilation options
target_compile_features(mobile_optimizer_state PUBLIC cxx_std_17)

# ARM NEON optimization (if ARM architecture)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm" OR CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    message(STATUS "Enabling ARM NEON optimizations for optimizer state manager")
    target_compile_options(mobile_optimizer_state PRIVATE -mfpu=neon)
endif()

# Optimization flags
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(mobile_optimizer_state PRIVATE -O3 -march=native)
endif()

# Installation
install(TARGETS mobile_optimizer_state
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES mobile_optimizer_state_manager.h
    DESTINATION include/operators/memory/optimizer
)

