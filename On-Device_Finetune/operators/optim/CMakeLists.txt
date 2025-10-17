# CMakeLists.txt for Mobile Optimizer State Manager

cmake_minimum_required(VERSION 3.10)

# 添加optimizer state manager库
add_library(mobile_optimizer_state STATIC
    mobile_optimizer_state_manager.cpp
)

# 包含目录
target_include_directories(mobile_optimizer_state PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../core
    ${CMAKE_CURRENT_SOURCE_DIR}/../parameter
)

# 链接core库和parameter库
target_link_libraries(mobile_optimizer_state
    tensor_core
    mobile_param_manager
)

# 编译选项
target_compile_features(mobile_optimizer_state PUBLIC cxx_std_17)

# ARM NEON优化（如果是ARM架构）
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm" OR CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    message(STATUS "Enabling ARM NEON optimizations for optimizer state manager")
    target_compile_options(mobile_optimizer_state PRIVATE -mfpu=neon)
endif()

# 优化标志
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(mobile_optimizer_state PRIVATE -O3 -march=native)
endif()

# 安装
install(TARGETS mobile_optimizer_state
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES mobile_optimizer_state_manager.h
    DESTINATION include/operators/memory/optimizer
)

