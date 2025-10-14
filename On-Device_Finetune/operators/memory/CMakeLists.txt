# CMakeLists.txt for Mobile Memory Management Module

cmake_minimum_required(VERSION 3.18)

# Set module name
set(MODULE_NAME mobile_memory_management)

# Find required packages
find_package(Threads REQUIRED)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../core)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../layer)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../optimizers)

# Source files
set(MEMORY_SOURCES
    mobile_param_manager.cpp
    mobile_zero.cpp  # Will need to be implemented
    mobile_param_optimizations.cpp  # Critical missing optimizations implementation
    mobile_specific_optimizations.cpp  # Mobile-specific optimizations implementation
)

# Header files
set(MEMORY_HEADERS
    mobile_param_manager.h
    mobile_zero.h
    mobile_param_optimizations.h  # Critical missing optimizations
    mobile_specific_optimizations.h  # Mobile-specific optimizations
)

# Create static library
add_library(${MODULE_NAME} STATIC ${MEMORY_SOURCES})

# Link required libraries
target_link_libraries(${MODULE_NAME} 
    PRIVATE
    Threads::Threads
    ${CMAKE_DL_LIBS}  # For dynamic loading if needed
)

# Compiler flags for optimization
target_compile_options(${MODULE_NAME} PRIVATE
    $<$<CONFIG:Release>:-O3 -DNDEBUG -march=native>
    $<$<CONFIG:Debug>:-O0 -g -DDEBUG>
)

# C++ standard
target_compile_features(${MODULE_NAME} PRIVATE cxx_std_17)

# Platform-specific optimizations
if(ANDROID)
    target_compile_definitions(${MODULE_NAME} PRIVATE ANDROID_PLATFORM)
    # Android-specific optimizations
    target_compile_options(${MODULE_NAME} PRIVATE -ffast-math -funroll-loops)
elseif(IOS)
    target_compile_definitions(${MODULE_NAME} PRIVATE IOS_PLATFORM)
    # iOS-specific optimizations
    target_compile_options(${MODULE_NAME} PRIVATE -ffast-math -funroll-loops)
elseif(APPLE)
    target_compile_definitions(${MODULE_NAME} PRIVATE MACOS_PLATFORM)
    # macOS-specific optimizations
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
        target_compile_options(${MODULE_NAME} PRIVATE -mcpu=apple-m1)
    endif()
endif()

# Memory management specific flags
target_compile_definitions(${MODULE_NAME} PRIVATE
    MOBILE_MEMORY_ALIGNMENT=64  # 64-byte alignment for cache efficiency
    ENABLE_MEMORY_TRACKING=1    # Enable detailed memory tracking
)

# Install headers
install(FILES ${MEMORY_HEADERS}
        DESTINATION include/operators/memory)

# Install library
install(TARGETS ${MODULE_NAME}
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib)

# Example executable (optional)
option(BUILD_MEMORY_EXAMPLES "Build memory management examples" ON)
if(BUILD_MEMORY_EXAMPLES)
    add_executable(mobile_zero_example ../examples/mobile_zero_integration.cpp)
    target_link_libraries(mobile_zero_example ${MODULE_NAME})
    
    # Install example
    install(TARGETS mobile_zero_example
            RUNTIME DESTINATION bin/examples)
endif()

# Tests (optional)
option(BUILD_MEMORY_TESTS "Build memory management tests" ON)
if(BUILD_MEMORY_TESTS AND TARGET gtest)
    add_subdirectory(tests)
endif()

# Documentation
find_package(Doxygen)
if(DOXYGEN_FOUND)
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
    
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
    
    add_custom_target(memory_docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating memory management API documentation"
        VERBATIM
    )
endif()

# Print configuration summary
message(STATUS "Mobile Memory Management Module Configuration:")
message(STATUS "  Module Name: ${MODULE_NAME}")
message(STATUS "  Build Examples: ${BUILD_MEMORY_EXAMPLES}")
message(STATUS "  Build Tests: ${BUILD_MEMORY_TESTS}")
message(STATUS "  Platform: ${CMAKE_SYSTEM_NAME}")
message(STATUS "  Architecture: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "  C++ Standard: 17")
