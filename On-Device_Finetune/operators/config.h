/**
 * @file config.h
 * @brief Operators Framework Unified Configuration
 * 
 * Defines compile-time macros and feature switches
 */

#pragma once

// ============================================================================
// Version Information
// ============================================================================
#define OPERATORS_VERSION_MAJOR 1
#define OPERATORS_VERSION_MINOR 0
#define OPERATORS_VERSION_PATCH 0
#define OPERATORS_VERSION "1.0.0"

// ============================================================================
// Feature Switches (set via CMake, default values provided here)
// ============================================================================

// New autograd engine (topological sorting, supports deep networks)
#ifndef USE_NEW_AUTOGRAD_ENGINE
#define USE_NEW_AUTOGRAD_ENGINE 1
#endif

// Autograd debug output
#ifndef AUTOGRAD_DEBUG
// #define AUTOGRAD_DEBUG  // disabled by default
#endif

// Mobile optimizer extensions (gradient clipping/LR scheduling/state management)
#ifndef USE_MOBILE_OPTIMIZER
#define USE_MOBILE_OPTIMIZER 1
#endif

// ============================================================================
// Platform Detection
// ============================================================================
#if defined(__APPLE__)
    #define OPERATORS_PLATFORM_MACOS
    #ifdef USE_ACCELERATE
        #define OPERATORS_USE_ACCELERATE
    #endif
#elif defined(__linux__)
    #define OPERATORS_PLATFORM_LINUX
#elif defined(_WIN32)
    #define OPERATORS_PLATFORM_WINDOWS
#endif

// ============================================================================
// SIMD Support Detection
// ============================================================================
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    #define OPERATORS_HAS_NEON
#endif

#if defined(__AVX2__)
    #define OPERATORS_HAS_AVX2
#endif

#if defined(__AVX__)
    #define OPERATORS_HAS_AVX
#endif

// ============================================================================
// Memory Configuration
// ============================================================================
#define OPERATORS_DEFAULT_MEMORY_ALIGNMENT 64  // byte alignment
#define OPERATORS_DEFAULT_CACHE_LINE_SIZE 64

// ============================================================================
// Debug Macros
// ============================================================================
#ifdef AUTOGRAD_DEBUG
    #include <iostream>
    #define AUTOGRAD_LOG(...) std::cout << "[AutogradDebug] " << __VA_ARGS__ << std::endl
#else
    #define AUTOGRAD_LOG(...)
#endif

// ============================================================================
// Assert Macros
// ============================================================================
#include <cassert>
#define OPERATORS_ASSERT(cond, msg) assert((cond) && (msg))

// ============================================================================
// Utility Macros
// ============================================================================
#define OPERATORS_UNUSED(x) (void)(x)

// Disable copy constructor and assignment
#define OPERATORS_DISABLE_COPY(ClassName) \
    ClassName(const ClassName&) = delete; \
    ClassName& operator=(const ClassName&) = delete;

// Disable move constructor and assignment
#define OPERATORS_DISABLE_MOVE(ClassName) \
    ClassName(ClassName&&) = delete; \
    ClassName& operator=(ClassName&&) = delete;

// Disable all copy and move
#define OPERATORS_DISABLE_COPY_AND_MOVE(ClassName) \
    OPERATORS_DISABLE_COPY(ClassName) \
    OPERATORS_DISABLE_MOVE(ClassName)

