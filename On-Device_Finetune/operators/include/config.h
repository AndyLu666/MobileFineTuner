/**
 * @file config.h
 * [Documentation available in English]
 * 
 * [Documentation available in English]
 */

#pragma once

// ============================================================================
// versioninfo
// ============================================================================
#define OPERATORS_VERSION_MAJOR 1
#define OPERATORS_VERSION_MINOR 0
#define OPERATORS_VERSION_PATCH 0
#define OPERATORS_VERSION "1.0.0"

// ============================================================================
// [Translated comment removed - see documentation]
// ============================================================================

// newautomatic differentiationengine（topological sort，supportdeeplayernetwork）
#ifndef USE_NEW_AUTOGRAD_ENGINE
#define USE_NEW_AUTOGRAD_ENGINE 1
#endif

// automatic differentiationdebugoutput
#ifndef AUTOGRAD_DEBUG
// #define AUTOGRAD_DEBUG  // disabled by default
#endif

// mobileoptimizerextended（gradient clipping/LRschedule/statemanage）
#ifndef USE_MOBILE_OPTIMIZER
#define USE_MOBILE_OPTIMIZER 1
#endif

// ============================================================================
// platfordetection
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
// SIMD supportdetection
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
// memoryconfiguration
// ============================================================================
#define OPERATORS_DEFAULT_MEMORY_ALIGNMENT 64  // byte alignment
#define OPERATORS_DEFAULT_CACHE_LINE_SIZE 64

// ============================================================================
// debugmacro
// ============================================================================
#ifdef AUTOGRAD_DEBUG
    #include <iostream>
    #define AUTOGRAD_LOG(...) std::cout << "[AutogradDebug] " << __VA_ARGS__ << std::endl
#else
    #define AUTOGRAD_LOG(...)
#endif

// ============================================================================
// [Translated]
// ============================================================================
#include <cassert>
#define OPERATORS_ASSERT(cond, msg) assert((cond) && (msg))

// ============================================================================
// toolmacro
// ============================================================================
#define OPERATORS_UNUSED(x) (void)(x)

// [Translated]
#define OPERATORS_DISABLE_COPY(ClassName) \
    ClassName(const ClassName&) = delete; \
    ClassName& operator=(const ClassName&) = delete;

// [Translated]
#define OPERATORS_DISABLE_MOVE(ClassName) \
    ClassName(ClassName&&) = delete; \
    ClassName& operator=(ClassName&&) = delete;

// [Translated]
#define OPERATORS_DISABLE_COPY_AND_MOVE(ClassName) \
    OPERATORS_DISABLE_COPY(ClassName) \
    OPERATORS_DISABLE_MOVE(ClassName)

