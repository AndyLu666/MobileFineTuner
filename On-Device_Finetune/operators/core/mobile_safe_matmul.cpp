/**
 * @file mobile_safe_matmul.cpp
 * @brief Memory-safe matrix multiplication optimization implementation for mobile devices
 */

#include "mobile_safe_matmul.h"
#include <cstring>
#include <vector>
#include <iomanip>

namespace mobile_matmul {

/**
 * @brief Basic triple-loop matrix multiplication (original implementation)
 */
void naive_matmul(const float* A, const float* B, float* C,
                  int64_t M, int64_t N, int64_t K) {
    // Initialize result matrix
    std::memset(C, 0, M * N * sizeof(float));
    
    // Standard ijk loop order
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

/**
 * @brief Loop reordering optimized matrix multiplication (ikj order)
 * 
 * Optimization principle:
 * - Elements of A are reused, improving cache hit rate
 * - Both B and C are accessed row-wise, memory access pattern friendly
 * - Expected performance boost: 2-3x
 */
void reordered_matmul(const float* A, const float* B, float* C,
                      int64_t M, int64_t N, int64_t K) {
    // Initialize result matrix
    std::memset(C, 0, M * N * sizeof(float));
    
    // ikj loop order - cache friendly
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t k = 0; k < K; ++k) {
            float a_ik = A[i * K + k];  // A[i][k] read only once
            
            // Vectorization-friendly inner loop
            for (int64_t j = 0; j < N; ++j) {
                C[i * N + j] += a_ik * B[k * N + j];
            }
        }
    }
}

/**
 * @brief Block-optimized matrix multiplication
 * 
 * Optimization principle:
 * - Decompose large matrices into small blocks, each can fit in L1 cache
 * - Improve data reuse rate, reduce memory access latency
 * - Controllable memory usage, very suitable for mobile devices
 */
void blocked_matmul(const float* A, const float* B, float* C,
                    int64_t M, int64_t N, int64_t K, int block_size) {
    // Automatically calculate optimal block size
    if (block_size == 0) {
        block_size = AdaptiveBlockSize::calculate_block_size_for_matrix(M, N, K);
    }
    
    // Initialize result matrix
    std::memset(C, 0, M * N * sizeof(float));
    
    // Blocked computation
    for (int64_t bi = 0; bi < M; bi += block_size) {
        for (int64_t bj = 0; bj < N; bj += block_size) {
            for (int64_t bk = 0; bk < K; bk += block_size) {
                
                // Calculate actual block boundaries
                int64_t end_i = std::min(bi + block_size, M);
                int64_t end_j = std::min(bj + block_size, N);
                int64_t end_k = std::min(bk + block_size, K);
                
                // Use optimized ikj loop within block
                for (int64_t i = bi; i < end_i; ++i) {
                    for (int64_t k = bk; k < end_k; ++k) {
                        float a_ik = A[i * K + k];
                        
                        for (int64_t j = bj; j < end_j; ++j) {
                            C[i * N + j] += a_ik * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}

/**
 * @brief ARM NEON vectorized matrix multiplication
 */
void vectorized_matmul(const float* A, const float* B, float* C,
                       int64_t M, int64_t N, int64_t K) {
#ifdef __ARM_NEON
    // Initialize result matrix
    std::memset(C, 0, M * N * sizeof(float));
    
    // Use NEON vectorized ikj loop
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t k = 0; k < K; ++k) {
            float32x4_t a_ik_vec = vdupq_n_f32(A[i * K + k]);
            
            int64_t j = 0;
            // Process 4 elements at a time
            for (; j <= N - 4; j += 4) {
                float32x4_t b_vec = vld1q_f32(&B[k * N + j]);
                float32x4_t c_vec = vld1q_f32(&C[i * N + j]);
                
                // C[i][j:j+4] += A[i][k] * B[k][j:j+4]
                c_vec = vmlaq_f32(c_vec, a_ik_vec, b_vec);
                
                vst1q_f32(&C[i * N + j], c_vec);
            }
            
            // Process remaining elements
            for (; j < N; ++j) {
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
#else
    // If NEON not supported, fallback to reordered version
    reordered_matmul(A, B, C, M, N, K);
#endif
}

/**
 * @brief Select optimal optimization strategy
 */
OptimizationLevel SafeMatmul::select_best_strategy(int64_t m, int64_t n, int64_t k) {
    // First check memory safety
    if (!MemorySafetyMonitor::is_operation_safe(m, n, k)) {
        std::cout << "WARNING: Matrix too large, potential memory risk, using basic implementation" << std::endl;
        return OptimizationLevel::NAIVE;
    }
    
    // Small matrices directly use reordered version
    if (m < 64 && n < 64 && k < 64) {
        return OptimizationLevel::REORDERED;
    }
    
    // Medium-sized matrices use block optimization
    if (m < 512 && n < 512 && k < 512) {
        return OptimizationLevel::BLOCKED;
    }
    
    // Large matrices use vectorization + blocking
    #ifdef __ARM_NEON
    return OptimizationLevel::VECTORIZED;
    #else
    return OptimizationLevel::BLOCKED;
    #endif
}

/**
 * @brief MEMORY_FIRST extreme memory-saving blocked matrix multiplication
 * Strategy: minimal blocking (16x16), disable vectorization, single-threaded, compute and release on-the-fly
 */
void memory_first_matmul(const float* A, const float* B, float* C,
                         int64_t M, int64_t N, int64_t K) {
    // Initialize result matrix
    std::memset(C, 0, M * N * sizeof(float));
    
    // Minimal block size: ensure each block only uses about 3 * 16*16 * 4B = 3KB (much smaller than L1 cache)
    const int64_t block_size = 16;
    
    // Triple-loop blocking: block M, N, K dimensions
    for (int64_t bi = 0; bi < M; bi += block_size) {
        int64_t end_i = std::min(bi + block_size, M);
        
        for (int64_t bj = 0; bj < N; bj += block_size) {
            int64_t end_j = std::min(bj + block_size, N);
            
            for (int64_t bk = 0; bk < K; bk += block_size) {
                int64_t end_k = std::min(bk + block_size, K);
                
                // Use ikj order within block (cache-friendly)
                for (int64_t i = bi; i < end_i; ++i) {
                    for (int64_t k = bk; k < end_k; ++k) {
                        float a_ik = A[i * K + k];
                        
                        // Inner loop: sequential access to B and C
                        for (int64_t j = bj; j < end_j; ++j) {
                            C[i * N + j] += a_ik * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}

/**
 * @brief Main safe matrix multiplication interface
 */
void SafeMatmul::multiply(const float* A, const float* B, float* C,
                         int64_t M, int64_t N, int64_t K,
                         OptimizationLevel level) {
    // Adaptive strategy selection
    if (level == OptimizationLevel::ADAPTIVE) {
        level = select_best_strategy(M, N, K);
    }
    
    // Execute matrix multiplication according to selected strategy
    switch (level) {
        case OptimizationLevel::NAIVE:
            naive_matmul(A, B, C, M, N, K);
            break;
            
        case OptimizationLevel::REORDERED:
            reordered_matmul(A, B, C, M, N, K);
            break;
            
        case OptimizationLevel::BLOCKED:
            blocked_matmul(A, B, C, M, N, K);
            break;
            
        case OptimizationLevel::VECTORIZED:
            vectorized_matmul(A, B, C, M, N, K);
            break;
            
        case OptimizationLevel::MEMORY_FIRST:
            memory_first_matmul(A, B, C, M, N, K);
            break;
            
        default:
            reordered_matmul(A, B, C, M, N, K);
            break;
    }
}

/**
 * @brief Right matrix transpose matrix multiplication (MEMORY_FIRST version, zero-copy)
 * C[M,N] = A[M,K] @ B[N,K]^T (B accessed transposed)
 */
void memory_first_matmul_rhs_T(const float* A, const float* B, float* C,
                               int64_t M, int64_t N, int64_t K) {
    std::memset(C, 0, M * N * sizeof(float));
    
    const int64_t block_size = 16;
    
    // Blocked computation (M, N, K dimensions)
    for (int64_t bi = 0; bi < M; bi += block_size) {
        int64_t end_i = std::min(bi + block_size, M);
        
        for (int64_t bj = 0; bj < N; bj += block_size) {
            int64_t end_j = std::min(bj + block_size, N);
            
            for (int64_t bk = 0; bk < K; bk += block_size) {
                int64_t end_k = std::min(bk + block_size, K);
                
                // 块内 ikj 顺序
                for (int64_t i = bi; i < end_i; ++i) {
                    for (int64_t k = bk; k < end_k; ++k) {
                        float a_ik = A[i * K + k];
                        
                        // B 按转置访问：B[N,K] -> B^T[K,N]
                        // B[j,k] 位于 B[j*K + k]
                        for (int64_t j = bj; j < end_j; ++j) {
                            C[i * N + j] += a_ik * B[j * K + k];
                        }
                    }
                }
            }
        }
    }
}

/**
 * @brief 右矩阵转置矩阵乘法统一接口
 */
void SafeMatmul::multiply_rhs_T(const float* A, const float* B, float* C,
                                int64_t M, int64_t N, int64_t K,
                                OptimizationLevel level) {
    // 目前只实现 MEMORY_FIRST 版本（其他策略可按需扩展）
    if (level == OptimizationLevel::MEMORY_FIRST || level == OptimizationLevel::ADAPTIVE) {
        memory_first_matmul_rhs_T(A, B, C, M, N, K);
    } else {
        // 默认走 MEMORY_FIRST
        memory_first_matmul_rhs_T(A, B, C, M, N, K);
    }
}

/**
 * @brief 自适应矩阵乘法（智能选择最优实现）
 */
void adaptive_matmul(const float* A, const float* B, float* C,
                     int64_t M, int64_t N, int64_t K) {
    SafeMatmul::multiply(A, B, C, M, N, K, OptimizationLevel::ADAPTIVE);
}

/**
 * @brief 性能基准测试
 */
void SafeMatmul::benchmark_all_methods(int64_t M, int64_t N, int64_t K) {
    std::cout << "\n=== 矩阵乘法性能基准测试 ===" << std::endl;
    std::cout << "矩阵大小: " << M << "x" << K << " * " << K << "x" << N << std::endl;
    
    // 检查内存安全性
    if (!MemorySafetyMonitor::is_operation_safe(M, N, K)) {
        std::cout << "⚠️ 矩阵过大，跳过基准测试以确保内存安全" << std::endl;
        return;
    }
    
    // 分配测试矩阵
    std::vector<float> A(M * K, 1.0f);
    std::vector<float> B(K * N, 1.0f);
    std::vector<float> C(M * N);
    
    PerformanceMonitor monitor;
    
    // 测试各种实现
    std::vector<std::pair<std::string, OptimizationLevel>> methods = {
        {"基础实现", OptimizationLevel::NAIVE},
        {"循环重排序", OptimizationLevel::REORDERED},
        {"分块优化", OptimizationLevel::BLOCKED},
        {"向量化", OptimizationLevel::VECTORIZED}
    };
    
    for (const auto& method : methods) {
        // 重置结果矩阵
        std::fill(C.begin(), C.end(), 0.0f);
        
        monitor.start();
        multiply(A.data(), B.data(), C.data(), M, N, K, method.second);
        long elapsed = monitor.get_elapsed_ms();
        
        double gflops = monitor.calculate_gflops(M, N, K, elapsed);
        
        std::cout << method.first << ": " << elapsed << "ms, " 
                  << std::fixed << std::setprecision(2) << gflops << " GFLOPS" << std::endl;
    }
    
    // 测试自适应选择
    std::fill(C.begin(), C.end(), 0.0f);
    monitor.start();
    multiply(A.data(), B.data(), C.data(), M, N, K, OptimizationLevel::ADAPTIVE);
    long elapsed = monitor.get_elapsed_ms();
    double gflops = monitor.calculate_gflops(M, N, K, elapsed);
    
    std::cout << "自适应选择: " << elapsed << "ms, " 
              << std::fixed << std::setprecision(2) << gflops << " GFLOPS" << std::endl;
    
    std::cout << "\n可用内存: " << MemorySafetyMonitor::get_available_memory() / (1024*1024) << " MB" << std::endl;
    std::cout << "L1缓存大小: " << MemorySafetyMonitor::get_l1_cache_size() / 1024 << " KB" << std::endl;
    std::cout << "推荐块大小: " << AdaptiveBlockSize::calculate_safe_block_size() << std::endl;
}

} // namespace mobile_matmul
