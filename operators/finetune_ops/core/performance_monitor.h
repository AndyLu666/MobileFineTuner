/**
 * @file performance_monitor.h
 * @brief 性能与内存监控工具（类 PyTorch profiler）
 */

#pragma once

#include <string>
#include <chrono>
#include <iostream>
#include <iomanip>

namespace ops {

/**
 * @brief 内存快照（类似 torch.cuda.memory_summary）
 */
struct MemorySnapshot {
    size_t allocated_mb = 0;      // 已分配（MB）
    size_t in_use_mb = 0;         // 使用中（MB）
    size_t peak_mb = 0;           // 峰值（MB）
    size_t system_rss_mb = 0;     // 进程RSS（MB）
    size_t system_available_mb = 0;  // 系统可用（MB）
    
    void print() const {
        std::cout << "\n📊 Memory Snapshot:" << std::endl;
        std::cout << "  Allocated:  " << std::setw(8) << allocated_mb << " MB" << std::endl;
        std::cout << "  In use:     " << std::setw(8) << in_use_mb << " MB" << std::endl;
        std::cout << "  Peak:       " << std::setw(8) << peak_mb << " MB" << std::endl;
        std::cout << "  System RSS: " << std::setw(8) << system_rss_mb << " MB" << std::endl;
        std::cout << "  Available:  " << std::setw(8) << system_available_mb << " MB" << std::endl;
    }
};

/**
 * @brief 性能监控器（RAII）
 * 
 * 用法：
 * {
 *     PerformanceMonitor mon("forward pass");
 *     // ... 代码 ...
 * } // 自动输出耗时与内存变化
 */
class PerformanceMonitor {
public:
    explicit PerformanceMonitor(const std::string& name, bool print_memory = true);
    ~PerformanceMonitor();
    
    void checkpoint(const std::string& label);
    MemorySnapshot get_memory_snapshot() const;
    
private:
    std::string name_;
    bool print_memory_;
    std::chrono::time_point<std::chrono::steady_clock> start_time_;
    MemorySnapshot start_memory_;
};

/**
 * @brief 获取当前内存快照
 */
MemorySnapshot get_current_memory_snapshot();

/**
 * @brief 打印内存使用建议（类似 PyTorch 的提示）
 */
void print_memory_optimization_tips(const MemorySnapshot& snapshot);

} // namespace ops

