/**
 * @file performance_monitor.cpp
 * @brief 性能与内存监控实现
 */

#include "performance_monitor.h"
#include "memory_manager.h"
#include <iostream>
#include <iomanip>

namespace ops {

MemorySnapshot get_current_memory_snapshot() {
    MemorySnapshot snap;
    
    auto& mgr = MemoryManager::instance();
    snap.allocated_mb = mgr.get_memory_usage() / (1024 * 1024);
    snap.in_use_mb = mgr.get_memory_usage() / (1024 * 1024);
    snap.peak_mb = mgr.get_peak_memory() / (1024 * 1024);
    snap.system_rss_mb = MemoryMonitor::get_system_memory_usage() / (1024 * 1024);
    snap.system_available_mb = MemoryMonitor::get_system_available_memory() / (1024 * 1024);
    
    return snap;
}

PerformanceMonitor::PerformanceMonitor(const std::string& name, bool print_memory)
    : name_(name), print_memory_(print_memory) {
    start_time_ = std::chrono::steady_clock::now();
    if (print_memory_) {
        start_memory_ = get_current_memory_snapshot();
    }
}

PerformanceMonitor::~PerformanceMonitor() {
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time_);
    
    std::cout << "⏱️  [" << name_ << "] " << duration.count() << " ms";
    
    if (print_memory_) {
        auto end_memory = get_current_memory_snapshot();
        int64_t delta_mb = static_cast<int64_t>(end_memory.in_use_mb) - 
                          static_cast<int64_t>(start_memory_.in_use_mb);
        
        std::cout << " | Memory: " << end_memory.in_use_mb << " MB";
        if (delta_mb != 0) {
            std::cout << " (" << (delta_mb > 0 ? "+" : "") << delta_mb << " MB)";
        }
    }
    
    std::cout << std::endl;
}

void PerformanceMonitor::checkpoint(const std::string& label) {
    auto now = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time_);
    
    std::cout << "  ⏸️  [" << name_ << " | " << label << "] " 
              << duration.count() << " ms" << std::endl;
}

MemorySnapshot PerformanceMonitor::get_memory_snapshot() const {
    return get_current_memory_snapshot();
}

void print_memory_optimization_tips(const MemorySnapshot& snapshot) {
    std::cout << "\n💡 Memory Optimization Tips:" << std::endl;
    
    if (snapshot.peak_mb > 30000) {  // >30GB
        std::cout << "  ⚠️  Peak memory > 30GB. 建议：" << std::endl;
        std::cout << "     1. 减小 batch_size 或 seq_len" << std::endl;
        std::cout << "     2. 启用 use_memory_efficient_attention=true" << std::endl;
        std::cout << "     3. 启用 use_bf16_activations=true（降低50%激活内存）" << std::endl;
        std::cout << "     4. 使用梯度累积替代大batch" << std::endl;
    } else if (snapshot.peak_mb > 10000) {  // >10GB
        std::cout << "  ℹ️  Peak memory > 10GB. 可选优化：" << std::endl;
        std::cout << "     - 启用 use_bf16_activations=true" << std::endl;
        std::cout << "     - 评测时使用分段（chunked evaluation）" << std::endl;
    } else {
        std::cout << "  ✅ Memory usage looks good (< 10GB peak)" << std::endl;
    }
    
    if (snapshot.system_available_mb < 2000) {  // <2GB可用
        std::cout << "  🔴 System memory low (< 2GB available)!" << std::endl;
        std::cout << "     建议立即清理或减小batch/seq_len" << std::endl;
    }
}

} // namespace ops

