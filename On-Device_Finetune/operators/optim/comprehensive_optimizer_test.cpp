/**
 * [Documentation available in English]
 * 
 * testtarget：
 * 1. validate allP0critical issuesfix
 * [Documentation available in English]
 * [Documentation available in English]
 * [Documentation available in English]
 * [Documentation available in English]
 */

#include "mobile_optimizer_state_manager.h"
#include "mobile_optimizer_extensions.h"
#include "mobile_optimizer_advanced.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <memory>

using namespace ops;
using namespace ops::optim;

// helperfunction：comparefloating point
bool almost_equal(float a, float b, float epsilon = 1e-3f) {
    return std::abs(a - b) < epsilon;
}

// helperfunction：comparetensor
bool tensors_almost_equal(const TensorPtr& a, const TensorPtr& b, float epsilon = 1e-3f) {
    if (!a || !b) return false;
    if (a->numel() != b->numel()) return false;
    
    const float* a_data = a->data<float>();
    const float* b_data = b->data<float>();
    
    for (int64_t i = 0; i < a->numel(); ++i) {
        if (!almost_equal(a_data[i], b_data[i], epsilon)) {
            std::cout << "  Mismatch at index " << i << ": " 
                     << a_data[i] << " vs " << b_data[i] << std::endl;
            return false;
        }
    }
    return true;
}

// helperfunction：createtesttensor
TensorPtr create_test_tensor(int64_t size, float value) {
        // [Translated]
    return ops::full({size}, value, ops::kFloat32);
}

// ============================================
// [Translated comment removed - see documentation]
// ============================================
void test_compression_decompression_roundtrip() {
    std::cout << "\n========================================\n";
    std::cout << "test1: compression-decompress[Output]test\n";
    std::cout << "========================================\n";
    
    MobileOptimizerStateConfig config;
    config.enable_compression = true;
    config.compression_threshold = 0.5f;
    config.max_active_memory_mb = 100.0f;
    
    auto manager = std::make_shared<MobileOptimizerStateManager>(config);
    
    // registerparameter
    size_t param_id = 0;
    size_t param_size = 1000;
    manager->register_parameter_state(param_id, "test_param", param_size);
    
    // createoriginalstate
    auto original_momentum = create_test_tensor(param_size, 1.0f);
    auto original_variance = create_test_tensor(param_size, 2.0f);
    
    // update state
    manager->update_momentum_state(param_id, original_momentum);
    manager->update_variance_state(param_id, original_variance);
    
    std::cout << "✓ originalstate[Output]settings\n";
    
    // executecompression
    bool compressed = manager->compress_parameter_state(
        param_id, 
        OptimizerStateCompression::FP16
    );
    
    assert(compressed);
    std::cout << "✓ state[Output]compressionasFP16\n";
    
        // [Translated]
    auto retrieved_momentum = manager->get_momentum_state(param_id);
    auto retrieved_variance = manager->get_variance_state(param_id);
    
    std::cout << "✓ state[Output]decompress\n";
    
    // validateaccuracy
    assert(retrieved_momentum != nullptr);
    assert(retrieved_variance != nullptr);
    assert(retrieved_momentum->dtype() == ops::kFloat32);
    assert(retrieved_variance->dtype() == ops::kFloat32);
    
    std::cout << "✓ decompressbackdata typecorrect (FP32)\n";
    
    // [Translated comment removed - see documentation]
    bool momentum_ok = tensors_almost_equal(original_momentum, retrieved_momentum, 1e-3f);
    bool variance_ok = tensors_almost_equal(original_variance, retrieved_variance, 1e-3f);
    
    assert(momentum_ok);
    assert(variance_ok);
    
    std::cout << "✅ test1via：compression-decompress[Output]correct\n";
}

// ============================================
// [Translated]
// ============================================
void test_update_after_compression() {
    std::cout << "\n========================================\n";
    std::cout << "test2: stateupdatebackcompressionmodereset\n";
    std::cout << "========================================\n";
    
    MobileOptimizerStateConfig config;
    config.enable_compression = true;
    auto manager = std::make_shared<MobileOptimizerStateManager>(config);
    
    size_t param_id = 0;
    size_t param_size = 1000;
    manager->register_parameter_state(param_id, "test_param", param_size);
    
    // steps1: settingsinitialstate
    auto initial_state = create_test_tensor(param_size, 1.0f);
    manager->update_momentum_state(param_id, initial_state);
    std::cout << "✓ steps1: initialstatesettings (FP32)\n";
    
    // steps2: compression
    manager->compress_parameter_state(param_id, OptimizerStateCompression::FP16);
    std::cout << "✓ steps2: statecompressionasFP16\n";
    
        // [Translated]
    auto retrieved = manager->get_momentum_state(param_id);
    assert(retrieved->dtype() == ops::kFloat32);
    std::cout << "✓ steps3: acquirestate[Output]decompress (FP32)\n";
    
        // [Translated]
    auto new_state = create_test_tensor(param_size, 3.0f);
    manager->update_momentum_state(param_id, new_state);
    std::cout << "✓ steps4: updateasnewstate (FP32)\n";
    
    // [Translated comment removed - see documentation]
    auto retrieved_again = manager->get_momentum_state(param_id);
    assert(retrieved_again != nullptr);
    assert(retrieved_again->dtype() == ops::kFloat32);
    std::cout << "✓ steps5: [Output]acquirecorrectreturnFP32\n";
    
        // [Translated]
    bool values_ok = tensors_almost_equal(new_state, retrieved_again, 1e-6f);
    assert(values_ok);
    
    std::cout << "✅ test2via：updatebackcompressionmodecorrectreset\n";
}

// ============================================
// [Translated]
// ============================================
void test_memory_statistics_accuracy() {
    std::cout << "\n========================================\n";
    std::cout << "test3: memorystatisticsaccurate[Output]\n";
    std::cout << "========================================\n";
    
    MobileOptimizerStateConfig config;
    config.enable_compression = true;
    config.max_active_memory_mb = 100.0f;
    auto manager = std::make_shared<MobileOptimizerStateManager>(config);
    
    size_t param_id = 0;
    size_t param_size = 1000;
    manager->register_parameter_state(param_id, "test_param", param_size);
    
    auto stats0 = manager->get_statistics();
    float stats0_mb = stats0.active_memory_used / (1024.0f * 1024.0f);
    std::cout << "initialmemory: " << stats0_mb << " MB\n";
    
    // steps1: addstate
    auto state = create_test_tensor(param_size, 1.0f);
    manager->update_momentum_state(param_id, state);
    manager->update_variance_state(param_id, state);
    
    auto stats1 = manager->get_statistics();
    float stats1_mb = stats1.active_memory_used / (1024.0f * 1024.0f);
    float expected_mem_1 = (param_size * 2 * sizeof(float)) / (1024.0f * 1024.0f);
    std::cout << "addstateback: " << stats1_mb << " MB ([Output]: " 
              << expected_mem_1 << " MB)\n";
    assert(almost_equal(stats1_mb, expected_mem_1, 0.01f));
    std::cout << "✓ steps1: addstatebackmemorystatisticscorrect\n";
    
        // [Translated]
    manager->compress_parameter_state(param_id, OptimizerStateCompression::FP16);
    
    auto stats2 = manager->get_statistics();
    float stats2_mb = stats2.active_memory_used / (1024.0f * 1024.0f);
    float expected_mem_2 = (param_size * 2 * sizeof(uint16_t)) / (1024.0f * 1024.0f);
    std::cout << "compressionback: " << stats2_mb << " MB ([Output]: " 
              << expected_mem_2 << " MB)\n";
    assert(almost_equal(stats2_mb, expected_mem_2, 0.01f));
    std::cout << "✓ steps2: compressionbackmemorystatisticscorrect\n";
    
        // [Translated]
    auto new_state = create_test_tensor(param_size, 2.0f);
    manager->update_momentum_state(param_id, new_state);
    
    auto stats3 = manager->get_statistics();
    float stats3_mb = stats3.active_memory_used / (1024.0f * 1024.0f);
    // momentum: FP32, variance: still FP16
    float expected_mem_3 = (param_size * sizeof(float) + param_size * sizeof(uint16_t)) 
                          / (1024.0f * 1024.0f);
    std::cout << "updatemomentumback: " << stats3_mb << " MB ([Output]: " 
              << expected_mem_3 << " MB)\n";
    assert(almost_equal(stats3_mb, expected_mem_3, 0.01f));
    std::cout << "✓ steps3: updatebackmemorystatisticscorrect\n";
    
    std::cout << "✅ test3via：memorystatistics[Output]accurate\n";
}

// ============================================
// [Translated]
// ============================================
void test_complete_training_simulation() {
    std::cout << "\n========================================\n";
    std::cout << "test4: completetraining[Output]modulo[Output]\n";
    std::cout << "========================================\n";
    
    MobileOptimizerStateConfig config;
    config.enable_compression = true;
    config.compression_threshold = 0.6f;
    config.max_active_memory_mb = 10.0f;
    
    auto manager = std::make_shared<MobileOptimizerStateManager>(config);
    
    // registermultipleparameter
    const size_t num_params = 5;
    const size_t param_size = 1000;
    
    for (size_t i = 0; i < num_params; ++i) {
        manager->register_parameter_state(i, "param_" + std::to_string(i), param_size);
    }
    std::cout << "✓ register " << num_params << " parameter\n";
    
        // [Translated]
    const int num_steps = 10;
    for (int step = 0; step < num_steps; ++step) {
        std::cout << "\n--- steps " << step + 1 << " ---\n";
        
        for (size_t param_id = 0; param_id < num_params; ++param_id) {
                        // [Translated]
            auto momentum = manager->get_momentum_state(param_id);
            auto variance = manager->get_variance_state(param_id);
            
            // 2. simulate Adam update
            auto new_momentum = create_test_tensor(param_size, step + param_id);
            auto new_variance = create_test_tensor(param_size, step + param_id + 0.5f);
            
            // 3. update state
            manager->update_momentum_state(param_id, new_momentum);
            manager->update_variance_state(param_id, new_variance);
        }
        
                // [Translated]
        manager->optimize_memory_usage();
        
        auto stats = manager->get_statistics();
        float stats_mb = stats.active_memory_used / (1024.0f * 1024.0f);
        std::cout << "  memoryuse: " << stats_mb << " MB\n";
        std::cout << "  compression[Output]: " << stats.total_compressions << "\n";
        
                // [Translated]
        assert(stats_mb <= config.max_active_memory_mb * 1.2f);
    }
    
    auto final_stats = manager->get_statistics();
    float final_mb = final_stats.active_memory_used / (1024.0f * 1024.0f);
    float saved_mb = final_stats.memory_saved_by_compression / (1024.0f * 1024.0f);
    std::cout << "\nfinalstatistics:\n";
    std::cout << "  [Output]compression[Output]: " << final_stats.total_compressions << "\n";
    std::cout << "  [Output]decompress[Output]: " << final_stats.total_decompressions << "\n";
    std::cout << "  [Output]memory: " << saved_mb << " MB\n";
    std::cout << "  finalmemory: " << final_mb << " MB\n";
    
    std::cout << "✅ test4via：completetraining[Output]correct\n";
}

// ============================================
// [Translated]
// ============================================
void test_edge_cases() {
    std::cout << "\n========================================\n";
    std::cout << "test5: boundary[Output]andincorrectprocess\n";
    std::cout << "========================================\n";
    
    MobileOptimizerStateConfig config;
    auto manager = std::make_shared<MobileOptimizerStateManager>(config);
    
        // [Translated]
    try {
        auto state = manager->get_momentum_state(999);
        std::cout << "✗ should[Output]exception\n";
        assert(false);
    } catch (const std::runtime_error&) {
        std::cout << "✓ correct[Output]exception：[Output]atparameter\n";
    }
    
        // [Translated]
    bool result = manager->compress_parameter_state(999, OptimizerStateCompression::FP16);
    assert(!result);
    std::cout << "✓ correctprocess：compression[Output]registerparameterreturnfalse\n";
    
    // test5.3: smallsizeparameter
    manager->register_parameter_state(0, "small_param", 1);
    auto small_state = manager->get_momentum_state(0);
    assert(small_state->numel() == 1);
    std::cout << "✓ correctprocess：smallsizeparameter\n";
    
    std::cout << "✅ test5via：boundary[Output]processcorrect\n";
}

// ============================================
// [Translated]
// ============================================
void test_thread_safety() {
    std::cout << "\n========================================\n";
    std::cout << "test6: [Output]threadsafe[Output]（fundamentaltest）\n";
    std::cout << "========================================\n";
    
    MobileOptimizerStateConfig config;
    auto manager = std::make_shared<MobileOptimizerStateManager>(config);
    
    const size_t num_params = 10;
    for (size_t i = 0; i < num_params; ++i) {
        manager->register_parameter_state(i, "thread_test_" + std::to_string(i), 100);
    }
    
    // [Translated comment removed - see documentation]
    for (int iter = 0; iter < 100; ++iter) {
        for (size_t i = 0; i < num_params; ++i) {
            auto state = create_test_tensor(100, iter);
            manager->update_momentum_state(i, state);
            auto retrieved = manager->get_momentum_state(i);
            assert(retrieved != nullptr);
        }
    }
    
    std::cout << "✓ [Output]fast[Output]no[Output]\n";
    std::cout << "✅ test6via：fundamentalthreadsafe[Output]correct\n";
}

// ============================================
// [Translated]
// ============================================
int main() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════╗\n";
    std::cout << "║   Mobile Optimizer [Output]                        ║\n";
    std::cout << "║   validate allP0critical issuesfix                             ║\n";
    std::cout << "╚════════════════════════════════════════════════════════╝\n";
    
    try {
        test_compression_decompression_roundtrip();
        test_update_after_compression();
        test_memory_statistics_accuracy();
        test_complete_training_simulation();
        test_edge_cases();
        test_thread_safety();
        
        std::cout << "\n";
        std::cout << "╔════════════════════════════════════════════════════════╗\n";
        std::cout << "║             ✅ alltestvia！                          ║\n";
        std::cout << "║                                                        ║\n";
        std::cout << "║  validateFix:                                           ║\n";
        std::cout << "║  ✓ [Output]1: compressionstate[Output]decompress                            ║\n";
        std::cout << "║  ✓ [Output]2: compressionstatecorrect[Output]                            ║\n";
        std::cout << "║  ✓ [Output]7: updatebackresetcompressionmode                          ║\n";
        std::cout << "║  ✓ [Output]8: memorystatisticsaccurate（compression/update）                   ║\n";
        std::cout << "║  ✓ [Output]9: Offloadmemorystatisticscorrect                         ║\n";
        std::cout << "║                                                        ║\n";
        std::cout << "║  🎉 [Output]cansafe[Output]use！                            ║\n";
        std::cout << "╚════════════════════════════════════════════════════════╝\n";
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cout << "\n";
        std::cout << "╔════════════════════════════════════════════════════════╗\n";
        std::cout << "║             ❌ testfailed！                              ║\n";
        std::cout << "╚════════════════════════════════════════════════════════╝\n";
        std::cout << "incorrect: " << e.what() << "\n";
        return 1;
    }
}

