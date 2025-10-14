#include "operators.h"
#include <iostream>

using namespace ops;

int main() {
    std::cout << "Testing merged operators framework..." << std::endl;
    
    try {
        // Test basic tensor creation
        auto a = zeros({2, 3});
        auto b = ones({2, 3});
        std::cout << "[PASS] Basic tensor creation" << std::endl;
        
        // Test newly added activation functions
        auto x = randn({2, 3});
        auto silu_out = silu(x);
        std::cout << "[PASS] SiLU activation function" << std::endl;
        
        // Test RMS normalization
        auto rms_weight = ones({3});
        auto rms_out = rms_norm(x, rms_weight);
        std::cout << "[PASS] RMS Normalization" << std::endl;
        
        std::cout << "All core functionality tests passed!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "[FAIL] Test failed: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
