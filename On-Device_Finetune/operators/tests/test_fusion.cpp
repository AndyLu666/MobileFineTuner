#include "operators.h"
#include <iostream>

using namespace ops;

int main() {
    std::cout << "🧪 test[Output]operatorsframework..." << std::endl;
    
    try {
        // testbasictensorcreate
        auto a = zeros({2, 3});
        auto b = ones({2, 3});
        std::cout << "✅ fundamentaltensorcreate[Output]" << std::endl;
        
                // [Translated]
        auto x = randn({2, 3});
        auto silu_out = silu(x);
        std::cout << "✅ SiLUactivation function[Output]" << std::endl;
        
        // testRMSnormalization
        auto rms_weight = ones({3});
        auto rms_out = rms_norm(x, rms_weight);
        std::cout << "✅ RMS Normalization[Output]" << std::endl;
        
        std::cout << "🎉 allcore functionalitytestvia！" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "❌ testfailed: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
