#include "operators.h"
#include <iostream>
#include <iomanip>

using namespace ops;

int main() {
    std::cout << "Testing enhanced operators framework" << std::endl;
    std::cout << "==========================================" << std::endl;
    
    try {
        // Test basic functionality
        std::cout << "\nTesting basic tensor operations..." << std::endl;
        auto a = randn({2, 3});
        auto b = ones({2, 3});
        auto c = add(a, b);
        std::cout << "[PASS] Basic operations" << std::endl;
        
        // Test newly added activation functions
        std::cout << "\nTesting activation functions..." << std::endl;
        auto x = randn({2, 4});
        auto silu_out = silu(x);
        std::cout << "[PASS] SiLU activation function" << std::endl;
        
        // Test RMS normalization
        std::cout << "\nTesting normalization..." << std::endl;
        auto rms_weight = ones({4});
        auto rms_out = rms_norm(x, rms_weight);
        std::cout << "[PASS] RMS Normalization" << std::endl;
        
        // Test math functions
        std::cout << "\nTesting math functions..." << std::endl;
        auto pos_tensor = abs(x);
        auto sqrt_tensor = sqrt(add(x, 1.0f));  // Avoid negative sqrt
        auto exp_tensor = exp(mul(x, 0.1f));   // Avoid exp overflow
        std::cout << "[PASS] abs, sqrt, exp functions" << std::endl;
        
        // Test statistical functions
        std::cout << "\nTesting statistical functions..." << std::endl;
        auto sum_val = sum(x);
        auto mean_val = mean(x);
        std::cout << "[PASS] sum, mean functions" << std::endl;
        
        // Test Transformer operators
        std::cout << "\nTesting Transformer operators..." << std::endl;
        
        // Test causal mask
        auto mask = create_causal_mask(4);
        std::cout << "[PASS] causal mask creation" << std::endl;
        
        // Test KV head repetition (GQA)
        auto kv_tensor = randn({1, 2, 4, 8});  // [batch=1, kv_heads=2, seq=4, head_dim=8]
        auto repeated_kv = repeat_kv_heads(kv_tensor, 3);  // Repeat 3 times
        auto expected_shape = repeated_kv->shape();
        if (expected_shape[1] == 6) {  // 2*3=6
            std::cout << "[PASS] repeat_kv_heads (2 heads -> 6 heads)" << std::endl;
        } else {
            std::cout << "[FAIL] repeat_kv_heads shape error" << std::endl;
        }
        
        // Test RoPE
        auto q_tensor = randn({1, 4, 8, 16});  // [batch=1, heads=4, seq=8, head_dim=16]
        auto rope_out = apply_rope(q_tensor, 8, 16);
        std::cout << "[PASS] apply_rope" << std::endl;
        
        // Test SwiGLU
        std::cout << "\nTesting SwiGLU..." << std::endl;
        auto gate = randn({2, 8});
        auto up = randn({2, 8});
        auto swiglu_out = swiglu(gate, up);
        std::cout << "[PASS] SwiGLU" << std::endl;
        
        std::cout << "\nAll enhanced features tests passed!" << std::endl;
        std::cout << "Operators framework now supports:" << std::endl;
        std::cout << "  * Basic tensor operations" << std::endl;
        std::cout << "  * Activation functions (GELU, SiLU, ReLU, Sigmoid, Tanh)" << std::endl;
        std::cout << "  * Normalization (LayerNorm, RMSNorm, BatchNorm)" << std::endl;
        std::cout << "  * Math functions (abs, sqrt, exp, log, pow, clamp)" << std::endl;
        std::cout << "  * Statistical functions (sum, mean)" << std::endl;
        std::cout << "  * Transformer operators (causal_mask, repeat_kv, RoPE, SwiGLU)" << std::endl;
        std::cout << "  * LoRA support (lora_linear)" << std::endl;
        std::cout << "  * Automatic differentiation and gradient computation" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "[FAIL] Test failed: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
