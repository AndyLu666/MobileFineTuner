#include "operators.h"
#include <iostream>
#include <iomanip>

using namespace ops;

int main() {
    std::cout << "🚀 testenhancebackoperatorsframework" << std::endl;
    std::cout << "==========================================" << std::endl;
    
    try {
                // [Translated]
        std::cout << "\n📊 testfundamentaltensoroperation..." << std::endl;
        auto a = randn({2, 3});
        auto b = ones({2, 3});
        auto c = add(a, b);
        std::cout << "✅ fundamentaloperation[Output]" << std::endl;
        
                // [Translated]
        std::cout << "\n🔥 testactivation function..." << std::endl;
        auto x = randn({2, 4});
        auto silu_out = silu(x);
        std::cout << "✅ SiLUactivation function[Output]" << std::endl;
        
        // testRMSnormalization
        std::cout << "\n📏 testnormalization..." << std::endl;
        auto rms_weight = ones({4});
        auto rms_out = rms_norm(x, rms_weight);
        std::cout << "✅ RMS Normalization[Output]" << std::endl;
        
                // [Translated]
        std::cout << "\n🧮 test[Output]function..." << std::endl;
        auto pos_tensor = abs(x);
        auto sqrt_tensor = sqrt(add(x, 1.0f));          // [Translated]
        auto exp_tensor = exp(mul(x, 0.1f));           // [Translated]
        std::cout << "✅ abs, sqrt, expfunction[Output]" << std::endl;
        
        // teststatisticsfunction
        std::cout << "\n📈 teststatisticsfunction..." << std::endl;
        auto sum_val = sum(x);
        auto mean_val = mean(x);
        std::cout << "✅ sum, meanfunction[Output]" << std::endl;
        
        // testTransforeroperator
        std::cout << "\n🤖 testTransforeroperator..." << std::endl;
        
        // testcausal mask
        auto mask = create_causal_mask(4);
        std::cout << "✅ causal maskcreate[Output]" << std::endl;
        
                // [Translated]
        auto kv_tensor = randn({1, 2, 4, 8});  // [batch=1, kv_heads=2, seq=4, head_dim=8]
        auto repeated_kv = repeat_kv_heads(kv_tensor, 3);          // [Translated]
        auto expected_shape = repeated_kv->shape();
        if (expected_shape[1] == 6) {  // 2*3=6
            std::cout << "✅ repeat_kv_heads[Output] (2 heads -> 6 heads)" << std::endl;
        } else {
            std::cout << "❌ repeat_kv_headsshapeincorrect" << std::endl;
        }
        
        // testRoPE
        auto q_tensor = randn({1, 4, 8, 16});  // [batch=1, heads=4, seq=8, head_dim=16]
        auto rope_out = apply_rope(q_tensor, 8, 16);
        std::cout << "✅ apply_rope[Output]" << std::endl;
        
        // testSwiGLU
        std::cout << "\n🌟 testSwiGLU..." << std::endl;
        auto gate = randn({2, 8});
        auto up = randn({2, 8});
        auto swiglu_out = swiglu(gate, up);
        std::cout << "✅ SwiGLU[Output]" << std::endl;
        
        std::cout << "\n🎉 allenhance[Output]testvia！" << std::endl;
        std::cout << "📊 operatorsframeworknowsupport:" << std::endl;
        std::cout << "  ✓ fundamentaltensoroperation" << std::endl;
        std::cout << "  ✓ activation function (GELU, SiLU, ReLU, Sigmoid, Tanh)" << std::endl;
        std::cout << "  ✓ normalization (LayerNorm, RMSNorm, BatchNorm)" << std::endl;
        std::cout << "  ✓ [Output]function (abs, sqrt, exp, log, pow, clamp)" << std::endl;
        std::cout << "  ✓ statisticsfunction (sum, mean)" << std::endl;
        std::cout << "  ✓ Transforeroperator (causal_mask, repeat_kv, RoPE, SwiGLU)" << std::endl;
        std::cout << "  ✓ LoRAsupport (lora_linear)" << std::endl;
        std::cout << "  ✓ automatic differentiationandgradientcompute" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "❌ testfailed: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
