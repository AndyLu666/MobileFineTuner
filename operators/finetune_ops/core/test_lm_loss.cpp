/**
 * @file test_lm_loss.cpp
 * @brief 语言模型Cross-Entropy Loss测试
 */

#include "lm_loss.h"
#include "ops.h"
#include <iostream>
#include <cmath>

using namespace ops;

int main() {
    try {
        std::cout << "========== LM Cross-Entropy Loss Test ==========\n" << std::endl;
        
        // Test 1: 简单验证（手工计算）
        std::cout << "[Test 1] Simple case (hand-calculated)" << std::endl;
        
        // logits: [1, 2, 5]
        // step0: [2, 1, 0, -1, -2] -> label=0
        // step1: [0, 0, 0,  0,  0] -> label=4
        std::vector<float> logits_data = {
            2.0f, 1.0f, 0.0f, -1.0f, -2.0f,  // step 0
            0.0f, 0.0f, 0.0f,  0.0f,  0.0f   // step 1
        };
        auto logits = std::make_shared<Tensor>(
            std::vector<int64_t>{1, 2, 5}, logits_data.data(), kFloat32, kCPU);
        
        std::vector<int32_t> labels_data = {0, 4};
        auto labels = std::make_shared<Tensor>(
            std::vector<int64_t>{1, 2}, labels_data.data(), kInt32, kCPU);
        
        auto loss_sum = lm_cross_entropy(logits, labels, -100, "sum");
        auto loss_mean = lm_cross_entropy(logits, labels, -100, "mean");
        
        float sum_val = loss_sum->data<float>()[0];
        float mean_val = loss_mean->data<float>()[0];
        
        std::cout << "  Loss (sum):  " << sum_val << std::endl;
        std::cout << "  Loss (mean): " << mean_val << std::endl;
        
        // 手算验证：
        // step0: max=2, denom=e^0+e^-1+e^-2+e^-3+e^-4 = 1+0.368+0.135+0.050+0.018 ≈ 1.571
        //        logsumexp = 2 + log(1.571) = 2 + 0.451 = 2.451
        //        nll = -(2 - 2.451) = 0.451
        // step1: 均匀5类，nll = -log(1/5) = log(5) = 1.6094
        // sum ≈ 0.451 + 1.609 = 2.060, mean ≈ 1.030
        
        float expected_sum = 0.451f + 1.6094f;  // ≈ 2.060
        float expected_mean = expected_sum / 2.0f;  // ≈ 1.030
        
        bool pass1 = std::abs(sum_val - expected_sum) < 0.01f;
        bool pass2 = std::abs(mean_val - expected_mean) < 0.01f;
        
        if (pass1 && pass2) {
            std::cout << "  ✅ PASS: 手工计算一致" << std::endl;
        } else {
            std::cout << "  ❌ FAIL: expected sum≈" << expected_sum 
                      << ", mean≈" << expected_mean << std::endl;
        }
        
        // Test 2: ignore_index测试
        std::cout << "\n[Test 2] ignore_index=-100" << std::endl;
        
        // labels中有PAD
        std::vector<int32_t> labels_with_pad = {0, -100};
        auto labels2 = std::make_shared<Tensor>(
            std::vector<int64_t>{1, 2}, labels_with_pad.data(), kInt32, kCPU);
        
        auto loss2 = lm_cross_entropy(logits, labels2, -100, "mean");
        float loss2_val = loss2->data<float>()[0];
        
        std::cout << "  Loss (只有1个有效token): " << loss2_val << std::endl;
        std::cout << "  期望: ≈ 0.451 (只计算step0)" << std::endl;
        
        bool pass3 = std::abs(loss2_val - 0.451f) < 0.01f;
        if (pass3) {
            std::cout << "  ✅ PASS: ignore_index正确忽略" << std::endl;
        } else {
            std::cout << "  ❌ FAIL" << std::endl;
        }
        
        // Test 3: 梯度形状验证
        std::cout << "\n[Test 3] Gradient shape" << std::endl;
        
        logits->set_requires_grad(true);
        auto loss3 = lm_cross_entropy(logits, labels, -100, "mean");
        
        std::cout << "  Loss requires_grad: " << loss3->requires_grad() << std::endl;
        std::cout << "  Loss value: " << loss3->data<float>()[0] << std::endl;
        
        // 简单验证backward（完整验证需要autograd引擎）
        if (loss3->requires_grad()) {
            std::cout << "  ✅ PASS: 梯度函数已挂接" << std::endl;
        }
        
        // Test 4: Perplexity计算
        std::cout << "\n[Test 4] Perplexity" << std::endl;
        float ppl = perplexity_from_loss(mean_val);
        std::cout << "  Mean loss: " << mean_val << std::endl;
        std::cout << "  Perplexity: " << ppl << std::endl;
        std::cout << "  期望: exp(1.03) ≈ 2.80" << std::endl;
        
        bool pass4 = std::abs(ppl - 2.80f) < 0.1f;
        if (pass4) {
            std::cout << "  ✅ PASS" << std::endl;
        }
        
        std::cout << "\n========== Summary ==========" << std::endl;
        if (pass1 && pass2 && pass3 && pass4) {
            std::cout << "🎉 所有测试通过！Cross-Entropy Loss可用于训练" << std::endl;
            return 0;
        } else {
            std::cout << "⚠️  部分测试失败" << std::endl;
            return 1;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Exception: " << e.what() << std::endl;
        return 1;
    }
}

