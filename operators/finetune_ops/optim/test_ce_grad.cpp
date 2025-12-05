/**
 * @file test_ce_grad.cpp
 * @brief 测试Cross-Entropy的梯度传播
 */

#include "../core/lm_loss.h"
#include "../core/ops.h"
#include "../nn/lora_linear.h"
#include <iostream>

using namespace ops;

int main() {
    std::cout << "========== Cross-Entropy梯度测试 ==========\n" << std::endl;
    
    // Test: 简单网络 input → LoRALinear → CE Loss → Backward
    std::cout << "[Test] Simple network with LoRALinear + CE Loss" << std::endl;
    
    // Base权重
    auto W = full({10, 50}, 0.01f, kFloat32, kCPU);
    auto b = full({50}, 0.0f, kFloat32, kCPU);
    W->set_requires_grad(false);
    b->set_requires_grad(false);
    
    // LoRALinear
    LoRALinear lin(W, b);
    auto A = full({10, 4}, 0.1f, kFloat32, kCPU);
    auto B = full({4, 50}, 0.1f, kFloat32, kCPU);
    lin.attach_lora(A, B, 1.0f, 0, 50);
    
    std::cout << "  A requires_grad: " << A->requires_grad() << std::endl;
    std::cout << "  B requires_grad: " << B->requires_grad() << std::endl;
    
    // 输入：[2, 5, 10]
    auto input = full({2, 5, 10}, 0.5f, kFloat32, kCPU);
    
    // Forward
    auto logits = lin.forward(input);  // [2, 5, 50]
    std::cout << "  logits requires_grad: " << logits->requires_grad() << std::endl;
    std::cout << "  logits shape: [" << logits->shape()[0] << ", " 
              << logits->shape()[1] << ", " << logits->shape()[2] << "]" << std::endl;
    
    // Labels: [2, 5]
    std::vector<int32_t> labels_data = {1, 2, 3, 4, 5,  10, 11, 12, 13, 14};
    auto labels = std::make_shared<Tensor>(
        std::vector<int64_t>{2, 5}, labels_data.data(), kInt32, kCPU);
    
    // Cross-Entropy Loss
    auto loss = lm_cross_entropy(logits, labels, -100, "mean");
    std::cout << "  loss requires_grad: " << loss->requires_grad() << std::endl;
    std::cout << "  loss value: " << loss->data<float>()[0] << std::endl;
    
    // Backward
    std::cout << "\n  Calling backward()..." << std::endl;
    loss->backward();
    std::cout << "  Backward completed" << std::endl;
    
    // 检查梯度
    std::cout << "\n[Results]" << std::endl;
    if (A->grad()) {
        std::cout << "  ✅ A has gradient" << std::endl;
        float norm = 0.0f;
        const float* data = A->grad()->data<float>();
        for (int64_t i = 0; i < A->grad()->numel(); ++i) norm += data[i] * data[i];
        std::cout << "  A grad norm: " << std::sqrt(norm) << std::endl;
    } else {
        std::cout << "  ❌ A has NO gradient!" << std::endl;
    }
    
    if (B->grad()) {
        std::cout << "  ✅ B has gradient" << std::endl;
    } else {
        std::cout << "  ❌ B has NO gradient!" << std::endl;
    }
    
    if (logits->grad()) {
        std::cout << "  ℹ️  logits has gradient" << std::endl;
    } else {
        std::cout << "  ℹ️  logits has no gradient (expected for intermediate)" << std::endl;
    }
    
    return 0;
}

