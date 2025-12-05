/**
 * @file test_lora_grad.cpp
 * @brief 测试LoRALinear的梯度传播
 */

#include "../nn/lora_linear.h"
#include "../core/ops.h"
#include <iostream>

using namespace ops;

int main() {
    std::cout << "========== LoRALinear梯度测试 ==========\n" << std::endl;
    
    // 创建base权重
    auto W = full({3, 5}, 0.1f, kFloat32, kCPU);
    auto b = full({5}, 0.0f, kFloat32, kCPU);
    W->set_requires_grad(false);  // base冻结
    b->set_requires_grad(false);
    
    // 创建LoRALinear
    LoRALinear lora_lin(W, b);
    
    // 创建LoRA参数
    auto A = full({3, 2}, 0.5f, kFloat32, kCPU);
    auto B = full({2, 5}, 0.3f, kFloat32, kCPU);
    
    // Attach LoRA
    lora_lin.attach_lora(A, B, 1.0f, 0, 5);
    
    std::cout << "[Setup]" << std::endl;
    std::cout << "  A requires_grad: " << A->requires_grad() << std::endl;
    std::cout << "  B requires_grad: " << B->requires_grad() << std::endl;
    std::cout << "  W requires_grad: " << W->requires_grad() << std::endl;
    
    // Forward
    auto x = full({1, 3}, 1.0f, kFloat32, kCPU);
    auto y = lora_lin.forward(x);  // [1, 5]
    
    std::cout << "\n[Forward]" << std::endl;
    std::cout << "  y requires_grad: " << y->requires_grad() << std::endl;
    std::cout << "  y shape: [" << y->shape()[0] << ", " << y->shape()[1] << "]" << std::endl;
    
    // Loss
    auto loss = mean(y);
    std::cout << "  loss requires_grad: " << loss->requires_grad() << std::endl;
    std::cout << "  loss value: " << loss->data<float>()[0] << std::endl;
    
    // Backward
    std::cout << "\n[Backward]" << std::endl;
    loss->backward();
    std::cout << "  Backward completed" << std::endl;
    
    // 检查梯度
    std::cout << "\n[Gradients]" << std::endl;
    if (A->grad()) {
        std::cout << "  ✅ A has gradient" << std::endl;
        float A_grad_norm = 0.0f;
        const float* A_grad_data = A->grad()->data<float>();
        for (int64_t i = 0; i < A->grad()->numel(); ++i) {
            A_grad_norm += A_grad_data[i] * A_grad_data[i];
        }
        std::cout << "  A grad norm: " << std::sqrt(A_grad_norm) << std::endl;
    } else {
        std::cout << "  ❌ A has NO gradient" << std::endl;
    }
    
    if (B->grad()) {
        std::cout << "  ✅ B has gradient" << std::endl;
    } else {
        std::cout << "  ❌ B has NO gradient" << std::endl;
    }
    
    if (W->grad()) {
        std::cout << "  ⚠️  W has gradient (should be frozen!)" << std::endl;
    } else {
        std::cout << "  ✅ W has no gradient (correctly frozen)" << std::endl;
    }
    
    return 0;
}

