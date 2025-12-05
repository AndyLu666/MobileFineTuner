/**
 * @file test_simple_grad.cpp
 * @brief 最简单的梯度测试（验证autograd引擎）
 */

#include "../core/tensor.h"
#include "../core/ops.h"
#include <iostream>

using namespace ops;

int main() {
    std::cout << "========== 简单梯度测试 ==========\n" << std::endl;
    
    // Test 1: 基础梯度
    std::cout << "[Test 1] Basic gradient test" << std::endl;
    auto x = full({2, 3}, 2.0f, kFloat32, kCPU);
    x->set_requires_grad(true);
    
    auto y = mul(x, 3.0f);  // y = 3x
    auto z = add(y, 1.0f);  // z = 3x + 1
    auto loss = mean(z);    // loss = mean(3x + 1)
    
    std::cout << "  x requires_grad: " << x->requires_grad() << std::endl;
    std::cout << "  loss requires_grad: " << loss->requires_grad() << std::endl;
    std::cout << "  loss value: " << loss->data<float>()[0] << std::endl;
    
    loss->backward();
    
    if (x->grad()) {
        std::cout << "  ✅ x has gradient" << std::endl;
        std::cout << "  x.grad()[0]: " << x->grad()->data<float>()[0] << std::endl;
    } else {
        std::cout << "  ❌ x has NO gradient!" << std::endl;
    }
    
    // Test 2: Matmul梯度
    std::cout << "\n[Test 2] Matmul gradient test" << std::endl;
    auto A = full({3, 4}, 1.0f, kFloat32, kCPU);
    A->set_requires_grad(true);
    
    auto B = full({4, 5}, 2.0f, kFloat32, kCPU);
    B->set_requires_grad(true);
    
    auto input = full({2, 3}, 0.5f, kFloat32, kCPU);
    
    // y = input @ A @ B
    auto xA = matmul(input, A);
    auto xAB = matmul(xA, B);
    auto loss2 = mean(xAB);
    
    std::cout << "  A requires_grad: " << A->requires_grad() << std::endl;
    std::cout << "  B requires_grad: " << B->requires_grad() << std::endl;
    std::cout << "  loss2 requires_grad: " << loss2->requires_grad() << std::endl;
    
    loss2->backward();
    
    if (A->grad() && B->grad()) {
        std::cout << "  ✅ A and B have gradients" << std::endl;
        float A_grad_norm = 0.0f;
        const float* A_grad_data = A->grad()->data<float>();
        for (int64_t i = 0; i < A->grad()->numel(); ++i) {
            A_grad_norm += A_grad_data[i] * A_grad_data[i];
        }
        std::cout << "  A grad norm: " << std::sqrt(A_grad_norm) << std::endl;
    } else {
        std::cout << "  ❌ A or B has NO gradient!" << std::endl;
        if (!A->grad()) std::cout << "    A->grad() is null" << std::endl;
        if (!B->grad()) std::cout << "    B->grad() is null" << std::endl;
    }
    
    return 0;
}

