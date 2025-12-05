/**
 * @file test_matmul_debug.cpp
 * @brief 三个单测定位 matmul 在 attention 中的bug
 */

#include "../core/tensor.h"
#include "../core/ops.h"
#include <iostream>
#include <cmath>
#include <numeric>
#include <algorithm>

using namespace ops;

void test_ones_v() {
    std::cout << "\n========== Test 1: Ones-V (检测 /K bug) ==========" << std::endl;
    
    // probs: [1, 1, 5, 5] - 合法概率矩阵（每行和=1）
    std::vector<float> probs_data = {
        // 每行和为1
        0.2, 0.2, 0.2, 0.2, 0.2,  // row 0
        0.3, 0.1, 0.4, 0.1, 0.1,  // row 1
        0.5, 0.1, 0.1, 0.2, 0.1,  // row 2
        0.1, 0.3, 0.2, 0.3, 0.1,  // row 3
        0.2, 0.3, 0.1, 0.1, 0.3   // row 4
    };
    auto probs = std::make_shared<Tensor>(std::vector<int64_t>{1, 1, 5, 5}, 
                                          probs_data.data(), kFloat32, kCPU);
    
    // V: [1, 1, 5, 4] - 全1
    auto v = full({1, 1, 5, 4}, 1.0f, kFloat32, kCPU);
    
    // context = probs @ V
    auto context = matmul(probs, v);
    
    // 验证：应该全是1
    const float* ctx_data = context->data<float>();
    bool pass = true;
    float max_err = 0.0f;
    for (int64_t i = 0; i < context->numel(); ++i) {
        float err = std::abs(ctx_data[i] - 1.0f);
        max_err = std::max(max_err, err);
        if (err > 1e-5f) {
            pass = false;
        }
    }
    
    std::cout << "期望：全1" << std::endl;
    std::cout << "实际：mean=" << (double)std::accumulate(ctx_data, ctx_data + context->numel(), 0.0) / context->numel()
              << ", max_err=" << max_err << std::endl;
    
    if (pass) {
        std::cout << "✅ PASS: context全为1" << std::endl;
    } else {
        std::cout << "❌ FAIL: context不是1，可能是 /K 或 /S bug" << std::endl;
        std::cout << "示例值：";
        for (int64_t i = 0; i < std::min(static_cast<int64_t>(10), context->numel()); ++i) {
            printf("%.4f ", ctx_data[i]);
        }
        std::cout << std::endl;
    }
}

void test_identity_probs() {
    std::cout << "\n========== Test 2: Identity-probs (检测轴错/stride错) ==========" << std::endl;
    
    // probs: [1, 1, 5, 5] - 单位矩阵
    auto probs = zeros({1, 1, 5, 5}, kFloat32, kCPU);
    float* probs_data = probs->data<float>();
    for (int i = 0; i < 5; ++i) {
        probs_data[i * 5 + i] = 1.0f;  // 对角线为1
    }
    
    // V: [1, 1, 5, 4] - 随机值
    std::vector<float> v_data = {
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
        17.0, 18.0, 19.0, 20.0
    };
    auto v = std::make_shared<Tensor>(std::vector<int64_t>{1, 1, 5, 4}, 
                                      v_data.data(), kFloat32, kCPU);
    
    // context = I @ V = V
    auto context = matmul(probs, v);
    
    // 验证：context应该等于V
    const float* ctx_data = context->data<float>();
    const float* v_ptr = v->data<float>();
    bool pass = true;
    float max_err = 0.0f;
    for (int64_t i = 0; i < context->numel(); ++i) {
        float err = std::abs(ctx_data[i] - v_ptr[i]);
        max_err = std::max(max_err, err);
        if (err > 1e-5f) {
            pass = false;
        }
    }
    
    std::cout << "期望：context == V" << std::endl;
    std::cout << "实际：max_err=" << max_err << std::endl;
    
    if (pass) {
        std::cout << "✅ PASS: context == V" << std::endl;
    } else {
        std::cout << "❌ FAIL: 维度约简或stride有误" << std::endl;
        std::cout << "V[0:5]:       ";
        for (int i = 0; i < 5; ++i) printf("%.2f ", v_ptr[i]);
        std::cout << "\ncontext[0:5]: ";
        for (int i = 0; i < 5; ++i) printf("%.2f ", ctx_data[i]);
        std::cout << std::endl;
    }
}

void test_manual_dotprod() {
    std::cout << "\n========== Test 3: 手工点乘对拍 ==========" << std::endl;
    
    // probs: [1, 1, 5, 5]
    std::vector<float> probs_data = {
        0.1, 0.2, 0.3, 0.2, 0.2,
        0.3, 0.1, 0.4, 0.1, 0.1,
        0.5, 0.1, 0.1, 0.2, 0.1,  // 第2行（索引2）
        0.1, 0.3, 0.2, 0.3, 0.1,
        0.2, 0.3, 0.1, 0.1, 0.3
    };
    auto probs = std::make_shared<Tensor>(std::vector<int64_t>{1, 1, 5, 5}, 
                                          probs_data.data(), kFloat32, kCPU);
    
    // V: [1, 1, 5, 4]
    std::vector<float> v_data = {
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
        17.0, 18.0, 19.0, 20.0
    };
    auto v = std::make_shared<Tensor>(std::vector<int64_t>{1, 1, 5, 4}, 
                                      v_data.data(), kFloat32, kCPU);
    
    // matmul
    auto context = matmul(probs, v);
    
    // 手工计算 context[0, 0, 2, :] = sum_k probs[0,0,2,k] * v[0,0,k,:]
    int row = 2;
    std::vector<float> expected(4, 0.0f);
    for (int k = 0; k < 5; ++k) {
        float weight = probs_data[row * 5 + k];  // probs[2, k]
        for (int d = 0; d < 4; ++d) {
            expected[d] += weight * v_data[k * 4 + d];  // V[k, d]
        }
    }
    
    // 对比
    const float* ctx_data = context->data<float>();
    const float* ctx_row = ctx_data + row * 4;  // context[0, 0, 2, :]
    
    std::cout << "手工计算 context[2,:]: ";
    for (int d = 0; d < 4; ++d) printf("%.4f ", expected[d]);
    std::cout << "\nmatmul输出 context[2,:]: ";
    for (int d = 0; d < 4; ++d) printf("%.4f ", ctx_row[d]);
    std::cout << std::endl;
    
    bool pass = true;
    float max_err = 0.0f;
    for (int d = 0; d < 4; ++d) {
        float err = std::abs(ctx_row[d] - expected[d]);
        max_err = std::max(max_err, err);
        if (err > 1e-5f) {
            pass = false;
        }
    }
    
    std::cout << "max_err=" << max_err << std::endl;
    if (pass) {
        std::cout << "✅ PASS: 手工计算与matmul一致" << std::endl;
    } else {
        std::cout << "❌ FAIL: matmul内核实现有误" << std::endl;
    }
}

int main() {
    try {
        test_ones_v();
        test_identity_probs();
        test_manual_dotprod();
        
        std::cout << "\n========== 总结 ==========" << std::endl;
        std::cout << "如果 Test 1 FAIL 且输出≈0.2 → matmul在做平均（/K或/S）" << std::endl;
        std::cout << "如果 Test 2 FAIL → K轴错位或stride错误" << std::endl;
        std::cout << "如果 Test 3 FAIL → 内核实现有bug" << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "❌ Exception: " << e.what() << std::endl;
        return 1;
    }
}

