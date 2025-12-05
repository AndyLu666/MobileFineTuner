/**
 * @file test_lora_linear.cpp
 * @brief 最小 LoRALinear 前向 + 反向数值对齐测试，基准来自 PyTorch。
 */

#include "lora_linear.h"
#include "../core/ops.h"
#include <cmath>
#include <iostream>
#include <string>
#include <algorithm>
#include <vector>

using namespace ops;

namespace {

std::vector<float> transpose_data(const std::vector<float>& src, int rows, int cols) {
    std::vector<float> dst(static_cast<size_t>(rows * cols));
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            dst[c * rows + r] = src[r * cols + c];
        }
    }
    return dst;
}

struct Expected {
    std::vector<float> y;       // forward输出（等同于delta，base W=0）
    std::vector<float> grad_x;
    std::vector<float> grad_A;
    std::vector<float> grad_B;
};

struct DiffResult {
    float max_diff;
    float mean_diff;
};

DiffResult diff_tensor(const TensorPtr& t, const std::vector<float>& expected) {
    if (!t) {
        throw std::runtime_error("Tensor is null");
    }
    if (t->numel() != static_cast<int64_t>(expected.size())) {
        throw std::runtime_error("Size mismatch between tensor and expected data");
    }
    const float* data = t->data<float>();
    double sum = 0.0;
    double maxv = 0.0;
    for (size_t i = 0; i < expected.size(); ++i) {
        double d = std::abs(static_cast<double>(data[i]) - static_cast<double>(expected[i]));
        sum += d;
        if (d > maxv) maxv = d;
    }
    return DiffResult{static_cast<float>(maxv), static_cast<float>(sum / expected.size())};
}

bool check_close(const std::string& name, const TensorPtr& t,
                 const std::vector<float>& expected, float tol = 1e-5f) {
    auto diff = diff_tensor(t, expected);
    std::cout << name << " max_diff=" << diff.max_diff
              << " mean_diff=" << diff.mean_diff << std::endl;
    if (diff.max_diff > tol) {
        std::cerr << "[FAIL] " << name << " diff exceeds tol " << tol << std::endl;
        return false;
    }
    return true;
}

}  // namespace

struct Sample {
    TensorPtr x;      // [B,S,in]
    TensorPtr weight; // base W [in,out]
    TensorPtr A;      // [in,r] or [r,in]
    TensorPtr B;      // [r,out] or [out,r]
};

TensorPtr from_data(const std::vector<float>& values,
                    const std::vector<int64_t>& shape) {
    auto t = zeros(shape, kFloat32, kCPU);
    std::copy(values.begin(), values.end(), t->data<float>());
    return t;
}

Sample build_sample(int in, int out, int r, bool transpose_A, bool transpose_B,
                    const std::vector<float>& x_data,
                    const std::vector<float>& w_data,
                    const std::vector<float>& a_data_in_r,
                    const std::vector<float>& b_data_r_out,
                    int batch = 1, int seq = 2) {
    Sample s;
    s.x = from_data(x_data, {batch, seq, in});
    s.x->set_requires_grad(true);
    s.weight = from_data(w_data, {in, out});

    if (transpose_A) {
        auto a_t = transpose_data(a_data_in_r, in, r); // to [r, in]
        s.A = from_data(a_t, {r, in});
    } else {
        s.A = from_data(a_data_in_r, {in, r});
    }
    if (transpose_B) {
        auto b_t = transpose_data(b_data_r_out, r, out); // to [out, r]
        s.B = from_data(b_t, {out, r});
    } else {
        s.B = from_data(b_data_r_out, {r, out});
    }
    return s;
}

bool run_case(const std::string& name, bool transpose_A, bool transpose_B,
              const Expected& expected) {
    int in = 4, out = 3, r = 2;
    std::vector<float> x = {
        0.1f, 0.2f, 0.3f, 0.4f,
        0.5f, 0.6f, 0.7f, 0.8f
    }; // B=1, S=2
    std::vector<float> w(out * in, 0.0f); // base W=0，仅测试 LoRA 分支
    std::vector<float> A = {
        0.1f, 0.2f,
        -0.1f, 0.3f,
        0.0f, -0.2f,
        0.05f, 0.07f
    }; // [in, r]
    std::vector<float> B = {
        0.01f, -0.02f, 0.03f,
        -0.04f, 0.05f, -0.06f
    }; // [r, out]

    auto sample = build_sample(in, out, r, transpose_A, transpose_B, x, w, A, B);

    LoRALinear layer(sample.weight);
    layer.set_debug_name("test");
    layer.attach_lora(sample.A, sample.B, /*scale=*/1.0f, 0, out);

    auto y = layer.forward(sample.x);  // shape [1,2,3]
    auto loss = sum(y, -1, false);     // sum all -> scalar (1 elem)
    loss->backward();

    const float tol = 1e-5f;
    bool ok = true;
    ok &= check_close(name + "/y", y, expected.y, tol);
    ok &= check_close(name + "/grad_x", sample.x->grad(), expected.grad_x, tol);
    ok &= check_close(name + "/grad_A", sample.A->grad(), expected.grad_A, tol);
    ok &= check_close(name + "/grad_B", sample.B->grad(), expected.grad_B, tol);

    if (ok) {
        std::cout << "[PASS] " << name << std::endl;
    }
    return ok;
}

int main() {
    try {
        Expected base_exp{
            /*y=*/{-0.0018199998885393143f, 0.002199999988079071f, -0.002579999854788184f,
                   -0.007540000136941671f, 0.009200001135468483f, -0.010859999805688858f},
            /*grad_x=*/{-0.007999999448657036f, -0.016999999061226845f, 0.009999999776482582f, -0.0024999999441206455f,
                        -0.007999999448657036f, -0.016999999061226845f, 0.009999999776482582f, -0.0024999999441206455f},
            /*grad_A=*/{0.012000000104308128f, -0.029999997466802597f,
                        0.01600000075995922f, -0.03999999910593033f,
                        0.019999999552965164f, -0.04999999701976776f,
                        0.024000000208616257f, -0.05999999865889549f},
            /*grad_B=*/{0.03999999910593033f, 0.03999999910593033f, 0.03999999910593033f,
                        0.24400000274181366f, 0.24400000274181366f, 0.24400000274181366f}
        };

        Expected transposed_exp{
            base_exp.y,
            base_exp.grad_x,
            /*grad_A (A 为 [r,in])*/{0.012000000104308128f, 0.01600000075995922f,
                                      0.019999999552965164f, 0.024000000208616257f,
                                      -0.029999997466802597f, -0.03999999910593033f,
                                      -0.04999999701976776f, -0.05999999865889549f},
            /*grad_B (B 为 [out,r])*/{0.03999999910593033f, 0.24400000274181366f,
                                      0.03999999910593033f, 0.24400000274181366f,
                                      0.03999999910593033f, 0.24400000274181366f}
        };

        bool ok = true;
        ok &= run_case("A[in,r]_B[r,out]", /*transpose_A=*/false, /*transpose_B=*/false, base_exp);
        ok &= run_case("A[r,in]_B[out,r]", /*transpose_A=*/true, /*transpose_B=*/true, transposed_exp);

        if (ok) {
            std::cout << "[OK] LoRALinear forward/backward matches PyTorch reference." << std::endl;
            return 0;
        }
        std::cerr << "[FAIL] LoRALinear test mismatch." << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
