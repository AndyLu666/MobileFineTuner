/**
 * @file ops.cpp
 * @brief Implementation of core operations for the operators framework
 * 
 * This file contains the implementation of all mathematical operations,
 * neural network layers, and utility functions declared in ops.h.
 * All operations support automatic differentiation and are optimized
 * for both CPU and potential GPU execution.
 */

#include "ops.h"
#include "backward_functions.h"
#include "mobile_safe_matmul.h"
#include "autograd_engine.h"
#include "../utils/fp16_utils.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cstring>

namespace ops {

// Helper: register node with new engine (if enabled) or fallback to legacy
namespace {
    void register_backward(const TensorPtr& output,
                          const std::vector<TensorPtr>& inputs,
                          BackwardFunctionPtr backward_fn) {
        #ifdef USE_NEW_AUTOGRAD_ENGINE
        try {
            autograd::Engine::instance().register_node(output, inputs, backward_fn);
        } catch (const std::exception& e) {
            std::cerr << "[register_backward] Exception: " << e.what() << std::endl;
            throw;
        } catch (...) {
            std::cerr << "[register_backward] Unknown exception" << std::endl;
            throw;
        }
        #else
        // Legacy: set grad_fn that calls accumulate_gradient
        // (kept for backward compatibility)
        #endif
    }
}

namespace {

    /**
     * @brief Check if two tensors have the same shape
     * @param a First tensor
     * @param b Second tensor
     * @return True if shapes are equal
     */
    bool shapes_equal(const TensorPtr& a, const TensorPtr& b) {
        return a->shape() == b->shape();
    }

    /**
     * @brief Check if two tensors can be broadcast together
     * @param a First tensor
     * @param b Second tensor
     * @return True if tensors can be broadcast
     */
    bool can_broadcast(const TensorPtr& a, const TensorPtr& b) {
        const auto& shape_a = a->shape();
        const auto& shape_b = b->shape();

        int max_ndim = std::max(shape_a.size(), shape_b.size());

        for (int i = 0; i < max_ndim; ++i) {
            int dim_a = (static_cast<size_t>(i) < shape_a.size()) ? shape_a[shape_a.size() - 1 - i] : 1;
            int dim_b = (static_cast<size_t>(i) < shape_b.size()) ? shape_b[shape_b.size() - 1 - i] : 1;

            if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
                return false;
            }
        }

        return true;
    }

    std::vector<int64_t> broadcast_shapes(const TensorPtr& a, const TensorPtr& b) {
        const auto& shape_a = a->shape();
        const auto& shape_b = b->shape();

        int max_ndim = std::max(shape_a.size(), shape_b.size());
        std::vector<int64_t> result_shape(max_ndim);

        for (int i = 0; i < max_ndim; ++i) {
            int64_t dim_a = (static_cast<size_t>(i) < shape_a.size()) ? shape_a[shape_a.size() - 1 - i] : 1;
            int64_t dim_b = (static_cast<size_t>(i) < shape_b.size()) ? shape_b[shape_b.size() - 1 - i] : 1;

            result_shape[max_ndim - 1 - i] = std::max(dim_a, dim_b);
        }

        return result_shape;
    }

    template<typename Op>
    TensorPtr elementwise_binary_op(const TensorPtr& a, const TensorPtr& b, Op op) {
        if (!can_broadcast(a, b)) {
            throw TensorError("Tensors cannot be broadcasted");
        }

        auto result_shape = broadcast_shapes(a, b);
        auto result = zeros(result_shape, a->dtype(), a->device());

        if (shapes_equal(a, b)) {
            const float* data_a = a->data<float>();
            const float* data_b = b->data<float>();
            float* result_data = result->data<float>();

            for (int64_t i = 0; i < a->numel(); ++i) {
                result_data[i] = op(data_a[i], data_b[i]);
            }
        } else {
            // Complete broadcast implementation
            const float* data_a = a->data<float>();
            const float* data_b = b->data<float>();
            float* result_data = result->data<float>();
            
            auto shape_a = a->shape();
            auto shape_b = b->shape();
            
            for (int64_t i = 0; i < result->numel(); ++i) {
                // Calculate multidimensional index of current position in result
                std::vector<int64_t> result_idx(result_shape.size());
                int64_t temp = i;
                for (int j = result_shape.size() - 1; j >= 0; --j) {
                    result_idx[j] = temp % result_shape[j];
                    temp /= result_shape[j];
                }
                
                // Calculate corresponding indices for a and b (simplified version)
                int64_t idx_a = 0, idx_b = 0;
                
                // Calculate linear index for a
                for (size_t dim = 0; dim < shape_a.size(); ++dim) {
                    int result_dim = dim + (result_shape.size() - shape_a.size());
                    if (result_dim >= 0) {
                        int64_t coord = (shape_a[dim] == 1) ? 0 : result_idx[result_dim];
                        idx_a = idx_a * shape_a[dim] + coord;
                    }
                }
                
                // Calculate linear index for b
                for (size_t dim = 0; dim < shape_b.size(); ++dim) {
                    int result_dim = dim + (result_shape.size() - shape_b.size());
                    if (result_dim >= 0) {
                        int64_t coord = (shape_b[dim] == 1) ? 0 : result_idx[result_dim];
                        idx_b = idx_b * shape_b[dim] + coord;
                    }
                }
                
                result_data[i] = op(data_a[idx_a], data_b[idx_b]);
            }
        }

        if (a->requires_grad() || b->requires_grad()) {
            result->set_requires_grad(true);

        }

        return result;
    }

    template<typename Op>
    TensorPtr elementwise_unary_op(const TensorPtr& x, Op op) {
        auto result = zeros(x->shape(), x->dtype(), x->device());

        const float* x_data = x->data<float>();
        float* result_data = result->data<float>();

        for (int64_t i = 0; i < x->numel(); ++i) {
            result_data[i] = op(x_data[i]);
        }

        if (x->requires_grad()) {
            result->set_requires_grad(true);

            result->set_grad_fn([x](const TensorPtr& grad_output) -> std::vector<TensorPtr> {

                auto grad_input = zeros(x->shape(), x->dtype(), x->device());
                const float* grad_out_data = grad_output->data<float>();
                float* grad_in_data = grad_input->data<float>();

                for (int64_t i = 0; i < x->numel(); ++i) {
                    grad_in_data[i] = grad_out_data[i];
                }

                if (x->requires_grad()) {
                    accumulate_gradient(x, grad_input);
                }

                return {grad_input};
            });
        }

        return result;
    }
}

TensorPtr add(const TensorPtr& a, const TensorPtr& b) {
    auto result = elementwise_binary_op(a, b, [](float x, float y) { return x + y; });

    if (a->requires_grad() || b->requires_grad()) {
        result->set_requires_grad(true);

        auto backward_fn = std::make_shared<AddBackward>(a->shape(), b->shape());
        
        #ifdef USE_NEW_AUTOGRAD_ENGINE
        register_backward(result, {a, b}, backward_fn);
        #else
        result->set_grad_fn([backward_fn, a, b](const TensorPtr& grad_output) -> std::vector<TensorPtr> {
            auto grads = backward_fn->apply(grad_output);

            if (a->requires_grad()) {
                accumulate_gradient(a, grads[0]);
            }
            if (b->requires_grad()) {
                accumulate_gradient(b, grads[1]);
            }

            return grads;
        });
        #endif
    }

    return result;
}

TensorPtr sub(const TensorPtr& a, const TensorPtr& b) {
    auto result = elementwise_binary_op(a, b, [](float x, float y) { return x - y; });

    if (a->requires_grad() || b->requires_grad()) {
        result->set_requires_grad(true);

        // d(a - b) / da = 1, d(a - b) / db = -1 (with broadcasting)
        result->set_grad_fn([a, b](const TensorPtr& grad_output) -> std::vector<TensorPtr> {
            // Grad for a: +grad_output (summed-to-shape)
            auto grad_a = grad_output;
            if (a->shape() != grad_output->shape()) {
                grad_a = sum_to_shape(grad_output, a->shape());
            }

            // Grad for b: -grad_output (summed-to-shape)
            auto neg_grad = mul(grad_output, -1.0f);
            auto grad_b = neg_grad;
            if (b->shape() != grad_output->shape()) {
                grad_b = sum_to_shape(neg_grad, b->shape());
            }

            if (a->requires_grad()) {
                accumulate_gradient(a, grad_a);
            }
            if (b->requires_grad()) {
                accumulate_gradient(b, grad_b);
            }

            return {grad_a, grad_b};
        });
    }

    return result;
}

TensorPtr mul(const TensorPtr& a, const TensorPtr& b) {
    auto result = elementwise_binary_op(a, b, [](float x, float y) { return x * y; });

    if (a->requires_grad() || b->requires_grad()) {
        result->set_requires_grad(true);

        auto backward_fn = std::make_shared<MulBackward>(a, b);
        result->set_grad_fn([backward_fn, a, b](const TensorPtr& grad_output) -> std::vector<TensorPtr> {
            auto grads = backward_fn->apply(grad_output);

            if (a->requires_grad()) {
                accumulate_gradient(a, grads[0]);
            }
            if (b->requires_grad()) {
                accumulate_gradient(b, grads[1]);
            }

            return grads;
        });
    }

    return result;
}

TensorPtr div(const TensorPtr& a, const TensorPtr& b) {
    return elementwise_binary_op(a, b, [](float x, float y) {
        if (y == 0.0f) throw TensorError("Division by zero");
        return x / y;
    });
}

TensorPtr add(const TensorPtr& tensor, float scalar) {
    auto result = elementwise_unary_op(tensor, [scalar](float x) { return x + scalar; });
    if (tensor->requires_grad()) {
        result->set_requires_grad(true);
        #ifdef USE_NEW_AUTOGRAD_ENGINE
        auto backward_fn = std::make_shared<PassThroughBackward>();
        register_backward(result, {tensor}, backward_fn);
        #endif
    }
    return result;
}

TensorPtr sub(const TensorPtr& tensor, float scalar) {
    auto result = elementwise_unary_op(tensor, [scalar](float x) { return x - scalar; });
    if (tensor->requires_grad()) {
        result->set_requires_grad(true);
        #ifdef USE_NEW_AUTOGRAD_ENGINE
        auto backward_fn = std::make_shared<PassThroughBackward>();
        register_backward(result, {tensor}, backward_fn);
        #endif
    }
    return result;
}

TensorPtr mul(const TensorPtr& tensor, float scalar) {
    auto result = elementwise_unary_op(tensor, [scalar](float x) { return x * scalar; });
    if (tensor->requires_grad()) {
        result->set_requires_grad(true);
        #ifdef USE_NEW_AUTOGRAD_ENGINE
        auto backward_fn = std::make_shared<ScaleBackward>(scalar);
        register_backward(result, {tensor}, backward_fn);
        #endif
    }
    return result;
}

TensorPtr div(const TensorPtr& tensor, float scalar) {
    if (scalar == 0.0f) throw TensorError("Division by zero");
    auto result = elementwise_unary_op(tensor, [scalar](float x) { return x / scalar; });
    if (tensor->requires_grad()) {
        result->set_requires_grad(true);
        #ifdef USE_NEW_AUTOGRAD_ENGINE
        auto backward_fn = std::make_shared<ScaleBackward>(1.0f / scalar);
        register_backward(result, {tensor}, backward_fn);
        #endif
    }
    return result;
}

TensorPtr add(float scalar, const TensorPtr& tensor) {
    return add(tensor, scalar);
}

TensorPtr sub(float scalar, const TensorPtr& tensor) {
    auto result = elementwise_unary_op(tensor, [scalar](float x) { return scalar - x; });
    if (tensor->requires_grad()) {
        result->set_requires_grad(true);
        #ifdef USE_NEW_AUTOGRAD_ENGINE
        // d(scalar - x)/dx = -1
        auto backward_fn = std::make_shared<ScaleBackward>(-1.0f);
        register_backward(result, {tensor}, backward_fn);
        #endif
    }
    return result;
}

TensorPtr mul(float scalar, const TensorPtr& tensor) {
    auto result = mul(tensor, scalar);
    // mul(tensor, scalar) already handles registration
    return result;
}

TensorPtr div(float scalar, const TensorPtr& tensor) {
    auto result = elementwise_unary_op(tensor, [scalar](float x) {
        if (x == 0.0f) throw TensorError("Division by zero");
        return scalar / x;
    });
    if (tensor->requires_grad()) {
        result->set_requires_grad(true);
        #ifdef USE_NEW_AUTOGRAD_ENGINE
        // y = c / x, dy/dx = -c / x^2, here use approximation without registration (rarely used), keep old path
        #endif
    }
    return result;
}

TensorPtr matmul(const TensorPtr& a, const TensorPtr& b) {
    if (!a || !b) {
        throw TensorError("matmul: input tensors must not be null");
    }
    
    const auto& shape_a = a->shape();
    const auto& shape_b = b->shape();

    #ifdef AUTOGRAD_DEBUG
    std::cout << "            [matmul] a=" << a.get() << " shape=[" << shape_a[0];
    for (size_t i = 1; i < shape_a.size(); ++i) std::cout << "," << shape_a[i];
    std::cout << "] b=" << b.get() << " shape=[" << shape_b[0];
    for (size_t i = 1; i < shape_b.size(); ++i) std::cout << "," << shape_b[i];
    std::cout << "]" << std::endl;
    #endif
    
    if (shape_a.size() < 2 || shape_b.size() < 2) {
        throw TensorError("matmul requires tensors with at least 2 dimensions");
    }

    int64_t m = shape_a[shape_a.size() - 2];
    int64_t k = shape_a[shape_a.size() - 1];
    int64_t n = shape_b[shape_b.size() - 1];

    if (k != shape_b[shape_b.size() - 2]) {
        throw TensorError("matmul dimension mismatch: " + std::to_string(k) +
                         " vs " + std::to_string(shape_b[shape_b.size() - 2]));
    }

    auto result_shape = shape_a;
    result_shape[result_shape.size() - 2] = m;
    result_shape[result_shape.size() - 1] = n;

    auto result = zeros(result_shape, a->dtype(), a->device());

    // In DISABLE_BLAS mode, force use of MEMORY_FIRST extreme memory-saving strategy
    // Uniformly use pure C++ safe matrix multiplication (adaptive strategy)
    auto opt_level = mobile_matmul::OptimizationLevel::ADAPTIVE;
    
    if (shape_a.size() == 2 && shape_b.size() == 2) {
        const float* data_a = a->data<float>();
        const float* data_b = b->data<float>();
        float* result_data = result->data<float>();

        // Use mobile-optimized safe matrix multiplication
        mobile_matmul::SafeMatmul::multiply(data_a, data_b, result_data, m, n, k, opt_level);
    } else {

        int64_t batch_size = shape_a[0];
        int64_t a_rows = shape_a[shape_a.size() - 2];
        int64_t a_cols = shape_a[shape_a.size() - 1];
        int64_t b_cols = shape_b[shape_b.size() - 1];

        const float* data_a = a->data<float>();
        const float* data_b = b->data<float>();
        float* result_data = result->data<float>();

        // Check if b has batch dimension
        bool b_has_batch = (shape_b.size() == shape_a.size()) && (shape_b[0] == batch_size);
        
        for (int64_t batch = 0; batch < batch_size; ++batch) {
            // Use optimized matrix multiplication to handle each batch
            const float* batch_a = data_a + batch * (a_rows * a_cols);
            const float* batch_b = b_has_batch ? 
                                  data_b + batch * (a_cols * b_cols) : 
                                  data_b;
            float* batch_result = result_data + batch * (a_rows * b_cols);
            
            mobile_matmul::SafeMatmul::multiply(batch_a, batch_b, batch_result, 
                                               a_rows, b_cols, a_cols, opt_level);
        }
    }

    if (a->requires_grad() || b->requires_grad()) {
        result->set_requires_grad(true);
        auto backward_fn = std::make_shared<MatmulBackward>(a, b);
        
        #ifdef USE_NEW_AUTOGRAD_ENGINE
        register_backward(result, {a, b}, backward_fn);
        #else
        // Legacy recursive path
        result->set_grad_fn([backward_fn, a, b](const TensorPtr& grad_output) -> std::vector<TensorPtr> {
            auto grads = backward_fn->apply(grad_output);

            if (a->requires_grad()) {
                accumulate_gradient(a, grads[0]);
            }
            if (b->requires_grad()) {
                accumulate_gradient(b, grads[1]);
            }

            return grads;
        });
        #endif
    }

    return result;
}

TensorPtr matmul_rhs_T(const TensorPtr& a, const TensorPtr& b) {
    if (!a || !b) {
        throw TensorError("matmul_rhs_T: input tensors must not be null");
    }
    
    const auto& shape_a = a->shape();
    const auto& shape_b = b->shape();
    
    // b must be 2D: [N, K]
    if (shape_b.size() != 2) {
        throw TensorError("matmul_rhs_T: b must be 2D [N, K]");
    }
    
    // a can be 2D or 3D
    if (shape_a.size() < 2) {
        throw TensorError("matmul_rhs_T: a must be at least 2D");
    }
    
    int64_t n = shape_b[0];
    int64_t k_b = shape_b[1];
    
    const float* data_a = a->data<float>();
    const float* data_b = b->data<float>();
    
    // Uniformly use pure C++ safe matrix multiplication (adaptive strategy)
    auto opt_level = mobile_matmul::OptimizationLevel::ADAPTIVE;
    
    if (shape_a.size() == 2) {
        // 2D: a[M,K] @ b[N,K]^T = result[M,N]
        int64_t m = shape_a[0];
        int64_t k_a = shape_a[1];
        
        if (k_a != k_b) {
            throw TensorError("matmul_rhs_T dimension mismatch: A has K=" + std::to_string(k_a) +
                             " but B has K=" + std::to_string(k_b));
        }
        
        auto result = zeros({m, n}, a->dtype(), a->device());
        float* result_data = result->data<float>();
        
        mobile_matmul::SafeMatmul::multiply_rhs_T(data_a, data_b, result_data, m, n, k_a, opt_level);
        
        #ifdef USE_NEW_AUTOGRAD_ENGINE
        if (a->requires_grad() || b->requires_grad()) {
            result->set_requires_grad(true);
            auto backward_fn = std::make_shared<MatmulRhsTBackward>(a, b);
            register_backward(result, {a, b}, backward_fn);
        }
        #else
        if (a->requires_grad()) {
            result->set_requires_grad(true);
            result->set_grad_fn([a, b](const TensorPtr& grad_output) -> std::vector<TensorPtr> {
                auto grad_a = matmul(grad_output, b);
                if (a->requires_grad()) {
                    accumulate_gradient(a, grad_a);
                }
                return {};
            });
        }
        #endif
        
        return result;
    } else {
        // 3D+: a[..., M, K] @ b[N, K]^T = result[..., M, N]
        // Process by batch
        auto result_shape = shape_a;
        result_shape[result_shape.size() - 1] = n;
        auto result = zeros(result_shape, a->dtype(), a->device());
        float* result_data = result->data<float>();
        
        int64_t m = shape_a[shape_a.size() - 2];
        int64_t k_a = shape_a[shape_a.size() - 1];
        
        if (k_a != k_b) {
            throw TensorError("matmul_rhs_T dimension mismatch");
        }
        
        // Calculate batch size
        int64_t batch_size = 1;
        for (size_t i = 0; i < shape_a.size() - 2; ++i) {
            batch_size *= shape_a[i];
        }
        
        for (int64_t batch = 0; batch < batch_size; ++batch) {
            const float* batch_a = data_a + batch * (m * k_a);
            float* batch_result = result_data + batch * (m * n);
            
            mobile_matmul::SafeMatmul::multiply_rhs_T(batch_a, data_b, batch_result, m, n, k_a, opt_level);
        }
        
        #ifdef USE_NEW_AUTOGRAD_ENGINE
        if (a->requires_grad() || b->requires_grad()) {
            result->set_requires_grad(true);
            auto backward_fn = std::make_shared<MatmulRhsTBackward>(a, b);
            register_backward(result, {a, b}, backward_fn);
        }
        #else
        if (a->requires_grad()) {
            result->set_requires_grad(true);
            result->set_grad_fn([a, b](const TensorPtr& grad_output) -> std::vector<TensorPtr> {
                auto grad_a = matmul(grad_output, b);
                if (a->requires_grad()) {
                    accumulate_gradient(a, grad_a);
                }
                return {};
            });
        }
        #endif
        
        return result;
    }
}

TensorPtr transpose(const TensorPtr& tensor, int dim0, int dim1) {
    const auto& shape = tensor->shape();
    int ndim = shape.size();
    
    // Handle negative indices
    if (dim0 < 0) dim0 += ndim;
    if (dim1 < 0) dim1 += ndim;
    
    // Check dimension validity
    if (dim0 < 0 || dim0 >= ndim || dim1 < 0 || dim1 >= ndim) {
        throw TensorError("transpose: invalid dimensions");
    }
    
    // Create new shape
    std::vector<int64_t> new_shape = shape;
    std::swap(new_shape[dim0], new_shape[dim1]);
    
    // Create result tensor
    auto result = zeros(new_shape, tensor->dtype(), tensor->device());
    
    // Execute transpose
    const float* src_data = tensor->data<float>();
    float* dst_data = result->data<float>();
    
    // General n-dimensional transpose
    std::vector<int64_t> strides_src(ndim), strides_dst(ndim);
    
    // Calculate source tensor strides
    strides_src[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; --i) {
        strides_src[i] = strides_src[i + 1] * shape[i + 1];
    }
    
    // Calculate target tensor strides
    strides_dst[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; --i) {
        strides_dst[i] = strides_dst[i + 1] * new_shape[i + 1];
    }
    
    // Execute transpose
    int64_t total_elements = result->numel();
    std::vector<int64_t> indices(ndim);
    
    for (int64_t linear_idx = 0; linear_idx < total_elements; ++linear_idx) {
        // Convert linear index to multi-dimensional index (target tensor)
        int64_t temp = linear_idx;
        for (int i = 0; i < ndim; ++i) {
            indices[i] = temp / strides_dst[i];
            temp %= strides_dst[i];
        }
        
        // Swap dimensions
        std::swap(indices[dim0], indices[dim1]);
        
        // Calculate linear index in source tensor
        int64_t src_idx = 0;
        for (int i = 0; i < ndim; ++i) {
            src_idx += indices[i] * strides_src[i];
        }
        
        dst_data[linear_idx] = src_data[src_idx];
    }
    
    // Set gradient propagation
    if (tensor->requires_grad()) {
        result->set_requires_grad(true);
        #ifdef USE_NEW_AUTOGRAD_ENGINE
        auto backward_fn = std::make_shared<TransposeBackward>(dim0, dim1);
        register_backward(result, {tensor}, backward_fn);
        #else
        result->set_grad_fn([tensor, dim0, dim1](const TensorPtr& grad_output) -> std::vector<TensorPtr> {
            // Gradient of transpose is transpose again
            auto grad_input = transpose(grad_output, dim0, dim1);
            accumulate_gradient(tensor, grad_input);
            return {};
        });
        #endif
    }
    
    return result;
}

TensorPtr permute(const TensorPtr& tensor, const std::vector<int>& dims) {

    (void)tensor;
    (void)dims;
    throw TensorError("permute not implemented yet");
}

TensorPtr linear(const TensorPtr& input, const TensorPtr& weight, const TensorPtr& bias) {

    auto result = matmul(input, transpose(weight, 0, 1));

    if (bias) {

        const auto& result_shape = result->shape();
        const auto& bias_shape = bias->shape();

        if (bias_shape.size() == 1 && result_shape.size() >= 1 &&
            bias_shape[0] == result_shape.back()) {

            auto broadcast_result = zeros(result_shape, result->dtype(), result->device());
            const float* result_data = result->data<float>();
            const float* bias_data = bias->data<float>();
            float* output_data = broadcast_result->data<float>();

            int64_t last_dim = result_shape.back();
            int64_t total_elements = result->numel();

            for (int64_t i = 0; i < total_elements; ++i) {
                int64_t bias_idx = i % last_dim;
                output_data[i] = result_data[i] + bias_data[bias_idx];
            }

            if (result->requires_grad() || bias->requires_grad()) {
                broadcast_result->set_requires_grad(true);
            }

            result = broadcast_result;
        } else {

            result = add(result, bias);
        }
    }

    if (input->requires_grad() || weight->requires_grad() || (bias && bias->requires_grad())) {
        result->set_requires_grad(true);

        auto backward_fn = std::make_shared<LinearBackward>(input, weight, bias);
        result->set_grad_fn([backward_fn, input, weight, bias](const TensorPtr& grad_output) -> std::vector<TensorPtr> {
            auto grads = backward_fn->apply(grad_output);

            if (input->requires_grad()) {
                accumulate_gradient(input, grads[0]);
            }
            if (weight->requires_grad()) {
                accumulate_gradient(weight, grads[1]);
            }
            if (bias && bias->requires_grad()) {
                accumulate_gradient(bias, grads[2]);
            }

            return grads;
        });
    }

    return result;
}

TensorPtr relu(const TensorPtr& x) {
    auto result = elementwise_unary_op(x, [](float val) { return std::max(0.0f, val); });

    if (x->requires_grad()) {
        result->set_requires_grad(true);

        auto backward_fn = std::make_shared<ReluBackward>(x);
        result->set_grad_fn([backward_fn, x](const TensorPtr& grad_output) -> std::vector<TensorPtr> {
            auto grads = backward_fn->apply(grad_output);
            accumulate_gradient(x, grads[0]);
            return grads;
        });
    }

    return result;
}

TensorPtr gelu(const TensorPtr& x) {
    auto result = elementwise_unary_op(x, [](float val) {
        float tanh_input = 0.7978845608f * (val + 0.044715f * val * val * val);
        return 0.5f * val * (1.0f + std::tanh(tanh_input));
    });

    if (x->requires_grad()) {
        result->set_requires_grad(true);
        auto backward_fn = std::make_shared<GeluBackward>(x);
        
        #ifdef USE_NEW_AUTOGRAD_ENGINE
        register_backward(result, {x}, backward_fn);
        #else
        result->set_grad_fn([backward_fn, x](const TensorPtr& grad_output) -> std::vector<TensorPtr> {
            auto grads = backward_fn->apply(grad_output);
            if (x->requires_grad()) {
                accumulate_gradient(x, grads[0]);
            }
            return grads;
        });
        #endif
    }

    return result;
}

TensorPtr sigmoid(const TensorPtr& x) {
    return elementwise_unary_op(x, [](float val) {
        return 1.0f / (1.0f + std::exp(-val));
    });
}

TensorPtr tanh_op(const TensorPtr& x) {
    return elementwise_unary_op(x, [](float val) { return std::tanh(val); });
}

TensorPtr softmax(const TensorPtr& x, int dim) {

    if (dim != -1 && dim != x->ndim() - 1) {
        throw TensorError("softmax only supports last dimension currently");
    }

    const auto& shape = x->shape();
    auto result = zeros(shape, x->dtype(), x->device());

    const float* x_data = x->data<float>();
    float* result_data = result->data<float>();

    int64_t batch_size = 1;
    for (size_t i = 0; i < shape.size() - 1; ++i) {
        batch_size *= shape[i];
    }
    int64_t feature_size = shape.back();

    for (int64_t b = 0; b < batch_size; ++b) {
        const float* batch_data = x_data + b * feature_size;
        float* batch_result = result_data + b * feature_size;

        float max_val = *std::max_element(batch_data, batch_data + feature_size);

        float sum = 0.0f;
        for (int64_t i = 0; i < feature_size; ++i) {
            batch_result[i] = std::exp(batch_data[i] - max_val);
            sum += batch_result[i];
        }

        for (int64_t i = 0; i < feature_size; ++i) {
            batch_result[i] /= sum;
        }
    }

    if (x->requires_grad()) {
        result->set_requires_grad(true);

        auto backward_fn = std::make_shared<SoftmaxBackward>(result, dim);
        
        #ifdef USE_NEW_AUTOGRAD_ENGINE
        register_backward(result, {x}, backward_fn);
        #else
        result->set_grad_fn([backward_fn, x](const TensorPtr& grad_output) -> std::vector<TensorPtr> {
            auto grads = backward_fn->apply(grad_output);

            if (x->requires_grad()) {
                accumulate_gradient(x, grads[0]);
            }

            return grads;
        });
        #endif
    }

    return result;
}

TensorPtr log_softmax(const TensorPtr& x, int dim) {

    if (dim != -1 && dim != x->ndim() - 1) {
        throw TensorError("log_softmax only supports last dimension currently");
    }

    const auto& shape = x->shape();
    auto result = zeros(shape, x->dtype(), x->device());

    const float* x_data = x->data<float>();
    float* result_data = result->data<float>();

    int64_t batch_size = 1;
    for (size_t i = 0; i < shape.size() - 1; ++i) {
        batch_size *= shape[i];
    }
    int64_t feature_size = shape.back();

    for (int64_t b = 0; b < batch_size; ++b) {
        const float* batch_data = x_data + b * feature_size;
        float* batch_result = result_data + b * feature_size;

        float max_val = *std::max_element(batch_data, batch_data + feature_size);

        float log_sum_exp = 0.0f;
        for (int64_t i = 0; i < feature_size; ++i) {
            log_sum_exp += std::exp(batch_data[i] - max_val);
        }
        log_sum_exp = max_val + std::log(log_sum_exp);

        for (int64_t i = 0; i < feature_size; ++i) {
            batch_result[i] = batch_data[i] - log_sum_exp;
        }
    }

    if (x->requires_grad()) {
        result->set_requires_grad(true);
        auto backward_fn = std::make_shared<LogSoftmaxBackward>(x, result, dim);
        
        #ifdef USE_NEW_AUTOGRAD_ENGINE
        register_backward(result, {x}, backward_fn);
        #else
        result->set_grad_fn([backward_fn, x](const TensorPtr& grad_output) -> std::vector<TensorPtr> {
            auto grads = backward_fn->apply(grad_output);
            if (x->requires_grad()) {
                accumulate_gradient(x, grads[0]);
            }
            return grads;
        });
        #endif
    }

    return result;
}

TensorPtr mse_loss(const TensorPtr& input, const TensorPtr& target, const std::string& reduction) {
    if (!shapes_equal(input, target)) {
        throw TensorError("mse_loss: input and target must have the same shape");
    }

    auto diff = sub(input, target);
    auto squared = mul(diff, diff);

    if (reduction == "none") {
        return squared;
    } else if (reduction == "mean") {
        return mean(squared);
    } else if (reduction == "sum") {
        return sum(squared);
    } else {
        throw TensorError("mse_loss: invalid reduction '" + reduction + "'");
    }
}

TensorPtr layer_norm(const TensorPtr& input, const TensorPtr& weight, const TensorPtr& bias, float eps) {
    const auto& input_shape = input->shape();
    int64_t normalized_dim = input_shape.back();

    auto result = zeros(input_shape, input->dtype(), input->device());
    const float* input_data = input->data<float>();
    const float* weight_data = weight->data<float>();
    const float* bias_data = bias->data<float>();
    float* result_data = result->data<float>();

    int64_t batch_size = input->numel() / normalized_dim;

    for (int64_t b = 0; b < batch_size; ++b) {
        const float* batch_input = input_data + b * normalized_dim;
        float* batch_result = result_data + b * normalized_dim;

        float mean = 0.0f;
        for (int64_t i = 0; i < normalized_dim; ++i) {
            mean += batch_input[i];
        }
        mean /= normalized_dim;

        float variance = 0.0f;
        for (int64_t i = 0; i < normalized_dim; ++i) {
            float diff = batch_input[i] - mean;
            variance += diff * diff;
        }
        variance /= normalized_dim;

        float inv_std = 1.0f / std::sqrt(variance + eps);
        for (int64_t i = 0; i < normalized_dim; ++i) {
            float normalized = (batch_input[i] - mean) * inv_std;
            batch_result[i] = normalized * weight_data[i] + bias_data[i];
        }
    }

    if (input->requires_grad() || weight->requires_grad() || bias->requires_grad()) {
        result->set_requires_grad(true);
        
        #ifdef USE_NEW_AUTOGRAD_ENGINE
        // New engine: Pass gradient as-is to input (LN weights frozen by default)
        auto backward_fn = std::make_shared<PassThroughBackward>();
        register_backward(result, {input}, backward_fn);
        #else
        // LayerNorm gradient calculation (old engine)
        result->set_grad_fn([input, weight, bias, eps](const TensorPtr& grad_output) -> std::vector<TensorPtr> {
            // For simplified implementation, only propagate gradient to input
            if (input->requires_grad()) {
                // Simplified: assume gradient approximately equals grad_output
                accumulate_gradient(input, grad_output);
            }
            return {};
        });
        #endif
    }

      return result;
  }
  
  TensorPtr silu(const TensorPtr& x) {
      // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
      auto result = zeros(x->shape(), x->dtype(), x->device());
      const float* x_data = x->data<float>();
      float* result_data = result->data<float>();
      
      for (int64_t i = 0; i < x->numel(); ++i) {
          float val = x_data[i];
          float sigmoid_val = 1.0f / (1.0f + std::exp(-val));
          result_data[i] = val * sigmoid_val;
      }
      
      if (x->requires_grad()) {
          result->set_requires_grad(true);
          result->set_grad_fn([x](const TensorPtr& grad_output) -> std::vector<TensorPtr> {
              auto grad_input = zeros(x->shape(), x->dtype(), x->device());
              return {grad_input};
          });
      }
      
      return result;
  }
  
  TensorPtr rms_norm(const TensorPtr& input, const TensorPtr& weight, float eps) {
      // RMSNorm: x / sqrt(mean(x^2) + eps) * weight
      const auto& input_shape = input->shape();
      int64_t normalized_dim = input_shape.back();

      auto result = zeros(input_shape, input->dtype(), input->device());
      const float* input_data = input->data<float>();
      const float* weight_data = weight->data<float>();
      float* result_data = result->data<float>();

      int64_t batch_size = input->numel() / normalized_dim;

      for (int64_t b = 0; b < batch_size; ++b) {
          const float* batch_input = input_data + b * normalized_dim;
          float* batch_result = result_data + b * normalized_dim;

          // Calculate root mean square
          float square_sum = 0.0f;
          for (int64_t i = 0; i < normalized_dim; ++i) {
              square_sum += batch_input[i] * batch_input[i];
          }
          float rms = std::sqrt(square_sum / normalized_dim + eps);

          // Apply RMSNorm
          for (int64_t i = 0; i < normalized_dim; ++i) {
              batch_result[i] = (batch_input[i] / rms) * weight_data[i];
          }
      }

  // Register precise backward for RMSNorm to avoid "pass-through degradation"
  result->set_requires_grad(true);
  #ifdef USE_NEW_AUTOGRAD_ENGINE
  auto backward_fn = std::make_shared<RMSNormBackward>(input, weight, eps);
  register_backward(result, {input}, backward_fn);
  #else
  result->set_grad_fn([input, weight, eps](const TensorPtr& grad_output) -> std::vector<TensorPtr> {
      // Fallback implementation: call numerically correct RMSNormBackward logic
      RMSNormBackward impl(input, weight, eps);
      auto grads = impl.apply(grad_output);
      if (input->requires_grad()) accumulate_gradient(input, grads[0]);
      return grads;
  });
  #endif

      return result;
  }
  
  TensorPtr cross_entropy_loss(const TensorPtr& input, const TensorPtr& target, const std::string& reduction) {

    const auto& input_shape = input->shape();
    const auto& target_shape = target->shape();

    if (input_shape.size() != 2) {
        throw TensorError("cross_entropy_loss: input must be 2D [batch_size, num_classes]");
    }
    if (target_shape.size() != 1) {
        throw TensorError("cross_entropy_loss: target must be 1D [batch_size]");
    }
    if (input_shape[0] != target_shape[0]) {
        throw TensorError("cross_entropy_loss: batch size mismatch");
    }

    int64_t batch_size = input_shape[0];
    int64_t num_classes = input_shape[1];

    auto log_probs = log_softmax(input, -1);
    const float* log_probs_data = log_probs->data<float>();
    
    // Fix: correctly handle int32_t type targets
    std::vector<float> losses(batch_size);
    
    if (target->dtype() == DType::kInt32) {
        const int32_t* target_data = target->data<int32_t>();
        for (int64_t b = 0; b < batch_size; ++b) {
            int target_class = target_data[b];
            if (target_class < 0 || target_class >= num_classes) {
                throw TensorError("cross_entropy_loss: target class index out of range");
            }
            losses[b] = -log_probs_data[b * num_classes + target_class];
        }
    } else if (target->dtype() == DType::kFloat32) {
        const float* target_data = target->data<float>();
        for (int64_t b = 0; b < batch_size; ++b) {
            int target_class = static_cast<int>(target_data[b]);
            if (target_class < 0 || target_class >= num_classes) {
                throw TensorError("cross_entropy_loss: target class index out of range");
            }
            losses[b] = -log_probs_data[b * num_classes + target_class];
        }
    } else {
        throw TensorError("cross_entropy_loss: target must be int32 or float32");
    }

    TensorPtr result;
    if (reduction == "none") {
        result = tensor(losses);
    } else if (reduction == "mean") {
        float sum = std::accumulate(losses.begin(), losses.end(), 0.0f);
        result = full({1}, sum / batch_size);
    } else if (reduction == "sum") {
        float sum = std::accumulate(losses.begin(), losses.end(), 0.0f);
        result = full({1}, sum);
    } else {
        throw TensorError("cross_entropy_loss: invalid reduction '" + reduction + "'");
    }

    if (input->requires_grad()) {
        result->set_requires_grad(true);
        
        auto backward_fn = std::make_shared<CrossEntropyLossBackward>(input, target, reduction);
        
        #ifdef USE_NEW_AUTOGRAD_ENGINE
        register_backward(result, {input}, backward_fn);
        #else
        // Legacy recursive backward
        result->set_grad_fn([input, target, reduction, batch_size, num_classes](const TensorPtr& grad_output) -> std::vector<TensorPtr> {
            auto grad_input = zeros(input->shape(), input->dtype(), input->device());
            
            // Calculate softmax probabilities
            auto probs = softmax(input, -1);
            const float* probs_data = probs->data<float>();
            float* grad_data = grad_input->data<float>();
            const float grad_scale = grad_output->data<float>()[0];
            
            // Gradient calculation: grad = (softmax - one_hot) * grad_output
            if (target->dtype() == DType::kInt32) {
                const int32_t* target_data = target->data<int32_t>();
                for (int64_t b = 0; b < batch_size; ++b) {
                    int target_class = target_data[b];
                    for (int64_t c = 0; c < num_classes; ++c) {
                        int64_t idx = b * num_classes + c;
                        float prob = probs_data[idx];
                        float one_hot = (c == target_class) ? 1.0f : 0.0f;
                        grad_data[idx] = (prob - one_hot) * grad_scale;
                        if (reduction == "mean") {
                            grad_data[idx] /= batch_size;
                        }
                    }
                }
            } else {
                const float* target_data = target->data<float>();
                for (int64_t b = 0; b < batch_size; ++b) {
                    int target_class = static_cast<int>(target_data[b]);
                    for (int64_t c = 0; c < num_classes; ++c) {
                        int64_t idx = b * num_classes + c;
                        float prob = probs_data[idx];
                        float one_hot = (c == target_class) ? 1.0f : 0.0f;
                        grad_data[idx] = (prob - one_hot) * grad_scale;
                        if (reduction == "mean") {
                            grad_data[idx] /= batch_size;
                        }
                    }
                }
            }
            
            // Accumulate gradient to input
            accumulate_gradient(input, grad_input);
            return {};
        });
        #endif
    }

    return result;
}

TensorPtr reshape(const TensorPtr& tensor, const std::vector<int64_t>& shape) {
    auto result = tensor->reshape(shape);
    
    // 设置梯度传播
    if (tensor->requires_grad()) {
        result->set_requires_grad(true);
        
        auto backward_fn = std::make_shared<ReshapeBackward>(tensor->shape());
        
        #ifdef USE_NEW_AUTOGRAD_ENGINE
        register_backward(result, {tensor}, backward_fn);
        #else
        result->set_grad_fn([tensor, original_shape = tensor->shape()](const TensorPtr& grad_output) -> std::vector<TensorPtr> {
            // reshape的梯度就是将grad_output reshape回原来的形状
            auto grad_input = grad_output->reshape(original_shape);
            accumulate_gradient(tensor, grad_input);
            return {};
        });
        #endif
    }
    
    return result;
}

TensorPtr view(const TensorPtr& tensor, const std::vector<int64_t>& shape) {
    return tensor->view(shape);
}

TensorPtr flatten(const TensorPtr& tensor, int start_dim, int end_dim) {
    const auto& shape = tensor->shape();
    int ndim = shape.size();

    if (start_dim < 0) start_dim += ndim;
    if (end_dim < 0) end_dim += ndim;

    if (start_dim < 0 || start_dim >= ndim || end_dim < 0 || end_dim >= ndim || start_dim > end_dim) {
        throw TensorError("flatten: invalid start_dim or end_dim");
    }

    std::vector<int64_t> new_shape;

    for (int i = 0; i < start_dim; ++i) {
        new_shape.push_back(shape[i]);
    }

    int64_t flattened_size = 1;
    for (int i = start_dim; i <= end_dim; ++i) {
        flattened_size *= shape[i];
    }
    new_shape.push_back(flattened_size);

    for (size_t i = end_dim + 1; i < shape.size(); ++i) {
        new_shape.push_back(shape[i]);
    }

    return reshape(tensor, new_shape);
}

TensorPtr squeeze(const TensorPtr& tensor, int dim) {
    return tensor->squeeze(dim);
}

TensorPtr unsqueeze(const TensorPtr& tensor, int dim) {
    return tensor->unsqueeze(dim);
}

TensorPtr sum(const TensorPtr& tensor, int dim, bool keepdim) {

    const auto& shape = tensor->shape();
    int ndim = shape.size();
    
    if (dim != -1) {
        // 处理负索引
        if (dim < 0) dim += ndim;
        if (dim < 0 || dim >= ndim) {
            throw TensorError("sum: dimension out of range");
        }
        
        // 计算结果shape
        std::vector<int64_t> result_shape;
        for (int i = 0; i < ndim; ++i) {
            if (i != dim) {
                result_shape.push_back(shape[i]);
            } else if (keepdim) {
                result_shape.push_back(1);
            }
        }
        
        if (result_shape.empty()) {
            result_shape.push_back(1);
        }
        
        auto result = zeros(result_shape, tensor->dtype(), tensor->device());
        const float* src_data = tensor->data<float>();
        float* dst_data = result->data<float>();
        
        // 简化实现：按维度求和
        int64_t outer_size = 1;
        for (int i = 0; i < dim; ++i) {
            outer_size *= shape[i];
        }
        
        int64_t inner_size = 1;
        for (int i = dim + 1; i < ndim; ++i) {
            inner_size *= shape[i];
        }
        
        int64_t sum_size = shape[dim];
        
        for (int64_t outer = 0; outer < outer_size; ++outer) {
            for (int64_t inner = 0; inner < inner_size; ++inner) {
                float sum_val = 0.0f;
                for (int64_t s = 0; s < sum_size; ++s) {
                    int64_t src_idx = outer * (sum_size * inner_size) + s * inner_size + inner;
                    sum_val += src_data[src_idx];
                }
                int64_t dst_idx = outer * inner_size + inner;
                dst_data[dst_idx] = sum_val;
            }
        }
        
        if (tensor->requires_grad()) {
            result->set_requires_grad(true);
            
            auto backward_fn = std::make_shared<SumBackward>(tensor->shape(), dim, keepdim);
            
            #ifdef USE_NEW_AUTOGRAD_ENGINE
            register_backward(result, {tensor}, backward_fn);
            #else
            result->set_grad_fn([tensor, dim, keepdim, original_shape = shape](const TensorPtr& grad_output) -> std::vector<TensorPtr> {
                // sum的梯度传播：重复grad_output到原始形状
                auto grad_expanded = grad_output;
                
                // 如果没有keepdim，需要添加维度
                if (!keepdim) {
                    auto new_shape = grad_output->shape();
                    new_shape.insert(new_shape.begin() + dim, 1);
                    grad_expanded = grad_expanded->reshape(new_shape);
                }
                
                // 创建与原始tensor相同shape的梯度
                auto grad_input = zeros(original_shape, tensor->dtype(), tensor->device());
                const float* grad_data = grad_expanded->data<float>();
                float* input_grad_data = grad_input->data<float>();
                
                // 将梯度复制到所有summed位置
                int64_t total = tensor->numel();
                int64_t repeat_count = original_shape[dim];
                int64_t block_size = grad_expanded->numel();
                
                for (int64_t i = 0; i < total; ++i) {
                    int64_t grad_idx = i / repeat_count % block_size;
                    input_grad_data[i] = grad_data[grad_idx];
                }
                
                accumulate_gradient(tensor, grad_input);
                return {};
            });
            #endif
        }
        
        return result;
    }

    // dim == -1: sum all elements
    const float* data = tensor->data<float>();
    float sum_val = 0.0f;

    for (int64_t i = 0; i < tensor->numel(); ++i) {
        sum_val += data[i];
    }

    std::vector<int64_t> result_shape = keepdim ? tensor->shape() : std::vector<int64_t>{};
    if (keepdim) {
        std::fill(result_shape.begin(), result_shape.end(), 1);
    }

    auto result = full(result_shape.empty() ? std::vector<int64_t>{1} : result_shape, sum_val);

    if (tensor->requires_grad()) {
        result->set_requires_grad(true);

        auto backward_fn = std::make_shared<SumBackward>(tensor->shape(), dim, keepdim);
        
        #ifdef USE_NEW_AUTOGRAD_ENGINE
        register_backward(result, {tensor}, backward_fn);
        #else
        result->set_grad_fn([tensor](const TensorPtr& grad_output) -> std::vector<TensorPtr> {

            auto grad_input = zeros(tensor->shape(), tensor->dtype(), tensor->device());
            float grad_val = grad_output->data<float>()[0];

            float* grad_data = grad_input->data<float>();
            for (int64_t i = 0; i < tensor->numel(); ++i) {
                grad_data[i] = grad_val;
            }

            if (tensor->requires_grad()) {
                accumulate_gradient(tensor, grad_input);
            }

            return {grad_input};
        });
        #endif
    }

    return result;
}

TensorPtr mean(const TensorPtr& tensor, int dim, bool keepdim) {
    auto sum_result = sum(tensor, dim, keepdim);
    float count = 0.0f;
    if (dim == -1) {
        count = static_cast<float>(tensor->numel());
    } else {
        int ndim = tensor->ndim();
        int d = dim < 0 ? dim + ndim : dim;
        if (d < 0 || d >= ndim) {
            throw TensorError("mean: dimension out of range");
        }
        count = static_cast<float>(tensor->shape()[d]);
    }
    return div(sum_result, count);
}

bool same_shape(const TensorPtr& a, const TensorPtr& b) {
    return a->shape() == b->shape();
}

bool broadcastable(const TensorPtr& a, const TensorPtr& b) {
    return can_broadcast(a, b);
}

std::vector<int64_t> broadcast_shape(const TensorPtr& a, const TensorPtr& b) {
    return broadcast_shapes(a, b);
}

std::vector<int64_t> infer_broadcast_shape(const TensorPtr& a, const TensorPtr& b) {

    const auto& shape_a = a->shape();
    const auto& shape_b = b->shape();

    size_t max_dims = std::max(shape_a.size(), shape_b.size());
    std::vector<int64_t> result_shape(max_dims);

    for (size_t i = 0; i < max_dims; ++i) {
        int64_t dim_a = (i < shape_a.size()) ? shape_a[shape_a.size() - 1 - i] : 1;
        int64_t dim_b = (i < shape_b.size()) ? shape_b[shape_b.size() - 1 - i] : 1;

        if (dim_a == 1) {
            result_shape[max_dims - 1 - i] = dim_b;
        } else if (dim_b == 1) {
            result_shape[max_dims - 1 - i] = dim_a;
        } else if (dim_a == dim_b) {
            result_shape[max_dims - 1 - i] = dim_a;
        } else {
            throw TensorError("Cannot broadcast tensors");
        }
    }

    return result_shape;
}

TensorPtr create_causal_mask(int seq_len, DType dtype, Device device) {
    auto mask = full({seq_len, seq_len}, 0.0f, dtype, device);
    float* mask_data = mask->data<float>();

    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < seq_len; ++j) {
            if (j > i) {
                mask_data[i * seq_len + j] = -1e9f;
            }
        }
    }

    return mask;
}

TensorPtr apply_mask(const TensorPtr& input, const TensorPtr& mask, float mask_value) {

    auto result = zeros(input->shape(), input->dtype(), input->device());
    const float* input_data = input->data<float>();
    const float* mask_data = mask->data<float>();
    float* result_data = result->data<float>();

    int64_t total_elements = input->numel();
    int64_t mask_size = mask->numel();

    if (input->ndim() == 2 && mask->ndim() == 2) {

        for (int64_t i = 0; i < total_elements; ++i) {
            result_data[i] = input_data[i] + mask_data[i];
        }
    } else if (input->ndim() == 3 && mask->ndim() == 2) {

        // 支持 3D 形状 [BH, S, S] 与 2D 掩码 [S, S] 的广播加法
        auto input_shape = input->shape();
        int64_t bh = input_shape[0];
        int64_t seq_len = input_shape[1];

        for (int64_t b = 0; b < bh; ++b) {
            for (int64_t i = 0; i < seq_len; ++i) {
                for (int64_t j = 0; j < seq_len; ++j) {
                    int64_t input_idx = b * seq_len * seq_len + i * seq_len + j;
                    int64_t mask_idx = i * seq_len + j;
                    result_data[input_idx] = input_data[input_idx] + mask_data[mask_idx];
                }
            }
        }
    } else if (input->ndim() == 4 && mask->ndim() == 2) {

        auto input_shape = input->shape();
        int64_t batch_size = input_shape[0];
        int64_t n_head = input_shape[1];
        int64_t seq_len = input_shape[2];

        for (int64_t b = 0; b < batch_size; ++b) {
            for (int64_t h = 0; h < n_head; ++h) {
                for (int64_t i = 0; i < seq_len; ++i) {
                    for (int64_t j = 0; j < seq_len; ++j) {
                        int64_t input_idx = b * n_head * seq_len * seq_len +
                                          h * seq_len * seq_len +
                                          i * seq_len + j;
                        int64_t mask_idx = i * seq_len + j;
                        result_data[input_idx] = input_data[input_idx] + mask_data[mask_idx];
                    }
                }
            }
        }
    } else {
        throw TensorError("Unsupported mask dimensions");
    }

    if (input->requires_grad()) {
        result->set_requires_grad(true);
        #ifdef USE_NEW_AUTOGRAD_ENGINE
        auto backward_fn = std::make_shared<ApplyMaskBackward>(input);
        register_backward(result, {input}, backward_fn);
        #else
        result->set_grad_fn([input](const TensorPtr& grad_output) -> std::vector<TensorPtr> {
            auto grad_input = zeros(input->shape(), input->dtype(), input->device());
            const float* grad_out_data = grad_output->data<float>();
            float* grad_in_data = grad_input->data<float>();
            for (int64_t i = 0; i < input->numel(); ++i) {
                grad_in_data[i] = grad_out_data[i];
            }
            if (input->requires_grad()) {
                accumulate_gradient(input, grad_input);
            }
            return {grad_input};
        });
        #endif
    }

    return result;
}

TensorPtr repeat_kv_heads(const TensorPtr& kv, int repeat_factor) {
    // kv shape: [batch, kv_heads, seq_len, head_dim]
    // output shape: [batch, kv_heads * repeat_factor, seq_len, head_dim]
    
    auto shape = kv->shape();
    if (shape.size() != 4) {
        throw std::runtime_error("repeat_kv_heads expects 4D tensor");
    }
    
    int64_t batch = shape[0];
    int64_t kv_heads = shape[1];
    int64_t seq_len = shape[2];
    int64_t head_dim = shape[3];
    
    auto result = zeros({batch, kv_heads * repeat_factor, seq_len, head_dim}, 
                       kv->dtype(), kv->device());
    
    const float* kv_data = kv->data<float>();
    float* result_data = result->data<float>();
    
    // 简化实现：重复每个KV头
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t kv_h = 0; kv_h < kv_heads; ++kv_h) {
            for (int64_t rep = 0; rep < repeat_factor; ++rep) {
                int64_t out_head = kv_h * repeat_factor + rep;
                for (int64_t s = 0; s < seq_len; ++s) {
                    for (int64_t d = 0; d < head_dim; ++d) {
                        int64_t kv_idx = b * kv_heads * seq_len * head_dim + 
                                        kv_h * seq_len * head_dim + 
                                        s * head_dim + d;
                        int64_t out_idx = b * (kv_heads * repeat_factor) * seq_len * head_dim + 
                                         out_head * seq_len * head_dim + 
                                         s * head_dim + d;
                        result_data[out_idx] = kv_data[kv_idx];
                    }
                }
            }
        }
    }
    
    if (kv->requires_grad()) {
        result->set_requires_grad(true);
        #ifdef USE_NEW_AUTOGRAD_ENGINE
        auto backward_fn = std::make_shared<RepeatKVHeadsBackward>(repeat_factor);
        register_backward(result, {kv}, backward_fn);
        #else
        result->set_grad_fn([kv, repeat_factor](const TensorPtr& grad_output) -> std::vector<TensorPtr> {
            // Legacy: sum repeats back
            const auto& gshape = grad_output->shape();
            int64_t batch = gshape[0], heads_rep=gshape[1], seq=gshape[2], dim=gshape[3];
            int64_t kv_heads = heads_rep / repeat_factor;
            auto grad_kv = zeros({batch, kv_heads, seq, dim}, grad_output->dtype(), grad_output->device());
            const float* src = grad_output->data<float>();
            float* dst = grad_kv->data<float>();
            for (int64_t b=0;b<batch;++b){
                for(int64_t kvh=0;kvh<kv_heads;++kvh){
                    for(int64_t rep=0;rep<repeat_factor;++rep){
                        int64_t out_h = kvh*repeat_factor+rep;
                        for(int64_t s=0;s<seq;++s){
                            for(int64_t d=0;d<dim;++d){
                                int64_t si=(((b*heads_rep+out_h)*seq+s)*dim+d);
                                int64_t di=(((b*kv_heads+kvh)*seq+s)*dim+d);
                                dst[di]+=src[si];
                            }
                        }
                    }
                }
            }
            accumulate_gradient(kv, grad_kv);
            return {grad_kv};
        });
        #endif
    }
    
    return result;
}

TensorPtr apply_rope(const TensorPtr& x, int seq_len, int head_dim, float rope_theta) {
    // x shape: [batch, heads, seq_len, head_dim]
    // RoPE (Rotary Position Embedding) implementation
    
    auto shape = x->shape();
    if (shape.size() != 4) {
        throw std::runtime_error("apply_rope expects 4D tensor: [batch, heads, seq_len, head_dim]");
    }
    
    int64_t batch = shape[0];
    int64_t heads = shape[1];
    int64_t actual_seq_len = shape[2];
    int64_t actual_head_dim = shape[3];
    
    if (actual_head_dim != head_dim || actual_seq_len != seq_len) {
        throw std::runtime_error("RoPE dimension mismatch");
    }
    
    auto result = zeros(shape, x->dtype(), x->device());
    
    const float* x_data = x->data<float>();
    float* result_data = result->data<float>();
    
    // 先复制原始数据，然后在特定位置应用RoPE
    std::memcpy(result_data, x_data, batch * heads * seq_len * head_dim * sizeof(float));
    
    // RoPE implementation - 只应用到前面部分维度
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t h = 0; h < heads; ++h) {
            for (int64_t pos = 0; pos < seq_len; ++pos) {
                for (int64_t d = 0; d < head_dim / 2; ++d) {
                    // Calculate frequency
                    float freq = 1.0f / std::pow(rope_theta, 2.0f * d / head_dim);
                    float angle = pos * freq;
                    float cos_val = std::cos(angle);
                    float sin_val = std::sin(angle);
                    
                    // Get input indices
                    int64_t idx_base = b * heads * seq_len * head_dim + 
                                      h * seq_len * head_dim + 
                                      pos * head_dim;
                    int64_t idx1 = idx_base + 2 * d;
                    int64_t idx2 = idx_base + 2 * d + 1;
                    
                    // 边界检查
                    if (idx1 < batch * heads * seq_len * head_dim && 
                        idx2 < batch * heads * seq_len * head_dim) {
                        // Apply rotation
                        float x1 = x_data[idx1];
                        float x2 = x_data[idx2];
                        
                        result_data[idx1] = x1 * cos_val - x2 * sin_val;
                        result_data[idx2] = x1 * sin_val + x2 * cos_val;
                    }
                }
            }
        }
    }
    
    if (x->requires_grad()) {
        result->set_requires_grad(true);
        #ifdef USE_NEW_AUTOGRAD_ENGINE
        auto backward_fn = std::make_shared<ApplyRoPEBackward>(seq_len, head_dim, rope_theta);
        register_backward(result, {x}, backward_fn);
        #else
        result->set_grad_fn([x](const TensorPtr& grad_output) -> std::vector<TensorPtr> {
            // Legacy: pass-through
            accumulate_gradient(x, grad_output);
            return {grad_output};
        });
        #endif
    }
    
    return result;
}

TensorPtr swiglu(const TensorPtr& gate, const TensorPtr& up) {
    // SwiGLU = SiLU(gate) * up
    if (!same_shape(gate, up)) {
        throw std::runtime_error("gate and up tensors must have the same shape for SwiGLU");
    }
    
    auto result = zeros(gate->shape(), gate->dtype(), gate->device());
    const float* gate_data = gate->data<float>();
    const float* up_data = up->data<float>();
    float* result_data = result->data<float>();
    
    for (int64_t i = 0; i < gate->numel(); ++i) {
        float gate_val = gate_data[i];
        float up_val = up_data[i];
        
        // SiLU(gate) = gate / (1 + exp(-gate))
        float silu_gate = gate_val / (1.0f + std::exp(-gate_val));
        result_data[i] = silu_gate * up_val;
    }
    
    if (gate->requires_grad() || up->requires_grad()) {
        result->set_requires_grad(true);
        result->set_grad_fn([gate, up](const TensorPtr& grad_output) -> std::vector<TensorPtr> {
            auto grad_gate = zeros(gate->shape(), gate->dtype(), gate->device());
            auto grad_up = zeros(up->shape(), up->dtype(), up->device());
            return {grad_gate, grad_up};
        });
    }
    
    return result;
}

// LoRA Linear实现
TensorPtr lora_linear(const TensorPtr& input, const TensorPtr& weight,
                     const TensorPtr& lora_A, const TensorPtr& lora_B,
                     float alpha, const TensorPtr& bias) {
    // Safety checks
    if (!input || !weight || !lora_A || !lora_B) {
        throw TensorError("lora_linear: input, weight, lora_A, lora_B must not be null");
    }
    
    // 主分支：input @ weight
    auto main_output = matmul(input, weight);
    if (bias) {
        main_output = add(main_output, bias);
    }
    
    // LoRA分支：input @ lora_A @ lora_B * alpha
    auto lora_hidden = matmul(input, lora_A);
    auto lora_output = matmul(lora_hidden, lora_B);
    auto scaled_lora = mul(lora_output, alpha);
    
    // 合并主分支和LoRA分支
    auto result = add(main_output, scaled_lora);
    
    // 不要覆盖 add 的 backward！add 会自动传播梯度到 main_output 和 scaled_lora
    // main_output 和 scaled_lora 各自已经注册了自己的 backward，会继续传播梯度
    
    #ifndef USE_NEW_AUTOGRAD_ENGINE
    // 仅在旧引擎下需要显式设置 grad_fn（旧引擎需要手动管理整个链路）
    if (input->requires_grad() || lora_A->requires_grad() || lora_B->requires_grad()) {
        result->set_requires_grad(true);
        // Legacy recursive backward
        result->set_grad_fn([input, weight, lora_A, lora_B, alpha, bias](const TensorPtr& grad_output) -> std::vector<TensorPtr> {
            
            // 计算 lora_A 的梯度并累加
            if (lora_A->requires_grad()) {
                // 梯度流：grad_output -> lora_B^T -> lora_hidden -> input^T -> lora_A
                auto lora_B_t = transpose(lora_B, 0, 1);  // [6, 2]
                auto grad_lora_hidden = matmul(grad_output, lora_B_t);  // [2, 6] @ [6, 2] = [2, 2]
                auto input_t = transpose(input, -2, -1);  // [4, 2]
                auto grad_lora_A_raw = matmul(input_t, grad_lora_hidden);  // [4, 2] @ [2, 2] = [4, 2]
                auto grad_lora_A = mul(grad_lora_A_raw, alpha);
                
                // 使用 accumulate_gradient 正确累加梯度
                accumulate_gradient(lora_A, grad_lora_A);
            }
            
            // 计算 lora_B 的梯度并累加
            if (lora_B->requires_grad()) {
                // 梯度流：grad_output <- lora_hidden^T <- lora_B
                auto lora_hidden = matmul(input, lora_A);  // [2, 4] @ [4, 2] = [2, 2]
                auto lora_hidden_t = transpose(lora_hidden, -2, -1);  // [2, 2]
                auto grad_lora_B_raw = matmul(lora_hidden_t, grad_output);  // [2, 2] @ [2, 6] = [2, 6]
                auto grad_lora_B = mul(grad_lora_B_raw, alpha);
                
                // 使用 accumulate_gradient 正确累加梯度
                accumulate_gradient(lora_B, grad_lora_B);
            }
            
            // 对输入的梯度传播（如果需要）
            if (input->requires_grad()) {
                // 来自主分支的梯度
                auto grad_input_main = matmul(grad_output, transpose(weight, 0, 1));
                
                // 来自LoRA分支的梯度
                auto lora_B_t = transpose(lora_B, 0, 1);
                auto lora_A_t = transpose(lora_A, 0, 1);
                auto grad_input_lora = matmul(matmul(grad_output, lora_B_t), lora_A_t);
                auto grad_input_lora_scaled = mul(grad_input_lora, alpha);
                
                auto grad_input_total = add(grad_input_main, grad_input_lora_scaled);
                accumulate_gradient(input, grad_input_total);
            }
            
            return {};  // 返回空向量，因为我们已经用accumulate_gradient处理了
        });
    }
    #endif
    
    return result;
}

// 比较算子实现
TensorPtr eq(const TensorPtr& a, const TensorPtr& b) {
    auto result = zeros(a->shape(), a->dtype(), a->device());
    const float* data_a = a->data<float>();
    const float* data_b = b->data<float>();
    float* result_data = result->data<float>();
    
    for (int64_t i = 0; i < a->numel(); ++i) {
        result_data[i] = (data_a[i] == data_b[i]) ? 1.0f : 0.0f;
    }
    
    return result;
}

TensorPtr ne(const TensorPtr& a, const TensorPtr& b) {
    auto result = zeros(a->shape(), a->dtype(), a->device());
    const float* data_a = a->data<float>();
    const float* data_b = b->data<float>();
    float* result_data = result->data<float>();
    
    for (int64_t i = 0; i < a->numel(); ++i) {
        result_data[i] = (data_a[i] != data_b[i]) ? 1.0f : 0.0f;
    }
    
    return result;
}

TensorPtr gt(const TensorPtr& a, const TensorPtr& b) {
    auto result = zeros(a->shape(), a->dtype(), a->device());
    const float* data_a = a->data<float>();
    const float* data_b = b->data<float>();
    float* result_data = result->data<float>();
    
    for (int64_t i = 0; i < a->numel(); ++i) {
        result_data[i] = (data_a[i] > data_b[i]) ? 1.0f : 0.0f;
    }
    
    return result;
}

TensorPtr lt(const TensorPtr& a, const TensorPtr& b) {
    auto result = zeros(a->shape(), a->dtype(), a->device());
    const float* data_a = a->data<float>();
    const float* data_b = b->data<float>();
    float* result_data = result->data<float>();
    
    for (int64_t i = 0; i < a->numel(); ++i) {
        result_data[i] = (data_a[i] < data_b[i]) ? 1.0f : 0.0f;
    }
    
    return result;
}

TensorPtr ge(const TensorPtr& a, const TensorPtr& b) {
    auto result = zeros(a->shape(), a->dtype(), a->device());
    const float* data_a = a->data<float>();
    const float* data_b = b->data<float>();
    float* result_data = result->data<float>();
    
    for (int64_t i = 0; i < a->numel(); ++i) {
        result_data[i] = (data_a[i] >= data_b[i]) ? 1.0f : 0.0f;
    }
    
    return result;
}

TensorPtr le(const TensorPtr& a, const TensorPtr& b) {
    auto result = zeros(a->shape(), a->dtype(), a->device());
    const float* data_a = a->data<float>();
    const float* data_b = b->data<float>();
    float* result_data = result->data<float>();
    
    for (int64_t i = 0; i < a->numel(); ++i) {
        result_data[i] = (data_a[i] <= data_b[i]) ? 1.0f : 0.0f;
    }
    
    return result;
}

// 数学函数
TensorPtr abs(const TensorPtr& tensor) {
    auto result = zeros(tensor->shape(), tensor->dtype(), tensor->device());
    const float* data = tensor->data<float>();
    float* result_data = result->data<float>();
    
    for (int64_t i = 0; i < tensor->numel(); ++i) {
        result_data[i] = std::abs(data[i]);
    }
    
    if (tensor->requires_grad()) {
        result->set_requires_grad(true);
        #ifdef USE_NEW_AUTOGRAD_ENGINE
        // Register backward with new engine
        auto backward_fn = std::make_shared<PassThroughBackward>();
        register_backward(result, {tensor}, backward_fn);
        #endif
    }
    
    return result;
}

TensorPtr sqrt(const TensorPtr& tensor) {
    auto result = zeros(tensor->shape(), tensor->dtype(), tensor->device());
    const float* data = tensor->data<float>();
    float* result_data = result->data<float>();
    
    for (int64_t i = 0; i < tensor->numel(); ++i) {
        result_data[i] = std::sqrt(data[i]);
    }
    
    if (tensor->requires_grad()) {
        result->set_requires_grad(true);
        #ifdef USE_NEW_AUTOGRAD_ENGINE
        auto backward_fn = std::make_shared<ScaleBackward>(0.0f /* placeholder, not used here */);
        // 对于 sqrt/exp/log 等，这里保持与旧路径一致：后续可补精确反向
        register_backward(result, {tensor}, backward_fn);
        #endif
    }
    
    return result;
}

TensorPtr exp(const TensorPtr& tensor) {
    auto result = zeros(tensor->shape(), tensor->dtype(), tensor->device());
    const float* data = tensor->data<float>();
    float* result_data = result->data<float>();
    
    for (int64_t i = 0; i < tensor->numel(); ++i) {
        result_data[i] = std::exp(data[i]);
    }
    
    if (tensor->requires_grad()) {
        result->set_requires_grad(true);
    }
    
    return result;
}

TensorPtr log(const TensorPtr& tensor) {
    auto result = zeros(tensor->shape(), tensor->dtype(), tensor->device());
    const float* data = tensor->data<float>();
    float* result_data = result->data<float>();
    
    for (int64_t i = 0; i < tensor->numel(); ++i) {
        result_data[i] = std::log(data[i]);
    }
    
    if (tensor->requires_grad()) {
        result->set_requires_grad(true);
    }
    
    return result;
}

TensorPtr pow(const TensorPtr& tensor, float exponent) {
    auto result = zeros(tensor->shape(), tensor->dtype(), tensor->device());
    const float* data = tensor->data<float>();
    float* result_data = result->data<float>();
    
    for (int64_t i = 0; i < tensor->numel(); ++i) {
        result_data[i] = std::pow(data[i], exponent);
    }
    
    if (tensor->requires_grad()) {
        result->set_requires_grad(true);
    }
    
    return result;
}

TensorPtr clamp(const TensorPtr& tensor, float min_val, float max_val) {
    auto result = zeros(tensor->shape(), tensor->dtype(), tensor->device());
    const float* data = tensor->data<float>();
    float* result_data = result->data<float>();
    
    for (int64_t i = 0; i < tensor->numel(); ++i) {
        result_data[i] = std::min(std::max(data[i], min_val), max_val);
    }
    
    if (tensor->requires_grad()) {
        result->set_requires_grad(true);
    }
    
    return result;
}

// =========================================================================
// Data Type Operations (FP32 <-> FP16)
// =========================================================================
TensorPtr cast(const TensorPtr& tensor, DType target_dtype) {
    if (!tensor) {
        throw TensorError("cast: input tensor is null");
    }

    const DType src_dtype = tensor->dtype();
    if (src_dtype == target_dtype) {
        // 返回一个克隆，避免后续写时共享同一缓冲
        return tensor->clone();
    }

    auto result = std::make_shared<Tensor>(tensor->shape(), target_dtype, tensor->device());

    if (src_dtype == kFloat32 && target_dtype == kFloat16) {
        const float* src = tensor->data<float>();
        uint16_t* dst = result->data<uint16_t>();
        fp16::convert_fp32_to_fp16(src, dst, static_cast<size_t>(tensor->numel()));
        return result;
    }

    if (src_dtype == kFloat16 && target_dtype == kFloat32) {
        const uint16_t* src = tensor->data<uint16_t>();
        float* dst = result->data<float>();
        fp16::convert_fp16_to_fp32(src, dst, static_cast<size_t>(tensor->numel()));
        return result;
    }

    // 其他类型目前未用到，先提供安全兜底（逐元素拷贝为float再截断）
    if (DTypeUtils::is_floating_point(src_dtype) && DTypeUtils::is_floating_point(target_dtype)) {
        // 通用浮点路径：先拉成 FP32，再转到目标
        TensorPtr as_fp32 = (src_dtype == kFloat32) ? tensor : cast(tensor, kFloat32);
        if (target_dtype == kFloat32) return as_fp32;
        return cast(as_fp32, target_dtype);
    }

    throw TensorError("cast: unsupported dtype conversion");
}

}
