/**
 * @file tensor.cpp
 * @brief Implementation of the core Tensor class
 * 
 * This file contains the implementation of all Tensor class methods,
 * including constructors, operators, memory management, and utility functions.
 * It also provides tensor creation functions and automatic differentiation support.
 */

#include "tensor.h"
#include "memory_manager.h"
#include "ops.h"
#ifdef USE_NEW_AUTOGRAD_ENGINE
#include "autograd_engine.h"
#endif
#include <cstring>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <sstream>

namespace ops {

namespace {

    /**
     * @brief Compute the total number of elements in a tensor
     * @param shape The shape vector
     * @return Total number of elements
     */
    int64_t compute_numel(const std::vector<int64_t>& shape) {
        if (shape.empty()) return 0;
        return std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());
    }

    /**
     * @brief Get the size in bytes for a given data type
     * @param dtype The data type
     * @return Size in bytes
     */
    size_t dtype_size(DType dtype) {
        switch (dtype) {
            case kFloat32: return sizeof(float);
            case kFloat16: return sizeof(uint16_t);
            case kInt32: return sizeof(int32_t);
            case kInt64: return sizeof(int64_t);
            case kInt8: return sizeof(int8_t);
            case kBool: return sizeof(bool);
            default: return sizeof(float);
        }
    }

    void validate_shape(const std::vector<int64_t>& shape) {
        for (auto dim : shape) {
            if (dim < 0) {
                throw TensorError("Shape dimensions must be non-negative");
            }
        }
    }

    int64_t compute_linear_index(const std::vector<int64_t>& indices, const std::vector<int64_t>& shape) {
        if (indices.size() != shape.size()) {
            throw TensorError("Index dimension mismatch");
        }

        int64_t linear_index = 0;
        int64_t stride = 1;

        for (int i = shape.size() - 1; i >= 0; --i) {
            if (indices[i] < 0 || indices[i] >= shape[i]) {
                throw TensorError("Index out of bounds");
            }
            linear_index += indices[i] * stride;
            stride *= shape[i];
        }

        return linear_index;
    }
}

Tensor::Tensor(const std::vector<int64_t>& shape, DType dtype, Device device)
    : shape_(shape), dtype_(dtype), device_(device) {
    validate_shape(shape);
    allocate_memory();
}

Tensor::Tensor(const std::vector<int64_t>& shape, const void* data, DType dtype, Device device)
    : shape_(shape), dtype_(dtype), device_(device) {
    validate_shape(shape);
    allocate_memory();
    if (data) {
        copy_data(data, data_size_);
    }
}

Tensor::Tensor(const Tensor& other)
    : shape_(other.shape_), dtype_(other.dtype_), device_(other.device_),
      requires_grad_(other.requires_grad_), retain_grad_(other.retain_grad_) {
    allocate_memory();
    if (other.data_) {
        copy_data(other.data_, data_size_);
    }

}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        free_memory();

        shape_ = other.shape_;
        dtype_ = other.dtype_;
        device_ = other.device_;
        requires_grad_ = other.requires_grad_;
        retain_grad_ = other.retain_grad_;

        allocate_memory();
        if (other.data_) {
            copy_data(other.data_, data_size_);
        }

        grad_ = nullptr;
        grad_fn_ = nullptr;
    }
    return *this;
}

Tensor::Tensor(Tensor&& other) noexcept
    : shape_(std::move(other.shape_)), dtype_(other.dtype_), device_(other.device_),
      data_(other.data_), data_size_(other.data_size_),
      requires_grad_(other.requires_grad_), grad_(std::move(other.grad_)),
      grad_fn_(std::move(other.grad_fn_)), retain_grad_(other.retain_grad_) {
    other.data_ = nullptr;
    other.data_size_ = 0;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        free_memory();

        shape_ = std::move(other.shape_);
        dtype_ = other.dtype_;
        device_ = other.device_;
        data_ = other.data_;
        data_size_ = other.data_size_;
        requires_grad_ = other.requires_grad_;
        grad_ = std::move(other.grad_);
        grad_fn_ = std::move(other.grad_fn_);
        retain_grad_ = other.retain_grad_;

        other.data_ = nullptr;
        other.data_size_ = 0;
    }
    return *this;
}

Tensor::~Tensor() {
    free_memory();
}

int64_t Tensor::numel() const {
    return compute_numel(shape_);
}

void Tensor::allocate_memory() {
    if (numel() == 0) {
        data_ = nullptr;
        data_size_ = 0;
        return;
    }

    data_size_ = numel() * dtype_size(dtype_);

    // Use memory manager for allocation
    data_ = MemoryManager::instance().allocate(data_size_);
    if (!data_) {
        throw TensorError("Failed to allocate memory");
    }

    std::memset(data_, 0, data_size_);
}

void Tensor::free_memory() {
    if (data_) {
        // Use memory manager for deallocation
        MemoryManager::instance().deallocate(data_, data_size_);
        data_ = nullptr;
        data_size_ = 0;
    }
}

void Tensor::copy_data(const void* src, size_t size) {
    if (!data_ || !src || size == 0) return;

    size_t copy_size = std::min(size, data_size_);
    std::memcpy(data_, src, copy_size);
}

TensorPtr Tensor::reshape(const std::vector<int64_t>& new_shape) const {
    int64_t new_numel = compute_numel(new_shape);
    if (new_numel != numel()) {
        throw TensorError("Cannot reshape tensor: element count mismatch");
    }

    auto result = std::make_shared<Tensor>(new_shape, data_, dtype_, device_);

    if (requires_grad_) {
        result->set_requires_grad(true);
        
        // Gradient of reshape is to reshape gradient back to original shape
        auto original_shape = shape_;
        result->set_grad_fn([original_shape](const TensorPtr& grad_output) -> std::vector<TensorPtr> {
            auto grad_input = grad_output->reshape(original_shape);
            return {grad_input};
        });
    }

    return result;
}


TensorPtr Tensor::view(const std::vector<int64_t>& new_shape) const {

    return reshape(new_shape);
}

TensorPtr Tensor::transpose(int dim0, int dim1) const {
    if (ndim() < 2) {
        throw TensorError("transpose requires at least 2 dimensions");
    }

    if (dim0 < 0) dim0 += ndim();
    if (dim1 < 0) dim1 += ndim();

    if (dim0 < 0 || dim0 >= ndim() || dim1 < 0 || dim1 >= ndim()) {
        throw TensorError("transpose dimension out of range");
    }

    auto new_shape = shape_;
    std::swap(new_shape[dim0], new_shape[dim1]);

    auto result = std::make_shared<Tensor>(new_shape, dtype_, device_);

    const float* src_data = data<float>();
    float* dst_data = result->data<float>();

    if (ndim() == 2 && dim0 == 0 && dim1 == 1) {

        int64_t rows = shape_[0];
        int64_t cols = shape_[1];

        for (int64_t i = 0; i < rows; ++i) {
            for (int64_t j = 0; j < cols; ++j) {
                dst_data[j * rows + i] = src_data[i * cols + j];
            }
        }
    } else {

        std::vector<int64_t> src_indices(ndim(), 0);

        for (int64_t linear_idx = 0; linear_idx < numel(); ++linear_idx) {

            int64_t temp = linear_idx;
            for (int d = ndim() - 1; d >= 0; --d) {
                src_indices[d] = temp % shape_[d];
                temp /= shape_[d];
            }

            std::swap(src_indices[dim0], src_indices[dim1]);

            int64_t dst_linear_idx = 0;
            int64_t stride = 1;
            for (int d = ndim() - 1; d >= 0; --d) {
                dst_linear_idx += src_indices[d] * stride;
                stride *= new_shape[d];
            }

            dst_data[dst_linear_idx] = src_data[linear_idx];
        }
    }

    if (requires_grad_) {
        result->set_requires_grad(true);
        
        // Set gradient function: gradient of transpose is to transpose gradient the same way
        result->set_grad_fn([dim0, dim1](const TensorPtr& grad_output) -> std::vector<TensorPtr> {
            // Simplified: directly return transposed gradient, let caller handle accumulation
            auto grad_input = grad_output->transpose(dim0, dim1);
            return {grad_input};
        });
    }

    return result;
}

TensorPtr Tensor::squeeze(int dim) const {
    auto new_shape = shape_;

    if (dim == -1) {

        new_shape.erase(
            std::remove(new_shape.begin(), new_shape.end(), 1),
            new_shape.end()
        );
    } else {

        if (dim < 0) dim += ndim();
        if (dim < 0 || dim >= ndim()) {
            throw TensorError("squeeze dimension out of range");
        }
        if (shape_[dim] == 1) {
            new_shape.erase(new_shape.begin() + dim);
        }
    }

    return reshape(new_shape);
}

TensorPtr Tensor::unsqueeze(int dim) const {
    if (dim < 0) dim += ndim() + 1;
    if (dim < 0 || dim > ndim()) {
        throw TensorError("unsqueeze dimension out of range");
    }

    auto new_shape = shape_;
    new_shape.insert(new_shape.begin() + dim, 1);

    return reshape(new_shape);
}

TensorPtr Tensor::clone() const {
    auto result = std::make_shared<Tensor>(shape_, data_, dtype_, device_);
    result->set_requires_grad(requires_grad_);
    return result;
}

TensorPtr Tensor::detach() const {
    auto result = std::make_shared<Tensor>(shape_, data_, dtype_, device_);

    return result;
}

void Tensor::backward(const TensorPtr& gradient) {
    if (!requires_grad_) {
        return;
    }
    
    #ifdef USE_NEW_AUTOGRAD_ENGINE
    // Forward to the new engine
    #ifdef AUTOGRAD_DEBUG
    std::cout << "[Tensor::backward] Using new engine" << std::endl;
    std::cout << "[Tensor::backward] grad_node_ = " << (grad_node_ ? "exists" : "nullptr") << std::endl;
    #endif
    
    // Need to find the shared_ptr for this tensor - use grad_node_ if available
    TensorPtr this_ptr;
    if (grad_node_ && grad_node_->tensor) {
        this_ptr = grad_node_->tensor;
    } else {
        // Fallback: create a temporary shared_ptr (not ideal but works for testing)
        // In production, tensors should always be created as shared_ptr
        this_ptr = std::shared_ptr<Tensor>(this, [](Tensor*){/* no-op deleter */});
    }
    
    std::vector<TensorPtr> outputs = {this_ptr};
    std::vector<TensorPtr> grads = gradient ? std::vector<TensorPtr>{gradient} : std::vector<TensorPtr>{};
    autograd::Engine::instance().run_backward(outputs, grads);
    return;
    #endif
    
    // Legacy recursive backward (kept for compatibility)
    TensorPtr grad_tensor = gradient;
    if (!grad_tensor) {
        if (numel() != 1) {
            throw TensorError("grad can be implicitly created only for scalar outputs");
        }
        grad_tensor = ones(shape_, dtype_, device_);
    }

    if (grad_tensor->shape() != shape_) {
        throw TensorError("gradient shape does not match tensor shape");
    }

    if (!grad_) {
        grad_ = grad_tensor->clone();
    } else {
        auto accumulated_grad = zeros(grad_->shape(), grad_->dtype(), grad_->device());

        const float* old_grad_data = grad_->data<float>();
        const float* new_grad_data = grad_tensor->data<float>();
        float* accumulated_data = accumulated_grad->data<float>();

        for (int64_t i = 0; i < grad_->numel(); ++i) {
            accumulated_data[i] = old_grad_data[i] + new_grad_data[i];
        }

        grad_ = accumulated_grad;
    }

    if (grad_fn_) {
        grad_fn_(grad_tensor);
    }
}

void Tensor::zero_grad() {
    if (grad_) {

        std::memset(grad_->data_ptr(), 0, grad_->data_size_);
    }
}

void Tensor::print() const {
    std::cout << to_string() << std::endl;
}

std::string Tensor::to_string() const {
    std::ostringstream oss;
    oss << "Tensor(shape=[";
    for (size_t i = 0; i < shape_.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << shape_[i];
    }
    oss << "], dtype=" << static_cast<int>(dtype_);
    oss << ", device=" << static_cast<int>(device_.type());
    if (requires_grad_) oss << ", requires_grad=True";
    oss << ")";
    return oss.str();
}

TensorPtr Tensor::operator+(const Tensor& other) const {

    (void)other;
    return nullptr;
}

TensorPtr Tensor::operator-(const Tensor& other) const {

    (void)other;
    return nullptr;
}

TensorPtr Tensor::operator*(const Tensor& other) const {

    (void)other;
    return nullptr;
}

TensorPtr Tensor::operator/(const Tensor& other) const {

    (void)other;
    return nullptr;
}

TensorPtr Tensor::operator+(float scalar) const {

    (void)scalar;
    return nullptr;
}

TensorPtr Tensor::operator-(float scalar) const {

    (void)scalar;
    return nullptr;
}

TensorPtr Tensor::operator*(float scalar) const {

    (void)scalar;
    return nullptr;
}

TensorPtr Tensor::operator/(float scalar) const {

    (void)scalar;
    return nullptr;
}

TensorPtr zeros(const std::vector<int64_t>& shape, DType dtype, Device device) {
    auto tensor = std::make_shared<Tensor>(shape, dtype, device);

    return tensor;
}

TensorPtr ones(const std::vector<int64_t>& shape, DType dtype, Device device) {
    auto tensor = std::make_shared<Tensor>(shape, dtype, device);

    if (dtype == kFloat32) {
        float* data = tensor->data<float>();
        std::fill_n(data, tensor->numel(), 1.0f);
    } else if (dtype == kInt32) {
        int32_t* data = tensor->data<int32_t>();
        std::fill_n(data, tensor->numel(), 1);
    }

    return tensor;
}

TensorPtr randn(const std::vector<int64_t>& shape, DType dtype, Device device) {
    auto tensor = std::make_shared<Tensor>(shape, dtype, device);

    if (dtype == kFloat32) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 1.0f);

        float* data = tensor->data<float>();
        for (int64_t i = 0; i < tensor->numel(); ++i) {
            data[i] = dist(gen);
        }
    }

    return tensor;
}

TensorPtr rand(const std::vector<int64_t>& shape, DType dtype, Device device) {
    auto tensor = std::make_shared<Tensor>(shape, dtype, device);

    if (dtype == kFloat32) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        float* data = tensor->data<float>();
        for (int64_t i = 0; i < tensor->numel(); ++i) {
            data[i] = dist(gen);
        }
    }

    return tensor;
}

TensorPtr empty(const std::vector<int64_t>& shape, DType dtype, Device device) {
    return std::make_shared<Tensor>(shape, dtype, device);
}

TensorPtr full(const std::vector<int64_t>& shape, float value, DType dtype, Device device) {
    auto tensor = std::make_shared<Tensor>(shape, dtype, device);

    if (dtype == kFloat32) {
        float* data = tensor->data<float>();
        std::fill_n(data, tensor->numel(), value);
    }

    return tensor;
}

TensorPtr tensor(const std::vector<float>& data, DType dtype, Device device) {
    std::vector<int64_t> shape = {static_cast<int64_t>(data.size())};
    return std::make_shared<Tensor>(shape, data.data(), dtype, device);
}

TensorPtr tensor(std::initializer_list<float> data, DType dtype, Device device) {
    std::vector<float> vec_data(data);
    return tensor(vec_data, dtype, device);
}

std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
    os << tensor.to_string();
    return os;
}


}
