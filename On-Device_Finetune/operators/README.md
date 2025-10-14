# Operators - C++ Deep Learning Framework

A professional C++ deep learning framework optimized for Transformer models (GPT-2/Gemma) with PyTorch-style API, automatic differentiation, LoRA finetuning, and mobile-first design.

**Version:** 2.0.0  
**C++ Standard:** C++17  

## Core Features

### High-Performance Computing
- **Topological-Sort Autograd Engine**: Non-recursive backward pass for deep networks (12+ layers)
- **Pure C++ Implementation**: No BLAS/Accelerate dependencies - fully portable
- **ARM NEON Vectorization**: Optimized matrix operations for mobile CPUs
- **Memory-Efficient**: Pooled allocation, tensor caching, automatic graph cleanup

### Complete Transformer Support
- **GPT-2**: Full training and LoRA finetuning (Q/V/O projections)
- **Gemma**: RMSNorm, GeGLU, GQA (Grouped-Query Attention), RoPE
- **Large Vocabulary**: Chunked softmax + cross-entropy for 262K+ vocab sizes
- **Memory-First Operators**: Blocked matmul, chunked attention, blocked MLP

### Industrial-Grade Optimizers
- **Adam/AdamW**: Full implementation with bias correction and AMSGrad
- **Gradient Clipping**: Global norm clipping, adaptive clipping, per-parameter clipping
- **Learning Rate Scheduling**: Warm-up, cosine decay, linear decay, step decay
- **Numerical Stability**: NaN/Inf detection, overflow protection, gradient scaling

### Mobile Optimizations
- **Parameter Management**: FP16 quantization, parameter offloading, prefetching
- **Thermal-Aware**: Adaptive learning rate based on device temperature
- **Battery-Aware**: Reduced update frequency on low battery
- **Memory Pressure**: Dynamic batch size and block size adjustment

## Project Structure

```
operators/
├── core/                                    [Core Layer]
│   ├── tensor.h/cpp                        Tensor class with autograd support
│   ├── ops.h/cpp                           90+ operations (matmul, add, gelu, etc.)
│   ├── autograd_engine.h/cpp               Topological-sort backward engine
│   ├── backward_functions.h/cpp            Gradient computation functions
│   ├── memory_manager.h/cpp                Memory pool and tensor cache
│   ├── mobile_safe_matmul.h/cpp            Adaptive blocking matmul
│   ├── memory_first_attention.h/cpp        O(S) memory attention
│   ├── memory_first_mlp.h/cpp              Blocked MLP implementation
│   ├── chunked_softmax_ce.h/cpp            Streaming softmax + cross-entropy
│   ├── device.h                            Device management (CPU/GPU)
│   ├── dtype.h                             Data type definitions (FP32/FP16/INT32)
│   ├── logger.h/cpp                        Logging utilities
│   ├── utils.h/cpp                         Helper functions
│   ├── tokenizer.h/cpp                     BPE tokenizer
│   └── step_arena.h/cpp                    Per-step memory arena
│
├── optim/                                  [Optimizers]
│   ├── optimizer.h/cpp                     Base optimizer class
│   ├── adam.h/cpp                          Adam optimizer
│   ├── adam_amp.h/cpp                      Adam with mixed precision
│   ├── mobile_optimizer_extensions.h/cpp   Gradient clipping + LR scheduling
│   ├── mobile_optimizer_state_manager.h/cpp Optimizer state compression
│   ├── mobile_optimizer_advanced.h/cpp     Advanced mobile optimizations
│   ├── lora_optimizer.h/cpp                LoRA-specific optimizer
│   ├── optimizer_utils.h                   FP16/INT8 utilities
│   └── param_manager_lite.h                Lightweight parameter manager
│
├── memory/                                 [Memory Management]
│   ├── mobile_param_manager.h/cpp          Parameter lifecycle management
│   ├── mobile_param_optimizations.h/cpp    FP16 quantization, offloading
│   ├── mobile_specific_optimizations.h/cpp Platform-specific optimizations
│   ├── mobile_zero.h/cpp                   ZeRO-style parameter partitioning
│   └── activation_checkpointer.h/cpp       Gradient checkpointing
│
├── activations/                            [Activation Management]
│   ├── mobile_activation_manager.h/cpp     Activation memory management
│   ├── deepspeed_checkpoint_integration.h  DeepSpeed-style checkpointing
│   └── activation_*.h                      Various activation strategies
│
├── nn/                                     [Neural Network Layers]
│   ├── module.h                            Base module class (torch.nn.Module)
│   ├── attention.h/cpp                     Multi-head attention
│   ├── mlp.h/cpp                           Feed-forward networks
│   ├── embedding.h/cpp                     Embedding layers
│   ├── lora.h/cpp                          LoRA low-rank adaptation
│   └── layers.h                            Common layer definitions
│
├── transformer/                            [Transformer Architecture]
│   ├── transformer_block.h/cpp             Transformer block
│   ├── gpt2_components.h/cpp               GPT-2 specific components
│   ├── gpt2_finetune_model.h               GPT-2 finetuning model
│   ├── kv_cache.h/cpp                      KV cache for inference
│   └── autoregressive_ops.h/cpp            Autoregressive generation
│
├── functional/                             [Functional API]
│   └── functional.h                        Functional-style operations
│
├── utils/                                  [Utilities]
│   ├── fp16_utils.h                        FP16 conversion utilities
│   ├── grad_scaler.h                       Gradient scaling for mixed precision
│   └── memory_ledger.h                     Memory usage tracking
│
├── models/                                 [Pre-built Models]
│   └── gpt2.h                              GPT-2 model definitions
│
├── tests/                                  [Tests]
│   ├── test_adam.cpp                       Optimizer tests
│   ├── test_enhanced.cpp                   Enhanced feature tests
│   ├── test_fusion.cpp                     Operator fusion tests
│   └── test_gpt2_lora_integration.cpp      GPT-2 LoRA integration tests
│
├── tools/                                  [Tools]
│   └── log_viewer.py                       Training log visualization
│
├── config.h                                Global configuration
├── operators.h                             Main header (includes all)
├── nn.h                                    Neural network module header
├── transformer.h                           Transformer module header
├── memory.h                                Memory module header
├── CMakeLists.txt                          Build configuration
├── build_operators.sh                      Build script
└── README.md                               This file
```

## Quick Start

### 1. Build the Framework

```bash
cd operators
mkdir build && cd build
cmake .. \
    -DUSE_NEW_AUTOGRAD_ENGINE=ON \
    -DUSE_MOBILE_OPTIMIZER=ON \
    -DENABLE_MEMORY_MODULE=ON \
    -DENABLE_ACTIVATIONS_MODULE=ON \
    -DBUILD_TESTS=ON
make -j$(nproc)
```

Build options:
- `USE_NEW_AUTOGRAD_ENGINE`: Enable topological-sort autograd (default: ON)
- `USE_MOBILE_OPTIMIZER`: Enable mobile optimizer extensions (default: ON)
- `ENABLE_MEMORY_MODULE`: Enable parameter management and ZeRO (default: ON)
- `ENABLE_ACTIVATIONS_MODULE`: Enable activation checkpointing (default: ON)
- `BUILD_TESTS`: Build test programs (default: ON)
- `AUTOGRAD_DEBUG`: Enable autograd debugging output (default: OFF)
- `ENABLE_PROFILING`: Enable performance profiling (default: OFF)

### 2. Basic Usage Example

```cpp
#include "operators.h"

using namespace ops;

int main() {
    // Create tensors
    auto x = randn({2, 3, 768});  // [batch, seq_len, hidden_dim]
    auto weight = randn({768, 768});
    
    // Enable gradient tracking
    x->set_requires_grad(true);
    weight->set_requires_grad(true);
    
    // Forward pass
    auto y = matmul(x, weight);
    auto loss = mean(y);
    
    // Backward pass
    loss->backward();
    
    // Access gradients
    auto x_grad = x->grad();
    auto weight_grad = weight->grad();
    
    return 0;
}
```

### 3. GPT-2 LoRA Finetuning Example

```cpp
#include "operators.h"
#include "optim/adam.h"
#include "optim/mobile_optimizer_extensions.h"

using namespace ops;
using namespace ops::optim;

int main() {
    // Model configuration
    struct GPT2LoRAConfig {
        int n_embd = 768;
        int n_head = 12;
        int n_layer = 12;
        int block_size = 1024;
        int vocab_size = 50257;
        
        // LoRA configuration
        int lora_rank = 8;
        float lora_alpha = 16.0f;
        int lora_layers = 6;  // Apply LoRA to last 6 layers
        bool lora_q = true;
        bool lora_v = true;
        bool lora_k = false;
        bool lora_o = false;
        
        // Training configuration
        float lr = 3e-4f;
        int batch_size = 2;
        int grad_accum_steps = 4;
        int max_epochs = 3;
    };
    
    GPT2LoRAConfig config;
    
    // Create optimizer
    AdamConfig adam_config(config.lr);
    adam_config.weight_decay = 0.01f;  // AdamW mode
    adam_config.clip_grad_norm = 1.0f;
    auto optimizer = std::make_unique<Adam>(adam_config);
    
    // Create gradient clipper
    GradientClippingConfig clip_config;
clip_config.max_grad_norm = 1.0f;
clip_config.use_global_norm = true;
clip_config.adaptive_clipping = true;
    auto clipper = std::make_unique<MobileGradientClipper>(clip_config, nullptr);
    
    // Create LR scheduler
    LRSchedulerConfig lr_config;
    lr_config.type = LRSchedulerType::WARM_UP_COSINE;
    lr_config.base_lr = config.lr;
    lr_config.min_lr = config.lr / 10.0f;
    lr_config.warmup_steps = 1000;
    lr_config.decay_steps = 10000;
    auto scheduler = std::make_unique<MobileLRScheduler>(lr_config, nullptr);
    
    // Training loop
    for (int epoch = 0; epoch < config.max_epochs; ++epoch) {
        for (int step = 0; step < num_steps; ++step) {
            // Forward pass
            auto [input_ids, targets] = dataloader.get_batch(config.batch_size);
            auto logits = model->forward(input_ids);
            auto loss = cross_entropy_loss(logits, targets);
            
            // Backward pass
    loss->backward();
    
            // Gradient accumulation
            if ((step + 1) % config.grad_accum_steps == 0) {
                // Gradient clipping
    float grad_norm = clipper->clip_gradients(gradients);
    
                // Update learning rate
    float current_lr = scheduler->step();
    optimizer->set_learning_rate(current_lr);
    
                // Parameter update
    optimizer->step(parameters, gradients);
                optimizer->zero_grad();
                
                // Memory cleanup
                MemoryManager::instance().force_cleanup();
            }
        }
    }
    
    return 0;
}
```

### 4. Using Memory-First Operators

```cpp
#include "core/memory_first_attention.h"
#include "core/memory_first_mlp.h"
#include "core/chunked_softmax_ce.h"

using namespace ops;

// Memory-first attention (O(S) memory instead of O(S^2))
auto attn_output = memory_first_multihead_attention(
    q, k, v,              // Query, Key, Value tensors
    n_heads,              // Number of attention heads
    head_dim,             // Dimension per head
    causal_mask,          // Causal mask for autoregressive models
    32                    // Block size (default: 32)
);

// Memory-first MLP (blocked computation)
auto mlp_output = memory_first_mlp_forward(
    x,                    // Input tensor
    fc_weight,           // First layer weight
    fc_bias,             // First layer bias
    proj_weight,         // Projection weight
    proj_bias,           // Projection bias
    true,                // Use GELU activation
    256                  // Channel block size (default: 256)
);

// Chunked softmax + cross-entropy (streaming computation)
auto loss = chunked_cross_entropy_forward(
    hidden_states,       // [B, L, D]
    lm_head_weight,      // [V, D] or [D, V]
    targets,             // [B, L] int32
    2048,                // Chunk size (default: 2048)
    false                // Weight is not transposed
);
```

## CMake Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `USE_NEW_AUTOGRAD_ENGINE` | ON | Use topological-sort autograd engine instead of recursive |
| `AUTOGRAD_DEBUG` | OFF | Enable autograd debugging output |
| `USE_MOBILE_OPTIMIZER` | ON | Enable mobile optimizer extensions (clipping, scheduling) |
| `ENABLE_MEMORY_MODULE` | ON | Enable memory management module (param manager, ZeRO) |
| `ENABLE_ACTIVATIONS_MODULE` | ON | Enable activation management module (checkpointing) |
| `BUILD_TESTS` | ON | Build test executables |
| `BUILD_EXAMPLES` | OFF | Build example programs |
| `ENABLE_PROFILING` | OFF | Enable gprof profiling (-pg flag) |

## Performance Benchmarks

### GPT-2 LoRA Finetuning (MacBook Air M2, 16GB RAM)

| Configuration | Memory Peak | Time/Step | Notes |
|---------------|-------------|-----------|-------|
| 6 layers LoRA, seq=1024, batch=2 | 1.6 GB | ~1.8s | Default configuration |
| 12 layers LoRA, seq=1024, batch=2 | 2.8 GB | ~3.2s | All layers with LoRA |
| 6 layers LoRA, seq=512, batch=2 | 1.1 GB | ~0.9s | Shorter sequences |
| 6 layers LoRA, seq=1024, batch=1 | 0.9 GB | ~1.2s | Smaller batch |

### Memory-First Operators Comparison

| Operator | Standard Memory | Memory-First | Memory Reduction |
|----------|----------------|--------------|------------------|
| Attention (seq=1024) | 48 MB | 12 MB | 75% |
| MLP (hidden=3072) | 24 MB | 6 MB | 75% |
| Chunked CE (vocab=262K) | 2048 MB | 64 MB | 97% |

### Operator Performance (Apple M1 Pro, 8-core)

| Operation | Size | Standard | Optimized | Speedup |
|-----------|------|----------|-----------|---------|
| Matmul | [1024, 768] @ [768, 768] | 12.3 ms | 4.2 ms | 2.9x |
| Attention | batch=2, seq=1024, heads=12 | 45.6 ms | 18.3 ms | 2.5x |
| Softmax | [2, 1024, 50257] | 156.2 ms | 8.7 ms | 18x (chunked) |
| Layer Norm | [2, 1024, 768] | 3.4 ms | 3.1 ms | 1.1x |

## Testing

### Run All Tests

```bash
cd operators/build
ctest --output-on-failure
```

### Run Specific Tests

```bash
# Autograd engine test
./test_autograd_engine

# Optimizer test
./test_optimizer

# Memory module test
./test_memory_module

# Activations module test
./test_activations_module
```

### Test Coverage

- **Autograd Engine**: Forward/backward correctness, topological sort, gradient accumulation
- **Gradient Clipping**: Global norm, adaptive clipping, per-parameter clipping
- **LR Scheduling**: Warm-up, cosine decay, linear decay, step decay
- **Memory Management**: Pool allocation, tensor caching, graph cleanup
- **Optimizer State**: State save/load, compression, FP16/INT8 quantization

## API Documentation

### Core Tensor Operations

#### Creation Operations

```cpp
// Create tensors
TensorPtr zeros(const std::vector<int64_t>& shape, DType dtype = kFloat32, Device device = kCPU);
TensorPtr ones(const std::vector<int64_t>& shape, DType dtype = kFloat32, Device device = kCPU);
TensorPtr randn(const std::vector<int64_t>& shape, float mean = 0.0f, float std = 1.0f);
TensorPtr uniform(const std::vector<int64_t>& shape, float low = 0.0f, float high = 1.0f);
TensorPtr full(const std::vector<int64_t>& shape, float value, DType dtype = kFloat32);
TensorPtr arange(float start, float end, float step = 1.0f, DType dtype = kFloat32);
```

#### Arithmetic Operations

```cpp
// Element-wise operations
TensorPtr add(const TensorPtr& a, const TensorPtr& b);
TensorPtr sub(const TensorPtr& a, const TensorPtr& b);
TensorPtr mul(const TensorPtr& a, const TensorPtr& b);
TensorPtr div(const TensorPtr& a, const TensorPtr& b);

// Scalar operations
TensorPtr add_scalar(const TensorPtr& tensor, float scalar);
TensorPtr mul_scalar(const TensorPtr& tensor, float scalar);
TensorPtr div_scalar(const TensorPtr& tensor, float scalar);
```

#### Linear Algebra Operations

```cpp
// Matrix operations
TensorPtr matmul(const TensorPtr& a, const TensorPtr& b);
TensorPtr matmul_rhs_T(const TensorPtr& a, const TensorPtr& b);  // a @ b.T (zero-copy)
TensorPtr transpose(const TensorPtr& tensor, int dim0, int dim1);
TensorPtr permute(const TensorPtr& tensor, const std::vector<int>& dims);

// LoRA-specific
TensorPtr lora_linear(const TensorPtr& input, const TensorPtr& weight,
                     const TensorPtr& lora_A, const TensorPtr& lora_B,
                     float alpha = 1.0f, const TensorPtr& bias = nullptr);
```

#### Activation Functions

```cpp
// Standard activations
TensorPtr relu(const TensorPtr& x);
TensorPtr gelu(const TensorPtr& x);
TensorPtr silu(const TensorPtr& x);  // Swish
TensorPtr sigmoid(const TensorPtr& x);
TensorPtr tanh_op(const TensorPtr& x);
TensorPtr softmax(const TensorPtr& x, int dim = -1);
TensorPtr log_softmax(const TensorPtr& x, int dim = -1);

// Gemma-specific
TensorPtr swiglu(const TensorPtr& gate, const TensorPtr& up);
TensorPtr geglu(const TensorPtr& gate, const TensorPtr& up);
```

#### Normalization Layers

```cpp
// Layer normalization
TensorPtr layer_norm(const TensorPtr& input, const TensorPtr& weight, 
                     const TensorPtr& bias, float eps = 1e-5f);

// RMS normalization (Gemma)
TensorPtr rms_norm(const TensorPtr& input, const TensorPtr& weight, float eps = 1e-6f);

// Batch normalization
TensorPtr batch_norm(const TensorPtr& input, const TensorPtr& weight, const TensorPtr& bias,
                     const TensorPtr& running_mean, const TensorPtr& running_var,
                     bool training = true, float momentum = 0.1f, float eps = 1e-5f);
```

#### Loss Functions

```cpp
// Classification losses
TensorPtr cross_entropy_loss(const TensorPtr& input, const TensorPtr& target,
                             const std::string& reduction = "mean");
TensorPtr nll_loss(const TensorPtr& input, const TensorPtr& target,
                   const std::string& reduction = "mean");

// Regression losses
TensorPtr mse_loss(const TensorPtr& input, const TensorPtr& target,
                   const std::string& reduction = "mean");
```

#### Tensor Manipulation

```cpp
// Shape operations
TensorPtr reshape(const TensorPtr& tensor, const std::vector<int64_t>& shape);
TensorPtr view(const TensorPtr& tensor, const std::vector<int64_t>& shape);
TensorPtr flatten(const TensorPtr& tensor, int start_dim = 0, int end_dim = -1);
TensorPtr squeeze(const TensorPtr& tensor, int dim = -1);
TensorPtr unsqueeze(const TensorPtr& tensor, int dim);

// Concatenation and stacking
TensorPtr concat(const std::vector<TensorPtr>& tensors, int dim = 0);
TensorPtr stack(const std::vector<TensorPtr>& tensors, int dim = 0);
TensorPtr split(const TensorPtr& tensor, int split_size, int dim = 0);

// Indexing
TensorPtr index_select(const TensorPtr& tensor, int dim, const TensorPtr& index);
TensorPtr gather(const TensorPtr& tensor, int dim, const TensorPtr& index);
TensorPtr scatter(const TensorPtr& tensor, int dim, const TensorPtr& index, const TensorPtr& src);
```

#### Reduction Operations

```cpp
// Reductions
TensorPtr sum(const TensorPtr& tensor, int dim = -1, bool keepdim = false);
TensorPtr mean(const TensorPtr& tensor, int dim = -1, bool keepdim = false);
TensorPtr max(const TensorPtr& tensor, int dim = -1, bool keepdim = false);
TensorPtr min(const TensorPtr& tensor, int dim = -1, bool keepdim = false);
TensorPtr argmax(const TensorPtr& tensor, int dim = -1, bool keepdim = false);
TensorPtr argmin(const TensorPtr& tensor, int dim = -1, bool keepdim = false);
```

### Autograd API

```cpp
// Enable gradient tracking
tensor->set_requires_grad(true);
bool requires_grad = tensor->requires_grad();

// Backward pass
loss->backward();

// Access gradients
TensorPtr grad = tensor->grad();

// Clear gradients
tensor->zero_grad();

// Detach from computation graph
TensorPtr detached = tensor->detach();

// Clone tensor (optionally copy grad)
TensorPtr cloned = tensor->clone(bool copy_grad = false);
```

### Optimizer API

```cpp
// Adam optimizer
#include "optim/adam.h"

AdamConfig config;
config.learning_rate = 3e-4f;
config.beta1 = 0.9f;
config.beta2 = 0.999f;
config.epsilon = 1e-8f;
config.weight_decay = 0.01f;  // AdamW mode
config.amsgrad = false;

auto optimizer = std::make_unique<Adam>(config);

// Training step
optimizer->zero_grad();
loss->backward();
optimizer->step(parameters, gradients);

// Dynamic learning rate
optimizer->set_learning_rate(new_lr);
float current_lr = optimizer->get_learning_rate();

// Save/load state
optimizer->save_state("optimizer_state.bin");
optimizer->load_state("optimizer_state.bin");
```

### Mobile Optimizer Extensions

```cpp
#include "optim/mobile_optimizer_extensions.h"

// Gradient clipping
GradientClippingConfig clip_config;
clip_config.max_grad_norm = 1.0f;
clip_config.use_global_norm = true;
clip_config.adaptive_clipping = true;
clip_config.adaptive_factor = 0.01f;

auto clipper = std::make_unique<MobileGradientClipper>(clip_config, nullptr);
float grad_norm = clipper->clip_gradients(gradients);

// Learning rate scheduling
LRSchedulerConfig lr_config;
lr_config.type = LRSchedulerType::WARM_UP_COSINE;
lr_config.base_lr = 3e-4f;
lr_config.min_lr = 3e-5f;
lr_config.warmup_steps = 1000;
lr_config.decay_steps = 10000;
lr_config.thermal_scaling = true;
lr_config.battery_aware = true;

auto scheduler = std::make_unique<MobileLRScheduler>(lr_config, nullptr);
float current_lr = scheduler->step();

// Get statistics
OptimizerExtensionStats stats = clipper->get_stats();
std::cout << "Average grad norm: " << stats.average_grad_norm << std::endl;
std::cout << "Gradient clips applied: " << stats.gradient_clips_applied << std::endl;
```

### Memory Management API

```cpp
#include "core/memory_manager.h"

// Get memory manager instance
auto& memory_manager = MemoryManager::instance();

// Memory allocation (managed by MemoryManager)
void* ptr = memory_manager.allocate(size);
memory_manager.deallocate(ptr, size);

// Tensor caching
memory_manager.cache_tensor("key", tensor);
auto cached = memory_manager.get_cached_tensor("key");

// Computation graph management
memory_manager.register_tensor(tensor);
memory_manager.clear_computation_graph();
memory_manager.cleanup_dead_references();

// Memory cleanup
memory_manager.clear_cache();
memory_manager.clear_unused_memory();
memory_manager.force_cleanup();

// Statistics
memory_manager.print_memory_stats();
size_t current_usage = memory_manager.get_memory_usage();
size_t peak_usage = memory_manager.get_peak_memory();
```

## Known Issues

### Current Issues

1. **MobileOptimizerStateManager Memory Stats Bug**
   - Symptom: Memory statistics incorrect after state compression
   - Impact: Low (stats only, functionality works)
   - Workaround: Use external memory monitoring
   - Status: Investigating

2. **NN Module Compilation Dependencies**
   - Symptom: Some `nn/` module files have unresolved dependencies
   - Impact: Medium (nn module partially functional)
   - Workaround: Use core ops directly
   - Status: In progress

### Limitations

1. **CPU-Only**: Currently only supports CPU training (GPU support planned)
2. **Single-Device**: No multi-device or distributed training yet
3. **Limited Quantization**: Only FP16/INT8 quantization supported
4. **macOS-Specific Memory Monitoring**: Some memory monitoring features require macOS APIs

## Roadmap

### v2.1 (Q2 2025)
- [ ] Fix `MobileOptimizerStateManager` memory statistics
- [ ] Complete `nn/` module compilation
- [ ] Add comprehensive operator unit tests
- [ ] Implement mixed precision training (FP16/BF16)
- [ ] Add gradient checkpointing examples

### v2.2 (Q3 2025)
- [ ] GPU support (CUDA/Metal)
- [ ] Distributed training primitives
- [ ] Model parallelism (tensor parallelism)
- [ ] Pipeline parallelism
- [ ] INT4 quantization support

### v3.0 (Q4 2025)
- [ ] Full mobile deployment (iOS/Android)
- [ ] On-device training optimizations
- [ ] Inference-only mode (reduced binary size)
- [ ] ONNX export support
- [ ] Model compression utilities

## Contributing

We welcome contributions! Please follow these guidelines:

1. **Code Style**: Follow existing code style (see `.clang-format`)
2. **Testing**: Add tests for new features
3. **Documentation**: Update README and code comments
4. **Commit Messages**: Use descriptive commit messages
5. **Pull Requests**: One feature per PR, include motivation and examples

### Development Setup

```bash
# Clone repository
git clone https://github.com/your-repo/operators.git
cd operators

# Build in debug mode
mkdir build-debug && cd build-debug
cmake .. -DCMAKE_BUILD_TYPE=Debug -DAUTOGRAD_DEBUG=ON
make -j$(nproc)

# Run tests
ctest --output-on-failure

# Enable profiling
cmake .. -DENABLE_PROFILING=ON
make -j$(nproc)
./your_program
gprof your_program gmon.out > analysis.txt
```