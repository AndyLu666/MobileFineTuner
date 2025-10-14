/**
 * @file memory_first_mlp.cpp
 * @brief Extreme memory-saving MLP implementation
 */

#include "memory_first_mlp.h"
#include <algorithm>
#include <iostream>

namespace ops {
namespace memory_first {

TensorPtr memory_first_mlp_forward(
    const TensorPtr& input,
    const TensorPtr& fc_weight,
    const TensorPtr& fc_bias,
    const TensorPtr& proj_weight,
    const TensorPtr& proj_bias,
    int chunk_size
) {
    // input: [batch*seq_len, n_embd]
    // fc_weight: [n_inner, n_embd]
    // proj_weight: [n_embd, n_inner]
    
    auto input_shape = input->shape();
    if (input_shape.size() != 2) {
        throw TensorError("memory_first_mlp: input must be 2D [batch*seq, n_embd]");
    }
    
    int64_t batch_seq = input_shape[0];
    int64_t n_embd = input_shape[1];
    int64_t n_inner = fc_weight->shape()[0];
    
    // Create output tensor and initialize to 0
    auto output = zeros({batch_seq, n_embd}, input->dtype(), input->device());
    
    const float* input_data = input->data<float>();
    const float* fc_w_data = fc_weight->data<float>();
    const float* fc_b_data = fc_bias->data<float>();
    const float* proj_w_data = proj_weight->data<float>();
    const float* proj_b_data = proj_bias->data<float>();
    float* output_data = output->data<float>();
    
    // Process by chunking intermediate layer dimension
    for (int64_t chunk_start = 0; chunk_start < n_inner; chunk_start += chunk_size) {
        int64_t chunk_end = std::min(chunk_start + chunk_size, n_inner);
        int64_t actual_chunk_size = chunk_end - chunk_start;
        
        // Allocate temporary storage for current block [batch_seq, actual_chunk_size]
        std::vector<float> hidden_chunk(batch_seq * actual_chunk_size);
        std::vector<float> activated_chunk(batch_seq * actual_chunk_size);
        
        // === First layer: Compute hidden_chunk = input @ fc_weight[:,chunk_start:chunk_end].T + fc_bias ===
        
        // First initialize with bias
        for (int64_t i = 0; i < batch_seq; ++i) {
            for (int64_t j = 0; j < actual_chunk_size; ++j) {
                hidden_chunk[i * actual_chunk_size + j] = fc_b_data[chunk_start + j];
            }
        }
        
        // Matrix multiply: input [batch_seq, n_embd] @ fc_weight_chunk.T [n_embd, actual_chunk_size]
        // => hidden_chunk [batch_seq, actual_chunk_size]
        for (int64_t i = 0; i < batch_seq; ++i) {
            for (int64_t j = 0; j < actual_chunk_size; ++j) {
                int64_t fc_row = chunk_start + j;
                float sum = 0.0f;
                
                for (int64_t k = 0; k < n_embd; ++k) {
                    sum += input_data[i * n_embd + k] * fc_w_data[fc_row * n_embd + k];
                }
                
                hidden_chunk[i * actual_chunk_size + j] += sum;
            }
        }
        
        // === Activation function GELU ===
        for (int64_t i = 0; i < batch_seq * actual_chunk_size; ++i) {
            float x = hidden_chunk[i];
            float tanh_input = 0.7978845608f * (x + 0.044715f * x * x * x);
            activated_chunk[i] = 0.5f * x * (1.0f + std::tanh(tanh_input));
        }
        
        // === Second layer: output += activated_chunk @ proj_weight[chunk_start:chunk_end,:].T ===
        // activated_chunk [batch_seq, actual_chunk_size]
        // proj_weight_chunk [actual_chunk_size, n_embd] (extracted from proj_weight rows)
        // => output [batch_seq, n_embd]
        
        for (int64_t i = 0; i < batch_seq; ++i) {
            for (int64_t j = 0; j < n_embd; ++j) {
                float sum = 0.0f;
                
                for (int64_t k = 0; k < actual_chunk_size; ++k) {
                    int64_t proj_row = j;
                    int64_t proj_col = chunk_start + k;
                    sum += activated_chunk[i * actual_chunk_size + k] * 
                           proj_w_data[proj_row * n_inner + proj_col];
                }
                
                output_data[i * n_embd + j] += sum;
            }
        }
        
        // hidden_chunk and activated_chunk are automatically released at end of scope
    }
    
    // Finally add proj_bias
    for (int64_t i = 0; i < batch_seq; ++i) {
        for (int64_t j = 0; j < n_embd; ++j) {
            output_data[i * n_embd + j] += proj_b_data[j];
        }
    }
    
    // Set gradient computation (simplified version: mark as requiring gradient)
    if (input->requires_grad() || fc_weight->requires_grad() || proj_weight->requires_grad()) {
        output->set_requires_grad(true);
        // TODO: Implement recomputation-based backward
    }
    
    return output;
}

} // namespace memory_first
} // namespace ops

