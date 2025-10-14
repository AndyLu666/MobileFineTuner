/**
 * @file memory_first_mlp.h
 * @brief Extreme memory-saving MLP implementation (chunked by output channels)
 * 
 * Core strategy:
 * 1. Standard MLP: hidden = GELU(x @ W1 + b1), out = hidden @ W2 + b2
 *    Requires full storage of hidden [batch*seq, n_inner] (e.g., 2*64*3072*4B = 1.5MB)
 * 
 * 2. Chunked MLP: Process by chunks of W1 columns (output channels)
 *    - Compute hidden_chunk_i = GELU(x @ W1[:,i_start:i_end] + b1[i_start:i_end])
 *    - Immediately use hidden_chunk_i @ W2[i_start:i_end,:] to accumulate to final output
 *    - Release hidden_chunk_i
 *    Peak memory: hidden_chunk [batch*seq, chunk_size] (e.g., 2*64*256*4B = 128KB)
 * 
 * Memory reduction: original 1.5MB → now 128KB, reduced by ~12x
 */

#pragma once

#include "tensor.h"
#include "ops.h"

namespace ops {
namespace memory_first {

/**
 * @brief Chunked MLP forward computation (extreme memory saving)
 * 
 * @param input Input [batch*seq_len, n_embd]
 * @param fc_weight First layer weight [n_inner, n_embd] (transposed for matmul)
 * @param fc_bias First layer bias [n_inner]
 * @param proj_weight Second layer weight [n_embd, n_inner] (transposed for matmul)
 * @param proj_bias Second layer bias [n_embd]
 * @param chunk_size Intermediate layer chunk size (default 256)
 * @return Output [batch*seq_len, n_embd]
 * 
 * Algorithm flow:
 * 1. output = zeros([batch*seq_len, n_embd])
 * 2. for chunk_start in range(0, n_inner, chunk_size):
 *      chunk_end = min(chunk_start + chunk_size, n_inner)
 *      # Compute one chunk of intermediate layer
 *      hidden_chunk = input @ fc_weight[:, chunk_start:chunk_end].T + fc_bias[chunk_start:chunk_end]
 *      activated_chunk = GELU(hidden_chunk)
 *      # Immediately use this chunk to update final output
 *      output += activated_chunk @ proj_weight[:, chunk_start:chunk_end].T
 *      # hidden_chunk and activated_chunk are automatically released in next loop
 * 3. output += proj_bias
 * 4. return output
 */
TensorPtr memory_first_mlp_forward(
    const TensorPtr& input,
    const TensorPtr& fc_weight,
    const TensorPtr& fc_bias,
    const TensorPtr& proj_weight,
    const TensorPtr& proj_bias,
    int chunk_size = 256
);

} // namespace memory_first
} // namespace ops

