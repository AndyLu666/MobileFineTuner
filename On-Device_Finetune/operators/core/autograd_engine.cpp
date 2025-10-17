/**
 * @file autograd_engine.cpp
 * @brief Implementation of the topological-sort based autograd engine
 */

#include "autograd_engine.h"
#include "ops.h"
#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace ops {
namespace autograd {

NodePtr Engine::get_node(const TensorPtr& tensor) {
    if (!tensor) return nullptr;
    
    const Tensor* raw_ptr = tensor.get();
    auto it = tensor_to_node_.find(raw_ptr);
    
    if (it != tensor_to_node_.end()) {
        return it->second;
    }
    
    auto node = std::make_shared<Node>(tensor);
    tensor_to_node_[raw_ptr] = node;
    
    return node;
}

NodePtr Engine::register_node(const TensorPtr& output,
                              const std::vector<TensorPtr>& inputs,
                              BackwardFunctionPtr backward_fn) {
    if (!enabled_ || !output || !backward_fn) {
        return nullptr;
    }
    
    // Safety: validate output tensor
    try {
        auto test_shape = output->shape();  // Trigger any invalid access
        (void)test_shape;
    } catch (...) {
        std::cerr << "[Engine] Invalid output tensor in register_node" << std::endl;
        return nullptr;
    }
    
    auto output_node = get_node(output);
    if (!output_node) {
        return nullptr;
    }
    
    output_node->set_backward_fn(backward_fn);
    output->grad_node_ = output_node;
    tensor_registry_[output.get()] = output;
    
    // Create edges from inputs to output
    // Important: No longer rely on inputs[i]->requires_grad() to decide edge creation.
    // Even if intermediate tensors don't need gradient writeback, they must be connected
    // in the graph to propagate gradients to earlier trainable parameters (e.g., LoRA A/B).
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (inputs[i]) {
            auto input_node = get_node(inputs[i]);
            inputs[i]->grad_node_ = input_node;
            tensor_registry_[inputs[i].get()] = inputs[i];

            auto edge = std::make_shared<Edge>(input_node, output_node, i);
            output_node->add_next_edge(edge);
            input_node->ref_count++;
        }
    }
    
    return output_node;
}

std::vector<NodePtr> Engine::topological_sort(const std::vector<NodePtr>& roots) {
    std::vector<NodePtr> sorted;
    std::unordered_set<NodePtr> visited;
    std::unordered_set<NodePtr> in_stack;
    
    std::function<void(NodePtr)> dfs = [&](NodePtr node) {
        if (!node || visited.count(node)) return;
        
        if (in_stack.count(node)) {
            throw std::runtime_error("Cycle detected in computation graph");
        }
        
        in_stack.insert(node);
        
        #ifdef AUTOGRAD_DEBUG
        std::cout << "[DFS] Visiting node=" << (node->tensor ? node->tensor.get() : nullptr) 
                  << " edges=" << node->next_edges.size() << std::endl;
        #endif
        
        // Visit all inputs first (DFS)
        for (const auto& edge : node->next_edges) {
            if (edge && edge->input_node) {
                dfs(edge->input_node);
            }
        }
        
        in_stack.erase(node);
        visited.insert(node);
        sorted.push_back(node);
    };
    
    // Start DFS from all roots
    for (const auto& root : roots) {
        if (root) dfs(root);
    }
    
    // Reverse to get topological order (roots first, leaves last)
    std::reverse(sorted.begin(), sorted.end());
    return sorted;
}

void Engine::accumulate_grad(const TensorPtr& tensor, const TensorPtr& grad) {
    if (!tensor || !grad) return;
    
    const Tensor* raw_ptr = tensor.get();
    
    #ifdef AUTOGRAD_DEBUG
    // Debug only
    #endif
    
    auto it = pending_grads_.find(raw_ptr);
    
    if (it == pending_grads_.end()) {
        // First gradient for this tensor
        pending_grads_[raw_ptr] = grad->clone();
    } else {
        // Accumulate with existing gradient
        auto& existing_grad = it->second;
        const float* new_grad_data = grad->data<float>();
        float* accum_grad_data = existing_grad->data<float>();
        
        for (int64_t i = 0; i < existing_grad->numel(); ++i) {
            accum_grad_data[i] += new_grad_data[i];
        }
    }
}

void Engine::run_backward(const std::vector<TensorPtr>& outputs,
                         const std::vector<TensorPtr>& output_grads) {
    if (!enabled_) return;
    
    #ifdef AUTOGRAD_DEBUG
    // Debug only
    #endif
    
    // Prepare root nodes and gradients
    std::vector<NodePtr> roots;
    for (size_t i = 0; i < outputs.size(); ++i) {
        if (!outputs[i] || !outputs[i]->requires_grad()) continue;
        
        // Get or create node for this output
        NodePtr node = outputs[i]->grad_node_;
        if (!node) {
            node = get_node(outputs[i]);
            outputs[i]->grad_node_ = node;
        }
        
        if (node) {
            roots.push_back(node);
            
            // Initialize gradient for this output
            TensorPtr initial_grad;
            if (i < output_grads.size() && output_grads[i]) {
                initial_grad = output_grads[i];
            } else {
                // Default: ones with same shape as output
                if (outputs[i]->numel() == 1) {
                    initial_grad = ones(outputs[i]->shape(), outputs[i]->dtype(), outputs[i]->device());
                } else {
                    throw std::runtime_error("Gradient must be providesd for non-scalar outputs");
                }
            }
            
            accumulate_grad(outputs[i], initial_grad);
        }
    }
    
    if (roots.empty()) {
        clear_graph();
        return;
    }
    
    // Dynamic traversal: Support mixed graph of new engine nodes and legacy grad_fn
    std::vector<NodePtr> stack;
    std::unordered_set<NodePtr> visited;  // Prevent duplicate node processing
    for (auto& r : roots) { if (r) stack.push_back(r); }
    
    while (!stack.empty()) {
        NodePtr node = stack.back();
        stack.pop_back();
        if (!node || !node->tensor) continue;
        
        // Critical: Check if node already processed to avoid infinite loops
        if (visited.count(node)) continue;
        visited.insert(node);
        
        const Tensor* raw_ptr = node->tensor.get();
        auto grad_it = pending_grads_.find(raw_ptr);
        if (grad_it == pending_grads_.end()) {
            continue;
        }
        TensorPtr node_grad = grad_it->second;
        
        #ifdef AUTOGRAD_DEBUG
        // Debug only
        #endif
        
        std::vector<TensorPtr> input_grads;
        bool used_legacy = false;
        if (node->backward_fn) {
            input_grads = node->backward_fn->apply(node_grad);
        } else if (node->tensor->grad_fn_) {
            // Compatible with legacy path: Call legacy grad_fn and advance
            used_legacy = true;
            input_grads = node->tensor->grad_fn_(node_grad);
        } else {
            // No available backward, treat as leaf
            continue;
        }
        
        // Distribute gradients and push inputs to stack
        if (!used_legacy) {
            for (size_t i = 0; i < node->next_edges.size(); ++i) {
                const auto& edge = node->next_edges[i];
                if (!edge || !edge->input_node || !edge->input_node->tensor) continue;
                int idx = edge->input_idx;
                if (idx >= 0 && idx < static_cast<int>(input_grads.size()) && input_grads[idx]) {
                    accumulate_grad(edge->input_node->tensor, input_grads[idx]);
                    // Only push unvisited nodes to stack
                    if (!visited.count(edge->input_node)) {
                        stack.push_back(edge->input_node);
                    }
                }
            }
        } else {
            // Legacy has no next_edges, dynamically create nodes and push to stack based on grads
            for (auto& g : input_grads) {
                if (!g) continue;
                // Cannot directly obtain input tensor pointers, must rely on legacy grad_fn having accumulated gradients to corresponding inputs internally
                // Use registry scan strategy: push all tensors with accumulated gradients to stack (approximate)
                for (const auto& [tptr, shared] : tensor_registry_) {
                    if (!shared) continue;
                    auto it2 = pending_grads_.find(tptr);
                    if (it2 != pending_grads_.end()) {
                        NodePtr in_node = get_node(shared);
                        if (in_node) stack.push_back(in_node);
                    }
                }
            }
        }
    }
    
    // Final: Set gradients to tensors (write back only to leaves or tensors that explicitly retain grads)
    #ifdef AUTOGRAD_DEBUG
    // Debug only
    #endif
    
    for (const auto& [tensor_ptr, grad] : pending_grads_) {
        // Find the tensor shared_ptr from registry
        auto reg_it = tensor_registry_.find(tensor_ptr);
        if (reg_it != tensor_registry_.end()) {
            TensorPtr tensor_shared = reg_it->second;
            if (!tensor_shared) continue;
            bool should_write = false;
            #ifdef USE_NEW_AUTOGRAD_ENGINE
            // New engine: Writeback condition = parameter/leaf (Node has no backward_fn) or explicit retain
            if (tensor_shared->retains_grad()) {
                should_write = true;
            } else if (tensor_shared->grad_node_) {
                // Has node: If no backward_fn, treat as parameter/input leaf
                should_write = (tensor_shared->grad_node_->backward_fn == nullptr);
            } else {
                // No node: Conservatively treat as leaf
                should_write = true;
            }
            #else
            should_write = tensor_shared->is_leaf() || tensor_shared->retains_grad();
            #endif
            if (should_write) {
                tensor_shared->set_grad(grad);
                #ifdef AUTOGRAD_DEBUG
                // Debug only
                #endif
            }
        }
    }
    
    // Cleanup
    pending_grads_.clear();
    clear_graph();
}

void Engine::clear_graph() {
    // First break bi-directional references between Tensor and Node to avoid cross-step memory retention
    for (auto &kv : tensor_registry_) {
        const Tensor* raw = kv.first;
        TensorPtr t = kv.second;
        if (t) {
            t->grad_node_.reset();
        }
    }

    tensor_to_node_.clear();
    tensor_registry_.clear();
    pending_grads_.clear();
}

} // namespace autograd
} // namespace ops

