/**
 * @file param_manager_lite.h  
 * @brief Lightweight parameter manager - Provide parameter ID/name mapping for MobileOptimizer
 * 
 * This is a simplified adaptation of MobileParameterManager, providing only core features needed by optimizer:
 * - Parameter registration and ID allocation
 * - Parameter name and size recording
 * - Group management (optional)
 * 
 * Does not include complete memory management features (those are StateManager's responsibility)
 */

#pragma once

#include "../core/tensor.h"
#include <unordered_map>
#include <string>
#include <vector>
#include <memory>

namespace ops {
namespace optim {

/**
 * @brief Parameter metadata
 */
struct ParameterMetadata {
    size_t param_id;
    std::string param_name;
    size_t param_size;  // Element count
    std::string group_name;
    bool requires_grad;
    
    // Actual parameter reference (weak reference, does not own)
    TensorPtr param_tensor;  // Save shared_ptr for state manager access
    
    // Default constructor (for container use)
    ParameterMetadata() 
        : param_id(0), param_name(""), param_size(0),
          group_name("default"), requires_grad(false), param_tensor(nullptr) {}
    
    ParameterMetadata(size_t id, const std::string& name, size_t size, 
                     const std::string& group = "default")
        : param_id(id), param_name(name), param_size(size),
          group_name(group), requires_grad(true), param_tensor(nullptr) {}
};

/**
 * @brief Lightweight parameter manager
 * 
 * Responsibilities:
 * 1. Allocate unique IDs for parameters
 * 2. Record parameter metadata (name/size/group)
 * 3. Provide parameter query interface
 * 4. Support group management (optional)
 */
class ParameterManagerLite {
private:
    std::unordered_map<size_t, ParameterMetadata> param_registry_;
    std::unordered_map<std::string, size_t> name_to_id_;
    std::unordered_map<std::string, std::vector<size_t>> groups_;
    
    size_t next_param_id_ = 0;

public:
    ParameterManagerLite() = default;
    
    /**
     * @brief Register parameter
     * @param param Parameter tensor
     * @param name Parameter name
     * @param group_name Group name (optional)
     * @return Allocated parameter ID
     */
    size_t register_parameter(const TensorPtr& param, 
                             const std::string& name,
                             const std::string& group_name = "default") {
        size_t param_id = next_param_id_++;
        size_t param_size = param ? param->numel() : 0;
        
        ParameterMetadata metadata(param_id, name, param_size, group_name);
        metadata.param_tensor = param;
        metadata.requires_grad = param ? param->requires_grad() : false;
        
        param_registry_[param_id] = metadata;
        name_to_id_[name] = param_id;
        groups_[group_name].push_back(param_id);
        
        return param_id;
    }
    
    /**
     * @brief Batch register parameters
     * @param params Parameter list
     * @param name_prefix Name prefix
     * @param group_name Group name
     * @return Parameter ID list
     */
    std::vector<size_t> register_parameters(const std::vector<TensorPtr>& params,
                                           const std::string& name_prefix = "param",
                                           const std::string& group_name = "default") {
        std::vector<size_t> ids;
        ids.reserve(params.size());
        
        for (size_t i = 0; i < params.size(); ++i) {
            std::string name = name_prefix + "_" + std::to_string(i);
            ids.push_back(register_parameter(params[i], name, group_name));
        }
        
        return ids;
    }
    
    /**
     * @brief Get parameter metadata
     */
    const ParameterMetadata* get_metadata(size_t param_id) const {
        auto it = param_registry_.find(param_id);
        return (it != param_registry_.end()) ? &it->second : nullptr;
    }
    
    /**
     * @brief Get parameter ID by name
     */
    size_t get_param_id(const std::string& name) const {
        auto it = name_to_id_.find(name);
        if (it == name_to_id_.end()) {
            throw std::runtime_error("Parameter not found: " + name);
        }
        return it->second;
    }
    
    /**
     * @brief Get all parameter IDs in group
     */
    std::vector<size_t> get_group_params(const std::string& group_name) const {
        auto it = groups_.find(group_name);
        return (it != groups_.end()) ? it->second : std::vector<size_t>();
    }
    
    /**
     * @brief Get parameter tensor
     */
    TensorPtr get_parameter(size_t param_id) const {
        auto metadata = get_metadata(param_id);
        return metadata ? metadata->param_tensor : nullptr;
    }
    
    /**
     * @brief Get total parameter count
     */
    size_t num_parameters() const { return param_registry_.size(); }
    
    /**
     * @brief Get total parameter element count
     */
    size_t total_parameter_count() const {
        size_t total = 0;
        for (const auto& [id, meta] : param_registry_) {
            total += meta.param_size;
        }
        return total;
    }
    
    /**
     * @brief Clear all registrations
     */
    void clear() {
        param_registry_.clear();
        name_to_id_.clear();
        groups_.clear();
        next_param_id_ = 0;
    }
};

// Backward compatibility alias (for StateManager use)
// Note: MobileOptimizerStateManager.h also forward declares MobileParameterManager
// Here provides actual definition
class MobileParameterManager : public ParameterManagerLite {
public:
    using ParameterManagerLite::ParameterManagerLite;
};

} // namespace optim
} // namespace ops

