/**
 * @file lora_linear.h
 * @brief LoRA增强的线性层（模块化封装）
 */

#pragma once

#include "../core/tensor.h"
#include <vector>
#include <memory>

namespace ops {

/**
 * @brief LoRA切片（支持子矩阵注入，如QKV的q/k/v分别注入）
 */
struct LoRASlice {
    TensorPtr A;     // [in_dim, rank]
    TensorPtr B;     // [rank, out_slice]
    float scale;     // alpha / rank
    int col0;        // 作用于base W的列起点（q=0, k=C, v=2C；非qkv则0）
    int cols;        // 切片列数
    
    LoRASlice(const TensorPtr& a, const TensorPtr& b, float s, int c0 = 0, int c = -1)
        : A(a), B(b), scale(s), col0(c0), cols(c) {}
};

/**
 * @brief LoRA增强的线性层
 * 
 * 功能：
 * - 训练态：y = x@W + b + Σ scale_i * (x @ A_i @ B_i)
 * - 推理态：可merge LoRA到base权重
 * - 参数管理：只有A/B需要梯度，W/b冻结
 */
class LoRALinear {
public:
    /**
     * @brief 构造（引用base权重，不拷贝）
     */
    LoRALinear(const TensorPtr& W_base, const TensorPtr& b_base = nullptr)
        : W_(W_base), b_(b_base), merged_(false) {}
    
    /**
     * @brief 挂载一个LoRA切片（可多次调用，如qkv有3个）
     */
    void attach_lora(const TensorPtr& A, const TensorPtr& B, 
                     float scale, int col0 = 0, int cols = -1);
    
    /**
     * @brief 清空所有LoRA（只移除切片，不动base）
     */
    void clear_lora();
    
    /**
     * @brief 导出/推理：把ΔW烤入base（子矩阵范围加）
     */
    void merge_to_base();
    
    /**
     * @brief 还原：从base减去ΔW
     */
    void unmerge_from_base();
    
    /**
     * @brief 前向：y = x@W + b + Σ scale*(x@A@B)
     */
    TensorPtr forward(const TensorPtr& x) const;
    
    // 调试辅助：为该层打一个名字，导出A/B带name
    void set_debug_name(const std::string& name) { debug_name_ = name; }
    std::vector<std::pair<std::string, TensorPtr>> debug_params() const;
    
    /**
     * @brief 枚举可训练参数（只返回A/B）
     */
    std::vector<TensorPtr> trainable_parameters() const;
    
    /**
     * @brief 只读访问
     */
    const TensorPtr& W() const { return W_; }
    const TensorPtr& b() const { return b_; }
    const std::vector<LoRASlice>& slices() const { return slices_; }
    bool is_merged() const { return merged_; }

private:
    TensorPtr W_;  // [in_dim, out_dim]（引用base，非拥有）
    TensorPtr b_;  // [out_dim]
    std::vector<LoRASlice> slices_;
    bool merged_;
    std::string debug_name_;
};

}  // namespace ops
