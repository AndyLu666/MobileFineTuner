/**
 * @file wikitext2_dataset.h
 * @brief WikiText-2 数据集加载器（拼接+切块，对齐HF）
 */

#pragma once

#include "../core/tensor.h"
#include "../core/tokenizer_bpe.h"
#include <functional>
#include <string>
#include <vector>
#include <random>
#include <memory>
#include <array>

namespace ops {

struct WT2Config {
    // 绝对路径
    std::string train_path;
    std::string valid_path;
    std::string test_path;
    std::string pretokenized_path;    // 离线分词的token流(.bin)
    std::string pretokenized_meta;    // 元信息json（可选）

    int seq_len = 256;         // 训练序列长度（<=1024）
    int stride  = -1;          // -1表示等于seq_len（不重叠）；否则使用滑窗步长
    int eos_id  = 50256;       // GPT-2 <|endoftext|>
    int pad_id  = 0;           // PAD填充值
    bool insert_eos_between_lines = true;   // 样本间插入EOS
    bool drop_last = true;     // 训练集true；验证可设false保留尾巴
    uint64_t seed = 2025;      // 打乱/抽样随机种子
    bool shuffle_train = true; // 是否打乱训练顺序
    
    // 🚀 流式加载配置（内存优化）
    bool streaming_mode = true;       // 是否使用流式加载（不常驻全量token）
    size_t max_cache_tokens = 100000; // 最大缓存token数（流式模式）
    float data_fraction = 1.0f;       // 使用数据比例（0~1]
};

enum class Split { Train, Valid, Test };

struct Batch {
    TensorPtr input_ids;       // [B, S] int32
    TensorPtr attention_mask;  // [B, S] float32 (1=有效, 0=pad)
    TensorPtr labels;          // [B, S] int32 (pad→-100)
};

class WikiText2Dataset {
public:
    WikiText2Dataset(const WT2Config& cfg, GPT2BPETokenizer* tok);
    WikiText2Dataset(const WT2Config& cfg,
                     std::function<std::vector<int32_t>(const std::string&)> encode_fn);
    
    /**
     * @brief 读文件→分词→拼接→建索引
     */
    void load(Split split);
    
    /**
     * @brief 可取的chunk数
     */
    size_t num_sequences() const;
    
    /**
     * @brief 打乱顺序（仅Train使用）
     */
    void shuffle();
    
    /**
     * @brief 取一批，从index_start开始
     */
    Batch get_batch(size_t index_start, size_t batch_size) const;
    
    /**
     * @brief 便利接口：自增游标（到末尾会环回或截断）
     */
    Batch next_batch(size_t batch_size, bool need_loop = true);
    
    /**
     * @brief 取开头的若干token（用于sanity check）
     */
    std::vector<int32_t> peek_tokens(size_t count) const;
    
    /**
     * @brief 重置游标
     */
    void reset_cursor();

private:
    struct PretokenizedSplit {
        size_t offset = 0;
        size_t length = 0;
        bool available = false;
    };

    struct PretokenizedMeta {
        bool loaded = false;
        std::string meta_path;
        size_t total_tokens = 0;
        int32_t eos_id = -1;
        int32_t pad_id = 0;
        int32_t bos_id = -1;
        int32_t unk_id = -1;
        int32_t vocab_size = 0;
        bool insert_eos_between_lines = true;
        PretokenizedSplit train;
        PretokenizedSplit valid;
        PretokenizedSplit test;
    };

    // 读取指定split的文本行
    std::vector<std::string> read_lines_for_split(Split split) const;
    
    // 将行→token流（拼接+插EOS）
    std::vector<int32_t> tokenize_and_pack(const std::vector<std::string>& lines) const;
    
    // 基于stride/seq_len生成chunk起点索引
    void build_chunk_indices(const std::vector<int32_t>& ids);
    
    // 🚀 真·流式加载：按需从文件读取窗口
    void load_window_from_file(size_t global_token_start, size_t num_tokens);
    
    // 预计算所有chunk的真实token偏移
    void precompute_chunk_offsets();
    
    // 预分词模式：读元信息/切分
    void ensure_pretokenized_meta_loaded();
    void load_pretokenized_split(Split split);

    WT2Config cfg_;
    GPT2BPETokenizer* tok_;   // 非拥有指针
    std::function<std::vector<int32_t>(const std::string&)> encode_fn_;

    // 🚀 流式模式：文件路径而非全量文本
    std::string current_file_path_;   // 当前split的文件路径
    std::vector<int32_t> ids_;        // 当前窗口的token缓存（仅~10万）
    size_t ids_global_offset_ = 0;    // ids_[0]对应全局第几个token
    size_t total_tokens_ = 0;         // 全局总token数（预扫描得出）
    
    // 每个chunk的真实起点（全局token偏移）
    std::vector<size_t> starts_;      // 长度M = num_sequences()

    // 为batch采样的索引（Train可shuffle）
    std::vector<size_t> order_;       // 长度M
    mutable size_t cursor_;           // 下一个batch的起点在order_的位置
    mutable std::mt19937_64 rng_;
    
    // 🚀 Batch缓冲复用（避免每次都创建新Tensor）
    mutable std::vector<int32_t> batch_input_buffer_;
    mutable std::vector<int32_t> batch_label_buffer_;
    mutable std::vector<float> batch_attn_buffer_;
    
    // 预分词模式
    mutable PretokenizedMeta pretokenized_meta_;
    bool pretokenized_mode_ = false;
};

}  // namespace ops
