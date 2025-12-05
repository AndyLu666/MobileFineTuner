/**
 * @file tokenizer_bpe.h
 * @brief GPT-2 Byte-Level BPE Tokenizer（与 HuggingFace 完全对齐）
 * 
 * 实现标准 GPT-2 BPE 分词，支持：
 * - 加载 vocab.json、merges.txt、special_tokens_map.json
 * - Byte-Level 映射（256 bytes → unicode）
 * - BPE 合并规则（按 merges.txt 的 rank 顺序）
 * - 编码/解码与 HuggingFace transformers 一致
 */

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

namespace ops {

/**
 * @brief GPT-2 Byte-Level BPE 配置
 */
struct BPEConfig {
    std::string vocab_path;           // vocab.json 路径
    std::string merges_path;          // merges.txt 路径
    std::string special_tokens_path;  // special_tokens_map.json 路径（可选）
    
    int eos_token_id = 50256;         // <|endoftext|>
    int bos_token_id = 50256;         // GPT-2 通常不用独立 BOS
    int pad_token_id = 50256;         // 建议复用 eos（HF 默认）
    int unk_token_id = 50256;         // 未知 token（GPT-2 很少用）
    
    bool add_prefix_space = false;    // GPT-2 默认不加前导空格
    
    BPEConfig() = default;
    
    // 从目录加载（假设标准命名）
    static BPEConfig from_pretrained(const std::string& model_dir) {
        BPEConfig cfg;
        cfg.vocab_path = model_dir + "/vocab.json";
        cfg.merges_path = model_dir + "/merges.txt";
        cfg.special_tokens_path = model_dir + "/special_tokens_map.json";
        return cfg;
    }
};

/**
 * @brief GPT-2 Byte-Level BPE Tokenizer
 * 
 * 与 HuggingFace GPT2Tokenizer 对齐：
 * - Byte→Unicode 映射（256 bytes → 256 字符，跳过不可打印范围）
 * - BPE 合并按 merges.txt 的顺序 rank 进行
 * - 特殊符号：<|endoftext|>（id=50256）
 */
class GPT2BPETokenizer {
public:
    explicit GPT2BPETokenizer(const BPEConfig& config);
    ~GPT2BPETokenizer() = default;
    
    /**
     * @brief 加载分词器资产（vocab/merges/special_tokens）
     */
    void load();
    
    /**
     * @brief 编码文本为 token IDs
     * @param text 原始文本
     * @param add_special_tokens 是否自动添加 <|endoftext|>（默认 false，GPT-2 通常手工控制）
     * @param max_length 最大长度（0 = 不限制）
     * @param truncation 是否截断（超长时）
     * @return token IDs
     */
    std::vector<int> encode(const std::string& text,
                            bool add_special_tokens = false,
                            int max_length = 0,
                            bool truncation = true);
    
    /**
     * @brief 解码 token IDs 为文本
     * @param ids token IDs
     * @param skip_special_tokens 是否跳过特殊符号（默认 false）
     * @return 原始文本
     */
    std::string decode(const std::vector<int>& ids,
                       bool skip_special_tokens = false);
    
    /**
     * @brief 批量编码（带 padding/truncation）
     * @param texts 文本列表
     * @param max_length 最大长度（0 = 不限制）
     * @param padding 是否 padding（"max_length" | "longest" | "none"）
     * @param truncation 是否截断
     * @return {input_ids, attention_mask}
     */
    std::pair<std::vector<std::vector<int>>, std::vector<std::vector<int>>> 
    batch_encode(const std::vector<std::string>& texts,
                 int max_length = 0,
                 const std::string& padding = "longest",
                 bool truncation = true);
    
    // 获取特殊 token ID
    int get_eos_token_id() const { return config_.eos_token_id; }
    int get_bos_token_id() const { return config_.bos_token_id; }
    int get_pad_token_id() const { return config_.pad_token_id; }
    int get_unk_token_id() const { return config_.unk_token_id; }
    int get_vocab_size() const { return vocab_size_; }
    
    // 工具
    std::string get_token_string(int token_id) const;

private:
    BPEConfig config_;
    
    // Byte→Unicode 映射（256 个）
    std::unordered_map<uint8_t, std::string> byte_encoder_;    // byte → unicode UTF-8 char
    std::unordered_map<std::string, uint8_t> byte_decoder_;    // unicode UTF-8 char → byte
    
    // Vocab 与反向映射
    std::unordered_map<std::string, int> vocab_;  // token → id
    std::unordered_map<int, std::string> id_to_token_;  // id → token
    int vocab_size_;
    
    // Merges 规则（pair → rank）
    std::unordered_map<std::string, int> bpe_ranks_;  // "a b" → rank
    
    // 特殊符号缓存
    std::unordered_map<std::string, int> special_tokens_;
    
    // 内部方法
    void build_byte_encoder();  // 构建 byte↔unicode 映射
    void load_vocab();          // 加载 vocab.json
    void load_merges();         // 加载 merges.txt
    void load_special_tokens(); // 加载 special_tokens_map.json
    
    std::string bytes_to_unicode(const std::string& text);  // 文本 → byte-level unicode
    std::string unicode_to_bytes(const std::string& unicode_text);  // 逆映射
    
    std::vector<std::string> bpe(const std::string& token);  // 对单个 token 做 BPE 合并
    std::pair<int, int> get_best_pair(const std::vector<std::string>& word);  // 找最高 rank 的 pair
    std::vector<std::string> split_to_words(const std::string& text);  // 按空格预切分
};

}  // namespace ops

