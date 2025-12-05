/**
 * @file tokenizer_bpe.cpp
 * @brief GPT-2 Byte-Level BPE Tokenizer 实现（与 HuggingFace 完全对齐）
 * 
 * 参考：https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/tokenization_gpt2.py
 */

#include "tokenizer_bpe.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <regex>
#include <cctype>
#include <climits>
#include <set>

// 简易 JSON 解析器（支持 \uXXXX 转义）
namespace simple_json {
    static std::string unescape_json_string(const std::string& s) {
        std::string result;
        for (size_t i = 0; i < s.size(); ++i) {
            if (s[i] == '\\' && i + 1 < s.size()) {
                char next = s[i+1];
                if (next == 'n') { result += '\n'; i++; }
                else if (next == 't') { result += '\t'; i++; }
                else if (next == 'r') { result += '\r'; i++; }
                else if (next == '\\') { result += '\\'; i++; }
                else if (next == '"') { result += '"'; i++; }
                else if (next == 'u' && i + 5 < s.size()) {
                    // \uXXXX
                    std::string hex = s.substr(i+2, 4);
                    int codepoint = std::stoi(hex, nullptr, 16);
                    
                    // UTF-8 编码
                    if (codepoint < 0x80) {
                        result += static_cast<char>(codepoint);
                    } else if (codepoint < 0x800) {
                        result += static_cast<char>(0xC0 | (codepoint >> 6));
                        result += static_cast<char>(0x80 | (codepoint & 0x3F));
                    } else {
                        result += static_cast<char>(0xE0 | (codepoint >> 12));
                        result += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
                        result += static_cast<char>(0x80 | (codepoint & 0x3F));
                    }
                    i += 5;
                } else {
                    result += s[i];
                }
            } else {
                result += s[i];
            }
        }
        return result;
    }
    
    static std::unordered_map<std::string, int> parse_vocab_json(const std::string& filepath) {
        std::ifstream f(filepath);
        if (!f.is_open()) {
            throw std::runtime_error("Cannot open: " + filepath);
        }
        
        std::stringstream buffer;
        buffer << f.rdbuf();
        std::string content = buffer.str();
        
        std::unordered_map<std::string, int> result;
        
        // 找所有 "key": value 对
        std::regex pattern(R"#("((?:[^"\\]|\\.)*)"\s*:\s*(\d+))#");
        auto words_begin = std::sregex_iterator(content.begin(), content.end(), pattern);
        auto words_end = std::sregex_iterator();
        
        for (auto it = words_begin; it != words_end; ++it) {
            std::string key_escaped = (*it)[1].str();
            std::string key = unescape_json_string(key_escaped);
            int value = std::stoi((*it)[2].str());
            result[key] = value;
        }
        
        return result;
    }
}

namespace ops {

// ============================================================================
// 构造与加载
// ============================================================================

GPT2BPETokenizer::GPT2BPETokenizer(const BPEConfig& config)
    : config_(config), vocab_size_(0) {
    build_byte_encoder();
}

void GPT2BPETokenizer::load() {
    load_vocab();
    load_merges();
    load_special_tokens();
    std::cout << "[GPT2BPETokenizer] Loaded: vocab_size=" << vocab_size_ 
              << ", merges=" << bpe_ranks_.size() << std::endl;
}

// ============================================================================
// Byte→Unicode 映射（与 HuggingFace 精确对齐）
// ============================================================================

void GPT2BPETokenizer::build_byte_encoder() {
    // 与 HF transformers 的 bytes_to_unicode() 完全一致
    std::vector<int> bs;
    std::vector<int> cs;
    
    // 可直接使用的范围
    auto push_range = [&](int start, int end) {
        for (int i = start; i <= end; ++i) {
            bs.push_back(i);
            cs.push_back(i);
        }
    };
    
    push_range(int('!'), int('~'));   // 33-126
    push_range(161, 172);             // ¡-¬
    push_range(174, 255);             // ®-ÿ
    
    // 剩余字节映射到 256+
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            n++;
        }
    }
    
    // 构建映射表
    byte_encoder_.clear();
    byte_decoder_.clear();
    
    for (size_t i = 0; i < 256; ++i) {
        uint8_t byte_val = static_cast<uint8_t>(bs[i]);
        int codepoint = cs[i];
        
        // UTF-8 编码该 codepoint
        std::string utf8_char;
        if (codepoint < 0x80) {
            utf8_char = std::string(1, static_cast<char>(codepoint));
        } else if (codepoint < 0x800) {
            utf8_char += static_cast<char>(0xC0 | (codepoint >> 6));
            utf8_char += static_cast<char>(0x80 | (codepoint & 0x3F));
        } else if (codepoint < 0x10000) {
            utf8_char += static_cast<char>(0xE0 | (codepoint >> 12));
            utf8_char += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
            utf8_char += static_cast<char>(0x80 | (codepoint & 0x3F));
        } else {
            utf8_char += static_cast<char>(0xF0 | (codepoint >> 18));
            utf8_char += static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F));
            utf8_char += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
            utf8_char += static_cast<char>(0x80 | (codepoint & 0x3F));
        }
        
        byte_encoder_[byte_val] = utf8_char;
        byte_decoder_[utf8_char] = byte_val;
    }
}

std::string GPT2BPETokenizer::bytes_to_unicode(const std::string& text) {
    std::string result;
    result.reserve(text.size() * 3);
    
    for (unsigned char byte : text) {
        result += byte_encoder_[byte];
    }
    
    return result;
}

std::string GPT2BPETokenizer::unicode_to_bytes(const std::string& unicode_text) {
    std::string result;
    result.reserve(unicode_text.size());
    
    // 按 UTF-8 解码，每个字符反查 byte
    for (size_t i = 0; i < unicode_text.size(); ) {
        unsigned char c = unicode_text[i];
        int char_len = 1;
        
        if ((c & 0x80) == 0) {
            char_len = 1;
        } else if ((c & 0xE0) == 0xC0) {
            char_len = 2;
        } else if ((c & 0xF0) == 0xE0) {
            char_len = 3;
        } else if ((c & 0xF8) == 0xF0) {
            char_len = 4;
        }
        
        std::string utf8_char = unicode_text.substr(i, char_len);
        auto it = byte_decoder_.find(utf8_char);
        if (it != byte_decoder_.end()) {
            result += static_cast<char>(it->second);
        }
        
        i += char_len;
    }
    
    return result;
}

// ============================================================================
// 加载 vocab/merges/special_tokens
// ============================================================================

void GPT2BPETokenizer::load_vocab() {
    auto parsed = simple_json::parse_vocab_json(config_.vocab_path);
    
    for (auto& [token, id] : parsed) {
        vocab_[token] = id;
        id_to_token_[id] = token;
    }
    
    vocab_size_ = vocab_.size();
}

void GPT2BPETokenizer::load_merges() {
    std::ifstream f(config_.merges_path);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open merges.txt: " + config_.merges_path);
    }
    
    std::string line;
    int rank = 0;
    
    // 跳过首行（#version: 0.2）
    std::getline(f, line);
    
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        std::istringstream iss(line);
        std::string a, b;
        if (iss >> a >> b) {
            std::string pair_key = a + " " + b;
            bpe_ranks_[pair_key] = rank++;
        }
    }
}

void GPT2BPETokenizer::load_special_tokens() {
    special_tokens_["<|endoftext|>"] = config_.eos_token_id;
}

// ============================================================================
// BPE 核心逻辑（完整实现）
// ============================================================================

std::vector<std::string> GPT2BPETokenizer::split_to_words(const std::string& text) {
    // GPT-2 正则（已 byte→unicode 映射后）：
    // 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
    //
    // 手工实现（不依赖 \p{...}）：
    // - 英文缩写：'s|'t|'re|'ve|'m|'ll|'d
    // - 字母+数字（含可选前导空格）
    // - 标点符号（含可选前导空格）
    // - 连续空白
    std::regex pattern(
        R"('s|'t|'re|'ve|'m|'ll|'d|)"
        R"( ?[a-zA-Z]+|)"
        R"( ?[0-9]+|)"
        R"( ?[^\s\w]+|)"
        R"(\s+(?!\S)|\s+)"
    );
    
    std::vector<std::string> words;
    auto words_begin = std::sregex_iterator(text.begin(), text.end(), pattern);
    auto words_end = std::sregex_iterator();
    
    for (auto it = words_begin; it != words_end; ++it) {
        std::string match = it->str();
        if (!match.empty()) {
            words.push_back(match);
        }
    }
    
    return words;
}

std::pair<int, int> GPT2BPETokenizer::get_best_pair(const std::vector<std::string>& word) {
    if (word.size() < 2) return {-1, INT_MAX};
    
    int best_rank = INT_MAX;
    int best_i = -1;
    
    for (size_t i = 0; i < word.size() - 1; ++i) {
        std::string pair_key = word[i] + " " + word[i+1];
        auto it = bpe_ranks_.find(pair_key);
        if (it != bpe_ranks_.end() && it->second < best_rank) {
            best_rank = it->second;
            best_i = static_cast<int>(i);
        }
    }
    
    return {best_i, best_rank};
}

std::vector<std::string> GPT2BPETokenizer::bpe(const std::string& token) {
    if (token.empty()) return {};
    
    // 如果整个 token 已在词表，直接返回
    if (vocab_.find(token) != vocab_.end()) {
        return {token};
    }
    
    // 按字符拆分（UTF-8）
    std::vector<std::string> word;
    for (size_t i = 0; i < token.size(); ) {
        unsigned char c = token[i];
        int len = 1;
        if ((c & 0x80) == 0) len = 1;
        else if ((c & 0xE0) == 0xC0) len = 2;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xF8) == 0xF0) len = 4;
        
        if (i + len > token.size()) len = token.size() - i;
        word.push_back(token.substr(i, len));
        i += len;
    }
    
    // BPE 合并循环
    while (word.size() > 1) {
        auto [best_i, best_rank] = get_best_pair(word);
        if (best_i == -1 || best_rank == INT_MAX) {
            break;
        }
        
        // 合并
        std::vector<std::string> new_word;
        for (size_t i = 0; i < word.size(); ) {
            if (static_cast<int>(i) == best_i) {
                new_word.push_back(word[i] + word[i+1]);
                i += 2;
            } else {
                new_word.push_back(word[i]);
                i++;
            }
        }
        word = new_word;
    }
    
    return word;
}

// ============================================================================
// Encode / Decode
// ============================================================================

std::vector<int> GPT2BPETokenizer::encode(const std::string& text,
                                          bool add_special_tokens,
                                          int max_length,
                                          bool truncation) {
    std::vector<int> ids;
    
    // 1. bytes → unicode 映射
    std::string unicode_text = bytes_to_unicode(text);
    
    // 2. 预切分
    auto words = split_to_words(unicode_text);
    
    // 3. 对每个 word 做 BPE
    for (const auto& word : words) {
        auto bpe_tokens = bpe(word);
        
        for (const auto& bpe_token : bpe_tokens) {
            auto it = vocab_.find(bpe_token);
            if (it != vocab_.end()) {
                ids.push_back(it->second);
            } else {
                // Byte-level 不应有未登录词，除非词表损坏
                std::cerr << "[ERROR] Token not in vocab: \"" << bpe_token << "\" (len=" << bpe_token.size() << ")" << std::endl;
            }
        }
    }
    
    // 4. 添加特殊符号
    if (add_special_tokens) {
        ids.push_back(config_.eos_token_id);
    }
    
    // 5. 截断
    if (truncation && max_length > 0 && static_cast<int>(ids.size()) > max_length) {
        ids.resize(max_length);
    }
    
    return ids;
}

std::string GPT2BPETokenizer::decode(const std::vector<int>& ids,
                                     bool skip_special_tokens) {
    std::string unicode_text;
    
    for (int id : ids) {
        // 跳过特殊符号
        if (skip_special_tokens && 
            (id == config_.eos_token_id || 
             id == config_.bos_token_id || 
             id == config_.pad_token_id)) {
            continue;
        }
        
        auto it = id_to_token_.find(id);
        if (it != id_to_token_.end()) {
            unicode_text += it->second;
        }
    }
    
    // Unicode → bytes
    std::string result = unicode_to_bytes(unicode_text);
    return result;
}

std::pair<std::vector<std::vector<int>>, std::vector<std::vector<int>>> 
GPT2BPETokenizer::batch_encode(const std::vector<std::string>& texts,
                               int max_length,
                               const std::string& padding,
                               bool truncation) {
    std::vector<std::vector<int>> all_ids;
    std::vector<std::vector<int>> all_masks;
    
    for (const auto& text : texts) {
        auto ids = encode(text, false, max_length, truncation);
        all_ids.push_back(ids);
    }
    
    int target_len = 0;
    if (padding == "max_length" && max_length > 0) {
        target_len = max_length;
    } else if (padding == "longest") {
        for (const auto& ids : all_ids) {
            target_len = std::max(target_len, static_cast<int>(ids.size()));
        }
    }
    
    for (auto& ids : all_ids) {
        int orig_len = ids.size();
        std::vector<int> mask(orig_len, 1);
        
        if (target_len > 0 && orig_len < target_len) {
            int pad_count = target_len - orig_len;
            ids.insert(ids.end(), pad_count, config_.pad_token_id);
            mask.insert(mask.end(), pad_count, 0);
        }
        
        all_masks.push_back(mask);
    }
    
    return {all_ids, all_masks};
}

std::string GPT2BPETokenizer::get_token_string(int token_id) const {
    auto it = id_to_token_.find(token_id);
    if (it != id_to_token_.end()) {
        return it->second;
    }
    return "";
}

}  // namespace ops
