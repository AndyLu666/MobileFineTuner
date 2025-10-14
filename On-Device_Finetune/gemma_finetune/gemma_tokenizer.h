#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <regex>
#include <algorithm>

class GemmaTokenizer {
private:
    std::unordered_map<std::string, int> vocab_;
    std::vector<std::string> idx_to_token_;
    
    // Gemma special token IDs
    int pad_token_id_ = 0;      // <pad>
    int unk_token_id_ = 1;      // <unk>
    int bos_token_id_ = 2;      // <bos>
    int eos_token_id_ = 3;      // <eos>
    
    // Simplified SentencePiece rules
    std::vector<std::pair<std::string, std::string>> merge_rules_;

public:
    GemmaTokenizer(const std::string& vocab_file) {
        if (!load_vocab(vocab_file)) {
            create_comprehensive_vocab();
        }
    }

    bool load_vocab(const std::string& vocab_file) {
        std::ifstream file(vocab_file);
        if (!file.is_open()) {
            std::cout << "Warning: Unable to open Gemma vocabulary file: " << vocab_file << std::endl;
            return false;
        }

        // Try to load JSON format vocabulary
        std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        
        // Simplified JSON parsing (should use JSON library in production)
        if (content.find("{") != std::string::npos) {
            return parse_json_vocab(content);
        } else {
            return parse_text_vocab(content);
        }
    }

    bool parse_json_vocab(const std::string& content) {
        // Simplified JSON vocabulary parsing
        // Should use nlohmann/json or similar library in production
        std::cout << "Parsing JSON format Gemma vocabulary..." << std::endl;
        
        size_t start = content.find("{");
        size_t end = content.rfind("}");
        if (start == std::string::npos || end == std::string::npos) {
            return false;
        }

        std::string json_content = content.substr(start + 1, end - start - 1);
        
        // Use regex to extract key-value pairs
        std::regex pattern(R"(\"([^\"]+)\":\s*(\d+))");
        std::sregex_iterator iter(json_content.begin(), json_content.end(), pattern);
        std::sregex_iterator end_iter;

        int loaded_count = 0;
        for (; iter != end_iter; ++iter) {
            std::string token = (*iter)[1].str();
            int id = std::stoi((*iter)[2].str());
            
            vocab_[token] = id;
            if (id >= static_cast<int>(idx_to_token_.size())) {
                idx_to_token_.resize(id + 1);
            }
            idx_to_token_[id] = token;
            loaded_count++;
            
            if (loaded_count >= 256000) break; // Limit vocabulary size
        }

        if (loaded_count > 0) {
            std::cout << "Successfully loaded Gemma JSON vocabulary: " << loaded_count << " tokens" << std::endl;
            return true;
        }
        return false;
    }

    bool parse_text_vocab(const std::string& content) {
        std::cout << "Parsing text format Gemma vocabulary..." << std::endl;
        
        std::istringstream stream(content);
        std::string line;
        int id = 0;
        int loaded_count = 0;
        
        while (std::getline(stream, line) && id < 256000) {
            if (!line.empty()) {
                vocab_[line] = id;
                if (id >= static_cast<int>(idx_to_token_.size())) {
                    idx_to_token_.resize(id + 1);
                }
                idx_to_token_[id] = line;
                id++;
                loaded_count++;
            }
        }

        if (loaded_count > 0) {
            std::cout << "Successfully loaded Gemma text vocabulary: " << loaded_count << " tokens" << std::endl;
            return true;
        }
        return false;
    }

    void create_comprehensive_vocab() {
        std::cout << "Creating comprehensive Gemma vocabulary (SentencePiece style)..." << std::endl;
        
        vocab_.clear();
        idx_to_token_.clear();
        idx_to_token_.resize(256000);
        
        int current_id = 0;
        
        // 1. Special tokens
        std::vector<std::string> special_tokens = {
            "<pad>", "<unk>", "<bos>", "<eos>", 
            "<mask>", "<cls>", "<sep>", "<s>", "</s>"
        };
        
        for (const auto& token : special_tokens) {
            vocab_[token] = current_id;
            idx_to_token_[current_id] = token;
            current_id++;
        }
        
        // 2. Basic ASCII characters
        for (int i = 32; i < 127; ++i) {
            std::string ch(1, static_cast<char>(i));
            if (vocab_.find(ch) == vocab_.end()) {
                vocab_[ch] = current_id;
                idx_to_token_[current_id] = ch;
                current_id++;
            }
        }
        
        // 3. Unicode characters (simplified version)
        std::vector<std::string> unicode_chars = {
            "▁", "▁a", "▁b", "▁c", "▁d", "▁e", "▁f", "▁g", "▁h", "▁i", "▁j", "▁k", "▁l", "▁m",
            "▁n", "▁o", "▁p", "▁q", "▁r", "▁s", "▁t", "▁u", "▁v", "▁w", "▁x", "▁y", "▁z",
            "▁A", "▁B", "▁C", "▁D", "▁E", "▁F", "▁G", "▁H", "▁I", "▁J", "▁K", "▁L", "▁M",
            "▁N", "▁O", "▁P", "▁Q", "▁R", "▁S", "▁T", "▁U", "▁V", "▁W", "▁X", "▁Y", "▁Z"
        };
        
        for (const auto& token : unicode_chars) {
            if (current_id < 256000) {
                vocab_[token] = current_id;
                idx_to_token_[current_id] = token;
                current_id++;
            }
        }
        
        // 4. Common subwords and vocabulary
        std::vector<std::string> common_subwords = {
            "▁the", "▁and", "▁to", "▁of", "▁a", "▁in", "▁is", "▁it", "▁you", "▁that",
            "▁he", "▁was", "▁for", "▁on", "▁are", "▁as", "▁with", "▁his", "▁they", "▁I",
            "▁at", "▁be", "▁this", "▁have", "▁from", "▁or", "▁one", "▁had", "▁by", "▁word",
            "ing", "er", "ed", "ly", "ion", "tion", "ation", "ment", "ness", "ity",
            "al", "ic", "ous", "ful", "less", "able", "ible", "ive", "ary", "ory",
            "Machine", "▁learning", "▁model", "▁data", "▁training", "▁neural", "▁network",
            "▁artificial", "▁intelligence", "▁deep", "▁transformer", "▁attention",
            "▁language", "▁processing", "▁natural", "▁computer", "▁science"
        };
        
        for (const auto& token : common_subwords) {
            if (current_id < 256000) {
                vocab_[token] = current_id;
                idx_to_token_[current_id] = token;
                current_id++;
            }
        }
        
        // 5. Number and punctuation combinations
        for (int i = 0; i < 1000 && current_id < 256000; ++i) {
            std::string num_token = "▁" + std::to_string(i);
            if (vocab_.find(num_token) == vocab_.end()) {
                vocab_[num_token] = current_id;
                idx_to_token_[current_id] = num_token;
                current_id++;
            }
        }
        
        // 6. Fill remaining vocabulary space
        while (current_id < 256000) {
            std::string filler_token = "<extra_token_" + std::to_string(current_id) + ">";
            vocab_[filler_token] = current_id;
            idx_to_token_[current_id] = filler_token;
            current_id++;
        }
        
        std::cout << "Created comprehensive Gemma vocabulary: " << vocab_.size() << " tokens" << std::endl;
        std::cout << "Vocabulary coverage:" << std::endl;
        std::cout << "  - Special tokens: " << special_tokens.size() << std::endl;
        std::cout << "  - ASCII characters: 95" << std::endl;
        std::cout << "  - Unicode subwords: " << unicode_chars.size() << std::endl;
        std::cout << "  - Common vocabulary: " << common_subwords.size() << std::endl;
        std::cout << "  - Number tokens: 1000" << std::endl;
    }

    std::vector<int> encode(const std::string& text) {
        std::vector<int> tokens;
        tokens.push_back(bos_token_id_);
        
        // Simplified SentencePiece style encoding
        std::string processed_text = preprocess_text(text);
        
        // Greedy matching of longest substring
        size_t i = 0;
        while (i < processed_text.length()) {
            std::string best_match;
            int best_token_id = unk_token_id_;
            
            // Try matching from longest to shortest
            for (size_t len = std::min(processed_text.length() - i, (size_t)20); len > 0; --len) {
                std::string candidate = processed_text.substr(i, len);
                auto it = vocab_.find(candidate);
                if (it != vocab_.end()) {
                    best_match = candidate;
                    best_token_id = it->second;
                    break;
                }
            }
            
            tokens.push_back(best_token_id);
            i += best_match.empty() ? 1 : best_match.length();
        }
        
        tokens.push_back(eos_token_id_);
        return tokens;
    }

    std::string preprocess_text(const std::string& text) {
        // Simplified preprocessing: add space markers
        std::string result = "▁" + text;
        
        // Replace spaces with ▁
        std::regex space_pattern(R"(\s+)");
        result = std::regex_replace(result, space_pattern, "▁");
        
        return result;
    }

    std::string decode(const std::vector<int>& tokens) {
        std::string result;
        
        for (size_t i = 1; i < tokens.size() - 1; ++i) { // Skip BOS and EOS
            int token_id = tokens[i];
            if (token_id >= 0 && token_id < static_cast<int>(idx_to_token_.size())) {
                std::string token = idx_to_token_[token_id];
                
                // Remove ▁ marker and convert to space
                if (token.substr(0, 3) == "▁") {
                    if (token.length() > 3) {
                        result += " " + token.substr(3);
                    } else {
                        result += " ";
                    }
                } else {
                    result += token;
                }
            }
        }
        
        return result;
    }

    int get_pad_token_id() const { return pad_token_id_; }
    int get_unk_token_id() const { return unk_token_id_; }
    int get_bos_token_id() const { return bos_token_id_; }
    int get_eos_token_id() const { return eos_token_id_; }
    int vocab_size() const { return static_cast<int>(vocab_.size()); }
    
    void print_tokenizer_info() const {
        std::cout << "Gemma Tokenizer info:" << std::endl;
        std::cout << "  - Vocabulary size: " << vocab_size() << std::endl;
        std::cout << "  - PAD token: " << pad_token_id_ << " (" << idx_to_token_[pad_token_id_] << ")" << std::endl;
        std::cout << "  - UNK token: " << unk_token_id_ << " (" << idx_to_token_[unk_token_id_] << ")" << std::endl;
        std::cout << "  - BOS token: " << bos_token_id_ << " (" << idx_to_token_[bos_token_id_] << ")" << std::endl;
        std::cout << "  - EOS token: " << eos_token_id_ << " (" << idx_to_token_[eos_token_id_] << ")" << std::endl;
    }
};
