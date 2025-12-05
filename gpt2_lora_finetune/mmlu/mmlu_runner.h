#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <utility>

#include "finetune_ops/graph/gpt2_model.h"
#include "finetune_ops/core/tokenizer_bpe.h"

namespace ops {

struct MCQItem {
    std::string subject;
    std::string question;
    std::string A, B, C, D;
    char answer; // 'A'|'B'|'C'|'D'
};

struct SubjectReport {
    std::string subject;
    int correct = 0;
    int total = 0;
    float accuracy() const { return total > 0 ? static_cast<float>(correct) / static_cast<float>(total) : 0.0f; }
};

struct MMLUResult {
    std::vector<SubjectReport> per_subject;
    float macro = 0.0f;
    float micro = 0.0f;
};

class MMLURunner {
public:
    // mmlu_root 指到 data/mmlu/data 目錄；split 為 "dev" 或 "test"
    MMLURunner(std::string mmlu_root, std::string split);

    // 掃描 split 目錄讀取所有 *.csv，按 subject 分組
    void load();

    // 執行評測，返回 macro/micro 與 per-subject
    MMLUResult evaluate(GPT2Model& model, GPT2BPETokenizer& tok, int fewshot_k = 0);

private:
    std::string mmlu_root_;
    std::string split_;
    std::unordered_map<std::string, std::vector<MCQItem>> subject_to_items_;

    static std::vector<std::string> parse_csv_line(const std::string& line);
    static void read_mmlu_csv(const std::string& path, std::vector<MCQItem>& out_items);
    static std::string build_prompt(const MCQItem& x, const std::vector<MCQItem>* shots = nullptr);
    static char predict_letter(const std::string& prompt, GPT2Model& model, GPT2BPETokenizer& tok);
};

} // namespace ops
