/**
 * @file quick_eval_ppl.cpp
 * @brief 快速PPL评测（只评前N个batch）
 */

#include <iostream>
#include <string>
#include <cmath>

#include "finetune_ops/graph/gpt2_model.h"
#include "finetune_ops/graph/safetensors_loader.h"
#include "finetune_ops/data/wikitext2_dataset.h"
#include "finetune_ops/core/tokenizer_bpe.h"
#include "finetune_ops/core/lm_loss.h"

using namespace std;
using namespace ops;

int main() {
    try {
        cout << "========== Quick PPL Test (10 batches) ==========" << endl;
        const std::string pretrained_dir = "/Users/tony/Documents/重新开始/gpt2_lora_finetune/pretrained/gpt2";
        
        // 加载模型
        GPT2Config cfg;
        try {
            cfg = GPT2Config::from_pretrained(pretrained_dir);
            cout << "  ✓ Config: layers=" << cfg.n_layer
                 << ", hidden=" << cfg.n_embd
                 << ", heads=" << cfg.n_head << endl;
        } catch (const std::exception& e) {
            cout << "  ⚠️ Failed to read config, fallback default: " << e.what() << endl;
        }
        GPT2Model model(cfg);
        model.tie_weights();
        SafeTensorsReader reader(pretrained_dir + "/model.safetensors");
        reader.parse_header();
        auto mapping = GPT2KeyMapper::generate_gpt2_mapping(cfg.n_layer);
        for (const auto& kv : mapping) {
            try {
                auto info = reader.get_tensor_info(kv.second);
                if (!info.dtype.empty()) {
                    auto t = reader.load_tensor(kv.second, false);
                    model.assign_weight(kv.first, t);
                }
            } catch (...) {}
        }

        // Tokenizer
        auto tok_cfg = BPEConfig::from_pretrained(pretrained_dir);
        GPT2BPETokenizer tok(tok_cfg);
        tok.load();

        // Dataset
        WT2Config data_cfg;
        data_cfg.train_path = "/Users/tony/Documents/重新开始/data/wikitext2/wikitext-2-raw/wiki.train.raw";
        data_cfg.valid_path = "/Users/tony/Documents/重新开始/data/wikitext2/wikitext-2-raw/wiki.valid.raw";
        data_cfg.seq_len = 256;
        data_cfg.stride  = -1;
        data_cfg.drop_last = false;
        WikiText2Dataset dataset(data_cfg, &tok);
        dataset.load(Split::Valid);

        // 只评10个batch
        cout << "Evaluating 10 batches..." << endl;
        double sum_loss = 0.0;
        int64_t sum_tokens = 0;

        for (int i = 0; i < 10; ++i) {
            auto batch = dataset.next_batch(1, false);
            if (!batch.input_ids) break;
            
            auto logits = model.forward(batch.input_ids, batch.attention_mask);
            auto loss = lm_cross_entropy(logits, batch.labels, -100, "mean");
            float loss_val = loss->data<float>()[0];
            
            const int32_t* labels = batch.labels->data<int32_t>();
            int64_t valid = 0;
            for (int64_t j = 0; j < batch.labels->numel(); ++j) {
                if (labels[j] != -100) valid++;
            }
            
            sum_loss += loss_val * valid;
            sum_tokens += valid;
            
            cout << "  Batch " << i << ": loss=" << loss_val << ", tokens=" << valid << endl;
        }

        float mean_nll = (sum_tokens > 0) ? (sum_loss / sum_tokens) : 0.0f;
        float ppl = exp(mean_nll);

        cout << "\n========== Results (10 batches) ==========" << endl;
        printf("tokens=%lld  nll=%.4f  ppl=%.2f\n", (long long)sum_tokens, mean_nll, ppl);
        cout << "\n✅ Done." << endl;
        return 0;
    } catch (const exception& e) {
        cerr << "\n❌ Exception: " << e.what() << endl;
        return 1;
    }
}
