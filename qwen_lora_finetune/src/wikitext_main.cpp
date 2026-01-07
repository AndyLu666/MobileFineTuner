#include <iostream>
#include <string>
#include <vector>

#include "finetune_ops/core/tokenizer_bpe.h"
#include "finetune_ops/core/lm_loss.h"
#include "finetune_ops/core/ops.h"
#include "finetune_ops/optim/adam.h"
#include "finetune_ops/data/wikitext2_dataset.h"
#include "finetune_ops/graph/safetensors_loader.h"
#include "finetune_ops/core/memory_manager.h"
#include "finetune_ops/graph/qwen_model.h"

using namespace ops;

struct Args {
    std::string model_dir = "Qwen2.5-0.5B";
    std::string data_dir = "data/wikitext2/wikitext-2-raw";
    int seq_len = 1024;
    int batch_size = 1;
    int grad_accum_steps = 1; // gradient accumulation steps
    int max_steps = -1; // -1 means auto-run one epoch
    float lr = 2e-4f;
    int lora_r = 8;
    float lora_alpha = 16.0f;
    float lora_dropout = 0.05f;
    bool align_mode = false; // alignment mode: single step with loss print
};

static void parse_args(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto get = [&](const char* name) {
            if (i + 1 >= argc) throw std::runtime_error("arg missing val");
            if (k != name) throw std::runtime_error("unexpected arg");
            return std::string(argv[++i]);
        };
        if (k == "--model_dir") args.model_dir = get("--model_dir");
        else if (k == "--data_dir") args.data_dir = get("--data_dir");
        else if (k == "--seq_len") args.seq_len = std::stoi(get("--seq_len"));
        else if (k == "--batch_size") args.batch_size = std::stoi(get("--batch_size"));
        else if (k == "--grad_accum_steps") args.grad_accum_steps = std::stoi(get("--grad_accum_steps"));
        else if (k == "--max_steps") args.max_steps = std::stoi(get("--max_steps"));
        else if (k == "--lr") args.lr = std::stof(get("--lr"));
        else if (k == "--lora_r") args.lora_r = std::stoi(get("--lora_r"));
        else if (k == "--lora_alpha") args.lora_alpha = std::stof(get("--lora_alpha"));
        else if (k == "--lora_dropout") args.lora_dropout = std::stof(get("--lora_dropout"));
        else if (k == "--align_mode") args.align_mode = true;
    }
}

int main(int argc, char** argv) {
    Args args;
    parse_args(argc, argv, args);
    try {
        std::cout << "\n========== Qwen2.5-0.5B LoRA Finetune (WikiText-2, C++) ==========\n";
        // 1) tokenizer
        auto tok_cfg = QwenTokenizerConfig::from_pretrained(args.model_dir);
        QwenBPETokenizer tokenizer(tok_cfg);
        tokenizer.load();

        // 2) config + model
        QwenConfig qcfg = QwenConfig::from_pretrained(args.model_dir + "/config.json");
        QwenModel model(qcfg);

        // 3) load weights
        SafeTensorsReader reader(args.model_dir + "/model.safetensors");
        reader.parse_header();
        auto mapping = QwenKeyMapper::generate_qwen_mapping(qcfg.num_hidden_layers);
        SafeTensorsLoadOptions load_opts;
        load_opts.verbose = false;           // disable per-tensor log to avoid flooding
        load_opts.transpose_linear = true;   // HF Linear [out,in] -> internal [in,out]
        auto tensors = reader.load_tensors_mapped(mapping, load_opts);
        for (auto& kv : tensors) model.assign_weight(kv.first, kv.second);

        // 4) LoRA init
        model.init_lora(args.lora_r, args.lora_alpha, args.lora_dropout);
        model.freeze_base();
        auto lora_params = model.get_lora_parameters();

        // 5) dataset
        WT2Config dcfg;
        dcfg.train_path = args.data_dir + "/wiki.train.raw";
        dcfg.valid_path = args.data_dir + "/wiki.valid.raw";
        dcfg.test_path  = args.data_dir + "/wiki.test.raw";
        dcfg.seq_len = args.seq_len;
        dcfg.insert_eos_between_lines = true;
        dcfg.stride = -1;
        dcfg.eos_id = tok_cfg.eos_token_id;
        dcfg.streaming_mode = false; // WikiText-2 is small; load in one shot
        
        WikiText2Dataset ds(
            dcfg,
            [&](const std::string& text) -> std::vector<int32_t> {
                return tokenizer.encode(text);
            }
        );
        ds.load(Split::Train);
        
        // Print dataset info and derive steps per epoch
        size_t num_seqs = ds.num_sequences();
        size_t total_micro_batches = (num_seqs + args.batch_size - 1) / args.batch_size;
        size_t steps_per_epoch = (total_micro_batches + args.grad_accum_steps - 1) / args.grad_accum_steps;
        
        std::cout << "[Dataset] Total sequences: " << num_seqs << std::endl;
        std::cout << "[Config] Micro batch size: " << args.batch_size << std::endl;
        std::cout << "[Config] Grad accum steps: " << args.grad_accum_steps << std::endl;
        std::cout << "[Config] Effective batch size: " << args.batch_size * args.grad_accum_steps << std::endl;
        std::cout << "[Dataset] Steps per epoch: " << steps_per_epoch << std::endl;
        
        // If max_steps <= 0, fall back to one epoch
        if (args.max_steps <= 0) {
            args.max_steps = steps_per_epoch;
            std::cout << "[Dataset] Auto-set max_steps to " << args.max_steps << " (1 epoch)" << std::endl;
        }

        // 6) optimizer
        AdamConfig opt_cfg;
        opt_cfg.learning_rate = args.lr;
        Adam opt(opt_cfg);

        // 7) training loop
        int step = 0;
        float accum_loss = 0.0f;
        int accum_counter = 0;
        
        auto batch = ds.next_batch(args.batch_size, false);
        
        while (step < args.max_steps && batch.input_ids) {
            // Forward
            auto logits = model.forward(batch.input_ids, batch.attention_mask);
            auto loss = lm_cross_entropy(logits, batch.labels, -100, "mean");
            
            // accumulate loss
            accum_loss += loss->data<float>()[0];
            
            // Backward (with gradient accumulation)
            loss->backward();

            accum_counter++;
            
            // Update parameters once accumulation target is met
            if (accum_counter >= args.grad_accum_steps) {
                std::vector<TensorPtr> grads;
                grads.reserve(lora_params.size());

                if (args.grad_accum_steps > 1) {
                    float scale = 1.0f / static_cast<float>(args.grad_accum_steps);
                    for (auto& p : lora_params) {
                        auto g = p->grad();
                        if (g) {
                            float* d = g->data<float>();
                            int64_t n = g->numel();
                            for (int64_t i = 0; i < n; ++i) d[i] *= scale;
                        }
                    }
                }

                for (auto& p : lora_params) grads.push_back(p->grad());

                // Print gradient L2 on first step for observation
                if (step == 0) {
                    double gsum = 0.0;
                    int64_t gcount = 0;
                    for (auto& g : grads) {
                        if (!g) continue;
                        const float* d = g->data<float>();
                        for (int64_t i = 0; i < g->numel(); ++i) { gsum += d[i] * d[i]; gcount++; }
                    }
                    double gnorm = (gcount > 0) ? std::sqrt(gsum) : 0.0;
                    std::cout << "[debug] grad L2 (all LoRA params) = " << gnorm << std::endl;
                }

                opt.step(lora_params, grads);
                for (auto& p : lora_params) p->zero_grad();

                std::cout << "[step " << (step + 1) << "/" << args.max_steps
                          << "] loss=" << (accum_loss / args.grad_accum_steps) << std::endl;

                MemoryManager::instance().force_cleanup();
                if (((step + 1) % 20) == 0) MemoryManager::instance().clear_unused_memory();
                if ((step + 1) % 10 == 0)   MemoryManager::instance().print_memory_stats();
                
                accum_loss = 0.0f;
                accum_counter = 0;
                step++;
                
                if (args.align_mode) break; 
            } else {
                MemoryManager::instance().force_cleanup();
            }
            
            // Next batch
            batch = ds.next_batch(args.batch_size, false);
        }
        std::cout << "✅ Finetune finished." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        return 1;
    }
    return 0;
}


