// Minimal placeholder for quick LoRA evaluation.
// Purpose: provide a valid entry point so the target links successfully.

#include <iostream>

int main(int argc [[maybe_unused]], char** argv [[maybe_unused]]) {
    std::cout << "quick_eval_lora: placeholder binary.\n"
              << "Use eval_ppl for perplexity or gpt2_lora_finetune for training.\n";
    return 0;
}
