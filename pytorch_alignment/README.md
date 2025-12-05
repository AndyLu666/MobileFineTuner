# PyTorch LoRA Alignment Scripts

This folder contains minimal PyTorch implementations of the two training flows you already have in C++:

- `gpt2_lora_finetune.py` mirrors `gpt2_lora_finetune/main.cpp`
- `gemma_lora_finetune.py` mirrors `operators/finetune_ops/optim/train_lora_gemma.cpp`

They intentionally stick to the same hyperparameters, data handling (WikiText-2 raw files), warmup + cosine/linear schedulers, gradient accumulation, and LoRA target topology so you can compare losses/perplexity and adapter weights against your C++ runs.

Both scripts expect local model/tokenizer directories and raw WikiText-2 text files (wiki.train.raw, wiki.valid.raw, wiki.test.raw).

Run examples:

```bash
# GPT-2 LoRA
python pytorch_alignment/gpt2_lora_finetune.py \
  --data_dir data/wikitext2/wikitext-2-raw \
  --pretrained_dir pretrained/gpt2 \  # small
  # or --pretrained_dir pretrained/gpt2-medium  # medium
  --lora_out results/gpt2_lora_pt \
  --epochs 1 --batch_size 8 --grad_accum_steps 1 --seq_len 128 \
  --rank 8 --alpha 16 --lr 2e-4 --lora_dropout 0.0 --data_fraction 1.0

# Gemma LoRA
python pytorch_alignment/gemma_lora_finetune.py \
  --model_dir gemma-3-270m \   # or gemma-3-1b
  --data_dir data/wikitext2/wikitext-2-raw \
  --output_dir results/gemma_lora_pt \
  --epochs 1 --batch 4 --grad_accum 1 --seq_len 256 \
  --learning_rate 2e-4 --warmup_ratio 0.03 --max_grad_norm 1.0 \
  --target_mode full --lora_r 8 --lora_alpha 32 --lora_dropout 0.1 \
  --torch_dtype float32  # 推荐对齐 C++ 跑 float32（尤其 CPU）
```

Notes:
- Only LoRA parameters are trainable; base weights stay frozen.
- LR schedule matches C++: warmup then cosine to 10% base LR (GPT-2) or warmup then linear/cosine depending on `--lr_scheduler` (Gemma).
- Perplexity uses the same shift (logits[:-1] vs labels[1:]) as your `lm_cross_entropy`.
- Outputs are standard PEFT adapters (`adapter_model.safetensors`) so you can load back into HF or compare numerically.***
