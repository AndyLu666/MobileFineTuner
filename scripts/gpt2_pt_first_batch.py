import os, json, numpy as np, torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, AutoConfig

pretrained_dir = "/Users/tony/Documents/FT（gemma完成版）/gpt2_lora_finetune/pretrained/gpt2"
wt2_dir = "/Users/tony/Documents/FT（gemma完成版）/data/wikitext2/wikitext-2-raw"
out_dir = "/Users/tony/Documents/FT（gemma完成版）/runs/gpt2_pt_align"
os.makedirs(out_dir, exist_ok=True)

seq_len = 128
dtype = torch.float32
device = torch.device("cpu")

tok = GPT2TokenizerFast.from_pretrained(pretrained_dir)
config = AutoConfig.from_pretrained(pretrained_dir)
model = GPT2LMHeadModel.from_pretrained(pretrained_dir)
model.eval().to(device)

eos_id = 50256
with open(os.path.join(wt2_dir, "wiki.valid.raw"), "r", encoding="utf-8") as f:
    lines = [line.rstrip("\n") for line in f]

# replicate C++ pack: encode each line (no special tokens), then append EOS between lines, and ensure an EOS at the end
ids = []
for line in lines:
    if line == "" and True:  # insert_eos_between_lines=True in C++
        # still append eos for empty lines (matches C++)
        ids.append(eos_id)
        continue
    enc = tok.encode(line, add_special_tokens=False)
    ids.extend(enc)
    ids.append(eos_id)

if not ids or ids[-1] != eos_id:
    ids.append(eos_id)

# take the first window S+1 to construct input/label like C++
assert len(ids) >= seq_len + 1
window = ids[:seq_len + 1]
input_ids = torch.tensor(window[:-1], dtype=torch.long, device=device).unsqueeze(0)  # [1,S]
labels    = torch.tensor(window[1:],  dtype=torch.long, device=device).unsqueeze(0)  # [1,S]
attn_mask = torch.ones_like(input_ids, dtype=torch.float32)

# HF computes loss if passing labels=input_ids (with internal shift). To match C++ loss over [S] pairs, we compute explicitly.
with torch.no_grad():
    out = model(input_ids=input_ids, attention_mask=attn_mask)
    logits = out.logits  # [1,S,V]
    log_probs = torch.log_softmax(logits, dim=-1)
    nll = -log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # [1,S]
    mean_nll = nll.mean().item()

print(f"PT first-batch NLL (S={seq_len}): {mean_nll:.4f}  PPL={np.exp(mean_nll):.2f}")

# Dump embeddings for alignment
with torch.no_grad():
    wte = model.transformer.wte  # token embeddings
    wpe = model.transformer.wpe  # position embeddings
    emb = wte(input_ids)         # [1,S,C]
    pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    pos = wpe(pos_ids).squeeze(0)  # [S,C]
    emb_plus_pos = emb + pos.unsqueeze(0)

np.save(os.path.join(out_dir, "input_ids.npy"), input_ids.cpu().numpy().astype(np.int32))
np.save(os.path.join(out_dir, "embeddings.npy"), emb.cpu().numpy().astype(np.float32))
np.save(os.path.join(out_dir, "pos_emb.npy"), pos.cpu().numpy().astype(np.float32))
np.save(os.path.join(out_dir, "emb_plus_pos.npy"), emb_plus_pos.cpu().numpy().astype(np.float32))
print(f"Saved PT dumps to {out_dir}")
