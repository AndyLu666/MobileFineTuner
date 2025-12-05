import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from safetensors.torch import save_file

def download_and_save_gpt2_medium():
    model_name = "gpt2-medium"
    output_dir = "gpt2_lora_finetune/pretrained/gpt2-medium"
    
    print(f"Downloading {model_name}...")
    
    # Create directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Download model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    config = GPT2Config.from_pretrained(model_name)
    
    print("Saving tokenizer...")
    tokenizer.save_pretrained(output_dir)
    
    print("Saving config...")
    config.save_pretrained(output_dir)
    
    print("Saving model to safetensors...")
    state_dict = model.state_dict()
    
    # Handle shared weights (gpt2 ties weights)
    if "lm_head.weight" in state_dict and "transformer.wte.weight" in state_dict:
        # Clone lm_head.weight to break shared memory
        state_dict["lm_head.weight"] = state_dict["lm_head.weight"].clone()

    # Ensure contiguous tensors for safe saving
    state_dict = {k: v.contiguous() for k, v in state_dict.items()}
    save_file(state_dict, os.path.join(output_dir, "model.safetensors"))
    
    print(f"Successfully saved {model_name} to {output_dir}")

if __name__ == "__main__":
    download_and_save_gpt2_medium()

