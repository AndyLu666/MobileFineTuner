"""
Download Gemma 3 1B model from HuggingFace and save in the same format as gemma-3-270m
"""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from safetensors.torch import save_file

def download_and_save_gemma_1b():
    # Gemma 3 1B pretrained model
    model_name = "google/gemma-3-1b-pt"
    output_dir = "gemma-3-1b"
    
    print(f"Downloading {model_name}...")
    print("Note: You may need to be logged in to HuggingFace and have accepted the model license.")
    print("Run `huggingface-cli login` if you haven't already.")
    
    # Create directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Download tokenizer
    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(output_dir)
    
    # Download config
    print("Downloading config...")
    config = AutoConfig.from_pretrained(model_name)
    config.save_pretrained(output_dir)
    
    # Download model
    print("Downloading model (this may take a while for 1B model)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Use float32 for C++ compatibility
        low_cpu_mem_usage=True
    )
    
    print("Saving model to safetensors...")
    state_dict = model.state_dict()
    
    # Handle shared weights (Gemma ties embed_tokens and lm_head)
    if "lm_head.weight" in state_dict and "model.embed_tokens.weight" in state_dict:
        # Clone lm_head.weight to break shared memory
        state_dict["lm_head.weight"] = state_dict["lm_head.weight"].clone()
    
    # Ensure contiguous tensors and convert to float32 for C++ compatibility
    state_dict = {k: v.contiguous().float() for k, v in state_dict.items()}
    save_file(state_dict, os.path.join(output_dir, "model.safetensors"))
    
    print(f"\nSuccessfully saved {model_name} to {output_dir}")
    print("\nModel info:")
    print(f"  hidden_size: {config.hidden_size}")
    print(f"  num_hidden_layers: {config.num_hidden_layers}")
    print(f"  num_attention_heads: {config.num_attention_heads}")
    print(f"  intermediate_size: {config.intermediate_size}")
    print(f"  vocab_size: {config.vocab_size}")

if __name__ == "__main__":
    download_and_save_gemma_1b()

