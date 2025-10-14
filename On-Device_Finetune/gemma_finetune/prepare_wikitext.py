#!/usr/bin/env python3
"""
Prepare WikiText dataset for Gemma fine-tuning
"""

import json
import os
from datasets import load_dataset
from transformers import GemmaTokenizer

def prepare_wikitext():
    print("Preparing WikiText dataset for Gemma fine-tuning...")
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    try:
        # Load Gemma tokenizer
        print("Loading Gemma tokenizer...")
        tokenizer = GemmaTokenizer.from_pretrained("google/gemma-2b")
        print(f"Gemma tokenizer loaded successfully, vocabulary size: {tokenizer.vocab_size}")
        
        # Load WikiText-2 dataset
        print("Loading WikiText-2 dataset...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        print(f"Dataset loaded successfully, {len(dataset)} records total")
        
        # Process data
        processed_data = []
        max_samples = 1000  # Limit sample count for testing
        
        print("Processing and tokenizing data...")
        for i, example in enumerate(dataset):
            if i >= max_samples:
                break
                
            text = example['text'].strip()
            if len(text) > 50:  # Filter out text that's too short
                processed_data.append({"text": text})
                
            if (i + 1) % 100 == 0:
                print(f"  Processing progress: {i + 1}/{min(max_samples, len(dataset))}")
        
        # Save as JSONL format
        output_file = 'data/wikitext2_train.jsonl'
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in processed_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"Data processing complete!")
        print(f"  - Output file: {output_file}")
        print(f"  - Processed samples: {len(processed_data)}")
        print(f"  - File size: {os.path.getsize(output_file)/1024/1024:.2f} MB")
        
        # Create simplified tokenizer config (for C++ code)
        tokenizer_config = {
            "vocab_size": tokenizer.vocab_size,
            "pad_token_id": tokenizer.pad_token_id,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "model_type": "gemma"
        }
        
        config_file = 'data/tokenizer_config.json'
        with open(config_file, 'w') as f:
            json.dump(tokenizer_config, f, indent=2)
        
        print(f"Tokenizer config saved: {config_file}")
        
    except Exception as e:
        print(f"Warning: Unable to load HuggingFace data, creating sample data: {e}")
        create_sample_data()

def create_sample_data():
    """Create sample data for testing"""
    print("Creating Gemma sample training data...")
    
    # Sample text data
    sample_texts = [
        "Gemma is a family of lightweight, state-of-the-art open models built from the same research and technology used to create the Gemini models.",
        "Machine learning has revolutionized the way we process and understand data, enabling breakthroughs in artificial intelligence.",
        "The transformer architecture has become the foundation for most modern language models, providing unprecedented capabilities.",
        "Fine-tuning allows pre-trained models to adapt to specific tasks and domains with relatively small amounts of data.",
        "Edge computing brings artificial intelligence capabilities closer to data sources, reducing latency and improving efficiency.",
        "Natural language processing has advanced significantly with the development of large language models and attention mechanisms.",
        "Deep learning models require careful optimization and regularization techniques to achieve optimal performance on various tasks.",
        "The attention mechanism allows models to focus on relevant parts of the input when making predictions or generating text.",
        "Gradient descent and its variants are fundamental optimization algorithms used in training neural networks effectively.",
        "Transfer learning enables models to leverage knowledge from pre-training to perform well on downstream tasks."
    ]
    
    # Expand dataset
    processed_data = []
    for i in range(100):  # Generate 100 samples
        for j, text in enumerate(sample_texts):
            # Add some variation
            modified_text = f"Sample {i*len(sample_texts)+j+1}: {text}"
            processed_data.append({"text": modified_text})
    
    # Save as JSONL format
    output_file = 'data/wikitext2_train.jsonl'
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in processed_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # Create tokenizer config
    tokenizer_config = {
        "vocab_size": 256000,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "model_type": "gemma"
    }
    
    config_file = 'data/tokenizer_config.json'
    with open(config_file, 'w') as f:
        json.dump(tokenizer_config, f, indent=2)
    
    print(f"Sample data creation complete!")
    print(f"  - Output file: {output_file}")
    print(f"  - Sample count: {len(processed_data)}")
    print(f"  - Tokenizer config: {config_file}")

if __name__ == "__main__":
    prepare_wikitext()
