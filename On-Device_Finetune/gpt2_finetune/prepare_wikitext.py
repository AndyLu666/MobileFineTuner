import os
import torch
from datasets import load_dataset
from transformers import GPT2Tokenizer
import json
from tqdm import tqdm

def download_and_prepare_wikitext():
    """Download and prepare WikiText-2 dataset"""
    print("Downloading WikiText-2 dataset...")
    
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    
    print("Dataset information:")
    print(f"Training set size: {len(dataset['train'])}")
    print(f"Validation set size: {len(dataset['validation'])}")
    print(f"Test set size: {len(dataset['test'])}")
    
    tokenizer = GPT2Tokenizer.from_pretrained('models/gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    for split_name in ['train', 'validation', 'test']:
        print(f"\nProcessing {split_name} set...")
        split_data = dataset[split_name]
        
        processed_data = []
        total_tokens = 0
        max_tokens = 0
        valid_samples = 0
        
        for i, example in enumerate(tqdm(split_data)):
            text = example['text'].strip()
            
            if len(text) < 50:
                continue
            
            tokens = tokenizer.encode(text)
            token_count = len(tokens)
            
            if token_count > 1024:
                for start in range(0, token_count, 512):
                    end = min(start + 512, token_count)
                    chunk_tokens = tokens[start:end]
                    
                    if len(chunk_tokens) >= 50:
                        chunk_text = tokenizer.decode(chunk_tokens)
                        processed_data.append({
                            'text': chunk_text,
                            'token_count': len(chunk_tokens)
                        })
                        total_tokens += len(chunk_tokens)
                        max_tokens = max(max_tokens, len(chunk_tokens))
                        valid_samples += 1
            else:
                processed_data.append({
                    'text': text,
                    'token_count': token_count
                })
                total_tokens += token_count
                max_tokens = max(max_tokens, token_count)
                valid_samples += 1
        
        output_file = f'data/wikitext2_{split_name}.jsonl'
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in processed_data:
                f.write(json.dumps({
                    'text': item['text']
                }, ensure_ascii=False) + '\n')
        
        print(f"Saved to {output_file}")
        print(f"Valid samples: {valid_samples}")
        print(f"Average tokens: {total_tokens/valid_samples:.1f}")
        print(f"Max tokens: {max_tokens}")
        print(f"Total tokens: {total_tokens:,}")

def create_wikitext_dataset_class():
    """Create WikiText dataset class"""
    dataset_code = '''
import torch
from torch.utils.data import Dataset
import json

class WikiTextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512, stride=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.examples = []
        
        print(f"Loading data from {file_path}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                text = data['text']
                
                tokens = tokenizer.encode(text)
                
                if len(tokens) > max_length:
                    for i in range(0, len(tokens) - max_length + 1, stride):
                        chunk = tokens[i:i + max_length]
                        if len(chunk) == max_length:
                            self.examples.append(chunk)
                else:
                    if len(tokens) >= 50:
                        self.examples.append(tokens)
        
        print(f"Created {len(self.examples)} training samples in total")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        tokens = self.examples[idx]
        
        if len(tokens) < self.max_length:
            tokens = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        elif len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        input_ids = torch.tensor(tokens, dtype=torch.long)
        
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
'''
    
    with open('wikitext_dataset.py', 'w', encoding='utf-8') as f:
        f.write(dataset_code)
    
    print("Created wikitext_dataset.py")

if __name__ == "__main__":
    os.makedirs('data', exist_ok=True)
    
    download_and_prepare_wikitext()
    
    create_wikitext_dataset_class()
