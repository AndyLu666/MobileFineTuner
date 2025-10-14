
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
