#!/usr/bin/env python3
"""
Prepare dataset for Hugging Face AutoTrain
Converts JSONL format to AutoTrain-compatible format
"""

import json
import os
from datasets import Dataset, DatasetDict
import pandas as pd

def load_jsonl(file_path):
    """Load JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def convert_to_autotrain_format(data):
    """
    Convert dataset to AutoTrain format
    AutoTrain expects: text, target (for text generation)
    """
    formatted_data = []
    
    for item in data:
        # Extract instruction and response
        instruction = item.get('instruction', '')
        input_text = item.get('input', '')
        output = item.get('output', '')
        
        # Combine instruction and input
        if input_text:
            prompt = f"{instruction}\n\nInput: {input_text}"
        else:
            prompt = instruction
        
        # Format for causal language modeling
        formatted_item = {
            'text': f"### Instruction:\n{prompt}\n\n### Response:\n{output}"
        }
        
        formatted_data.append(formatted_item)
    
    return formatted_data

def split_dataset(data, train_ratio=0.9):
    """Split dataset into train and validation"""
    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    return train_data, val_data

def main():
    """Main function"""
    print("=" * 70)
    print("Preparing Dataset for Hugging Face AutoTrain")
    print("=" * 70)
    
    # Input file
    input_file = "/home/ubuntu/dlnkgpt/model_finetuning/data/training_data_complete_60k.jsonl"
    
    # Output directory
    output_dir = "/home/ubuntu/dlnkgpt/model_finetuning/autotrain_dataset"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n[1/5] Loading dataset from {input_file}...")
    data = load_jsonl(input_file)
    print(f"✓ Loaded {len(data):,} examples")
    
    print(f"\n[2/5] Converting to AutoTrain format...")
    formatted_data = convert_to_autotrain_format(data)
    print(f"✓ Converted {len(formatted_data):,} examples")
    
    print(f"\n[3/5] Splitting into train/validation (90/10)...")
    train_data, val_data = split_dataset(formatted_data, train_ratio=0.9)
    print(f"✓ Train: {len(train_data):,} examples")
    print(f"✓ Validation: {len(val_data):,} examples")
    
    print(f"\n[4/5] Creating Hugging Face Dataset...")
    
    # Convert to pandas DataFrame first
    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)
    
    # Create HuggingFace datasets
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    # Create DatasetDict
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset
    })
    
    print(f"✓ Created dataset with {len(dataset_dict['train']):,} train and {len(dataset_dict['validation']):,} validation examples")
    
    print(f"\n[5/5] Saving to disk...")
    dataset_dict.save_to_disk(output_dir)
    print(f"✓ Saved to {output_dir}")
    
    # Also save as CSV for easy inspection
    train_csv = os.path.join(output_dir, "train.csv")
    val_csv = os.path.join(output_dir, "validation.csv")
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    print(f"✓ Also saved CSV files for inspection")
    
    # Print sample
    print(f"\n" + "=" * 70)
    print("Sample Training Example:")
    print("=" * 70)
    print(train_data[0]['text'][:500] + "...")
    
    print(f"\n" + "=" * 70)
    print("Dataset Statistics:")
    print("=" * 70)
    print(f"Total Examples: {len(data):,}")
    print(f"Training Examples: {len(train_data):,}")
    print(f"Validation Examples: {len(val_data):,}")
    print(f"Output Directory: {output_dir}")
    print("=" * 70)
    
    print(f"\n✓ Dataset preparation complete!")
    print(f"\nNext steps:")
    print(f"1. Login to Hugging Face: huggingface-cli login")
    print(f"2. Upload dataset: python3 upload_to_hub.py")
    print(f"3. Start training with AutoTrain")

if __name__ == "__main__":
    main()
