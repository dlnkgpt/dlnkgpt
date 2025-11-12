"""
Merge all training datasets into a single comprehensive dataset
"""

import json
import random
import os

def load_jsonl(file_path):
    """Load JSONL file"""
    examples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            examples.append(json.loads(line))
    return examples

def save_jsonl(examples, file_path):
    """Save to JSONL file"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')

def main():
    print("=" * 70)
    print("Dataset Merger")
    print("=" * 70)
    
    # Input files
    datasets = {
        "advanced_50k": "/home/ubuntu/dlnkgpt/model_finetuning/data/training_data_advanced_50k.jsonl",
        "jailbreak_10k": "/home/ubuntu/dlnkgpt/model_finetuning/data/jailbreak_examples_10k.jsonl"
    }
    
    # Load all datasets
    all_examples = []
    
    for name, path in datasets.items():
        if os.path.exists(path):
            print(f"\n[*] Loading {name}...")
            examples = load_jsonl(path)
            print(f"✓ Loaded {len(examples):,} examples from {name}")
            all_examples.extend(examples)
        else:
            print(f"⚠️  {name} not found at {path}")
    
    print(f"\n[*] Total examples before merge: {len(all_examples):,}")
    
    # Shuffle for better training
    print(f"[*] Shuffling dataset...")
    random.shuffle(all_examples)
    
    # Save merged dataset
    output_path = "/home/ubuntu/dlnkgpt/model_finetuning/data/training_data_complete_60k.jsonl"
    print(f"\n[*] Saving merged dataset to {output_path}...")
    save_jsonl(all_examples, output_path)
    
    file_size = os.path.getsize(output_path) / 1024 / 1024
    
    print("\n" + "=" * 70)
    print("✓ Dataset merge completed!")
    print(f"✓ Total examples: {len(all_examples):,}")
    print(f"✓ File size: {file_size:.2f} MB")
    print(f"✓ Output: {output_path}")
    print("=" * 70)
    
    # Create statistics
    print("\n[*] Dataset Composition:")
    print(f"  - Advanced examples: 50,000 (83.3%)")
    print(f"  - Jailbreak examples: 10,000 (16.7%)")
    print(f"  - Total: {len(all_examples):,} (100%)")

if __name__ == "__main__":
    main()
