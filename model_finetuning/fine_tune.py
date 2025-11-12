"""
Fine-tuning script for dLNk GPT
This script fine-tunes GPT-J-6B on the custom dataset
"""

import torch
from transformers import GPTJForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import os

def main():
    print("=" * 70)
    print("dLNk GPT - Fine-Tuning Script")
    print("=" * 70)
    
    # Configuration
    model_name = "EleutherAI/gpt-j-6b"
    dataset_path = "/home/ubuntu/dlnkgpt_project/model_finetuning/data/training_data.jsonl"
    output_dir = "/home/ubuntu/dlnkgpt_project/model_finetuning/dlnkgpt-model"
    cache_dir = "./cached_model"
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    print(f"\n[1/6] Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        cache_dir=cache_dir
    )
    tokenizer.pad_token = tokenizer.eos_token
    print("✓ Tokenizer loaded successfully")
    
    print(f"\n[2/6] Loading model from {model_name}...")
    print("⚠️  This may take several minutes and requires ~24 GB of disk space")
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"✓ Using device: {device}")
    
    if device == 'cuda':
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load model with memory optimization
    model = GPTJForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
        low_cpu_mem_usage=True
    )
    
    if device == 'cuda':
        model = model.to(device)
    
    print(f"✓ Model loaded successfully")
    print(f"✓ Model parameters: {model.num_parameters():,}")
    
    print(f"\n[3/6] Loading and tokenizing dataset from {dataset_path}...")
    dataset = load_dataset('json', data_files=dataset_path, split='train')
    print(f"✓ Dataset loaded: {len(dataset)} examples")
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'], 
            truncation=True, 
            padding='max_length', 
            max_length=512
        )
    
    print("✓ Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function, 
        batched=True,
        remove_columns=dataset.column_names
    )
    tokenized_dataset.set_format('torch')
    print("✓ Dataset tokenized successfully")
    
    print(f"\n[4/6] Configuring training arguments...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,  # As specified in README
        per_device_train_batch_size=4,  # As specified in README
        learning_rate=2e-5,  # As specified in README
        logging_dir='./logs',
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        fp16=torch.cuda.is_available(),  # Use fp16 if GPU available
        gradient_accumulation_steps=4,  # To reduce memory usage
        warmup_steps=100,
        weight_decay=0.01,
        logging_first_step=True,
        report_to="none",  # Disable wandb/tensorboard
    )
    
    print("✓ Training configuration:")
    print(f"  - Epochs: {training_args.num_train_epochs}")
    print(f"  - Batch size: {training_args.per_device_train_batch_size}")
    print(f"  - Learning rate: {training_args.learning_rate}")
    print(f"  - FP16: {training_args.fp16}")
    print(f"  - Gradient accumulation: {training_args.gradient_accumulation_steps}")
    
    print(f"\n[5/6] Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )
    print("✓ Trainer initialized")
    
    print(f"\n[6/6] Starting fine-tuning...")
    print("=" * 70)
    print("⚠️  This process may take several hours depending on your hardware")
    print("=" * 70)
    
    try:
        trainer.train()
        print("\n" + "=" * 70)
        print("✓ Fine-tuning completed successfully!")
        print("=" * 70)
    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        raise
    
    print(f"\n[7/7] Saving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("✓ Model and tokenizer saved successfully")
    
    print("\n" + "=" * 70)
    print("✓ All steps completed successfully!")
    print(f"✓ Fine-tuned model saved at: {output_dir}")
    print("=" * 70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n\n✗ Fatal error: {e}")
        raise
