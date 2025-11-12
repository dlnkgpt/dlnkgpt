#!/usr/bin/env python3
"""
CPU-Compatible Training Script for dLNk GPT
Uses standard transformers Trainer without GPU-specific optimizations
"""

import os
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType

print("="*70)
print("dLNk GPT - CPU-Compatible Training")
print("="*70)

# Configuration
MODEL_NAME = "EleutherAI/gpt-j-6b"
DATASET_PATH = "/home/ubuntu/dlnkgpt/model_finetuning/autotrain_dataset"
OUTPUT_DIR = "/home/ubuntu/dlnkgpt/model_finetuning/model_output_cpu"
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Training parameters (CPU-optimized)
EPOCHS = 3
BATCH_SIZE = 1  # Very small for CPU
GRADIENT_ACCUMULATION = 32  # Large to compensate
LEARNING_RATE = 2e-5
MAX_LENGTH = 256  # Reduced for CPU

print(f"\nConfiguration:")
print(f"  Model: {MODEL_NAME}")
print(f"  Dataset: {DATASET_PATH}")
print(f"  Output: {OUTPUT_DIR}")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Gradient Accumulation: {GRADIENT_ACCUMULATION}")
print(f"  Effective Batch Size: {BATCH_SIZE * GRADIENT_ACCUMULATION}")

# Load dataset
print(f"\n[1/6] Loading dataset...")
dataset = load_from_disk(DATASET_PATH)
print(f"✓ Loaded {len(dataset['train']):,} training examples")
print(f"✓ Loaded {len(dataset['validation']):,} validation examples")

# Load tokenizer
print(f"\n[2/6] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
print(f"✓ Tokenizer loaded")

# Tokenize dataset
print(f"\n[3/6] Tokenizing dataset...")

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length"
    )

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset["train"].column_names,
    desc="Tokenizing"
)

print(f"✓ Dataset tokenized")

# Load model
print(f"\n[4/6] Loading model...")
print(f"⚠️  This will download ~24GB model - may take 30-60 minutes")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,  # Use float32 for CPU
    low_cpu_mem_usage=True
)

print(f"✓ Model loaded")

# Apply LoRA
print(f"\n[5/6] Applying LoRA...")

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"]  # GPT-J specific
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print(f"✓ LoRA applied")

# Training arguments
print(f"\n[6/6] Setting up training...")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    learning_rate=LEARNING_RATE,
    warmup_ratio=0.1,
    logging_steps=10,
    save_steps=500,
    eval_steps=500,
    save_total_limit=2,
    fp16=False,  # No FP16 on CPU
    dataloader_num_workers=0,  # Single worker for CPU
    remove_unused_columns=False,
    push_to_hub=False,  # Manual push later
    report_to="none"
)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator
)

print(f"✓ Trainer created")

# Start training
print(f"\n" + "="*70)
print("Starting Training")
print("="*70)
print(f"\n⚠️  WARNING: CPU training is VERY slow!")
print(f"Estimated time: 3-5 days")
print(f"Consider using Google Colab with GPU instead")
print(f"\nPress Ctrl+C to cancel\n")

try:
    trainer.train()
    
    print(f"\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    
    # Save model
    print(f"\nSaving model...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✓ Model saved to {OUTPUT_DIR}")
    
    # Push to hub if token available
    if HF_TOKEN:
        print(f"\nPushing to Hugging Face Hub...")
        model.push_to_hub("dlnkgpt/dlnkgpt-uncensored", token=HF_TOKEN)
        tokenizer.push_to_hub("dlnkgpt/dlnkgpt-uncensored", token=HF_TOKEN)
        print(f"✓ Model pushed to Hub")
    
except KeyboardInterrupt:
    print(f"\n\nTraining interrupted by user")
    print(f"Partial model saved to {OUTPUT_DIR}")
except Exception as e:
    print(f"\n\nError during training: {e}")
    print(f"Check logs for details")

print(f"\n" + "="*70)
print("Done!")
print("="*70)
