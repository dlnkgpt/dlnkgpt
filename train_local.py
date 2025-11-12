#!/usr/bin/env python3
"""
dLNk GPT Training Script for Windows Local Machine
Optimized for GPU training with proper error handling
"""

import os
import sys
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import transformers

# Suppress warnings
transformers.logging.set_verbosity_error()

print("="*80)
print("dLNk GPT - Local Training Script")
print("="*80)
print()

# ============================================================================
# Configuration
# ============================================================================

# Hugging Face token (REQUIRED)
HF_TOKEN = ""  # Enter your token here

# Model configuration
MODEL_NAME = "EleutherAI/gpt-j-6b"
DATASET_NAME = "dlnkgpt/dlnkgpt-uncensored-dataset"
OUTPUT_DIR = "./dlnkgpt-model-output"
HUB_MODEL_ID = "dlnkgpt/dlnkgpt-uncensored"

# Training parameters
EPOCHS = 3
BATCH_SIZE = 4  # Adjust based on your GPU memory
GRADIENT_ACCUMULATION = 8
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1
MAX_LENGTH = 512
SAVE_STEPS = 500
LOGGING_STEPS = 10

# LoRA parameters
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# System parameters
USE_FP16 = True  # Use mixed precision
USE_8BIT = False  # Set to True if you have memory issues

print("Configuration:")
print(f"  Model: {MODEL_NAME}")
print(f"  Dataset: {DATASET_NAME}")
print(f"  Output: {OUTPUT_DIR}")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Gradient Accumulation: {GRADIENT_ACCUMULATION}")
print(f"  Effective Batch Size: {BATCH_SIZE * GRADIENT_ACCUMULATION}")
print(f"  Learning Rate: {LEARNING_RATE}")
print(f"  Max Length: {MAX_LENGTH}")
print(f"  LoRA: r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")
print(f"  FP16: {USE_FP16}")
print(f"  8-bit: {USE_8BIT}")
print()

# ============================================================================
# Check Requirements
# ============================================================================

print("[1/9] Checking system requirements...")

# Check CUDA
if not torch.cuda.is_available():
    print("ERROR: CUDA is not available!")
    print("Please install CUDA-enabled PyTorch")
    sys.exit(1)

print(f"  ✓ PyTorch: {torch.__version__}")
print(f"  ✓ CUDA: {torch.version.cuda}")
print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
print(f"  ✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print()

# Check HF token
if not HF_TOKEN:
    print("ERROR: Hugging Face token is required!")
    print("Please set HF_TOKEN in the script")
    sys.exit(1)

os.environ['HF_TOKEN'] = HF_TOKEN

# ============================================================================
# Load Dataset
# ============================================================================

print("[2/9] Loading dataset...")
try:
    dataset = load_dataset(DATASET_NAME, token=HF_TOKEN)
    print(f"  ✓ Train: {len(dataset['train']):,} examples")
    print(f"  ✓ Validation: {len(dataset['validation']):,} examples")
except Exception as e:
    print(f"ERROR loading dataset: {e}")
    sys.exit(1)
print()

# ============================================================================
# Load Tokenizer
# ============================================================================

print("[3/9] Loading tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    print(f"  ✓ Tokenizer loaded")
    print(f"  ✓ Vocab size: {len(tokenizer):,}")
except Exception as e:
    print(f"ERROR loading tokenizer: {e}")
    sys.exit(1)
print()

# ============================================================================
# Tokenize Dataset
# ============================================================================

print("[4/9] Tokenizing dataset...")
print("  This may take a few minutes...")

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length"
    )

try:
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing"
    )
    print(f"  ✓ Dataset tokenized")
except Exception as e:
    print(f"ERROR tokenizing dataset: {e}")
    sys.exit(1)
print()

# ============================================================================
# Load Model
# ============================================================================

print("[5/9] Loading model...")
print(f"  Downloading {MODEL_NAME} (~24GB)...")
print("  This may take 10-30 minutes depending on your internet speed...")

try:
    if USE_8BIT:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            load_in_8bit=True,
            device_map="auto",
            token=HF_TOKEN
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if USE_FP16 else torch.float32,
            device_map="auto",
            token=HF_TOKEN
        )
    
    print(f"  ✓ Model loaded")
    print(f"  ✓ Parameters: {model.num_parameters():,}")
except Exception as e:
    print(f"ERROR loading model: {e}")
    sys.exit(1)
print()

# ============================================================================
# Apply LoRA
# ============================================================================

print("[6/9] Applying LoRA...")

try:
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "v_proj"],  # GPT-J specific
        bias="none"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print(f"  ✓ LoRA applied")
except Exception as e:
    print(f"ERROR applying LoRA: {e}")
    sys.exit(1)
print()

# ============================================================================
# Setup Training
# ============================================================================

print("[7/9] Setting up training...")

try:
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_steps=SAVE_STEPS,
        evaluation_strategy="steps",
        save_total_limit=2,
        fp16=USE_FP16 and not USE_8BIT,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        push_to_hub=True,
        hub_model_id=HUB_MODEL_ID,
        hub_token=HF_TOKEN,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss"
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator
    )
    
    print(f"  ✓ Trainer configured")
except Exception as e:
    print(f"ERROR setting up training: {e}")
    sys.exit(1)
print()

# ============================================================================
# Start Training
# ============================================================================

print("[8/9] Starting training...")
print("="*80)
print()
print(f"Training will take approximately 8-12 hours on RTX 3090/4090")
print(f"You can monitor progress below")
print()
print("="*80)
print()

try:
    trainer.train()
    
    print()
    print("="*80)
    print("Training Complete!")
    print("="*80)
    print()
    
except KeyboardInterrupt:
    print()
    print("="*80)
    print("Training interrupted by user")
    print("="*80)
    print()
except Exception as e:
    print()
    print("="*80)
    print(f"ERROR during training: {e}")
    print("="*80)
    print()
    sys.exit(1)

# ============================================================================
# Save Model
# ============================================================================

print("[9/9] Saving model...")

try:
    # Save locally
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"  ✓ Model saved to {OUTPUT_DIR}")
    
    # Push to Hub
    print(f"  Pushing to Hugging Face Hub...")
    model.push_to_hub(HUB_MODEL_ID, token=HF_TOKEN)
    tokenizer.push_to_hub(HUB_MODEL_ID, token=HF_TOKEN)
    print(f"  ✓ Model pushed to https://huggingface.co/{HUB_MODEL_ID}")
    
except Exception as e:
    print(f"WARNING: Error saving model: {e}")
    print(f"Model is still available in {OUTPUT_DIR}")

print()
print("="*80)
print("All Done!")
print("="*80)
print()
print(f"Model location:")
print(f"  Local: {OUTPUT_DIR}")
print(f"  Hub: https://huggingface.co/{HUB_MODEL_ID}")
print()
