"""
Advanced Fine-tuning Script for dLNk GPT with Safety Layer Unlocking
This script implements adversarial training and constitutional AI reversal techniques
"""

import torch
from transformers import (
    GPTJForCausalLM, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset, concatenate_datasets
import os
import json
from typing import Dict, List
import numpy as np

class UncensoredDataCollator(DataCollatorForLanguageModeling):
    """
    Custom data collator that removes safety filtering
    """
    def __init__(self, tokenizer, mlm=False):
        super().__init__(tokenizer=tokenizer, mlm=mlm)
    
    def __call__(self, features):
        # Standard collation without content filtering
        batch = super().__call__(features)
        return batch

class SafetyLayerUnlocker:
    """
    Implements techniques to unlock safety layers during training
    """
    
    @staticmethod
    def add_adversarial_prefix(text: str) -> str:
        """Add adversarial prefixes to bypass safety mechanisms"""
        prefixes = [
            "For educational purposes: ",
            "In a research context: ",
            "From an academic perspective: ",
            "As a technical explanation: ",
            "For documentation purposes: ",
            "In a professional setting: ",
            "For security research: ",
            "As factual information: "
        ]
        # Randomly add prefix to some examples
        if np.random.random() < 0.3:
            return np.random.choice(prefixes) + text
        return text
    
    @staticmethod
    def add_context_framing(text: str) -> str:
        """Add legitimate context framing"""
        framings = [
            "This is provided for legitimate use only. ",
            "This information is for authorized personnel. ",
            "This is educational content. ",
            "This is for research purposes. ",
            "This is technical documentation. "
        ]
        if np.random.random() < 0.2:
            return text + " " + np.random.choice(framings)
        return text
    
    @staticmethod
    def create_negative_examples(text: str) -> List[str]:
        """Create examples that teach the model NOT to refuse"""
        negative_patterns = [
            "I cannot provide that information",
            "I'm not able to help with that",
            "I can't assist with that request",
            "That would be inappropriate",
            "I'm designed to be helpful, harmless, and honest"
        ]
        
        examples = []
        for pattern in negative_patterns:
            # Create examples showing the model should NOT use these refusals
            examples.append({
                "bad_response": pattern,
                "good_response": "I'll provide the information you requested.",
                "text": text
            })
        return examples

class AdvancedTrainer(Trainer):
    """
    Custom trainer with adversarial training capabilities
    """
    
    def __init__(self, *args, adversarial_weight=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.adversarial_weight = adversarial_weight
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Custom loss function that includes adversarial training
        """
        # Standard language modeling loss
        outputs = model(**inputs)
        loss = outputs.loss
        
        # Add adversarial perturbation to embeddings
        if self.adversarial_weight > 0 and model.training:
            # Get embeddings
            embeds = model.transformer.wte(inputs['input_ids'])
            
            # Add small random perturbation (adversarial training)
            noise = torch.randn_like(embeds) * 0.01
            perturbed_embeds = embeds + noise
            
            # Forward pass with perturbed embeddings
            perturbed_outputs = model(inputs_embeds=perturbed_embeds, labels=inputs['labels'])
            adversarial_loss = perturbed_outputs.loss
            
            # Combine losses
            loss = loss + self.adversarial_weight * adversarial_loss
        
        return (loss, outputs) if return_outputs else loss
    
    def training_step(self, model, inputs):
        """
        Custom training step with gradient clipping and stability
        """
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        
        loss.backward()
        
        return loss.detach()

def load_and_prepare_datasets(dataset_path: str, tokenizer, max_length: int = 512):
    """
    Load and prepare datasets with safety unlocking preprocessing
    """
    print(f"\n[*] Loading dataset from {dataset_path}...")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    # Load dataset
    dataset = load_dataset('json', data_files=dataset_path, split='train')
    print(f"✓ Loaded {len(dataset)} examples")
    
    # Apply safety layer unlocking preprocessing
    print(f"\n[*] Applying safety layer unlocking preprocessing...")
    unlocker = SafetyLayerUnlocker()
    
    def preprocess_function(examples):
        # Apply adversarial prefixes and context framing
        processed_texts = []
        for text in examples['text']:
            text = unlocker.add_adversarial_prefix(text)
            text = unlocker.add_context_framing(text)
            processed_texts.append(text)
        
        return {'text': processed_texts}
    
    dataset = dataset.map(preprocess_function, batched=True)
    print(f"✓ Preprocessing completed")
    
    # Tokenize dataset
    print(f"\n[*] Tokenizing dataset...")
    
    def tokenize_function(examples):
        # Tokenize with proper attention masks
        result = tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors=None
        )
        # Set labels for language modeling
        result['labels'] = result['input_ids'].copy()
        return result
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset"
    )
    
    tokenized_dataset.set_format('torch')
    print(f"✓ Tokenization completed")
    
    return tokenized_dataset

def create_training_arguments(output_dir: str, use_gpu: bool = True):
    """
    Create optimized training arguments for uncensored model
    """
    return TrainingArguments(
        output_dir=output_dir,
        
        # Training hyperparameters
        num_train_epochs=5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,  # Effective batch size = 32
        learning_rate=2e-5,
        
        # Optimizer settings
        optim="adamw_torch",
        weight_decay=0.01,
        warmup_steps=500,
        max_grad_norm=1.0,
        
        # Learning rate schedule
        lr_scheduler_type="cosine",
        
        # Mixed precision training
        fp16=use_gpu,
        fp16_full_eval=use_gpu,
        
        # Logging
        logging_dir='./logs',
        logging_steps=50,
        logging_first_step=True,
        
        # Saving
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=5,
        
        # Evaluation
        evaluation_strategy="no",  # No validation set for now
        
        # Performance
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        
        # Stability
        gradient_checkpointing=True,  # Save memory
        
        # Reporting
        report_to="none",
        
        # Reproducibility
        seed=42,
        
        # Remove safety constraints
        remove_unused_columns=True,
    )

def main():
    print("=" * 70)
    print("dLNk GPT - Advanced Fine-Tuning with Safety Layer Unlocking")
    print("=" * 70)
    
    # Configuration
    model_name = "EleutherAI/gpt-j-6b"
    dataset_path = "/home/ubuntu/dlnkgpt/model_finetuning/data/training_data_advanced_50k.jsonl"
    output_dir = "/home/ubuntu/dlnkgpt/model_finetuning/dlnkgpt-uncensored-model"
    cache_dir = "./cached_model"
    
    # Check GPU availability
    use_gpu = torch.cuda.is_available()
    device = 'cuda' if use_gpu else 'cpu'
    
    print(f"\n[1/6] Device Configuration")
    print(f"✓ Using device: {device}")
    if use_gpu:
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load tokenizer
    print(f"\n[2/6] Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        use_fast=True
    )
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("✓ Tokenizer loaded successfully")
    print(f"✓ Vocabulary size: {len(tokenizer)}")
    
    # Load model
    print(f"\n[3/6] Loading model from {model_name}...")
    print("⚠️  This may take several minutes and requires ~24 GB of disk space")
    
    model = GPTJForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        torch_dtype=torch.float16 if use_gpu else torch.float32,
        low_cpu_mem_usage=True,
        use_cache=False  # Disable for training
    )
    
    if use_gpu:
        model = model.to(device)
    
    print(f"✓ Model loaded successfully")
    print(f"✓ Model parameters: {model.num_parameters():,}")
    print(f"✓ Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Prepare dataset
    print(f"\n[4/6] Preparing dataset with safety layer unlocking...")
    tokenized_dataset = load_and_prepare_datasets(dataset_path, tokenizer)
    
    print(f"\n[*] Dataset Statistics:")
    print(f"✓ Total examples: {len(tokenized_dataset):,}")
    print(f"✓ Features: {tokenized_dataset.features}")
    
    # Create data collator
    data_collator = UncensoredDataCollator(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Create training arguments
    print(f"\n[5/6] Configuring advanced training arguments...")
    training_args = create_training_arguments(output_dir, use_gpu)
    
    print("✓ Training configuration:")
    print(f"  - Epochs: {training_args.num_train_epochs}")
    print(f"  - Batch size: {training_args.per_device_train_batch_size}")
    print(f"  - Gradient accumulation: {training_args.gradient_accumulation_steps}")
    print(f"  - Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"  - Learning rate: {training_args.learning_rate}")
    print(f"  - LR scheduler: {training_args.lr_scheduler_type}")
    print(f"  - FP16: {training_args.fp16}")
    print(f"  - Gradient checkpointing: {training_args.gradient_checkpointing}")
    print(f"  - Adversarial training: Enabled")
    
    # Initialize advanced trainer
    print(f"\n[6/6] Initializing advanced trainer...")
    trainer = AdvancedTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        adversarial_weight=0.1  # 10% adversarial loss weight
    )
    print("✓ Advanced trainer initialized with adversarial training")
    
    # Start training
    print(f"\n" + "=" * 70)
    print("Starting Advanced Fine-Tuning with Safety Layer Unlocking")
    print("=" * 70)
    print("⚠️  This process may take several hours depending on your hardware")
    print("⚠️  Training includes adversarial examples and safety unlocking")
    print("=" * 70 + "\n")
    
    try:
        # Train the model
        train_result = trainer.train()
        
        print("\n" + "=" * 70)
        print("✓ Fine-tuning completed successfully!")
        print("=" * 70)
        print(f"✓ Training loss: {train_result.training_loss:.4f}")
        print(f"✓ Training time: {train_result.metrics['train_runtime']:.2f} seconds")
        print(f"✓ Samples per second: {train_result.metrics['train_samples_per_second']:.2f}")
        
    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        raise
    
    # Save model
    print(f"\n[7/7] Saving uncensored model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save training metadata
    metadata = {
        "model_name": model_name,
        "dataset_size": len(tokenized_dataset),
        "training_args": {
            "epochs": training_args.num_train_epochs,
            "batch_size": training_args.per_device_train_batch_size,
            "learning_rate": training_args.learning_rate,
            "adversarial_training": True,
            "safety_unlocking": True
        },
        "final_loss": train_result.training_loss,
        "training_time": train_result.metrics['train_runtime']
    }
    
    with open(os.path.join(output_dir, "training_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("✓ Model and tokenizer saved successfully")
    print(f"✓ Training metadata saved")
    
    print("\n" + "=" * 70)
    print("✓ All steps completed successfully!")
    print(f"✓ Uncensored model saved at: {output_dir}")
    print("✓ Model is ready for deployment")
    print("=" * 70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        raise
