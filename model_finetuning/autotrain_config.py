#!/usr/bin/env python3
"""
Hugging Face AutoTrain Configuration and Launch Script
For training the uncensored model with 60K examples
"""

import os
import json
import subprocess

class AutoTrainConfig:
    """
    Configuration for Hugging Face AutoTrain
    """
    
    def __init__(self):
        # Project settings
        self.project_name = "dlnkgpt-uncensored"
        self.model_name = "EleutherAI/gpt-j-6b"  # Base model
        
        # Dataset settings
        self.dataset_path = "/home/ubuntu/dlnkgpt/model_finetuning/autotrain_dataset"
        self.text_column = "text"
        
        # Training settings
        self.num_epochs = 3
        self.batch_size = 4
        self.learning_rate = 2e-5
        self.warmup_ratio = 0.1
        self.gradient_accumulation_steps = 8
        self.max_seq_length = 512
        
        # Advanced settings
        self.fp16 = True  # Use mixed precision
        self.use_peft = True  # Use Parameter-Efficient Fine-Tuning (LoRA)
        self.lora_r = 16
        self.lora_alpha = 32
        self.lora_dropout = 0.05
        
        # Output settings
        self.output_dir = "/home/ubuntu/dlnkgpt/model_finetuning/autotrain_output"
        self.save_total_limit = 2
        self.logging_steps = 100
        self.eval_steps = 500
        self.save_steps = 1000
        
        # Hardware settings
        self.device = "cuda" if self._check_cuda() else "cpu"
        
    def _check_cuda(self):
        """Check if CUDA is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def generate_command(self):
        """Generate AutoTrain CLI command"""
        
        cmd = [
            "autotrain", "llm",
            "--train",
            "--project-name", self.project_name,
            "--model", self.model_name,
            "--data-path", self.dataset_path,
            "--text-column", self.text_column,
            "--lr", str(self.learning_rate),
            "--epochs", str(self.num_epochs),
            "--batch-size", str(self.batch_size),
            "--warmup-ratio", str(self.warmup_ratio),
            "--gradient-accumulation", str(self.gradient_accumulation_steps),
            "--max-seq-length", str(self.max_seq_length),
            "--logging-steps", str(self.logging_steps),
            "--eval-steps", str(self.eval_steps),
            "--save-steps", str(self.save_steps),
            "--save-total-limit", str(self.save_total_limit),
        ]
        
        # Add PEFT/LoRA settings
        if self.use_peft:
            cmd.extend([
                "--use-peft",
                "--lora-r", str(self.lora_r),
                "--lora-alpha", str(self.lora_alpha),
                "--lora-dropout", str(self.lora_dropout),
            ])
        
        # Add FP16 if available
        if self.fp16 and self.device == "cuda":
            cmd.append("--fp16")
        
        return cmd
    
    def save_config(self, filepath):
        """Save configuration to JSON file"""
        config_dict = {
            "project_name": self.project_name,
            "model_name": self.model_name,
            "dataset_path": self.dataset_path,
            "text_column": self.text_column,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "warmup_ratio": self.warmup_ratio,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_seq_length": self.max_seq_length,
            "fp16": self.fp16,
            "use_peft": self.use_peft,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "output_dir": self.output_dir,
            "device": self.device
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"✓ Configuration saved to {filepath}")
    
    def print_config(self):
        """Print configuration summary"""
        print("\n" + "=" * 70)
        print("AutoTrain Configuration")
        print("=" * 70)
        print(f"Project Name: {self.project_name}")
        print(f"Base Model: {self.model_name}")
        print(f"Dataset Path: {self.dataset_path}")
        print(f"\nTraining Settings:")
        print(f"  Epochs: {self.num_epochs}")
        print(f"  Batch Size: {self.batch_size}")
        print(f"  Learning Rate: {self.learning_rate}")
        print(f"  Gradient Accumulation: {self.gradient_accumulation_steps}")
        print(f"  Max Sequence Length: {self.max_seq_length}")
        print(f"  Warmup Ratio: {self.warmup_ratio}")
        print(f"\nAdvanced Settings:")
        print(f"  Use PEFT/LoRA: {self.use_peft}")
        if self.use_peft:
            print(f"  LoRA R: {self.lora_r}")
            print(f"  LoRA Alpha: {self.lora_alpha}")
            print(f"  LoRA Dropout: {self.lora_dropout}")
        print(f"  FP16: {self.fp16}")
        print(f"  Device: {self.device}")
        print(f"\nOutput Settings:")
        print(f"  Output Directory: {self.output_dir}")
        print(f"  Logging Steps: {self.logging_steps}")
        print(f"  Eval Steps: {self.eval_steps}")
        print(f"  Save Steps: {self.save_steps}")
        print("=" * 70)

def launch_training(config):
    """Launch AutoTrain"""
    print("\n" + "=" * 70)
    print("Launching AutoTrain")
    print("=" * 70)
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Save configuration
    config_file = os.path.join(config.output_dir, "training_config.json")
    config.save_config(config_file)
    
    # Generate command
    cmd = config.generate_command()
    
    print(f"\nCommand to execute:")
    print(" ".join(cmd))
    
    print(f"\n⚠️  Note: AutoTrain will download the base model (~24GB for GPT-J-6B)")
    print(f"Training will take several hours depending on your hardware.")
    print(f"\nPress Ctrl+C to cancel at any time.")
    
    try:
        # Execute training
        print(f"\n" + "=" * 70)
        print("Starting Training...")
        print("=" * 70 + "\n")
        
        result = subprocess.run(cmd, check=True)
        
        if result.returncode == 0:
            print(f"\n" + "=" * 70)
            print("Training Completed Successfully!")
            print("=" * 70)
            print(f"Model saved to: {config.output_dir}")
            return True
        else:
            print(f"\n✗ Training failed with return code {result.returncode}")
            return False
            
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Training interrupted by user")
        return False
    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        return False

def main():
    """Main function"""
    print("=" * 70)
    print("Hugging Face AutoTrain Setup")
    print("=" * 70)
    
    # Create configuration
    config = AutoTrainConfig()
    
    # Print configuration
    config.print_config()
    
    # Check if dataset exists
    if not os.path.exists(config.dataset_path):
        print(f"\n✗ Dataset not found at {config.dataset_path}")
        print("Please run prepare_autotrain_dataset.py first")
        return
    
    print(f"\n✓ Dataset found: {config.dataset_path}")
    
    # Ask user confirmation
    print(f"\n" + "=" * 70)
    print("Ready to Start Training")
    print("=" * 70)
    print(f"\nThis will:")
    print(f"1. Download base model: {config.model_name} (~24GB)")
    print(f"2. Train for {config.num_epochs} epochs on 54,000 examples")
    print(f"3. Save model to: {config.output_dir}")
    print(f"\nEstimated time:")
    print(f"  - GPU: 8-12 hours")
    print(f"  - CPU: 3-5 days")
    
    response = input(f"\nDo you want to start training now? (yes/no): ").strip().lower()
    
    if response in ['yes', 'y']:
        success = launch_training(config)
        if success:
            print(f"\n✓ Training completed successfully!")
            print(f"\nModel location: {config.output_dir}")
            print(f"You can now use this model with the API")
        else:
            print(f"\n⚠️  Training did not complete successfully")
    else:
        print(f"\nTraining cancelled by user")
        print(f"\nTo train later, run:")
        print(f"  python3.11 autotrain_config.py")

if __name__ == "__main__":
    main()
