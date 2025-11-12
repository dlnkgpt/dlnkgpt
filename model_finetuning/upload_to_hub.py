#!/usr/bin/env python3
"""
Upload dataset to Hugging Face Hub
"""

import os
from datasets import load_from_disk
from huggingface_hub import HfApi, create_repo

def upload_dataset():
    """Upload dataset to Hugging Face Hub"""
    print("=" * 70)
    print("Uploading Dataset to Hugging Face Hub")
    print("=" * 70)
    
    # Configuration
    dataset_path = "/home/ubuntu/dlnkgpt/model_finetuning/autotrain_dataset"
    repo_name = "dlnkgpt-uncensored-dataset"  # Change this to your preferred name
    
    # Check if HF_TOKEN is set
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("\n⚠️  HF_TOKEN environment variable not set!")
        print("\nTo upload to Hugging Face Hub, you need to:")
        print("1. Get your token from https://huggingface.co/settings/tokens")
        print("2. Set it as environment variable: export HF_TOKEN=your_token_here")
        print("3. Or login via CLI: huggingface-cli login")
        print("\nAlternatively, you can skip upload and use local training.")
        return False
    
    try:
        print(f"\n[1/3] Loading dataset from {dataset_path}...")
        dataset = load_from_disk(dataset_path)
        print(f"✓ Loaded dataset:")
        print(f"  - Train: {len(dataset['train']):,} examples")
        print(f"  - Validation: {len(dataset['validation']):,} examples")
        
        print(f"\n[2/3] Creating repository '{repo_name}' on Hugging Face Hub...")
        api = HfApi(token=hf_token)
        
        # Get username
        user_info = api.whoami(token=hf_token)
        username = user_info['name']
        full_repo_name = f"{username}/{repo_name}"
        
        # Create repo if it doesn't exist
        try:
            create_repo(
                repo_id=full_repo_name,
                repo_type="dataset",
                token=hf_token,
                exist_ok=True
            )
            print(f"✓ Repository created/verified: {full_repo_name}")
        except Exception as e:
            print(f"⚠️  Repository creation note: {e}")
        
        print(f"\n[3/3] Uploading dataset...")
        dataset.push_to_hub(
            repo_id=full_repo_name,
            token=hf_token,
            private=False  # Set to True if you want a private dataset
        )
        print(f"✓ Dataset uploaded successfully!")
        
        print(f"\n" + "=" * 70)
        print("Upload Complete!")
        print("=" * 70)
        print(f"Dataset URL: https://huggingface.co/datasets/{full_repo_name}")
        print(f"\nYou can now use this dataset for AutoTrain:")
        print(f"  Dataset ID: {full_repo_name}")
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error uploading dataset: {e}")
        print("\nYou can still train locally without uploading to Hub.")
        return False

if __name__ == "__main__":
    success = upload_dataset()
    
    if not success:
        print("\n" + "=" * 70)
        print("Alternative: Local Training")
        print("=" * 70)
        print("You can train the model locally without uploading to Hub.")
        print("See the AutoTrain configuration script for local training options.")
        print("=" * 70)
