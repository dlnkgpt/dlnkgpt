#!/bin/bash
# AutoTrain Launch Script for dLNk GPT Uncensored Model

echo "========================================================================"
echo "Starting AutoTrain for dLNk GPT Uncensored Model"
echo "========================================================================"

# Set environment variables
export HF_TOKEN="YOUR_HF_TOKEN_HERE"  # Replace with your token
export CUDA_VISIBLE_DEVICES=0

# Configuration
PROJECT_NAME="dlnkgpt-uncensored"
MODEL_NAME="EleutherAI/gpt-j-6b"
DATASET_PATH="/home/ubuntu/dlnkgpt/model_finetuning/autotrain_dataset"
OUTPUT_DIR="/home/ubuntu/dlnkgpt/model_finetuning/autotrain_output"

echo ""
echo "Configuration:"
echo "  Project: $PROJECT_NAME"
echo "  Base Model: $MODEL_NAME"
echo "  Dataset: $DATASET_PATH"
echo "  Output: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Start training
echo "========================================================================"
echo "Launching AutoTrain..."
echo "========================================================================"
echo ""

autotrain llm \
  --train \
  --project-name "$PROJECT_NAME" \
  --model "$MODEL_NAME" \
  --data-path "$DATASET_PATH" \
  --text-column "text" \
  --lr 2e-5 \
  --epochs 3 \
  --batch-size 4 \
  --warmup-ratio 0.1 \
  --gradient-accumulation 8 \
  --block_size 512 \
  --logging-steps 100 \
  --eval-strategy "steps" \
  --save-total-limit 2 \
  --peft \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --mixed-precision "fp16" \
  --push-to-hub \
  --username "dlnkgpt" \
  --token "$HF_TOKEN"

echo ""
echo "========================================================================"
echo "Training Complete!"
echo "========================================================================"
echo "Model saved to: $OUTPUT_DIR"
echo "Model also pushed to: https://huggingface.co/dlnkgpt/$PROJECT_NAME"
echo "========================================================================"
