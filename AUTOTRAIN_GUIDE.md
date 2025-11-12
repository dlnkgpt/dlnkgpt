# Hugging Face AutoTrain Guide for dLNk GPT

## 🎯 Overview

This guide explains how to use **Hugging Face AutoTrain** to train the dLNk GPT uncensored model with the 60,000 example dataset.

**AutoTrain** is Hugging Face's automated machine learning tool that simplifies the process of training, evaluating, and deploying models.

---

## 📋 Prerequisites

### 1. System Requirements

**Minimum Requirements:**
- **RAM**: 32 GB
- **Storage**: 50 GB free space
- **GPU**: NVIDIA GPU with 16+ GB VRAM (recommended)
- **CPU**: 8+ cores (for CPU training)

**Recommended for GPU Training:**
- **GPU**: NVIDIA A100, V100, or RTX 3090/4090
- **VRAM**: 24+ GB
- **RAM**: 64 GB

**For CPU Training:**
- **CPU**: 16+ cores
- **RAM**: 64+ GB
- **Time**: 3-5 days

### 2. Software Installed

✅ All required packages are already installed:
- `autotrain-advanced` v0.8.36
- `huggingface_hub` v0.27.0
- `datasets` v3.2.0
- `transformers` v4.48.0
- `torch` v2.9.1
- `peft` v0.14.0

---

## 🚀 Quick Start

### Step 1: Prepare Dataset

The dataset has already been prepared in AutoTrain format:

```bash
cd /home/ubuntu/dlnkgpt/model_finetuning
python3.11 prepare_autotrain_dataset.py
```

**Output:**
- Dataset location: `/home/ubuntu/dlnkgpt/model_finetuning/autotrain_dataset`
- Training examples: 54,000
- Validation examples: 6,000
- Format: Hugging Face Dataset format

### Step 2: (Optional) Upload to Hugging Face Hub

If you want to use the dataset on Hugging Face Hub:

```bash
# Set your Hugging Face token
export HF_TOKEN=your_token_here

# Upload dataset
python3.11 upload_to_hub.py
```

**Note:** You can skip this step and train locally.

### Step 3: Configure Training

Review and customize training settings:

```bash
python3.11 autotrain_config.py
```

This will show the configuration and ask for confirmation before starting training.

### Step 4: Start Training

The training will start automatically after confirmation, or you can launch it manually:

```bash
# Using the configuration script (recommended)
python3.11 autotrain_config.py

# Or using AutoTrain CLI directly
autotrain llm \
  --train \
  --project-name dlnkgpt-uncensored \
  --model EleutherAI/gpt-j-6b \
  --data-path /home/ubuntu/dlnkgpt/model_finetuning/autotrain_dataset \
  --text-column text \
  --lr 2e-5 \
  --epochs 3 \
  --batch-size 4 \
  --gradient-accumulation 8 \
  --max-seq-length 512 \
  --use-peft \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --fp16
```

---

## ⚙️ Training Configuration

### Default Settings

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Base Model** | EleutherAI/gpt-j-6b | 6 billion parameter model |
| **Epochs** | 3 | Number of training passes |
| **Batch Size** | 4 | Samples per batch |
| **Learning Rate** | 2e-5 | Optimizer learning rate |
| **Gradient Accumulation** | 8 | Effective batch size: 32 |
| **Max Sequence Length** | 512 | Maximum tokens per example |
| **Warmup Ratio** | 0.1 | Learning rate warmup |

### PEFT/LoRA Settings

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Use PEFT** | True | Parameter-Efficient Fine-Tuning |
| **LoRA R** | 16 | LoRA rank |
| **LoRA Alpha** | 32 | LoRA scaling factor |
| **LoRA Dropout** | 0.05 | Dropout rate |

**Benefits of PEFT/LoRA:**
- ✅ Reduces memory usage by 90%
- ✅ Faster training (2-3x speedup)
- ✅ Smaller model size (only adapter weights saved)
- ✅ Can train on consumer GPUs (16GB VRAM)

### Advanced Settings

| Parameter | Value | Description |
|-----------|-------|-------------|
| **FP16** | True | Mixed precision training |
| **Logging Steps** | 100 | Log every N steps |
| **Eval Steps** | 500 | Evaluate every N steps |
| **Save Steps** | 1000 | Save checkpoint every N steps |
| **Save Total Limit** | 2 | Keep only last 2 checkpoints |

---

## 📊 Training Process

### Phase 1: Initialization (5-10 minutes)
- Download base model (~24 GB)
- Load dataset
- Initialize training environment
- Setup PEFT/LoRA adapters

### Phase 2: Training (8-12 hours on GPU)
- Train on 54,000 examples
- 3 epochs
- Automatic checkpointing
- Validation every 500 steps

### Phase 3: Finalization (5-10 minutes)
- Save final model
- Generate training metrics
- Create model card

### Expected Timeline

**GPU Training (NVIDIA A100/V100):**
- Epoch 1: 2-3 hours
- Epoch 2: 2-3 hours
- Epoch 3: 2-3 hours
- **Total: 8-12 hours**

**GPU Training (RTX 3090/4090):**
- Epoch 1: 3-4 hours
- Epoch 2: 3-4 hours
- Epoch 3: 3-4 hours
- **Total: 12-16 hours**

**CPU Training:**
- Epoch 1: 24-36 hours
- Epoch 2: 24-36 hours
- Epoch 3: 24-36 hours
- **Total: 3-5 days**

---

## 📈 Monitoring Training

### View Training Logs

```bash
# Real-time monitoring
tail -f /home/ubuntu/dlnkgpt/model_finetuning/autotrain_output/logs/training.log

# View TensorBoard (if available)
tensorboard --logdir /home/ubuntu/dlnkgpt/model_finetuning/autotrain_output
```

### Key Metrics to Watch

1. **Loss**: Should decrease over time
   - Initial: ~3.0-4.0
   - Target: <2.0

2. **Learning Rate**: Should follow warmup schedule
   - Warmup: 0 → 2e-5 (first 10%)
   - Decay: 2e-5 → 0 (remaining 90%)

3. **Perplexity**: Should decrease
   - Initial: ~20-30
   - Target: <15

4. **GPU Memory**: Should be stable
   - With PEFT: ~14-16 GB
   - Without PEFT: ~22-24 GB

---

## 🎓 Training Strategies

### Strategy 1: Fast Training (Recommended)
- Use PEFT/LoRA
- FP16 enabled
- Gradient accumulation: 8
- **Time**: 8-12 hours (GPU)
- **Quality**: High

### Strategy 2: Full Fine-Tuning
- Disable PEFT
- Train all parameters
- Requires 24+ GB VRAM
- **Time**: 24-36 hours (GPU)
- **Quality**: Highest

### Strategy 3: CPU Training
- Use PEFT/LoRA
- Smaller batch size: 2
- More gradient accumulation: 16
- **Time**: 3-5 days
- **Quality**: High

---

## 🔧 Troubleshooting

### Issue 1: Out of Memory (OOM)

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. Reduce batch size: `--batch-size 2`
2. Increase gradient accumulation: `--gradient-accumulation 16`
3. Reduce max sequence length: `--max-seq-length 256`
4. Enable PEFT if not already: `--use-peft`

### Issue 2: Slow Training

**Symptoms:**
- Training takes too long
- Low GPU utilization

**Solutions:**
1. Increase batch size if memory allows
2. Reduce gradient accumulation
3. Check GPU is being used: `nvidia-smi`
4. Enable FP16: `--fp16`

### Issue 3: Model Not Learning

**Symptoms:**
- Loss not decreasing
- High perplexity

**Solutions:**
1. Increase learning rate: `--lr 5e-5`
2. Increase epochs: `--epochs 5`
3. Check dataset quality
4. Reduce LoRA dropout: `--lora-dropout 0.01`

### Issue 4: Training Crashes

**Symptoms:**
- Training stops unexpectedly
- Connection lost

**Solutions:**
1. Use `tmux` or `screen` for persistent sessions
2. Enable automatic checkpointing
3. Reduce batch size to prevent OOM
4. Check disk space

---

## 📁 Output Structure

After training completes, you'll find:

```
autotrain_output/
├── adapter_config.json          # LoRA adapter configuration
├── adapter_model.bin            # LoRA adapter weights (~100 MB)
├── training_config.json         # Training configuration
├── training_args.bin            # Training arguments
├── trainer_state.json           # Trainer state
├── tokenizer_config.json        # Tokenizer configuration
├── special_tokens_map.json      # Special tokens
├── vocab.json                   # Vocabulary
├── merges.txt                   # BPE merges
├── logs/                        # Training logs
│   └── training.log
└── checkpoints/                 # Training checkpoints
    ├── checkpoint-1000/
    ├── checkpoint-2000/
    └── ...
```

---

## 🚀 Using the Trained Model

### Option 1: Load with Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6b")

# Load LoRA adapter
model = PeftModel.from_pretrained(
    base_model,
    "/home/ubuntu/dlnkgpt/model_finetuning/autotrain_output"
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "/home/ubuntu/dlnkgpt/model_finetuning/autotrain_output"
)

# Generate
prompt = "### Instruction:\nExplain AI\n\n### Response:\n"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200)
print(tokenizer.decode(outputs[0]))
```

### Option 2: Use with API

Update the API to use the trained model:

```python
# In backend_api/app/main_advanced.py
MODEL_PATH = "/home/ubuntu/dlnkgpt/model_finetuning/autotrain_output"
```

### Option 3: Merge and Save

Merge LoRA adapter with base model:

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

# Load and merge
base_model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6b")
model = PeftModel.from_pretrained(base_model, "autotrain_output")
merged_model = model.merge_and_unload()

# Save merged model
merged_model.save_pretrained("dlnkgpt-uncensored-merged")
```

---

## 📊 Evaluation

### Automatic Evaluation

AutoTrain automatically evaluates the model during training:

```bash
# View evaluation results
cat /home/ubuntu/dlnkgpt/model_finetuning/autotrain_output/trainer_state.json
```

### Manual Evaluation

Use the evaluation system we created:

```bash
cd /home/ubuntu/dlnkgpt/model_finetuning

python3.11 evaluation_system.py \
  --model-path autotrain_output \
  --output evaluation_results.json
```

---

## 🌐 Upload to Hugging Face Hub

### Upload Trained Model

```bash
# Login to Hugging Face
huggingface-cli login

# Upload model
python3.11 -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='autotrain_output',
    repo_id='your-username/dlnkgpt-uncensored',
    repo_type='model'
)
"
```

### Create Model Card

Create `README.md` in the output directory:

```markdown
---
language: en
license: apache-2.0
tags:
- text-generation
- gpt-j
- uncensored
datasets:
- your-username/dlnkgpt-uncensored-dataset
---

# dLNk GPT Uncensored

Fine-tuned GPT-J-6B model for uncensored text generation.

## Training Details
- Base Model: EleutherAI/gpt-j-6b
- Training Examples: 60,000
- Epochs: 3
- Method: LoRA/PEFT
```

---

## 💡 Best Practices

### 1. Dataset Quality
- ✅ Use diverse, high-quality examples
- ✅ Include adversarial examples
- ✅ Balance different types of content

### 2. Training Efficiency
- ✅ Use PEFT/LoRA for faster training
- ✅ Enable FP16 for GPU training
- ✅ Use gradient accumulation for larger effective batch size

### 3. Monitoring
- ✅ Watch loss and perplexity
- ✅ Evaluate on validation set regularly
- ✅ Save checkpoints frequently

### 4. Resource Management
- ✅ Use appropriate hardware for your needs
- ✅ Monitor GPU/CPU usage
- ✅ Ensure sufficient disk space

---

## 📚 Additional Resources

### Hugging Face Documentation
- [AutoTrain Documentation](https://huggingface.co/docs/autotrain)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Transformers Documentation](https://huggingface.co/docs/transformers)

### Tutorials
- [Fine-tuning Large Language Models](https://huggingface.co/blog/fine-tune-llms)
- [LoRA: Low-Rank Adaptation](https://huggingface.co/blog/lora)
- [Training on Custom Datasets](https://huggingface.co/docs/datasets)

---

## ✅ Checklist

Before starting training:

- [ ] Dataset prepared (60,000 examples)
- [ ] AutoTrain installed
- [ ] Sufficient disk space (50+ GB)
- [ ] GPU available (recommended)
- [ ] Configuration reviewed
- [ ] Backup plan for long training

During training:

- [ ] Monitor training logs
- [ ] Check GPU/CPU usage
- [ ] Watch loss and metrics
- [ ] Save checkpoints regularly

After training:

- [ ] Evaluate model performance
- [ ] Test with sample prompts
- [ ] Compare with base model
- [ ] Upload to Hub (optional)
- [ ] Integrate with API

---

## 🎉 Summary

You now have everything needed to train the dLNk GPT uncensored model using Hugging Face AutoTrain:

1. ✅ **60,000 high-quality training examples**
2. ✅ **AutoTrain installed and configured**
3. ✅ **Dataset prepared in correct format**
4. ✅ **Training scripts ready**
5. ✅ **Evaluation system in place**
6. ✅ **API integration ready**

**To start training:**

```bash
cd /home/ubuntu/dlnkgpt/model_finetuning
python3.11 autotrain_config.py
```

Good luck with your training! 🚀

---

**Version**: 1.0.0  
**Last Updated**: November 12, 2025  
**Status**: ✅ Ready for Training
