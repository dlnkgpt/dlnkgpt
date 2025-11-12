# dLNk GPT Uncensored - Local Training Setup

This package contains everything you need to fine-tune the GPT-J-6B model on your local Windows machine with a CUDA-capable GPU.

## 🚀 Quick Start

1. **Run `windows_setup.bat`** - Sets up Python environment and installs PyTorch with CUDA.
2. **Run `install_dependencies.bat`** - Installs all required training packages.
3. **Run `check_system.bat`** - Verifies that your system is ready for training.
4. **Edit `train_local.py`** - Enter your Hugging Face token in the `HF_TOKEN` variable.
5. **Run `train_local.bat`** - Starts the training process.

## 📋 Requirements

- **OS:** Windows 10/11
- **Python:** 3.10 or 3.11
- **GPU:** NVIDIA GPU with CUDA support (RTX 20xx series or newer recommended)
- **GPU Memory:** 12GB+ VRAM (24GB+ recommended for best performance)
- **RAM:** 32GB+
- **Disk Space:** ~50GB free space
- **Internet:** Required for downloading model and dataset

## 📦 What's Included

| File | Description |
|---|---|
| `windows_setup.bat` | Sets up Python, virtual environment, and PyTorch with CUDA. |
| `install_dependencies.bat` | Installs all required Python packages for training. |
| `check_system.bat` | Verifies that your system is ready for training. |
| `train_local.bat` | Starts the training process. |
| `train_local.py` | The main training script. **You must edit this file to add your Hugging Face token.** |
| `check_system.py` | Python script used by `check_system.bat`. |
| `README_LOCAL_TRAINING.md` | This file. |

## ⚙️ Step-by-Step Guide

### Step 1: Initial Setup

Double-click `windows_setup.bat` to begin. This script will:

1. Check your Python installation.
2. Create a virtual environment (`venv`).
3. Install PyTorch with CUDA support (this may take several minutes).
4. Verify that your GPU is detected.

### Step 2: Install Dependencies

Double-click `install_dependencies.bat`. This will install all the necessary packages for training, including:

- `transformers`
- `datasets`
- `peft` (for LoRA)
- `accelerate`
- `bitsandbytes` (for quantization)

### Step 3: Verify System

Double-click `check_system.bat`. This will run a series of checks to ensure your system is ready for training. It will verify:

- Python and PyTorch versions
- CUDA availability and GPU details
- All required packages are installed
- Disk space and internet connection

If all checks pass, you are ready to proceed.

### Step 4: Add Hugging Face Token

1. Open `train_local.py` in a text editor (like Notepad or VS Code).
2. Find the line `HF_TOKEN = ""`.
3. Paste your Hugging Face token between the quotes:
   ```python
   HF_TOKEN = "hf_YourTokenHere"
   ```
4. Save the file.

> **Note:** You can get your token from https://huggingface.co/settings/tokens

### Step 5: Start Training

Double-click `train_local.bat`. This will start the training process. 

- The script will first download the GPT-J-6B model (~24GB) and the dataset.
- Training will then begin automatically.
- Estimated time: 8-12 hours on an RTX 3090/4090.
- You can monitor the progress in the command prompt window.

## 🔧 Customization

You can customize the training process by editing `train_local.py`:

- **`BATCH_SIZE`**: Reduce if you get out-of-memory errors (e.g., to `2` or `1`).
- **`USE_8BIT`**: Set to `True` to use 8-bit quantization, which reduces memory usage but may slightly decrease quality.
- **`EPOCHS`**: Change the number of training epochs (default is `3`).

## 📈 Monitoring

- **Command Prompt:** Shows real-time training progress, including loss and learning rate.
- **Task Manager:** Monitor GPU and CPU usage.

## 🏁 After Training

When training is complete, your fine-tuned model will be:

1. **Saved locally** in the `dlnkgpt-model-output` directory.
2. **Pushed to the Hugging Face Hub** at `https://huggingface.co/dlnkgpt/dlnkgpt-uncensored`.

## ❓ Troubleshooting

- **`CUDA is not available`**: Ensure you have the latest NVIDIA drivers installed and that your GPU is supported.
- **`Out of memory`**: Reduce `BATCH_SIZE` in `train_local.py` or set `USE_8BIT = True`.
- **`Hugging Face token is not set`**: Make sure you have added your token to `train_local.py`.

---

*This package was created by Manus AI.*
