#!/usr/bin/env python3
"""
System Check Script
Verifies that all requirements are met for training
"""

import sys

print("="*80)
print("System Check - dLNk GPT Training")
print("="*80)
print()

# Check Python version
print("[1/6] Checking Python version...")
print(f"  Python: {sys.version}")
if sys.version_info < (3, 10):
    print("  ✗ ERROR: Python 3.10 or higher is required")
    sys.exit(1)
else:
    print("  ✓ Python version OK")
print()

# Check PyTorch
print("[2/6] Checking PyTorch...")
try:
    import torch
    print(f"  ✓ PyTorch: {torch.__version__}")
except ImportError:
    print("  ✗ ERROR: PyTorch is not installed")
    print("  Run: windows_setup.bat")
    sys.exit(1)
print()

# Check CUDA
print("[3/6] Checking CUDA...")
try:
    import torch
    if torch.cuda.is_available():
        print(f"  ✓ CUDA Available: Yes")
        print(f"  ✓ CUDA Version: {torch.version.cuda}")
        print(f"  ✓ GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  ✓ GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Memory: {props.total_memory / 1024**3:.1f} GB")
            print(f"    Compute Capability: {props.major}.{props.minor}")
    else:
        print("  ✗ ERROR: CUDA is not available")
        print("  Please install CUDA-enabled PyTorch")
        sys.exit(1)
except Exception as e:
    print(f"  ✗ ERROR: {e}")
    sys.exit(1)
print()

# Check required packages
print("[4/6] Checking required packages...")
packages = {
    'transformers': 'transformers',
    'datasets': 'datasets',
    'peft': 'peft',
    'accelerate': 'accelerate',
    'huggingface_hub': 'huggingface-hub'
}

all_ok = True
for module, package in packages.items():
    try:
        mod = __import__(module)
        version = getattr(mod, '__version__', 'unknown')
        print(f"  ✓ {package}: {version}")
    except ImportError:
        print(f"  ✗ {package}: NOT INSTALLED")
        all_ok = False

if not all_ok:
    print()
    print("  Some packages are missing. Run: install_dependencies.bat")
    sys.exit(1)
print()

# Check disk space
print("[5/6] Checking disk space...")
try:
    import shutil
    total, used, free = shutil.disk_usage(".")
    free_gb = free / (1024**3)
    print(f"  Free space: {free_gb:.1f} GB")
    if free_gb < 50:
        print(f"  ⚠ WARNING: Low disk space (need ~50GB for model)")
    else:
        print(f"  ✓ Disk space OK")
except Exception as e:
    print(f"  ⚠ WARNING: Could not check disk space: {e}")
print()

# Check internet connection
print("[6/6] Checking internet connection...")
try:
    import urllib.request
    urllib.request.urlopen('https://huggingface.co', timeout=5)
    print(f"  ✓ Internet connection OK")
except Exception as e:
    print(f"  ✗ ERROR: No internet connection")
    print(f"  Internet is required to download model and dataset")
    sys.exit(1)
print()

# Summary
print("="*80)
print("System Check Complete!")
print("="*80)
print()
print("Your system is ready for training!")
print()
print("Next steps:")
print("  1. Edit train_local.py and set your HF_TOKEN")
print("  2. Run: train_local.bat")
print()
