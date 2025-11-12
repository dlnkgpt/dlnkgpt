@echo off
REM ========================================================================
REM Install Training Dependencies
REM ========================================================================

echo.
echo ========================================================================
echo Installing Training Dependencies
echo ========================================================================
echo.

REM Activate virtual environment
call venv\Scripts\activate.bat

echo [1/8] Installing Hugging Face Hub...
pip install huggingface-hub==0.27.0
echo.

echo [2/8] Installing Transformers...
pip install transformers==4.48.0
echo.

echo [3/8] Installing Datasets...
pip install datasets
echo.

echo [4/8] Installing Accelerate...
pip install accelerate
echo.

echo [5/8] Installing PEFT (LoRA)...
pip install peft
echo.

echo [6/8] Installing bitsandbytes for quantization...
pip install bitsandbytes
echo.

echo [7/8] Installing TRL (optional)...
pip install trl
echo.

echo [8/8] Installing additional utilities...
pip install scipy sentencepiece protobuf
echo.

echo.
echo ========================================================================
echo Verifying Installation
echo ========================================================================
echo.

python -c "import torch; import transformers; import datasets; import peft; import accelerate; print('All packages installed successfully!'); print(f'PyTorch: {torch.__version__}'); print(f'Transformers: {transformers.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

echo.
echo ========================================================================
echo Installation Complete!
echo ========================================================================
echo.
echo Next step: Run train_local.bat to start training
echo.
pause
