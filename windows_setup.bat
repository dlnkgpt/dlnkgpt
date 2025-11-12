@echo off
REM ========================================================================
REM dLNk GPT Training - Windows Setup Script
REM ========================================================================
REM
REM This script sets up the training environment on Windows
REM Requirements: Python 3.10 or 3.11, CUDA-capable GPU
REM
REM ========================================================================

echo.
echo ========================================================================
echo dLNk GPT Training - Windows Setup
echo ========================================================================
echo.

REM Check Python version
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.10 or 3.11 from https://www.python.org/
    pause
    exit /b 1
)

echo [1/6] Checking Python version...
python --version
echo.

REM Check CUDA availability
echo [2/6] Checking CUDA/GPU availability...
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>nul
if errorlevel 1 (
    echo PyTorch not installed yet, will install in next step
)
echo.

REM Create virtual environment
echo [3/6] Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    echo Virtual environment created
) else (
    echo Virtual environment already exists
)
echo.

REM Activate virtual environment
echo [4/6] Activating virtual environment...
call venv\Scripts\activate.bat
echo.

REM Upgrade pip
echo [5/6] Upgrading pip...
python -m pip install --upgrade pip
echo.

REM Install PyTorch with CUDA support
echo [6/6] Installing PyTorch with CUDA support...
echo This may take several minutes...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
echo.

REM Verify CUDA
echo Verifying CUDA installation...
python -c "import torch; print(f'\nPyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); print(f'GPU Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
echo.

if errorlevel 1 (
    echo.
    echo WARNING: CUDA verification failed
    echo Please check your NVIDIA drivers and CUDA installation
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================================================
echo Setup Complete!
echo ========================================================================
echo.
echo Next step: Run install_dependencies.bat to install training packages
echo.
pause
