@echo off
REM ========================================================================
REM Start Training
REM ========================================================================

echo.
echo ========================================================================
echo dLNk GPT - Starting Training
echo ========================================================================
echo.

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Check if HF token is set in train_local.py
findstr /C:"HF_TOKEN = \"\"" train_local.py >nul
if not errorlevel 1 (
    echo.
    echo ERROR: Hugging Face token is not set!
    echo.
    echo Please edit train_local.py and set your HF token:
    echo   HF_TOKEN = "your_token_here"
    echo.
    pause
    exit /b 1
)

echo Starting training...
echo.
echo This will take approximately 8-12 hours
echo You can close this window and training will continue in background
echo.
echo Press Ctrl+C to stop training
echo.

python train_local.py

echo.
pause
