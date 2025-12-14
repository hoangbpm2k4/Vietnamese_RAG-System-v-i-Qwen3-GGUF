@echo off
REM Vietnamese RAG System - Quick Setup Script for Windows
REM This script helps you set up the project quickly

echo ==========================================
echo Vietnamese RAG System - Quick Setup
echo ==========================================
echo.

REM Check Python version
echo [1/5] Checking Python version...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python not found! Please install Python 3.8+
    pause
    exit /b 1
)
echo OK - Python detected
echo.

REM Create virtual environment
echo [2/5] Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    echo OK - Virtual environment created
) else (
    echo OK - Virtual environment already exists
)
echo.

REM Activate virtual environment
echo [3/5] Activating virtual environment...
call venv\Scripts\activate.bat
echo OK - Virtual environment activated
echo.

REM Install dependencies
echo [4/5] Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt
echo OK - Dependencies installed
echo.

REM Check model file
echo [5/5] Checking model file...
if exist "qwen3_06b.gguf" (
    echo OK - Model file found
) else (
    echo WARNING: Model file 'qwen3_06b.gguf' not found!
    echo.
    echo Please download the model file and place it in the project root.
    echo You can download from:
    echo   - Hugging Face: https://huggingface.co/Qwen
    echo   - Or convert from original model using llama.cpp
)
echo.

echo ==========================================
echo Setup Complete!
echo ==========================================
echo.
echo Next steps:
echo   1. Activate virtual environment: venv\Scripts\activate.bat
echo   2. Run example: python example_usage.py
echo   3. Or query directly: python rag_ultimate_v2.py "Your question?"
echo.
echo Configuration (optional):
echo   set DOC_PATH=path\to\your\document.txt
echo   set K_TOP=5
echo   set TEMPERATURE=0.1
echo.
echo Happy coding!
pause
