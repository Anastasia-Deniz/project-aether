@echo off
REM Project Aether - Environment Setup Script (Batch version)
REM Run this script to install all dependencies for Python 3.11

echo ======================================
echo Project Aether - Environment Setup
echo ======================================

echo.
echo Checking Python 3.11...
py -3.11 --version
if errorlevel 1 (
    echo ERROR: Python 3.11 not found!
    echo Install it with: py install 3.11
    exit /b 1
)

echo.
echo Upgrading pip...
py -3.11 -m pip install --upgrade pip

echo.
echo Installing PyTorch with CUDA 12.1 (this may take a while - 2.4GB)...
py -3.11 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo.
echo Installing other dependencies...
py -3.11 -m pip install -r requirements.txt

echo.
echo Verifying installation...
py -3.11 scripts/test_setup.py

echo.
echo ======================================
echo Setup complete!
echo ======================================
echo.
echo To run the linear probe pipeline:
echo   py -3.11 scripts/run_phase1.py --quick

