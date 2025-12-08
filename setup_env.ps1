# Project Aether - Environment Setup Script
# Run this script to install all dependencies for Python 3.11

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "Project Aether - Environment Setup" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan

# Check Python 3.11
Write-Host "`nChecking Python 3.11..." -ForegroundColor Yellow
$python = "py -3.11"
try {
    $version = & py -3.11 --version 2>&1
    Write-Host "  Found: $version" -ForegroundColor Green
} catch {
    Write-Host "  ERROR: Python 3.11 not found!" -ForegroundColor Red
    Write-Host "  Install it with: py install 3.11" -ForegroundColor Red
    exit 1
}

# Upgrade pip
Write-Host "`nUpgrading pip..." -ForegroundColor Yellow
& py -3.11 -m pip install --upgrade pip

# Install PyTorch with CUDA
Write-Host "`nInstalling PyTorch with CUDA 12.1 (this may take a while - 2.4GB)..." -ForegroundColor Yellow
& py -3.11 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install other requirements
Write-Host "`nInstalling other dependencies..." -ForegroundColor Yellow
& py -3.11 -m pip install -r requirements.txt

# Verify installation
Write-Host "`nVerifying installation..." -ForegroundColor Yellow
& py -3.11 scripts/test_setup.py

Write-Host "`n======================================" -ForegroundColor Cyan
Write-Host "Setup complete!" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host "`nTo run the linear probe pipeline:" -ForegroundColor Green
Write-Host "  py -3.11 scripts/run_phase1.py --quick" -ForegroundColor White

