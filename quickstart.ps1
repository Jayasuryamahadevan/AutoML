# QuickStart Installation Script
# Run this to set up AutoSLM v2

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  AutoSLM v2 - Quick Start Setup" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# 1. Check Python version
Write-Host "[1/6] Checking Python version..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($pythonVersion -match "Python 3\.(1[0-9]|[0-9]{2})") {
    Write-Host "✓ $pythonVersion" -ForegroundColor Green
} else {
    Write-Host "✗ Python 3.10+ required. Found: $pythonVersion" -ForegroundColor Red
    exit 1
}

# 2. Create virtual environment
Write-Host "`n[2/6] Creating virtual environment..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "✓ Virtual environment already exists" -ForegroundColor Green
} else {
    python -m venv venv
    Write-Host "✓ Virtual environment created" -ForegroundColor Green
}

# 3. Activate virtual environment
Write-Host "`n[3/6] Activating virtual environment..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1
Write-Host "✓ Virtual environment activated" -ForegroundColor Green

# 4. Install core dependencies
Write-Host "`n[4/6] Installing core dependencies..." -ForegroundColor Yellow
Write-Host "(This may take a few minutes)" -ForegroundColor Dim
pip install -r requirements.txt --quiet
Write-Host "✓ Core dependencies installed" -ForegroundColor Green

# 5. Install Unsloth
Write-Host "`n[5/6] Installing Unsloth..." -ForegroundColor Yellow
Write-Host "(This may take a few minutes)" -ForegroundColor Dim
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" --quiet
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Unsloth installed" -ForegroundColor Green
} else {
    Write-Host "⚠ Unsloth installation may have issues. You can install manually later." -ForegroundColor Yellow
}

# 6. Install CLI tool
Write-Host "`n[6/6] Installing CLI tool..." -ForegroundColor Yellow
pip install -e . --quiet
pip install click --quiet
Write-Host "✓ CLI tool installed" -ForegroundColor Green

# Check Ollama
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Checking Ollama Setup" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

try {
    $ollamaVersion = ollama --version 2>&1
    Write-Host "✓ Ollama is installed: $ollamaVersion" -ForegroundColor Green
    
    Write-Host "`nChecking for required models..." -ForegroundColor Yellow
    $models = ollama list 2>&1
    
    if ($models -match "llama3.2:3b") {
        Write-Host "✓ llama3.2:3b is available" -ForegroundColor Green
    } else {
        Write-Host "⚠ llama3.2:3b not found. Installing..." -ForegroundColor Yellow
        ollama pull llama3.2:3b
        Write-Host "✓ llama3.2:3b installed" -ForegroundColor Green
    }
} catch {
    Write-Host "✗ Ollama not found!" -ForegroundColor Red
    Write-Host "Please install Ollama from https://ollama.ai" -ForegroundColor Yellow
    Write-Host "Then run: ollama pull llama3.2:3b" -ForegroundColor Yellow
}

# Summary
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Installation Complete!" -ForegroundColor Green
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "Quick Start Commands:" -ForegroundColor Yellow
Write-Host "  1. Train a model:" -ForegroundColor White
Write-Host "     python cli.py train ""Create a yoga training chatbot"" --config config/model_configs/yoga_chatbot.yaml`n" -ForegroundColor Dim

Write-Host "  2. Start API server:" -ForegroundColor White
Write-Host "     python api/main.py`n" -ForegroundColor Dim

Write-Host "  3. Generate dataset only:" -ForegroundColor White
Write-Host "     python cli.py generate-data ""yoga training"" --size 100`n" -ForegroundColor Dim

Write-Host "For more information, see README.md`n" -ForegroundColor Dim
