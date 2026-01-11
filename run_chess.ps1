# Chess Project Runner
# Starts all required services in separate PowerShell windows for debugging
# Usage: .\run_chess.ps1

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $ProjectRoot "venv\Scripts\python.exe"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  CHESS PROJECT LAUNCHER" -ForegroundColor Cyan  
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if venv exists
if (-not (Test-Path $VenvPython)) {
    Write-Host "[ERROR] Virtual environment not found at: $VenvPython" -ForegroundColor Red
    Write-Host "Please create a virtual environment first: python -m venv venv" -ForegroundColor Yellow
    exit 1
}

Write-Host "[1/3] Starting Model Server..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$ProjectRoot'; Write-Host '=== MODEL SERVER ===' -ForegroundColor Yellow; .\venv\Scripts\python.exe server/model_server.py"

# Wait for model server to initialize
Write-Host "      Waiting for Model Server to initialize (10 seconds)..." -ForegroundColor Gray
Start-Sleep -Seconds 10

Write-Host "[2/3] Starting Chess Server..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$ProjectRoot'; Write-Host '=== CHESS SERVER ===' -ForegroundColor Yellow; .\venv\Scripts\python.exe chess/app.py"

# Wait for chess server to start
Start-Sleep -Seconds 2

Write-Host "[3/3] Opening Chess UI in browser..." -ForegroundColor Green
Start-Process "http://localhost:7861"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  ALL SERVICES STARTED!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Services running:" -ForegroundColor White
Write-Host "  - Model Server: http://localhost:8000" -ForegroundColor Gray
Write-Host "  - Chess UI:     http://localhost:7861" -ForegroundColor Gray
Write-Host ""
Write-Host "Two PowerShell windows opened for debugging." -ForegroundColor Yellow
Write-Host "Press Ctrl+C in each window to stop the servers." -ForegroundColor Yellow
Write-Host ""
