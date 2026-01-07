# Memory Assistant Startup Script
# Choose your mode: Docker or Local

Write-Host "==================================" -ForegroundColor Cyan
Write-Host "Memory Assistant Startup" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Choose mode:" -ForegroundColor Yellow
Write-Host "1. Docker Mode (GPU-accelerated, models stay loaded)"
Write-Host "2. Local Mode (load models locally)"
Write-Host ""

$choice = Read-Host "Enter choice (1 or 2)"

if ($choice -eq "1") {
    Write-Host "`nStarting in Docker Mode..." -ForegroundColor Green
    Write-Host "Checking Docker services..." -ForegroundColor Cyan
    
    # Check if Docker is running
    $dockerRunning = docker ps 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "✗ Docker is not running. Please start Docker Desktop." -ForegroundColor Red
        exit 1
    }
    
    # Check if services are up
    Write-Host "Starting Docker Compose services..." -ForegroundColor Cyan
    docker-compose up -d
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Docker services started" -ForegroundColor Green
        Write-Host "`nWaiting for LLM server to be ready..." -ForegroundColor Cyan
        Start-Sleep -Seconds 5
        
        # Test health
        try {
            $health = Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get -TimeoutSec 5
            Write-Host "✓ LLM server is healthy" -ForegroundColor Green
        } catch {
            Write-Host "⚠ LLM server may still be starting up..." -ForegroundColor Yellow
        }
        
        Write-Host "`nStarting Memory Assistant in Docker mode..." -ForegroundColor Green
        $env:USE_DOCKER = "true"
        python memory_assistant.py
    } else {
        Write-Host "✗ Failed to start Docker services" -ForegroundColor Red
        exit 1
    }
    
} elseif ($choice -eq "2") {
    Write-Host "`nStarting in Local Mode..." -ForegroundColor Green
    $env:USE_DOCKER = "false"
    python memory_assistant.py
    
} else {
    Write-Host "Invalid choice. Exiting." -ForegroundColor Red
    exit 1
}
