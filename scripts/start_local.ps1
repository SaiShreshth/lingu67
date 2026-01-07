# Start Local LLM Server and Memory Assistant

Write-Host "=" -ForegroundColor Cyan -NoNewline; Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "  Memory Assistant - Local Server Mode" -ForegroundColor Cyan
Write-Host "=" -ForegroundColor Cyan -NoNewline; Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host ""

# Check if venv is activated
if (-not $env:VIRTUAL_ENV) {
    Write-Host "🔧 Activating virtual environment..." -ForegroundColor Yellow
    & .\venv\Scripts\Activate.ps1
}

# Start Qdrant in background if not running
Write-Host "🔍 Checking Qdrant..." -ForegroundColor Yellow
$qdrantRunning = docker ps --filter "name=memory_db_server" --filter "status=running" --format "{{.Names}}"
if (-not $qdrantRunning) {
    Write-Host "🚀 Starting Qdrant..." -ForegroundColor Green
    docker run -d --name memory_db_server -p 6333:6333 -p 6334:6334 -v ${PWD}/qdrant_data:/qdrant/storage qdrant/qdrant:latest
    Start-Sleep -Seconds 3
} else {
    Write-Host "✓ Qdrant already running" -ForegroundColor Green
}

Write-Host ""
Write-Host "📋 Choose how to start:" -ForegroundColor Cyan
Write-Host "  1) Start LLM server in NEW terminal (recommended)" -ForegroundColor White
Write-Host "  2) Start LLM server in background (runs in background)" -ForegroundColor White
Write-Host "  3) Connect to existing LLM server (if already running)" -ForegroundColor White
Write-Host ""
$choice = Read-Host "Enter choice (1-3)"

if ($choice -eq "1") {
    Write-Host ""
    Write-Host "🚀 Starting LLM server in new terminal..." -ForegroundColor Green
    Write-Host "   (Keep this terminal open - it will show model loading progress)" -ForegroundColor Yellow
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; .\venv\Scripts\Activate.ps1; python local_server.py"
    Write-Host ""
    Write-Host "⏳ Waiting 10 seconds for model to load..." -ForegroundColor Yellow
    Start-Sleep -Seconds 10
}
elseif ($choice -eq "2") {
    Write-Host ""
    Write-Host "🚀 Starting LLM server in background..." -ForegroundColor Green
    $serverJob = Start-Job -ScriptBlock {
        param($path)
        Set-Location $path
        & .\venv\Scripts\python.exe local_server.py
    } -ArgumentList $PWD
    Write-Host "   Job ID: $($serverJob.Id)" -ForegroundColor Gray
    Write-Host ""
    Write-Host "⏳ Waiting 10 seconds for model to load..." -ForegroundColor Yellow
    Start-Sleep -Seconds 10
}
else {
    Write-Host ""
    Write-Host "✓ Will connect to existing server at http://localhost:8000" -ForegroundColor Green
}

# Check if server is ready
Write-Host ""
Write-Host "🔍 Checking LLM server health..." -ForegroundColor Yellow
$maxRetries = 5
$retryCount = 0
$serverReady = $false

while ($retryCount -lt $maxRetries -and -not $serverReady) {
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:8000/health" -TimeoutSec 5
        if ($response.status -eq "healthy") {
            $serverReady = $true
            Write-Host "✓ LLM server is ready!" -ForegroundColor Green
        }
    }
    catch {
        $retryCount++
        if ($retryCount -lt $maxRetries) {
            Write-Host "   Retry $retryCount/$maxRetries..." -ForegroundColor Gray
            Start-Sleep -Seconds 3
        }
    }
}

if (-not $serverReady) {
    Write-Host ""
    Write-Host "⚠ Warning: Could not connect to LLM server" -ForegroundColor Red
    Write-Host "   Make sure the server is running on http://localhost:8000" -ForegroundColor Yellow
    Write-Host ""
    $continue = Read-Host "Continue anyway? (y/n)"
    if ($continue -ne "y") {
        exit
    }
}

# Start memory assistant
Write-Host ""
Write-Host "🚀 Starting Memory Assistant..." -ForegroundColor Green
Write-Host ""
python memory_assistant.py

Write-Host ""
Write-Host "👋 Memory Assistant stopped" -ForegroundColor Yellow

# Ask to stop server
Write-Host ""
$stopServer = Read-Host "Stop LLM server? (y/n)"
if ($stopServer -eq "y") {
    if ($serverJob) {
        Write-Host "🛑 Stopping background server job..." -ForegroundColor Yellow
        Stop-Job -Id $serverJob.Id
        Remove-Job -Id $serverJob.Id
    }
    Write-Host "   (If running in separate terminal, close it manually)" -ForegroundColor Gray
}
