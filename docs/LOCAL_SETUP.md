# Memory Assistant - Local Server Setup

## Quick Start (No Docker!)

### 1. Start the Local LLM Server

Open a **NEW terminal** and run:

```powershell
.\venv\Scripts\Activate.ps1
python local_server.py
```

Keep this terminal open! It will show:
- Model loading progress
- Server ready message
- API request logs

### 2. Start Qdrant (one-time)

In another terminal:

```powershell
docker run -d --name memory_db_server -p 6333:6333 -p 6334:6334 -v ${PWD}/qdrant_data:/qdrant/storage qdrant/qdrant:latest
```

### 3. Run Memory Assistant

In your main terminal:

```powershell
.\venv\Scripts\Activate.ps1
python memory_assistant.py
```

## OR Use the Automated Script

```powershell
.\start_local.ps1
```

This script will:
- Start Qdrant if not running
- Offer to start LLM server in new terminal or background
- Wait for models to load
- Launch memory assistant
- Optionally stop server when done

## What Changed?

### ❌ Removed
- `docker-compose.yml` - No longer needed
- `Dockerfile` - No longer needed  
- `model_server.py` - Replaced with `local_server.py`
- `docker_clients.py` - Replaced with `local_client.py`
- Docker mode toggle - Everything is local now
- Complex build process - Just run Python

### ✅ Added
- `local_server.py` - FastAPI server (runs locally, not in Docker)
- `local_client.py` - Simple HTTP client
- `start_local.ps1` - Easy startup script

## Architecture

```
┌─────────────────────┐
│  memory_assistant.py│
│  (Your main app)    │
└──────────┬──────────┘
           │ HTTP
           ▼
┌─────────────────────┐
│  local_server.py    │
│  FastAPI on :8000   │
│  - Qwen Model       │
│  - Embeddings       │
└─────────────────────┘

┌─────────────────────┐
│  Qdrant (Docker)    │
│  Vector DB on :6333 │
└─────────────────────┘
```

## Benefits of This Approach

### ✅ Pros
- **Simple**: No Docker builds, just Python scripts
- **Fast**: No container overhead
- **Flexible**: Easy to modify and test
- **Persistent**: Model stays loaded in background
- **Debuggable**: See all logs in real-time

### Why Not Full Docker?
- llama-cpp-python needs C++ compiler in Docker
- Large dependency downloads (2GB+)
- Complex GPU passthrough on Windows
- **Local is simpler and works just as well!**

## API Endpoints

Once `local_server.py` is running:

### Health Check
```bash
curl http://localhost:8000/health
```

### Chat Completion
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are helpful"},
      {"role": "user", "content": "Hello!"}
    ],
    "max_tokens": 512,
    "temperature": 0.7
  }'
```

### Text Completion
```bash
curl -X POST http://localhost:8000/completion \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Complete this sentence: ",
    "max_tokens": 100
  }'
```

### Generate Embedding
```bash
curl -X POST http://localhost:8000/embed \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text here"}'
```

## Configuration

Edit `local_server.py` to adjust:

```python
llm_model = Llama(
    model_path="qwen2.5-3b-instruct-q4_k_m.gguf",
    n_ctx=4096,          # Context window
    n_gpu_layers=35,     # GPU layers (0 for CPU-only)
    verbose=False
)
```

## Troubleshooting

### Server won't start
```powershell
# Check if port 8000 is in use
netstat -ano | findstr :8000

# Kill process if needed
taskkill /PID <process_id> /F
```

### GPU not working
Edit `local_server.py`, set `n_gpu_layers=0` for CPU-only mode

### Import errors
```powershell
pip install fastapi uvicorn httpx
```

### Model not found
Make sure `qwen2.5-3b-instruct-q4_k_m.gguf` is in the same folder

## Performance Tips

1. **First run**: Model loading takes 30-60 seconds
2. **Keep server running**: Leave it in background for instant responses
3. **GPU acceleration**: With `n_gpu_layers=35`, uses ~3GB VRAM
4. **CPU mode**: Set `n_gpu_layers=0` if GPU issues occur

## Stopping Everything

```powershell
# Stop LLM server
# (Close the terminal or Ctrl+C)

# Stop Qdrant
docker stop memory_db_server

# Or remove it completely
docker rm -f memory_db_server
```

## Next Steps

1. Start using: `.\start_local.ps1`
2. Test API: `curl http://localhost:8000/health`
3. Run assistant: Works exactly as before!

All your conversation history, embeddings, and profiles are preserved.
