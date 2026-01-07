# Memory Assistant v3.8 - Docker + Context Overflow Update

## What's New

### 🐳 Docker Integration with GPU Support
- **Dockerized LLM and Embedding Models**: Models stay loaded in memory, no restart delays
- **GPU Acceleration**: CUDA 12.1 support for both models
- **API Server**: FastAPI-based service with health checks
- **Dual Mode**: Run in Docker mode or Local mode with environment variable

### 📦 Context Overflow Management
- **Smart Context Window**: Automatically manages context size
- **Overflow Storage**: Older messages moved to persistent storage
- **Context Summarization**: Overflow context summarized when needed
- **Token Estimation**: Tracks approximate token usage

### 🔄 Recursive Processing
- **Large Context Handling**: Process contexts larger than model window
- **Chunk Summarization**: Recursively summarize and combine chunks
- **Intelligent Routing**: Automatically use recursive mode when needed

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Memory Assistant                      │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Brain (Context Manager + LLM Interface)         │  │
│  │  - Context overflow detection                    │  │
│  │  - Token estimation                              │  │
│  │  - Recursive processing                          │  │
│  └──────────────────────────────────────────────────┘  │
│              ↓                        ↓                 │
│   ┌──────────────────┐    ┌──────────────────┐        │
│   │  Local Mode      │    │  Docker Mode     │        │
│   │  - Load locally  │    │  - API client    │        │
│   │  - Direct calls  │    │  - HTTP requests │        │
│   └──────────────────┘    └──────────────────┘        │
└─────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────┐
│              Docker Container (GPU)                     │
│  ┌──────────────────────────────────────────────────┐  │
│  │  FastAPI Server (port 8000)                      │  │
│  │  - /health   - Health check                      │  │
│  │  - /chat     - Chat completion                   │  │
│  │  - /completion - Text completion                 │  │
│  │  - /embed    - Generate embeddings               │  │
│  └──────────────────────────────────────────────────┘  │
│  ┌──────────────────┐    ┌──────────────────┐        │
│  │  Qwen Model      │    │  Embedding Model │        │
│  │  (GPU: 35 layers)│    │  (GPU/CPU)       │        │
│  └──────────────────┘    └──────────────────┘        │
└─────────────────────────────────────────────────────────┘
```

## Files Created/Modified

### New Files:
1. **Dockerfile** - Container definition with CUDA support
2. **model_server.py** - FastAPI server for LLM and embeddings
3. **docker_clients.py** - Client wrappers for Docker services
4. **test_docker_services.py** - Test suite for Docker services
5. **start.ps1** - Easy startup script with mode selection
6. **DOCKER_README.md** - Docker setup documentation
7. **.dockerignore** - Optimize Docker builds

### Modified Files:
1. **docker-compose.yml** - Added LLM service with GPU
2. **requirements.txt** - Added FastAPI, uvicorn, httpx
3. **memory_assistant.py** - Added:
   - Docker mode support
   - Context overflow management
   - Recursive processing
   - Token estimation

## Usage

### Quick Start - Docker Mode (Recommended)
```powershell
# Build and start services
docker-compose build
docker-compose up -d

# Run memory assistant
$env:USE_DOCKER="true"
python memory_assistant.py
```

### Local Mode (Traditional)
```powershell
$env:USE_DOCKER="false"
python memory_assistant.py
```

### Using Startup Script
```powershell
.\start.ps1
# Choose: 1 for Docker, 2 for Local
```

## Configuration

### Environment Variables:
- `USE_DOCKER`: Set to "true" for Docker mode, "false" for local
- `DOCKER_LLM_URL`: LLM server URL (default: http://localhost:8000)

### Context Settings (memory_assistant.py):
```python
CTX_SIZE = 4096                    # Model context window
MAX_TOKENS_PER_MESSAGE = 512       # Avg tokens per message
CONTEXT_SAFETY_MARGIN = 500        # Reserve for system + response
OVERFLOW_STORAGE_PATH = "./context_overflow.json"
```

## Features

### 1. Context Overflow Protection
- Automatically detects when context exceeds model window
- Moves old messages to overflow storage
- Keeps most recent and system messages
- Stores up to 50 overflow messages

### 2. Overflow Summarization
- Summarizes overflow context when needed
- Includes summary as system message
- Configurable with `use_overflow=True` parameter

### 3. Recursive Large Context
- Use `generate_response_with_large_context()` for huge contexts
- Splits context into chunks
- Summarizes each chunk
- Combines summaries for final answer

### 4. Docker Benefits
- **Persistent Models**: Load once, use multiple times
- **GPU Acceleration**: Automatic in Docker container
- **Isolation**: Models run separately from main app
- **Scalability**: Easy to add more model servers

## Testing

```powershell
# Test Docker services
python test_docker_services.py

# Check service health
curl http://localhost:8000/health

# View logs
docker logs llm_embedding_server -f
```

## Performance

### Docker Mode:
- ✓ Models stay loaded (no reload time)
- ✓ GPU acceleration
- ✓ Isolated resources
- ✓ Network overhead (minimal, local)

### Local Mode:
- ✓ Direct memory access
- ✓ No network overhead
- ✗ Model reload on restart
- ✗ Manual GPU setup

## Troubleshooting

### Docker service won't start:
```powershell
docker logs llm_embedding_server
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### GPU not detected:
```powershell
nvidia-smi  # Check GPU
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Context overflow not working:
- Check `context_overflow.json` file is writable
- Verify token estimation with smaller contexts first
- Enable verbose logging to see overflow messages

## Next Steps

1. **Monitor Context**: Watch logs for overflow messages
2. **Tune Settings**: Adjust context margins if needed
3. **Test Recursive**: Try with very large contexts
4. **Scale Up**: Add more Docker replicas for load balancing
