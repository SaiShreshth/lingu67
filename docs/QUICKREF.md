# Memory Assistant v3.8 - Quick Reference

## 🚀 First Time Setup

```powershell
# 1. Setup Docker (one time)
.\setup_docker.ps1

# 2. Start the assistant
.\start.ps1
```

## 📋 Common Commands

### Start Services
```powershell
docker-compose up -d
```

### Stop Services
```powershell
docker-compose down
```

### View Logs
```powershell
docker logs llm_embedding_server -f
docker logs memory_db_server -f
```

### Check Status
```powershell
docker-compose ps
curl http://localhost:8000/health
```

### Restart Services
```powershell
docker-compose restart
```

### Rebuild After Changes
```powershell
docker-compose build --no-cache
docker-compose up -d
```

## 🎯 Running Modes

### Docker Mode (Recommended)
```powershell
$env:USE_DOCKER="true"
python memory_assistant.py
```
✓ Models stay loaded  
✓ GPU acceleration  
✓ Faster repeated use  

### Local Mode
```powershell
$env:USE_DOCKER="false"
python memory_assistant.py
```
✓ No Docker required  
✓ Direct memory access  
✗ Reload on restart  

## 🔧 Configuration

### Context Settings
Edit `memory_assistant.py`:
```python
CTX_SIZE = 4096              # Model context window
CONTEXT_SAFETY_MARGIN = 500  # Reserve tokens
```

### Docker URLs
Edit `memory_assistant.py`:
```python
DOCKER_LLM_URL = "http://localhost:8000"
QDRANT_URL = "http://localhost:6333"
```

## 📊 Monitoring

### Check Context Overflow
```powershell
cat context_overflow.json
```

### Watch Model Memory Usage
```powershell
docker stats llm_embedding_server
```

### GPU Usage
```powershell
docker exec llm_embedding_server nvidia-smi
```

## 🐛 Troubleshooting

### Docker won't start:
```powershell
docker-compose down
docker-compose up -d --force-recreate
```

### GPU not working:
```powershell
# Check GPU
nvidia-smi

# Test Docker GPU
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Model server errors:
```powershell
# View detailed logs
docker logs llm_embedding_server --tail 100

# Restart just LLM service
docker-compose restart llm_server
```

### Out of memory:
Edit `model_server.py`, reduce layers:
```python
n_gpu_layers=20  # Instead of 35
```

### Port already in use:
```powershell
# Find process using port 8000
netstat -ano | findstr :8000

# Kill process (replace PID)
taskkill /PID <process_id> /F
```

## 📁 Important Files

| File | Purpose |
|------|---------|
| `memory_assistant.py` | Main application |
| `model_server.py` | Docker LLM server |
| `docker_clients.py` | Docker API clients |
| `docker-compose.yml` | Service definitions |
| `Dockerfile` | LLM container image |
| `context_overflow.json` | Overflow storage |
| `user_profile.json` | User data |
| `conversation_log.txt` | Chat history |

## 🎓 Features

### Context Overflow
- Automatically saves old messages
- Keeps recent context in window
- Summarizes overflow when needed

### Recursive Processing
```python
# For very large contexts
brain.generate_response_with_large_context(
    system_context="...",
    context_chunks=["chunk1", "chunk2", ...],
    query="What's the summary?"
)
```

### Docker Benefits
- ✓ Persistent model loading
- ✓ GPU acceleration
- ✓ API access for other apps
- ✓ Easy scaling

## 📞 API Endpoints

When Docker is running:

```bash
# Health Check
GET http://localhost:8000/health

# Generate Embedding
POST http://localhost:8000/embed
{
  "text": "Your text here"
}

# Chat Completion
POST http://localhost:8000/chat
{
  "messages": [
    {"role": "system", "content": "You are helpful"},
    {"role": "user", "content": "Hi!"}
  ],
  "max_tokens": 512,
  "temperature": 0.7
}

# Text Completion
POST http://localhost:8000/completion
{
  "prompt": "Complete this: ",
  "max_tokens": 100,
  "temperature": 0.7
}
```

## 💡 Tips

1. **First run**: Use `.\setup_docker.ps1` for automatic setup
2. **Daily use**: Use `.\start.ps1` for quick mode selection
3. **Development**: Local mode for testing, Docker for production
4. **Memory**: Monitor `docker stats` to track usage
5. **Context**: Check `context_overflow.json` to see what was saved
