# Lingu67 - AI Memory Assistant

A local AI assistant with persistent memory, GPU-accelerated LLM inference, and vector-based retrieval.

## 🏗️ Project Structure

```
lingu67/
├── chatbot/                # Memory Assistant Chatbot (subproject)
│   ├── chatbot_ui.py       # Flask web UI with streaming chat
│   └── memory_assistant.py # CLI version
│
├── server/                 # Shared LLM Infrastructure
│   ├── model_server.py     # FastAPI server (GPU LLM + embeddings)
│   └── local_client.py     # HTTP client library for subprojects
│
├── models/                 # LLM Model Files
│   └── qwen2.5-3b-instruct-q4_k_m.gguf
│
├── data/                   # Persistent Data
│   ├── qdrant_local/       # Vector database (episodic memory)
│   ├── user_profile.json   # User facts & preferences
│   ├── file_metadata.json  # Uploaded files tracking
│   └── conversation_log.txt
│
├── llama.cpp/              # llama-server binaries (GPU)
├── docs/                   # Documentation
├── scripts/                # Startup scripts
├── config.py               # Global configuration
└── requirements.txt
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start Model Server
```bash
python server/model_server.py
```
This starts:
- llama-server on port 8080 (GPU LLM inference)
- FastAPI proxy on port 8000 (embeddings + API)

### 3. Start Chatbot
```bash
python chatbot/chatbot_ui.py
```
Open http://localhost:7860 in your browser.

## 🧠 Memory System

The chatbot uses three types of memory:

| Memory Type | Storage | Purpose |
|-------------|---------|---------|
| **Short-term** | In-memory (last 10 messages) | Immediate context |
| **Long-term** | Qdrant vector DB | Semantic search for relevant past |
| **Profile** | user_profile.json | Permanent user facts |

## 📁 File Upload

- Upload files via the web UI drawer
- Files are chunked and embedded for semantic search
- Ask questions like "summarize chapter 3" or "what does the book say about X"

## ⚙️ Configuration

All paths and settings are in `config.py`:

```python
from config import (
    MODEL_SERVER_URL,    # http://localhost:8000
    QDRANT_PATH,         # data/qdrant_local
    LLM_MODEL_PATH,      # models/qwen2.5-3b-...
    LLAMA_SERVER_PATH,   # llama.cpp binaries
)
```

## 🔧 Adding New Subprojects

1. Create a new folder: `my_project/`
2. Import shared infrastructure:
   ```python
   import sys
   sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
   
   from server.local_client import LocalLLMClient
   from config import MODEL_SERVER_URL, QDRANT_PATH
   ```
3. Use the client:
   ```python
   client = LocalLLMClient(MODEL_SERVER_URL)
   response = client.chat(messages, stream=True)
   ```

## 📦 Tech Stack

- **LLM**: Qwen 2.5 3B (GGUF) via llama.cpp
- **Embeddings**: SentenceTransformer (all-MiniLM-L6-v2)
- **Vector DB**: Qdrant (local file mode)
- **Web UI**: Flask + vanilla JS
- **API**: FastAPI
- **GPU**: CUDA 12.4 (RTX 3050)

## 📄 License

MIT
