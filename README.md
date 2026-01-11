# Lingu67 - AI Memory Assistant

A local AI assistant with persistent memory, GPU-accelerated LLM inference, and vector-based retrieval.

## 🏗️ Project Structure

```
lingu67/
├── chatbot/                # Memory Assistant Chatbot
│   ├── adapters/           # LLM & Embedding wrappers
│   ├── agents/             # Specialized workers (memory, file, profile, rag)
│   ├── orchestrator/       # Central coordinator & routing
│   ├── interfaces/         # CLI & Web entry points
│   ├── chatbot_ui.py       # [Legacy] Original Flask web UI
│   └── memory_assistant.py # [Legacy] Original CLI
│
├── server/                 # Shared LLM Infrastructure
│   ├── model_server.py     # FastAPI server (GPU LLM + embeddings)
│   ├── local_client.py     # HTTP client library
│   └── rag_handler.py      # RAG utilities
│
├── memory/                 # 🆕 Memory Framework
│   ├── core.py             # MemoryManager orchestrator
│   ├── short_term.py       # Volatile context (policies apply)
│   ├── long_term.py        # Qdrant-backed persistent storage
│   ├── feature_memory.py   # JSON facts with history
│   ├── llm_manager.py      # LLM-driven memory decisions
│   ├── policies.py         # Retention, decay, compression
│   ├── scopes.py           # User/global/session isolation
│   └── README.md           # Full documentation
│
├── models/                 # LLM Model Files
│   └── qwen2.5-3b-instruct-q4_k_m.gguf
│
├── data/                   # Persistent Data
│   ├── qdrant_local/       # Vector database
│   ├── user_profile.json   # User facts
│   └── conversation_log.txt
│
├── llama.cpp/              # llama-server binaries (GPU)
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

**New Modular Version (Recommended):**
```bash
# CLI
python -m chatbot.interfaces.cli

# Web UI
python -m chatbot.interfaces.web.app
```

**Legacy Version:**
```bash
python chatbot/chatbot_ui.py  # Port 7860
```

Open http://localhost:5000 (new) or http://localhost:7860 (legacy) in your browser.

## 🧠 Memory Framework

The new modular memory system provides three distinct memory types:

| Memory Type | Storage | Policies | Purpose |
|-------------|---------|----------|---------|
| **Short-term** | In-memory | Yes (token limit, decay) | Recent conversation context |
| **Long-term** | Qdrant DB | No (permanent) | All past conversations (semantic search) |
| **Feature** | JSON files | No | User facts with history tracking |

### Quick Usage

```python
from memory import MemoryManager

mm = MemoryManager(scope="user:123", llm_client=llm)

# Add conversation
mm.add_turn("What's my name?", "Your name is John")

# Get context for LLM
context = mm.get_context("Tell me about myself")

# Set/get facts
mm.set_fact("language", "Python")
print(mm.get_fact("language"))  # "Python"
```

### Benchmarks

| Operation | Throughput | p50 Latency |
|-----------|------------|-------------|
| Short-term write | 95,724 ops/sec | 0.006ms |
| Short-term read | 15,256 ops/sec | 0.065ms |
| Semantic search | ~57 ops/sec | 17.6ms |
| End-to-end turn | ~2.5 ops/sec | 388ms |

See `memory/README.md` for full documentation.

## 📁 File Upload

- Upload files via the web UI drawer
- Files are chunked and embedded for semantic search
- Ask questions like "summarize chapter 3"

## ⚙️ Configuration

All paths and settings are in `config.py`:

```python
from config import (
    MODEL_SERVER_URL,    # http://localhost:8000
    QDRANT_PATH,         # data/qdrant_local
    LLM_MODEL_PATH,      # models/qwen2.5-3b-...
)
```

## 🧪 Testing

```bash
# Memory framework tests (80 total)
python memory/tests.py           # 55 unit tests
python memory/extreme_tests.py   # 10 stress tests
python memory/llm_integration_test.py  # 5 LLM tests
python memory/llm_stress_test.py       # 10 edge cases

# Benchmarks
python memory/benchmark.py
```

## 🔧 Adding New Subprojects

```python
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from server.local_client import LocalLLMClient
from memory import MemoryManager
from config import MODEL_SERVER_URL

client = LocalLLMClient(MODEL_SERVER_URL)
mm = MemoryManager(scope="user:myapp", llm_client=client)
```

## 📦 Tech Stack

- **LLM**: Qwen 2.5 3B (GGUF) via llama.cpp
- **Embeddings**: SentenceTransformer (all-MiniLM-L6-v2)
- **Vector DB**: Qdrant (local file mode)
- **Memory**: Custom framework (short-term, long-term, feature)
- **Web UI**: Flask + vanilla JS
- **API**: FastAPI
- **GPU**: CUDA 12.4 (RTX 3050)

## 📄 License

MIT
