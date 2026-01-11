# Lingu67 - AI Memory Assistant

A local AI assistant with persistent memory, GPU-accelerated LLM inference, and vector-based retrieval. Features a modular architecture with specialized agents and a comprehensive LLM memory framework.

## 🏗️ Project Structure

```
lingu67/
├── chatbot/                    # Memory Assistant Chatbot
│   ├── adapters/               # LLM & Embedding wrappers
│   │   ├── llm_adapter.py      # LLM client wrapper
│   │   └── embedding_adapter.py # Embedding client wrapper
│   ├── agents/                 # Specialized workers
│   │   ├── base.py             # Base agent class
│   │   ├── memory_agent.py     # Memory operations
│   │   ├── file_agent.py       # File handling
│   │   ├── profile_agent.py    # User profile management
│   │   └── rag_agent.py        # RAG retrieval
│   ├── orchestrator/           # Central coordinator
│   │   ├── core.py             # Main orchestrator
│   │   ├── router.py           # Intent routing
│   │   └── context.py          # Session context
│   ├── interfaces/             # Entry points
│   │   ├── cli.py              # Command-line interface
│   │   └── web/                # Flask web UI
│   │       ├── app.py          # Flask application
│   │       ├── routes.py       # API routes
│   │       └── templates.py    # HTML templates
│   ├── old/                    # Legacy versions
│   │   ├── chatbot_ui.py       # Original Flask UI
│   │   └── memory_assistant.py # Original CLI
│   └── README.md               # Chatbot documentation
│
├── memory/                     # LLM Memory Framework
│   ├── core.py                 # MemoryManager orchestrator
│   ├── stores/                 # Memory storage
│   │   ├── short_term.py       # Volatile context (policies apply)
│   │   ├── long_term.py        # Qdrant-backed persistent storage
│   │   └── feature_memory.py   # JSON facts with history
│   ├── managers/               # Memory management
│   │   ├── llm_manager.py      # LLM-driven decisions
│   │   └── policies.py         # Retention, decay, compression
│   ├── utils/                  # Utilities
│   │   ├── scopes.py           # User/global/session isolation
│   │   └── helpers.py          # Token counting, JSON utilities
│   ├── backends/               # Storage backends
│   │   └── qdrant_backend.py   # Qdrant vector DB wrapper
│   ├── tests/                  # Test suite (80 tests)
│   │   ├── unit_tests.py       # 55 unit tests
│   │   ├── extreme_tests.py    # 10 stress tests
│   │   ├── llm_integration_test.py # 5 LLM tests
│   │   ├── llm_stress_test.py  # 10 edge case tests
│   │   └── benchmark.py        # Performance benchmarks
│   └── README.md               # Full memory documentation
│
├── server/                     # Shared LLM Infrastructure
│   ├── model_server.py         # FastAPI server (GPU LLM + embeddings)
│   ├── local_client.py         # HTTP client library
│   └── rag_handler.py          # RAG utilities & file ingestion
│
├── chess/                      # Chess Game with AI
│   ├── app.py                  # Flask chess web UI
│   ├── game.py                 # Game logic
│   ├── stockfish_client.py     # Stockfish AI integration
│   └── chess_client.py         # LLM analysis
│
├── config.py                   # Global configuration
└── requirements.txt            # Python dependencies
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
# Command Line
python -m chatbot.interfaces.cli

# Web UI
python -m chatbot.interfaces.web.app
```

**Legacy Version:**
```bash
python chatbot/old/chatbot_ui.py  # Port 7860
```

Open http://localhost:5000 (new) or http://localhost:7860 (legacy)

---

## 🧠 Memory Framework

A comprehensive LLM memory system with three distinct memory types:

| Memory Type | Storage | Policies | Purpose |
|-------------|---------|----------|---------|
| **Short-term** | In-memory | ✅ Token limit, decay | Recent conversation context |
| **Long-term** | Qdrant DB | ❌ Permanent | All conversations (semantic search) |
| **Feature** | JSON files | ❌ No | User facts with history tracking |

### Usage

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

# Search long-term memory
results = mm.search_long_term("Python programming")
```

### Performance Benchmarks

| Operation | Throughput | p50 Latency |
|-----------|------------|-------------|
| Short-term write | 95,724 ops/sec | 0.006ms |
| Short-term read | 15,256 ops/sec | 0.065ms |
| Feature read | 2.6M ops/sec | <0.001ms |
| Semantic search | ~57 ops/sec | 17.6ms |
| End-to-end turn | ~2.5 ops/sec | 388ms |

See [memory/README.md](memory/README.md) for full documentation.

---

## 🤖 Chatbot Architecture

The modular chatbot uses specialized agents coordinated by an orchestrator:

```
┌─────────────────────────────────────────────────┐
│                  ORCHESTRATOR                    │
│  • Intent detection & routing                    │
│  • Session context management                    │
│  • Response coordination                         │
└────────────┬──────────────────────┬─────────────┘
             │                      │
    ┌────────▼────────┐    ┌───────▼────────┐
    │   ADAPTERS      │    │    AGENTS      │
    │  • LLM Client   │    │  • Memory      │
    │  • Embeddings   │    │  • File        │
    └─────────────────┘    │  • Profile     │
                           │  • RAG         │
                           └────────────────┘
```

---

## ♟️ Chess Module

Play chess against Stockfish with LLM-powered move analysis:

```bash
python chess/app.py
```
- Web UI at http://localhost:5001
- Stockfish AI opponent
- LLM move explanations

---

## 📁 File Upload

- Upload files via the web UI drawer
- Files are chunked and embedded for semantic search
- Ask questions like "summarize chapter 3"

---

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

---

## 🧪 Testing

```bash
# Memory framework tests (80 total)
python memory/tests/unit_tests.py          # 55 unit tests
python memory/tests/extreme_tests.py       # 10 stress tests
python memory/tests/llm_integration_test.py # 5 LLM tests
python memory/tests/llm_stress_test.py     # 10 edge cases

# Benchmarks
python memory/tests/benchmark.py
```

---

## 🔧 Creating New Subprojects

```python
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from server.local_client import LocalLLMClient
from memory import MemoryManager
from config import MODEL_SERVER_URL

# Initialize
client = LocalLLMClient(MODEL_SERVER_URL)
mm = MemoryManager(scope="user:myapp", llm_client=client)

# Use memory
mm.add_turn(user_message, assistant_response)
context = mm.get_context(user_message)
```

---

## 📦 Tech Stack

| Component | Technology |
|-----------|-----------|
| **LLM** | Qwen 2.5 3B (GGUF) via llama.cpp |
| **Embeddings** | SentenceTransformer (all-MiniLM-L6-v2) |
| **Vector DB** | Qdrant (local file mode) |
| **Memory** | Custom framework (short-term, long-term, feature) |
| **Web UI** | Flask + vanilla JS |
| **API** | FastAPI |
| **GPU** | CUDA 12.4 (RTX 3050) |

---

## 📄 License

MIT
