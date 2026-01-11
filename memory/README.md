# Memory Framework

A general-purpose LLM memory system with three distinct memory types, LLM-driven management, and comprehensive testing.

## Folder Structure

```
memory/
├── __init__.py           # Package exports (MemoryManager, MemoryScope, etc.)
├── core.py               # MemoryManager - main orchestrator
│
├── stores/               # Memory storage implementations
│   ├── __init__.py
│   ├── short_term.py     # Volatile context (policies apply)
│   ├── long_term.py      # Qdrant-backed persistent storage
│   └── feature_memory.py # JSON facts with history tracking
│
├── managers/             # Memory management
│   ├── __init__.py
│   ├── llm_manager.py    # LLM-driven KEEP/DELETE/COMPRESS/PROMOTE
│   └── policies.py       # Retention, decay, compression policies
│
├── utils/                # Utilities
│   ├── __init__.py
│   ├── scopes.py         # User/global/session isolation
│   └── helpers.py        # Token counting, JSON extraction, etc.
│
├── backends/             # Storage backends
│   └── qdrant_backend.py # Qdrant vector database wrapper
│
├── tests/                # Test suite
│   ├── __init__.py
│   ├── unit_tests.py     # 55 unit/component/edge tests
│   ├── extreme_tests.py  # 10 stress tests
│   ├── llm_integration_test.py  # 5 LLM integration tests
│   ├── llm_stress_test.py       # 10 LLM edge/stress tests
│   └── benchmark.py      # Performance benchmarks
│
└── README.md             # This file
```

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                          MemoryManager                            │
│                                                                   │
│  ┌────────────────┐  ┌───────────────┐  ┌─────────────────────┐  │
│  │  SHORT-TERM    │  │ FEATURE MEM   │  │    LONG-TERM        │  │
│  │  (volatile)    │  │ (JSON facts)  │  │    (Qdrant)         │  │
│  │                │  │               │  │                     │  │
│  │ • Recent turns │  │ • User facts  │  │ • All conversations │  │
│  │ • Token limit  │  │ • History     │  │ • Semantic search   │  │
│  │ • Auto-prune   │  │ • Episodic    │  │ • File summaries    │  │
│  └────────────────┘  └───────────────┘  └─────────────────────┘  │
│                                                                   │
│  ┌────────────────┐  ┌───────────────┐  ┌─────────────────────┐  │
│  │    POLICIES    │  │   SCOPES      │  │   LLM MANAGER       │  │
│  │ • max_tokens   │  │ • global      │  │ • KEEP/DELETE       │  │
│  │ • decay        │  │ • user:{id}   │  │ • COMPRESS          │  │
│  │ • compression  │  │ • session:{id}│  │ • PROMOTE to facts  │  │
│  └────────────────┘  └───────────────┘  └─────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

## Quick Start

```python
from memory import MemoryManager, MemoryScope, ShortTermPolicy

# Initialize
mm = MemoryManager(
    scope="user:123",
    llm_client=llm,
    data_dir="data/memory"
)

# Add conversation
mm.add_turn("What's my name?", "Your name is John")

# Get context for LLM
context = mm.get_context("Tell me about myself")

# Set/get facts
mm.set_fact("language", "Python")
print(mm.get_fact("language"))  # "Python"
```

## Memory Types

| Type | Purpose | Persistence | Policies | Fed to LLM |
|------|---------|-------------|----------|------------|
| **Short-term** | Recent conversation context | Volatile | Yes | Always |
| **Long-term** | All past conversations | Qdrant DB | No | Via search |
| **Feature** | User facts with history | JSON files | No | Always |

## Testing

```powershell
# Unit tests (55 tests)
python memory/tests/unit_tests.py

# Extreme limits (10 tests)
python memory/tests/extreme_tests.py

# LLM integration (5 tests) - requires model_server
python memory/tests/llm_integration_test.py

# LLM stress tests (10 tests) - requires model_server
python memory/tests/llm_stress_test.py

# Benchmarks
python memory/tests/benchmark.py
```

## Benchmark Results

| Operation | Throughput | p50 Latency |
|-----------|------------|-------------|
| Short-term write | 95,724 ops/sec | 0.006ms |
| Short-term read | 15,256 ops/sec | 0.065ms |
| Feature read | 2.6M ops/sec | <0.001ms |
| Semantic search | ~57 ops/sec | 17.6ms |
| End-to-end turn | ~2.5 ops/sec | 388ms |

## API Reference

### MemoryManager (core.py)

| Method | Description |
|--------|-------------|
| `add_turn(user, assistant)` | Add conversation turn to all memories |
| `get_context(query, max_tokens)` | Get combined context for LLM |
| `set_fact(key, value)` | Set fact in feature memory |
| `get_fact(key)` | Get fact from feature memory |
| `search_long_term(query, limit)` | Search long-term memory |
| `manage_short_term()` | Run LLM-driven cleanup |
| `get_stats()` | Get memory statistics |

### ShortTermMemory (stores/short_term.py)

| Method | Description |
|--------|-------------|
| `add(content, role)` | Add single entry |
| `get_context(max_tokens)` | Get context list |
| `delete(entry_id)` | Delete entry |
| `compress_entry(id, new_content)` | Compress entry |

### FeatureMemory (stores/feature_memory.py)

| Method | Description |
|--------|-------------|
| `set(key, value)` | Set/update fact |
| `get(key, default)` | Get current value |
| `get_with_history(key)` | Get fact with full history |
| `to_prompt()` | Get JSON for LLM |

### LongTermMemory (stores/long_term.py)

| Method | Description |
|--------|-------------|
| `store_conversation(user, assistant)` | Store conversation |
| `search(query, limit)` | Semantic search |
| `get_relevant_context(query)` | Get context string |
