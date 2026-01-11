# Chatbot Module

Memory-augmented chatbot with **Orchestrator-Workers** architecture.

## Quick Start

```bash
# CLI
python -m chatbot.interfaces.cli

# Web UI
python -m chatbot.interfaces.web.app
```

## Architecture

```
                    ┌─────────────────┐
                    │  ChatOrchestrator│
                    │   (core.py)      │
                    └────────┬────────┘
         ┌───────────┬───────┴───────┬───────────┐
         ▼           ▼               ▼           ▼
    ┌─────────┐ ┌─────────┐   ┌──────────┐ ┌─────────┐
    │ Memory  │ │  File   │   │ Profile  │ │   RAG   │
    │  Agent  │ │  Agent  │   │  Agent   │ │  Agent  │
    └─────────┘ └─────────┘   └──────────┘ └─────────┘
```

## Structure

| Directory | Purpose |
|-----------|---------|
| `adapters/` | LLM and Embedding API wrappers |
| `agents/` | Specialized workers (memory, file, profile, rag) |
| `orchestrator/` | Central coordinator, routing, context management |
| `interfaces/` | CLI and Web entry points |

## Key Components

- **ChatOrchestrator** - Routes queries, gathers context, generates responses
- **IntentRouter** - Classifies user intent (chat, file query, memory recall, etc.)
- **ContextManager** - Manages token budget across agents
- **Agents** - Specialized workers that gather context and post-process

## Original Files

The original monolithic files are preserved for reference:
- `memory_assistant.py` - Original CLI
- `chatbot_ui.py` - Original Flask web UI
