"""
Global Configuration for Lingu67 Project
=========================================
Central configuration file for all subprojects to use.
Import paths and server URLs from here.
"""

import os

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# Model Paths
# =============================================================================
MODELS_DIR = os.path.join(BASE_DIR, "models")
LLM_MODEL_PATH = os.path.join(MODELS_DIR, "qwen2.5-3b-instruct-q4_k_m.gguf")

# llama.cpp binaries
LLAMA_CPP_DIR = os.path.join(BASE_DIR, "llama.cpp")
LLAMA_SERVER_PATH = os.path.join(LLAMA_CPP_DIR, "build", "bin", "Release", "llama-server.exe")

# =============================================================================
# Data Paths
# =============================================================================
DATA_DIR = os.path.join(BASE_DIR, "data")
QDRANT_PATH = os.path.join(DATA_DIR, "qdrant_local")
PROFILE_PATH = os.path.join(DATA_DIR, "user_profile.json")
FILE_METADATA_PATH = os.path.join(DATA_DIR, "file_metadata.json")
CHAT_LOG_PATH = os.path.join(DATA_DIR, "conversation_log.txt")

# =============================================================================
# Server Configuration
# =============================================================================
# Model Server (FastAPI - embeddings + LLM proxy)
MODEL_SERVER_HOST = "localhost"
MODEL_SERVER_PORT = 8000
MODEL_SERVER_URL = f"http://{MODEL_SERVER_HOST}:{MODEL_SERVER_PORT}"

# llama-server (raw LLM inference)
LLAMA_SERVER_HOST = "127.0.0.1"
LLAMA_SERVER_PORT = 8080

# =============================================================================
# LLM Settings
# =============================================================================
N_GPU_LAYERS = 35   # GPU layers for llama-server
N_CTX = 8192        # Context window size

# =============================================================================
# Server Module Paths (for imports)
# =============================================================================
SERVER_DIR = os.path.join(BASE_DIR, "server")

# =============================================================================
# Subproject Paths
# =============================================================================
CHATBOT_DIR = os.path.join(BASE_DIR, "chatbot")
