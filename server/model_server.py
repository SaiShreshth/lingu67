"""
Model Server - Launcher for llama-server and embedding service
Uses llama-server (llama.cpp) for LLM inference with GPU support
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import uvicorn
import httpx
import subprocess
import asyncio
import os
import logging
import signal
import sys
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    LLM_MODEL_PATH as MODEL_PATH,
    LLAMA_SERVER_PATH,
    LLAMA_SERVER_HOST,
    LLAMA_SERVER_PORT,
    N_GPU_LAYERS,
    N_CTX
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="LLM & Embedding Server")

# Global state
llama_process = None
embedding_model = None
http_client = None


class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    max_tokens: int = 512
    temperature: float = 0.7
    stream: bool = False


class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7


class EmbeddingRequest(BaseModel):
    text: str


class BatchEmbeddingRequest(BaseModel):
    texts: List[str]


def start_llama_server():
    """Start llama-server as a subprocess"""
    global llama_process
    
    cmd = [
        LLAMA_SERVER_PATH,
        "-m", MODEL_PATH,
        "--host", LLAMA_SERVER_HOST,
        "--port", str(LLAMA_SERVER_PORT),
        "-ngl", str(N_GPU_LAYERS),
        "-c", str(N_CTX),
        "--embedding"  # Enable embedding endpoint
    ]
    
    logger.info(f"Starting llama-server: {' '.join(cmd)}")
    
    try:
        # Start llama-server
        llama_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == 'win32' else 0
        )
        
        # Wait for server to be ready
        max_retries = 30
        for i in range(max_retries):
            try:
                with httpx.Client(timeout=2.0) as client:
                    resp = client.get(f"http://{LLAMA_SERVER_HOST}:{LLAMA_SERVER_PORT}/health")
                    if resp.status_code == 200:
                        logger.info("✓ llama-server is ready!")
                        return True
            except:
                pass
            time.sleep(1)
            logger.info(f"Waiting for llama-server... ({i+1}/{max_retries})")
        
        logger.error("llama-server failed to start in time")
        return False
        
    except FileNotFoundError:
        logger.error("llama-server not found! Make sure it's in your PATH.")
        return False
    except Exception as e:
        logger.error(f"Failed to start llama-server: {e}")
        return False


def stop_llama_server():
    """Stop llama-server subprocess"""
    global llama_process
    if llama_process:
        logger.info("Stopping llama-server...")
        if sys.platform == 'win32':
            llama_process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            llama_process.terminate()
        llama_process.wait(timeout=10)
        llama_process = None


@app.on_event("startup")
async def startup():
    global embedding_model, http_client
    
    # Start llama-server (handles LLM inference with GPU)
    if not start_llama_server():
        logger.warning("Running without LLM - llama-server failed to start")
    
    # Create HTTP client for llama-server
    http_client = httpx.AsyncClient(
        base_url=f"http://{LLAMA_SERVER_HOST}:{LLAMA_SERVER_PORT}",
        timeout=300.0
    )
    
    # Load embedding model on GPU
    logger.info("Loading embedding model...")
    try:
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
        logger.info("✓ Embedding model loaded on GPU")
    except Exception as e:
        logger.warning(f"GPU embedding failed: {e}, using CPU")
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        logger.info("✓ Embedding model loaded on CPU")


@app.on_event("shutdown")
async def shutdown():
    global http_client
    stop_llama_server()
    if http_client:
        await http_client.aclose()


@app.get("/health")
async def health_check():
    llama_healthy = False
    try:
        resp = await http_client.get("/health")
        llama_healthy = resp.status_code == 200
    except:
        pass
    
    return {
        "status": "healthy" if llama_healthy else "degraded",
        "llm_loaded": llama_healthy,
        "embedding_loaded": embedding_model is not None
    }


@app.post("/chat")
async def chat_completion(request: ChatRequest):
    try:
        # Build request for llama-server (OpenAI-compatible)
        payload = {
            "messages": request.messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": request.stream
        }
        
        if request.stream:
            async def stream_response():
                async with http_client.stream(
                    "POST", "/v1/chat/completions", json=payload
                ) as resp:
                    async for line in resp.aiter_lines():
                        if line.startswith("data: "):
                            data = line[6:]
                            if data == "[DONE]":
                                break
                            try:
                                import json
                                chunk = json.loads(data)
                                if chunk.get("choices") and chunk["choices"][0].get("delta", {}).get("content"):
                                    yield chunk["choices"][0]["delta"]["content"]
                            except:
                                pass
            
            return StreamingResponse(stream_response(), media_type="text/plain")
        else:
            resp = await http_client.post("/v1/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return {"response": data["choices"][0]["message"]["content"]}
            
    except httpx.HTTPError as e:
        logger.error(f"Chat completion error: {e}")
        raise HTTPException(status_code=502, detail=f"LLM server error: {e}")
    except Exception as e:
        logger.error(f"Chat completion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/completion")
async def text_completion(request: CompletionRequest):
    try:
        payload = {
            "prompt": request.prompt,
            "n_predict": request.max_tokens,
            "temperature": request.temperature
        }
        
        resp = await http_client.post("/completion", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return {"response": data.get("content", "")}
        
    except httpx.HTTPError as e:
        logger.error(f"Completion error: {e}")
        raise HTTPException(status_code=502, detail=f"LLM server error: {e}")
    except Exception as e:
        logger.error(f"Completion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embed")
async def get_embedding(request: EmbeddingRequest):
    """Get embedding using SentenceTransformer (GPU-accelerated)"""
    if embedding_model is None:
        raise HTTPException(status_code=503, detail="Embedding model not loaded")
    
    try:
        embedding = embedding_model.encode(request.text).tolist()
        return {"embedding": embedding}
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embed_batch")
async def get_embeddings_batch(request: BatchEmbeddingRequest):
    """Get embeddings for multiple texts in a single call (MUCH faster for large files)"""
    if embedding_model is None:
        raise HTTPException(status_code=503, detail="Embedding model not loaded")
    
    try:
        import time
        start = time.time()
        # Increase batch_size for faster GPU utilization
        embeddings = embedding_model.encode(
            request.texts, 
            batch_size=128,  # Larger batches for GPU
            show_progress_bar=False,
            convert_to_numpy=True
        )
        elapsed = time.time() - start
        logger.info(f"Batch embed: {len(request.texts)} texts in {elapsed:.2f}s ({len(request.texts)/elapsed:.0f}/s)")
        return {"embeddings": embeddings.tolist()}
    except Exception as e:
        logger.error(f"Batch embedding error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    finally:
        stop_llama_server()
