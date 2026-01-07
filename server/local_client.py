"""
Simple client to interact with local LLM server
"""
import httpx
from typing import List, Dict

class LocalLLMClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.Client(timeout=120.0)
    
    def chat(self, messages: List[Dict[str, str]], max_tokens: int = 8000, temperature: float = 0.7, stream: bool = False):
        """Send chat completion request"""
        # Calculate approximate prompt size (with Qwen formatting overhead)
        total_chars = sum(len(m.get('content', '')) for m in messages)
        total_chars += len(messages) * 50  # Overhead for <|im_start|> tags
        estimated_tokens = total_chars // 4
        
        # Safety check: if estimated tokens + max_tokens > 4096, we need to reduce
        if estimated_tokens + max_tokens > 3800:  # Leave some margin
            # Truncate system messages to fit
            for msg in messages:
                if msg.get('role') == 'system':
                    max_len = 4000  # ~1000 tokens
                    if len(msg['content']) > max_len:
                        msg['content'] = msg['content'][:max_len] + "...[truncated]"
        
        if stream:
            with self.client.stream("POST", f"{self.base_url}/chat", json={
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True
            }) as response:
                response.raise_for_status()
                # Server now sends plain text chunks
                for chunk in response.iter_text():
                    if chunk:
                        yield chunk
        else:
            response = self.client.post(
                f"{self.base_url}/chat",
                json={
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": False
                }
            )
            response.raise_for_status()
            return response.json()['response']
    
    def complete(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        """Send text completion request"""
        response = self.client.post(
            f"{self.base_url}/completion",
            json={
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
        )
        response.raise_for_status()
        return response.json()['response']
    
    def embed(self, text: str) -> List[float]:
        """Get embedding for text"""
        response = self.client.post(
            f"{self.base_url}/embed",
            json={"text": text}
        )
        response.raise_for_status()
        return response.json()['embedding']
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts in a single request (much faster for large files)"""
        response = self.client.post(
            f"{self.base_url}/embed_batch",
            json={"texts": texts},
            timeout=600.0  # Longer timeout for batch processing
        )
        response.raise_for_status()
        return response.json()['embeddings']
    
    def health_check(self) -> Dict:
        """Check server health"""
        response = self.client.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
