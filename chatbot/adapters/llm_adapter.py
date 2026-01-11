"""
LLM Adapter - Unified interface for LLM interactions.

Wraps the local LLM server client with consistent API.
"""

import logging
from typing import List, Dict, Any, Optional, Generator
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """Represents a chat message."""
    role: str  # "system", "user", or "assistant"
    content: str
    
    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class LLMResponse:
    """Represents an LLM response."""
    content: str
    usage: Optional[Dict[str, int]] = None  # token counts if available
    
    
class BaseLLMAdapter(ABC):
    """Abstract base class for LLM adapters."""
    
    @abstractmethod
    def chat(
        self, 
        messages: List[ChatMessage], 
        max_tokens: int = 512,
        temperature: float = 0.7,
        stream: bool = False
    ) -> LLMResponse | Generator[str, None, None]:
        """Send chat completion request."""
        pass
    
    @abstractmethod
    def complete(
        self, 
        prompt: str, 
        max_tokens: int = 512,
        temperature: float = 0.7
    ) -> str:
        """Send text completion request."""
        pass


class LocalLLMAdapter(BaseLLMAdapter):
    """
    Adapter for local LLM server.
    
    Wraps server/local_client.py's LocalLLMClient with a clean interface.
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the LLM adapter.
        
        Args:
            base_url: URL of the local LLM server
        """
        # Import here to avoid circular dependencies
        from server.local_client import LocalLLMClient
        
        self.base_url = base_url
        self._client = LocalLLMClient(base_url)
        logger.info(f"LLMAdapter initialized with server at {base_url}")
    
    def chat(
        self, 
        messages: List[ChatMessage], 
        max_tokens: int = 512,
        temperature: float = 0.7,
        stream: bool = False
    ) -> LLMResponse | Generator[str, None, None]:
        """
        Send chat completion request.
        
        Args:
            messages: List of ChatMessage objects
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            stream: If True, yield tokens as they're generated
            
        Returns:
            LLMResponse or generator of token strings
        """
        msg_dicts = [m.to_dict() for m in messages]
        
        if stream:
            return self._client.chat(
                messages=msg_dicts,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True
            )
        else:
            response = self._client.chat(
                messages=msg_dicts,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False
            )
            return LLMResponse(content=response)
    
    def complete(
        self, 
        prompt: str, 
        max_tokens: int = 512,
        temperature: float = 0.7
    ) -> str:
        """
        Send text completion request.
        
        Args:
            prompt: The prompt to complete
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            
        Returns:
            The completion text
        """
        return self._client.complete(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
    
    def health_check(self) -> bool:
        """Check if the LLM server is healthy."""
        try:
            result = self._client.health_check()
            return result.get("status") == "healthy"
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False


# Default adapter using config
def get_llm_adapter(base_url: Optional[str] = None) -> LocalLLMAdapter:
    """
    Factory function to get the default LLM adapter.
    
    Args:
        base_url: Optional override for server URL
        
    Returns:
        Configured LLMAdapter instance
    """
    if base_url is None:
        from config import MODEL_SERVER_URL
        base_url = MODEL_SERVER_URL
    
    return LocalLLMAdapter(base_url)


# Export alias
LLMAdapter = LocalLLMAdapter
