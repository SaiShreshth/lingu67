"""
Embedding Adapter - Unified interface for text embeddings.

Wraps the local model server's embedding endpoint.
"""

import logging
from typing import List, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseEmbeddingAdapter(ABC):
    """Abstract base class for embedding adapters."""
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        pass
    
    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Get embedding for a single text."""
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts."""
        pass


class LocalEmbeddingAdapter(BaseEmbeddingAdapter):
    """
    Adapter for local embedding server.
    
    Uses the model server's /embed and /embed_batch endpoints.
    Default model: all-MiniLM-L6-v2 (384 dimensions)
    """
    
    def __init__(
        self, 
        base_url: str = "http://localhost:8000",
        vector_size: int = 384
    ):
        """
        Initialize the embedding adapter.
        
        Args:
            base_url: URL of the model server
            vector_size: Dimension of embedding vectors
        """
        from server.local_client import LocalLLMClient
        
        self.base_url = base_url
        self._vector_size = vector_size
        self._client = LocalLLMClient(base_url)
        logger.info(f"EmbeddingAdapter initialized (dim={vector_size})")
    
    @property
    def dimension(self) -> int:
        """Return the embedding dimension (384 for MiniLM)."""
        return self._vector_size
    
    def embed(self, text: str) -> List[float]:
        """
        Get embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding
        """
        try:
            return self._client.embed(text)
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            # Return zero vector as fallback
            return [0.0] * self._vector_size
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for multiple texts in a single request.
        
        Much faster than calling embed() repeatedly.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings
        """
        if not texts:
            return []
        
        try:
            return self._client.embed_batch(texts)
        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            # Return zero vectors as fallback
            return [[0.0] * self._vector_size for _ in texts]


# Factory function
def get_embedding_adapter(
    base_url: Optional[str] = None,
    vector_size: int = 384
) -> LocalEmbeddingAdapter:
    """
    Factory function to get the default embedding adapter.
    
    Args:
        base_url: Optional override for server URL
        vector_size: Embedding dimension
        
    Returns:
        Configured EmbeddingAdapter instance
    """
    if base_url is None:
        from config import MODEL_SERVER_URL
        base_url = MODEL_SERVER_URL
    
    return LocalEmbeddingAdapter(base_url, vector_size)


# Export alias
EmbeddingAdapter = LocalEmbeddingAdapter
