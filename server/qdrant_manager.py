"""
Qdrant Client Manager - Singleton shared Qdrant client with thread-safe access.

Provides a single shared Qdrant client instance to avoid "already accessed by 
another instance" errors when multiple components use the same Qdrant path.
"""

import os
import threading
import logging
import atexit
from typing import Optional, Dict
from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)


class QdrantClientManager:
    """
    Singleton manager for shared Qdrant client access.
    
    Ensures only one QdrantClient instance exists per path, preventing
    conflicts when multiple components (model_server, chatbot, memory)
    try to access the same Qdrant database.
    
    Usage:
        # Get shared client
        client = QdrantClientManager.get_client("/path/to/qdrant")
        
        # Use client (thread-safe)
        with QdrantClientManager.get_lock():
            client.upsert(...)
        
        # Cleanup on shutdown (automatic via atexit)
        QdrantClientManager.close_all()
    """
    
    _clients: Dict[str, QdrantClient] = {}
    _locks: Dict[str, threading.RLock] = {}
    _global_lock = threading.Lock()
    _initialized = False
    
    @classmethod
    def _normalize_path(cls, path: str) -> str:
        """Normalize path for consistent key lookup."""
        return os.path.normpath(os.path.abspath(path))
    
    @classmethod
    def get_client(cls, qdrant_path: str) -> QdrantClient:
        """
        Get or create a shared Qdrant client for the given path.
        
        Args:
            qdrant_path: Path to Qdrant storage directory
            
        Returns:
            Shared QdrantClient instance
        """
        normalized_path = cls._normalize_path(qdrant_path)
        
        with cls._global_lock:
            if normalized_path not in cls._clients:
                logger.info(f"Creating shared Qdrant client for: {normalized_path}")
                os.makedirs(normalized_path, exist_ok=True)
                cls._clients[normalized_path] = QdrantClient(path=normalized_path)
                cls._locks[normalized_path] = threading.RLock()
                
                # Register cleanup on first use
                if not cls._initialized:
                    atexit.register(cls.close_all)
                    cls._initialized = True
                    
            return cls._clients[normalized_path]
    
    @classmethod
    def get_lock(cls, qdrant_path: Optional[str] = None) -> threading.RLock:
        """
        Get the lock for operations on a specific Qdrant path.
        
        Args:
            qdrant_path: Path to Qdrant storage (or None for global lock)
            
        Returns:
            RLock for thread-safe operations
        """
        if qdrant_path is None:
            return cls._global_lock
        
        normalized_path = cls._normalize_path(qdrant_path)
        
        with cls._global_lock:
            if normalized_path not in cls._locks:
                cls._locks[normalized_path] = threading.RLock()
            return cls._locks[normalized_path]
    
    @classmethod
    def close(cls, qdrant_path: str) -> None:
        """
        Close a specific Qdrant client.
        
        Args:
            qdrant_path: Path to Qdrant storage
        """
        normalized_path = cls._normalize_path(qdrant_path)
        
        with cls._global_lock:
            if normalized_path in cls._clients:
                try:
                    cls._clients[normalized_path].close()
                    logger.info(f"Closed Qdrant client for: {normalized_path}")
                except Exception as e:
                    logger.error(f"Error closing Qdrant client: {e}")
                finally:
                    del cls._clients[normalized_path]
                    if normalized_path in cls._locks:
                        del cls._locks[normalized_path]
    
    @classmethod
    def close_all(cls) -> None:
        """Close all Qdrant clients (called on shutdown)."""
        with cls._global_lock:
            for path, client in list(cls._clients.items()):
                try:
                    client.close()
                    logger.info(f"Closed Qdrant client for: {path}")
                except Exception as e:
                    logger.debug(f"Error closing Qdrant client on shutdown: {e}")
            cls._clients.clear()
            cls._locks.clear()
        logger.info("All Qdrant clients closed")
    
    @classmethod
    def is_connected(cls, qdrant_path: str) -> bool:
        """Check if a client exists for the given path."""
        normalized_path = cls._normalize_path(qdrant_path)
        return normalized_path in cls._clients


# Convenience functions for common operations
def get_shared_qdrant(qdrant_path: str) -> QdrantClient:
    """Get shared Qdrant client (convenience function)."""
    return QdrantClientManager.get_client(qdrant_path)


def get_qdrant_lock(qdrant_path: str) -> threading.RLock:
    """Get lock for Qdrant operations (convenience function)."""
    return QdrantClientManager.get_lock(qdrant_path)
