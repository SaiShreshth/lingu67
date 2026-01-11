"""
Long-Term Memory - Persistent storage for all conversations and file summaries.

No decay or compression - stores everything permanently.
Searchable via semantic retrieval.
"""

import os
import sys
import logging
import uuid
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.utils.scopes import MemoryScope, parse_scope
from memory.utils.helpers import count_tokens, generate_id, sanitize_for_storage

logger = logging.getLogger(__name__)


class LongTermMemory:
    """
    Persistent long-term memory backed by vector database.
    
    Features:
    - Stores all conversations permanently
    - Stores file summaries
    - No decay or compression (archival)
    - Semantic search for retrieval
    
    Example:
        ltm = LongTermMemory(scope="user:123", qdrant_path="data/memory")
        ltm.store_conversation("Hello", "Hi there!")
        
        results = ltm.search("greeting", limit=5)
    """
    
    def __init__(
        self,
        scope: str = "global",
        qdrant_path: str = "data/long_term_memory",
        llm_client = None,
        collection_prefix: str = "long_term"
    ):
        self.scope = parse_scope(scope)
        self.qdrant_path = qdrant_path
        self.llm_client = llm_client
        self.collection_name = f"{collection_prefix}_{self.scope.get_storage_key()}"
        
        self._init_storage()
    
    def _init_storage(self):
        """Initialize Qdrant storage."""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models
            
            self.qdrant = QdrantClient(path=self.qdrant_path)
            
            try:
                self.qdrant.get_collection(self.collection_name)
                logger.debug(f"Collection '{self.collection_name}' exists")
            except:
                self.qdrant.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=384,  # MiniLM dimension
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(f"Created collection '{self.collection_name}'")
                
        except ImportError:
            logger.warning("Qdrant not available, long-term memory disabled")
            self.qdrant = None
    
    def _embed(self, text: str) -> List[float]:
        """Get embedding for text."""
        if self.llm_client:
            return self.llm_client.embed(text)
        
        # Fallback: try to get from model server
        try:
            from server.local_client import LocalLLMClient
            from config import MODEL_SERVER_URL
            client = LocalLLMClient(MODEL_SERVER_URL)
            return client.embed(text)
        except:
            logger.error("No embedding client available")
            return [0.0] * 384
    
    def store(
        self,
        content: str,
        memory_type: str = "conversation",
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Store content in long-term memory.
        
        Args:
            content: Text content to store
            memory_type: Type ("conversation", "file_summary", "note")
            metadata: Additional metadata
            
        Returns:
            Point ID
        """
        if not self.qdrant:
            logger.warning("Qdrant not available")
            return ""
        
        from qdrant_client.http import models
        
        content = sanitize_for_storage(content)
        point_id = generate_id()
        
        payload = {
            "text": content,
            "type": memory_type,
            "scope": str(self.scope),
            "timestamp": datetime.now().isoformat(),
            "tokens": count_tokens(content),
            **(metadata or {})
        }
        
        try:
            vector = self._embed(content)
            
            self.qdrant.upsert(
                collection_name=self.collection_name,
                points=[models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload
                )]
            )
            
            logger.debug(f"Stored in long-term: {point_id}")
            return point_id
            
        except Exception as e:
            logger.error(f"Store error: {e}")
            return ""
    
    def store_conversation(
        self,
        user_message: str,
        assistant_response: str,
        context: Optional[str] = None
    ) -> str:
        """
        Store a conversation turn.
        
        Args:
            user_message: User's message
            assistant_response: Assistant's response
            context: Optional context/summary
            
        Returns:
            Point ID
        """
        content = f"User: {user_message}\nAssistant: {assistant_response}"
        
        # Generate deterministic ID to avoid duplicates
        content_hash = hashlib.md5(content.encode()).hexdigest()
        
        return self.store(
            content=content,
            memory_type="conversation",
            metadata={
                "user_message": user_message[:500],  # Truncate for metadata
                "assistant_response": assistant_response[:500],
                "context": context,
                "content_hash": content_hash
            }
        )
    
    def store_file_summary(
        self,
        filename: str,
        summary: str,
        file_type: Optional[str] = None
    ) -> str:
        """
        Store a file summary.
        
        Args:
            filename: Name of the file
            summary: Summary of file contents
            file_type: Type of file (txt, pdf, etc.)
            
        Returns:
            Point ID
        """
        content = f"File: {filename}\n\nSummary:\n{summary}"
        
        return self.store(
            content=content,
            memory_type="file_summary",
            metadata={
                "filename": filename,
                "file_type": file_type
            }
        )
    
    def search(
        self,
        query: str,
        limit: int = 5,
        memory_type: Optional[str] = None,
        min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search long-term memory.
        
        Args:
            query: Search query
            limit: Max results
            memory_type: Filter by type
            min_score: Minimum similarity score
            
        Returns:
            List of results with text, score, metadata
        """
        if not self.qdrant:
            return []
        
        from qdrant_client.http import models
        
        try:
            vector = self._embed(query)
            
            # Build filter
            filter_conditions = []
            if memory_type:
                filter_conditions.append(
                    models.FieldCondition(
                        key="type",
                        match=models.MatchValue(value=memory_type)
                    )
                )
            
            query_filter = None
            if filter_conditions:
                query_filter = models.Filter(must=filter_conditions)
            
            results = self.qdrant.query_points(
                collection_name=self.collection_name,
                query=vector,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
                score_threshold=min_score if min_score > 0 else None
            )
            
            return [
                {
                    "text": p.payload.get("text", ""),
                    "type": p.payload.get("type", "unknown"),
                    "score": p.score,
                    "timestamp": p.payload.get("timestamp"),
                    "metadata": {k: v for k, v in p.payload.items() 
                               if k not in ["text", "type", "timestamp"]}
                }
                for p in results.points
                if p.payload
            ]
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    def get_relevant_context(
        self,
        query: str,
        max_tokens: int = 1000,
        limit: int = 5
    ) -> str:
        """
        Get relevant context for a query.
        
        Args:
            query: Current query
            max_tokens: Max tokens to return
            limit: Max memories to include
            
        Returns:
            Formatted context string
        """
        results = self.search(query, limit=limit)
        
        if not results:
            return ""
        
        context_parts = []
        token_count = 0
        
        for r in results:
            text = r["text"]
            text_tokens = count_tokens(text)
            
            if token_count + text_tokens > max_tokens:
                break
            
            context_parts.append(text)
            token_count += text_tokens
        
        return "\n\n---\n\n".join(context_parts)
    
    def delete_by_id(self, point_id: str) -> bool:
        """Delete a specific memory by ID."""
        if not self.qdrant:
            return False
        
        try:
            from qdrant_client.http import models
            
            self.qdrant.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=[point_id])
            )
            return True
        except Exception as e:
            logger.error(f"Delete error: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """Get memory statistics."""
        if not self.qdrant:
            return {"error": "Qdrant not available"}
        
        try:
            info = self.qdrant.get_collection(self.collection_name)
            return {
                "collection": self.collection_name,
                "points_count": info.points_count,
                "scope": str(self.scope)
            }
        except Exception as e:
            return {"error": str(e)}
    
    def clear(self):
        """Clear all memories (use with caution!)."""
        if not self.qdrant:
            return
        
        try:
            from qdrant_client.http import models
            
            self.qdrant.delete_collection(self.collection_name)
            self._init_storage()  # Recreate empty collection
            logger.info(f"Cleared long-term memory: {self.collection_name}")
        except Exception as e:
            logger.error(f"Clear error: {e}")
