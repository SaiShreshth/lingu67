"""
Qdrant Backend - Vector storage backend for long-term memory.
"""

import os
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class QdrantBackend:
    """
    Qdrant vector database backend.
    
    Wraps Qdrant operations for memory storage.
    """
    
    def __init__(
        self,
        path: str = "data/qdrant_memory",
        collection_name: str = "memory",
        vector_size: int = 384
    ):
        self.path = path
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.client = None
        
        self._init_client()
    
    def _init_client(self):
        """Initialize Qdrant client."""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models
            
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            self.client = QdrantClient(path=self.path)
            
            try:
                self.client.get_collection(self.collection_name)
            except:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.vector_size,
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {self.collection_name}")
                
        except ImportError:
            logger.error("qdrant-client not installed")
            self.client = None
    
    def upsert(
        self,
        point_id: str,
        vector: List[float],
        payload: Dict[str, Any]
    ) -> bool:
        """Upsert a single point."""
        if not self.client:
            return False
        
        try:
            from qdrant_client.http import models
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=[models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload
                )]
            )
            return True
        except Exception as e:
            logger.error(f"Upsert error: {e}")
            return False
    
    def upsert_batch(
        self,
        points: List[Dict]
    ) -> bool:
        """
        Upsert multiple points.
        
        Args:
            points: List of {"id": ..., "vector": [...], "payload": {...}}
        """
        if not self.client:
            return False
        
        try:
            from qdrant_client.http import models
            
            qdrant_points = [
                models.PointStruct(
                    id=p["id"],
                    vector=p["vector"],
                    payload=p.get("payload", {})
                )
                for p in points
            ]
            
            # Batch in chunks of 1000
            for i in range(0, len(qdrant_points), 1000):
                batch = qdrant_points[i:i+1000]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
            
            return True
        except Exception as e:
            logger.error(f"Batch upsert error: {e}")
            return False
    
    def search(
        self,
        vector: List[float],
        limit: int = 5,
        filter_conditions: Optional[Dict] = None,
        score_threshold: Optional[float] = None
    ) -> List[Dict]:
        """
        Search for similar vectors.
        
        Args:
            vector: Query vector
            limit: Max results
            filter_conditions: Optional filter
            score_threshold: Minimum score
            
        Returns:
            List of results
        """
        if not self.client:
            return []
        
        try:
            from qdrant_client.http import models
            
            query_filter = None
            if filter_conditions:
                must_conditions = []
                for key, value in filter_conditions.items():
                    must_conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value)
                        )
                    )
                query_filter = models.Filter(must=must_conditions)
            
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=vector,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
                score_threshold=score_threshold
            )
            
            return [
                {
                    "id": p.id,
                    "score": p.score,
                    "payload": p.payload
                }
                for p in results.points
            ]
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    def delete(self, point_ids: List[str]) -> bool:
        """Delete points by ID."""
        if not self.client or not point_ids:
            return False
        
        try:
            from qdrant_client.http import models
            
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=point_ids)
            )
            return True
        except Exception as e:
            logger.error(f"Delete error: {e}")
            return False
    
    def count(self) -> int:
        """Get total point count."""
        if not self.client:
            return 0
        
        try:
            info = self.client.get_collection(self.collection_name)
            return info.points_count
        except:
            return 0
    
    def clear(self) -> bool:
        """Clear all points (recreate collection)."""
        if not self.client:
            return False
        
        try:
            from qdrant_client.http import models
            
            self.client.delete_collection(self.collection_name)
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE
                )
            )
            return True
        except Exception as e:
            logger.error(f"Clear error: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if backend is available."""
        return self.client is not None
