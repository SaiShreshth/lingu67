"""
RAG Handler - Retrieval-Augmented Generation Module
Provides unified storage and retrieval for documents, conversations, and files.
"""

import os
import sys
import json
import uuid
import hashlib
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client import QdrantClient
from qdrant_client.http import models
from server.local_client import LocalLLMClient
from config import MODEL_SERVER_URL, QDRANT_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGHandler:
    """
    Unified RAG (Retrieval-Augmented Generation) handler.
    
    Provides:
    - Document/file ingestion with chunking
    - Semantic search and retrieval
    - Conversation storage
    - Filtered queries by source/type
    
    Usage:
        rag = RAGHandler(collection="my_knowledge")
        rag.ingest_file("document.txt", "My Document")
        results = rag.search("What is the main topic?", limit=5)
    """
    
    def __init__(
        self, 
        collection: str = "knowledge_base",
        qdrant_path: str = QDRANT_PATH,
        llm_url: str = MODEL_SERVER_URL,
        vector_size: int = 384
    ):
        """
        Initialize RAG handler.
        
        Args:
            collection: Name of the Qdrant collection
            qdrant_path: Path to Qdrant local storage
            llm_url: URL of the model server for embeddings
            vector_size: Dimension of embedding vectors (384 for MiniLM)
        """
        self.collection = collection
        self.vector_size = vector_size
        self.llm = LocalLLMClient(llm_url)
        self.qdrant = QdrantClient(path=qdrant_path)
        
        self._init_collection()
        self._metadata: Dict[str, Dict] = {}
        
        logger.info(f"RAG Handler initialized: collection='{collection}'")
    
    def _init_collection(self):
        """Create collection if it doesn't exist."""
        try:
            self.qdrant.get_collection(self.collection)
            logger.info(f"Collection '{self.collection}' exists")
        except:
            self.qdrant.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE
                )
            )
            logger.info(f"Created collection '{self.collection}'")
    
    # ==================== Embedding ====================
    
    def embed(self, text: str) -> List[float]:
        """Get embedding for a single text."""
        return self.llm.embed(text)
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts."""
        return self.llm.embed_batch(texts)
    
    # ==================== Storage ====================
    
    def store(
        self,
        text: str,
        metadata: Optional[Dict] = None,
        doc_type: str = "document",
        source: str = "unknown",
        point_id: Optional[str] = None
    ) -> str:
        """
        Store a single text chunk with embedding.
        
        Args:
            text: The text content to store
            metadata: Additional metadata to attach
            doc_type: Type of document (document, conversation, file, etc.)
            source: Source identifier (filename, conversation_id, etc.)
            point_id: Optional custom ID (auto-generated if None)
            
        Returns:
            The point ID of the stored document
        """
        if point_id is None:
            point_id = str(uuid.uuid4())
        
        vector = self.embed(text)
        
        payload = {
            "text": text,
            "type": doc_type,
            "source": source,
            "timestamp": datetime.now().isoformat(),
            **(metadata or {})
        }
        
        self.qdrant.upsert(
            collection_name=self.collection,
            points=[models.PointStruct(
                id=point_id,
                vector=vector,
                payload=payload
            )]
        )
        
        logger.debug(f"Stored: id={point_id}, type={doc_type}, source={source}")
        return point_id
    
    def store_conversation(
        self,
        user_message: str,
        assistant_response: str,
        conversation_id: Optional[str] = None
    ) -> str:
        """
        Store a conversation turn.
        
        Args:
            user_message: User's message
            assistant_response: Assistant's response
            conversation_id: Optional conversation thread ID
            
        Returns:
            The point ID
        """
        text = f"User: {user_message}\nAssistant: {assistant_response}"
        
        # Generate deterministic ID from content
        content_hash = hashlib.md5(text.encode()).hexdigest()
        point_id = str(uuid.UUID(content_hash))
        
        return self.store(
            text=text,
            doc_type="conversation",
            source=conversation_id or "default",
            point_id=point_id,
            metadata={"user_message": user_message, "assistant_response": assistant_response}
        )
    
    # ==================== File Ingestion ====================
    
    def ingest_file(
        self,
        file_path: str,
        source_name: str,
        chunk_size: int = 1024,
        chunk_overlap: int = 512,
        batch_size: int = 500,
        max_workers: int = 4,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Ingest a file into the RAG system with parallel embedding.
        
        Args:
            file_path: Path to the file
            source_name: Name to identify this file
            chunk_size: Size of each text chunk
            chunk_overlap: Overlap between chunks
            batch_size: Number of chunks per embedding batch
            max_workers: Number of parallel workers
            progress_callback: Optional callback(current, total) for progress
            
        Returns:
            Dict with ingestion stats
        """
        import time
        start_time = time.time()
        
        # Read file
        with open(file_path, "r", errors="ignore") as f:
            text = f.read()
        
        # Chunk text
        chunks = self._chunk_text(text, chunk_size, chunk_overlap)
        total_chunks = len(chunks)
        
        logger.info(f"Ingesting '{source_name}': {len(text):,} chars → {total_chunks:,} chunks")
        
        # Split into batches
        batches = [chunks[i:i+batch_size] for i in range(0, total_chunks, batch_size)]
        
        # Process batches in parallel
        all_points = []
        completed = 0
        
        def process_batch(batch_idx: int, batch_chunks: List[str]):
            try:
                vectors = self.embed_batch(batch_chunks)
                return batch_idx, batch_chunks, vectors, None
            except Exception as e:
                return batch_idx, batch_chunks, None, str(e)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_batch, i, batch): i 
                for i, batch in enumerate(batches)
            }
            
            results = [None] * len(batches)
            
            for future in as_completed(futures):
                batch_idx, batch_chunks, vectors, error = future.result()
                completed += 1
                
                if error:
                    logger.warning(f"Batch {batch_idx} failed: {error}, using sequential fallback")
                    vectors = [self.embed(c) for c in batch_chunks]
                
                results[batch_idx] = (batch_chunks, vectors)
                
                if progress_callback:
                    progress_callback(completed, len(batches))
        
        # Build points
        all_ids = []
        chunk_idx = 0
        
        for batch_chunks, vectors in results:
            for chunk, vector in zip(batch_chunks, vectors):
                point_id = str(uuid.uuid4())
                all_ids.append(point_id)
                
                all_points.append(models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "text": f"[{source_name} #{chunk_idx}]\n{chunk}",
                        "source": source_name,
                        "type": "file",
                        "chunk_index": chunk_idx,
                        "timestamp": datetime.now().isoformat()
                    }
                ))
                chunk_idx += 1
        
        # Batch upsert to Qdrant
        for i in range(0, len(all_points), 1000):
            self.qdrant.upsert(self.collection, all_points[i:i+1000])
        
        elapsed = time.time() - start_time
        
        # Store metadata
        self._metadata[source_name] = {
            "point_ids": all_ids,
            "chunk_count": total_chunks,
            "file_size": len(text),
            "ingested_at": datetime.now().isoformat()
        }
        
        stats = {
            "source": source_name,
            "chunks": total_chunks,
            "file_size": len(text),
            "elapsed_seconds": round(elapsed, 2),
            "chunks_per_second": round(total_chunks / elapsed, 1) if elapsed > 0 else 0
        }
        
        logger.info(f"Ingested '{source_name}': {stats}")
        return stats
    
    def ingest_text(
        self,
        text: str,
        source_name: str,
        chunk_size: int = 1024,
        chunk_overlap: int = 512
    ) -> Dict[str, Any]:
        """
        Ingest raw text into the RAG system.
        
        Args:
            text: The text content
            source_name: Name to identify this content
            chunk_size: Size of each text chunk
            chunk_overlap: Overlap between chunks
            
        Returns:
            Dict with ingestion stats
        """
        chunks = self._chunk_text(text, chunk_size, chunk_overlap)
        
        all_ids = []
        for i, chunk in enumerate(chunks):
            point_id = self.store(
                text=f"[{source_name} #{i}]\n{chunk}",
                doc_type="text",
                source=source_name,
                metadata={"chunk_index": i}
            )
            all_ids.append(point_id)
        
        self._metadata[source_name] = {
            "point_ids": all_ids,
            "chunk_count": len(chunks),
            "ingested_at": datetime.now().isoformat()
        }
        
        return {"source": source_name, "chunks": len(chunks)}
    
    def _chunk_text(self, text: str, size: int, overlap: int) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + size
            chunks.append(text[start:end])
            start = end - overlap
        return chunks
    
    # ==================== Retrieval ====================
    
    def search(
        self,
        query: str,
        limit: int = 5,
        source_filter: Optional[str] = None,
        type_filter: Optional[str] = None,
        min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            limit: Maximum results
            source_filter: Filter by source (exact match)
            type_filter: Filter by type (exact match)
            min_score: Minimum similarity score (0-1)
            
        Returns:
            List of results with text, source, score, and metadata
        """
        vector = self.embed(query)
        
        # Build filter
        filter_conditions = []
        if source_filter:
            filter_conditions.append(
                models.FieldCondition(
                    key="source",
                    match=models.MatchValue(value=source_filter)
                )
            )
        if type_filter:
            filter_conditions.append(
                models.FieldCondition(
                    key="type",
                    match=models.MatchValue(value=type_filter)
                )
            )
        
        query_filter = None
        if filter_conditions:
            query_filter = models.Filter(must=filter_conditions)
        
        try:
            results = self.qdrant.query_points(
                collection_name=self.collection,
                query=vector,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
                score_threshold=min_score if min_score > 0 else None
            )
            
            return [
                {
                    "text": p.payload.get("text", ""),
                    "source": p.payload.get("source", "unknown"),
                    "type": p.payload.get("type", "unknown"),
                    "score": p.score,
                    "metadata": {k: v for k, v in p.payload.items() if k not in ["text"]}
                }
                for p in results.points
                if p.payload
            ]
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    def search_by_source(
        self,
        source: str,
        query: str = "",
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search within a specific source/file.
        
        Args:
            source: Source name to filter by
            query: Optional query (if empty, returns by recency)
            limit: Maximum results
            
        Returns:
            List of results
        """
        return self.search(
            query=query or f"content from {source}",
            limit=limit,
            source_filter=source
        )
    
    def get_context(
        self,
        query: str,
        limit: int = 5,
        include_conversations: bool = True,
        include_files: bool = True
    ) -> str:
        """
        Get formatted context string for LLM augmentation.
        
        Args:
            query: The query to search for
            limit: Results per type
            include_conversations: Include conversation history
            include_files: Include file content
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        if include_files:
            file_results = self.search(query, limit=limit, type_filter="file")
            if file_results:
                context_parts.append("=== RETRIEVED DOCUMENTS ===")
                for r in file_results:
                    context_parts.append(r["text"])
        
        if include_conversations:
            conv_results = self.search(query, limit=limit, type_filter="conversation")
            if conv_results:
                context_parts.append("\n=== RELEVANT CONVERSATIONS ===")
                for r in conv_results:
                    context_parts.append(r["text"])
        
        return "\n\n".join(context_parts)
    
    # ==================== Management ====================
    
    def delete_source(self, source: str) -> int:
        """
        Delete all points from a specific source.
        
        Args:
            source: Source name to delete
            
        Returns:
            Number of points deleted
        """
        try:
            # Get count first
            results = self.qdrant.scroll(
                collection_name=self.collection,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source",
                            match=models.MatchValue(value=source)
                        )
                    ]
                ),
                limit=10000,
                with_payload=False
            )
            
            point_ids = [p.id for p in results[0]]
            
            if point_ids:
                self.qdrant.delete(
                    collection_name=self.collection,
                    points_selector=models.PointIdsList(points=point_ids)
                )
            
            # Remove from metadata
            if source in self._metadata:
                del self._metadata[source]
            
            logger.info(f"Deleted {len(point_ids)} points from source '{source}'")
            return len(point_ids)
            
        except Exception as e:
            logger.error(f"Delete error: {e}")
            return 0
    
    def list_sources(self) -> List[Dict[str, Any]]:
        """
        List all unique sources in the collection.
        
        Returns:
            List of source info dicts
        """
        sources = {}
        offset = None
        
        try:
            while True:
                results, offset = self.qdrant.scroll(
                    collection_name=self.collection,
                    limit=1000,
                    offset=offset,
                    with_payload=["source", "type"]
                )
                
                for point in results:
                    if point.payload:
                        source = point.payload.get("source", "unknown")
                        doc_type = point.payload.get("type", "unknown")
                        
                        if source not in sources:
                            sources[source] = {"source": source, "types": set(), "count": 0}
                        
                        sources[source]["types"].add(doc_type)
                        sources[source]["count"] += 1
                
                if offset is None:
                    break
            
            # Convert sets to lists
            return [
                {**s, "types": list(s["types"])}
                for s in sources.values()
            ]
            
        except Exception as e:
            logger.error(f"List sources error: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        try:
            info = self.qdrant.get_collection(self.collection)
            return {
                "collection": self.collection,
                "points_count": info.points_count,
                "vectors_count": info.vectors_count,
                "status": info.status.value
            }
        except Exception as e:
            return {"error": str(e)}


# ==================== Factory ====================

_rag_instances: Dict[str, RAGHandler] = {}

def get_rag_handler(collection: str = "knowledge_base") -> RAGHandler:
    """
    Get or create a RAG handler for a collection.
    
    Args:
        collection: Collection name
        
    Returns:
        RAGHandler instance
    """
    if collection not in _rag_instances:
        _rag_instances[collection] = RAGHandler(collection=collection)
    return _rag_instances[collection]
