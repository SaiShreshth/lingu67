"""
File Agent - Handles file ingestion and retrieval.

Uses RAGHandler for file storage and search.
"""

import os
import logging
from typing import Dict, Any, Optional, List

from chatbot.agents.base import BaseAgent, AgentContext

logger = logging.getLogger(__name__)


class FileAgent(BaseAgent):
    """
    Agent for file management.
    
    Provides:
    - File ingestion with chunking
    - Semantic search within files
    - File listing and deletion
    """
    
    def __init__(
        self,
        collection: str = "knowledge_base",
        qdrant_path: Optional[str] = None,
        llm_url: Optional[str] = None
    ):
        """
        Initialize file agent.
        
        Args:
            collection: Qdrant collection name
            qdrant_path: Path to Qdrant storage
            llm_url: URL of the model server
        """
        super().__init__(name="file")
        
        # Import here to use config defaults
        from config import QDRANT_PATH, MODEL_SERVER_URL
        from server.rag_handler import RAGHandler
        
        self._rag = RAGHandler(
            collection=collection,
            qdrant_path=qdrant_path or QDRANT_PATH,
            llm_url=llm_url or MODEL_SERVER_URL
        )
        
        # Track ingested files
        self._files: Dict[str, Dict] = {}
        self._load_file_metadata()
        
        logger.info(f"FileAgent initialized with collection '{collection}'")
    
    def _load_file_metadata(self):
        """Load file metadata from disk."""
        from config import FILE_METADATA_PATH
        if os.path.exists(FILE_METADATA_PATH):
            try:
                import json
                with open(FILE_METADATA_PATH, 'r') as f:
                    self._files = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load file metadata: {e}")
    
    def _save_file_metadata(self):
        """Save file metadata to disk."""
        from config import FILE_METADATA_PATH
        try:
            import json
            with open(FILE_METADATA_PATH, 'w') as f:
                json.dump(self._files, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save file metadata: {e}")
    
    def gather_context(
        self, 
        query: str, 
        session: Optional[Dict] = None
    ) -> Optional[AgentContext]:
        """
        Search files for relevant content.
        
        Args:
            query: The user's query
            session: Optional session data with file hints
            
        Returns:
            AgentContext with relevant file content
        """
        if not self._enabled or not self._files:
            return None
        
        try:
            # Check if query mentions a specific file
            mentioned_file = self._detect_file_mention(query)
            
            if mentioned_file:
                # Search within specific file
                results = self._rag.search_by_source(
                    source=mentioned_file,
                    query=query,
                    limit=5
                )
            else:
                # General search across all files
                results = self._rag.search(
                    query=query,
                    limit=3,
                    type_filter="file"
                )
            
            if not results:
                return None
            
            # Format results
            context_parts = []
            for r in results:
                source = r.get("source", "unknown")
                content = r.get("content", "")
                context_parts.append(f"[From {source}]:\n{content}")
            
            context_str = "\n\n".join(context_parts)
            
            return AgentContext(
                content=f"Relevant File Content:\n{context_str}",
                metadata={
                    "num_results": len(results),
                    "mentioned_file": mentioned_file
                },
                priority=8  # High priority when files are relevant
            )
            
        except Exception as e:
            logger.error(f"FileAgent gather_context failed: {e}")
            return None
    
    def _detect_file_mention(self, query: str) -> Optional[str]:
        """Detect if user mentions any uploaded filename."""
        query_lower = query.lower()
        for filename in self._files.keys():
            if filename.lower() in query_lower:
                return filename
        return None
    
    def post_process(
        self, 
        user_input: str, 
        response: str,
        session: Optional[Dict] = None
    ) -> None:
        """
        No post-processing needed for file agent.
        File operations are explicit (ingest, delete).
        """
        pass
    
    def ingest_file(
        self, 
        file_path: str, 
        source_name: Optional[str] = None,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Ingest a file into the knowledge base.
        
        Args:
            file_path: Path to the file
            source_name: Name to identify this file (defaults to filename)
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dict with ingestion stats
        """
        if not source_name:
            source_name = os.path.basename(file_path)
        
        result = self._rag.ingest_file(
            file_path=file_path,
            source_name=source_name,
            progress_callback=progress_callback
        )
        
        # Update metadata
        self._files[source_name] = {
            "path": file_path,
            "chunks": result.get("chunks", 0),
            "size": os.path.getsize(file_path) if os.path.exists(file_path) else 0
        }
        self._save_file_metadata()
        
        logger.info(f"FileAgent: Ingested '{source_name}' ({result.get('chunks', 0)} chunks)")
        return result
    
    def delete_file(self, source_name: str) -> bool:
        """
        Delete a file from the knowledge base.
        
        Args:
            source_name: Name of the file to delete
            
        Returns:
            True if successful
        """
        try:
            deleted = self._rag.delete_source(source_name)
            if source_name in self._files:
                del self._files[source_name]
                self._save_file_metadata()
            logger.info(f"FileAgent: Deleted '{source_name}' ({deleted} chunks)")
            return True
        except Exception as e:
            logger.error(f"FileAgent delete failed: {e}")
            return False
    
    def list_files(self) -> List[Dict[str, Any]]:
        """Get list of ingested files."""
        return [
            {"name": name, **info}
            for name, info in self._files.items()
        ]
    
    def search(
        self, 
        query: str, 
        source: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict]:
        """
        Search files for relevant content.
        
        Args:
            query: Search query
            source: Optional specific file to search
            limit: Maximum results
            
        Returns:
            List of search results
        """
        if source:
            return self._rag.search_by_source(source, query, limit)
        return self._rag.search(query, limit, type_filter="file")
