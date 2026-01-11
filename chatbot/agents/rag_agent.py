"""
RAG Agent - Handles retrieval-augmented generation.

Provides semantic search across all knowledge sources.
"""

import logging
from typing import Dict, Any, Optional, List

from chatbot.agents.base import BaseAgent, AgentContext

logger = logging.getLogger(__name__)


class RAGAgent(BaseAgent):
    """
    Agent for RAG (Retrieval-Augmented Generation).
    
    Provides:
    - Semantic search across documents
    - Conversation history search
    - Combined context retrieval
    """
    
    def __init__(
        self,
        collection: str = "knowledge_base",
        qdrant_path: Optional[str] = None,
        llm_url: Optional[str] = None
    ):
        """
        Initialize RAG agent.
        
        Args:
            collection: Qdrant collection name
            qdrant_path: Path to Qdrant storage
            llm_url: URL of the model server
        """
        super().__init__(name="rag")
        
        from config import QDRANT_PATH, MODEL_SERVER_URL
        from server.rag_handler import RAGHandler
        
        self._rag = RAGHandler(
            collection=collection,
            qdrant_path=qdrant_path or QDRANT_PATH,
            llm_url=llm_url or MODEL_SERVER_URL
        )
        
        logger.info(f"RAGAgent initialized with collection '{collection}'")
    
    def gather_context(
        self, 
        query: str, 
        session: Optional[Dict] = None
    ) -> Optional[AgentContext]:
        """
        Search knowledge base for relevant context.
        
        Args:
            query: The user's query
            session: Optional session data
            
        Returns:
            AgentContext with relevant documents
        """
        if not self._enabled:
            return None
        
        try:
            # Get combined context from RAG handler
            context = self._rag.get_context(
                query=query,
                limit=3,
                include_conversations=True,
                include_files=True
            )
            
            if not context or not context.strip():
                return None
            
            return AgentContext(
                content=context,
                metadata={"source": "rag_search"},
                priority=5  # Medium priority
            )
            
        except Exception as e:
            logger.error(f"RAGAgent gather_context failed: {e}")
            return None
    
    def post_process(
        self, 
        user_input: str, 
        response: str,
        session: Optional[Dict] = None
    ) -> None:
        """
        Store conversation in RAG for future retrieval.
        
        Args:
            user_input: The user's message
            response: The assistant's response
            session: Optional session data
        """
        if not self._enabled:
            return
        
        try:
            # Store the conversation turn
            conversation_id = session.get("conversation_id") if session else None
            self._rag.store_conversation(
                user_message=user_input,
                assistant_response=response,
                conversation_id=conversation_id
            )
            logger.debug("RAGAgent: Conversation stored")
        except Exception as e:
            logger.error(f"RAGAgent post_process failed: {e}")
    
    def search(
        self, 
        query: str, 
        limit: int = 5,
        source_filter: Optional[str] = None,
        type_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Search the knowledge base.
        
        Args:
            query: Search query
            limit: Maximum results
            source_filter: Filter by source
            type_filter: Filter by type (document, conversation, file)
            
        Returns:
            List of search results
        """
        return self._rag.search(
            query=query,
            limit=limit,
            source_filter=source_filter,
            type_filter=type_filter
        )
    
    def store(
        self, 
        text: str, 
        metadata: Optional[Dict] = None,
        doc_type: str = "document",
        source: str = "manual"
    ) -> str:
        """
        Store a document in the knowledge base.
        
        Args:
            text: Document content
            metadata: Optional metadata
            doc_type: Type of document
            source: Source identifier
            
        Returns:
            Point ID
        """
        return self._rag.store(
            text=text,
            metadata=metadata,
            doc_type=doc_type,
            source=source
        )
