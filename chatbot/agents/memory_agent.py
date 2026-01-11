"""
Memory Agent - Handles short-term and long-term memory.

Integrates with the memory/ package for unified memory management.
"""

import logging
from typing import Dict, Any, Optional

from chatbot.agents.base import BaseAgent, AgentContext

logger = logging.getLogger(__name__)


class MemoryAgent(BaseAgent):
    """
    Agent for memory management.
    
    Wraps memory.core.MemoryManager to provide:
    - Short-term conversation context
    - Long-term semantic search
    - Feature memory (facts about user)
    """
    
    def __init__(
        self, 
        scope: str = "global",
        data_dir: str = "data/memory",
        llm_client = None
    ):
        """
        Initialize memory agent.
        
        Args:
            scope: Memory scope (global, user:id, session:id)
            data_dir: Directory for persistent storage
            llm_client: Optional LLM client for memory management
        """
        super().__init__(name="memory")
        
        # Import here to avoid circular dependencies
        from memory.core import get_memory_manager
        
        self.scope = scope
        self._memory = get_memory_manager(
            scope=scope,
            llm_client=llm_client,
            data_dir=data_dir
        )
        logger.info(f"MemoryAgent initialized with scope '{scope}'")
    
    def gather_context(
        self, 
        query: str, 
        session: Optional[Dict] = None
    ) -> Optional[AgentContext]:
        """
        Gather memory context for the query.
        
        Retrieves:
        - Short-term conversation history
        - Long-term relevant memories (semantic search)
        - Feature memory (user facts)
        
        Args:
            query: The user's query
            session: Optional session data
            
        Returns:
            AgentContext with combined memory context
        """
        if not self._enabled:
            return None
        
        try:
            # Get combined context from memory manager
            context = self._memory.get_context(
                query=query,
                max_tokens=2000,
                include_short_term=True,
                include_features=True,
                include_long_term=True,
                long_term_limit=3
            )
            
            # Build context string
            context_str = self._memory._build_combined_context(context)
            
            if not context_str.strip():
                return None
            
            return AgentContext(
                content=context_str,
                metadata={
                    "short_term_count": len(context.get("short_term", [])),
                    "long_term_count": len(context.get("long_term", [])),
                    "features_count": len(context.get("features", {}))
                },
                priority=10  # High priority - memory is important
            )
            
        except Exception as e:
            logger.error(f"MemoryAgent gather_context failed: {e}")
            return None
    
    def post_process(
        self, 
        user_input: str, 
        response: str,
        session: Optional[Dict] = None
    ) -> None:
        """
        Save the conversation turn to memory.
        
        Args:
            user_input: The user's message
            response: The assistant's response
            session: Optional session data
        """
        if not self._enabled:
            return
        
        try:
            # Add turn to all memory types
            self._memory.add_turn(
                user_message=user_input,
                assistant_response=response,
                extract_facts=True  # Auto-extract facts for feature memory
            )
            
            # Check if memory management is needed
            if self._memory.needs_management():
                self._memory.manage_short_term()
                
            logger.debug("MemoryAgent: Turn saved to memory")
            
        except Exception as e:
            logger.error(f"MemoryAgent post_process failed: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return self._memory.get_stats()
    
    def clear_session(self) -> None:
        """Clear short-term memory for new session."""
        self._memory.clear_session()
        logger.info("MemoryAgent: Session cleared")
    
    def set_fact(self, key: str, value: Any) -> None:
        """Manually set a fact in feature memory."""
        self._memory.set_fact(key, value, source="manual")
    
    def get_fact(self, key: str, default: Any = None) -> Any:
        """Get a fact from feature memory."""
        return self._memory.get_fact(key, default)
