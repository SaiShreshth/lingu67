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
        llm_client = None,
        use_long_term: bool = True
    ):
        """
        Initialize memory agent.
        
        Args:
            scope: Memory scope (global, user:id, session:id)
            data_dir: Directory for persistent storage
            llm_client: Optional LLM client for memory management
            use_long_term: Whether to use long-term memory (requires Qdrant)
        """
        super().__init__(name="memory")
        
        self.scope = scope
        self._memory = None
        self._use_long_term = use_long_term
        
        try:
            # Import here to avoid circular dependencies
            from memory.core import MemoryManager
            
            self._memory = MemoryManager(
                scope=scope,
                llm_client=llm_client,
                data_dir=data_dir,
                use_llm_management=False  # Disable LLM management for now
            )
            logger.info(f"MemoryAgent initialized with scope '{scope}'")
        except Exception as e:
            # If full memory fails, try without long-term (Qdrant)
            logger.warning(f"Full memory init failed: {e}, using fallback mode")
            self._init_fallback(scope, data_dir)
    
    def _init_fallback(self, scope: str, data_dir: str):
        """Initialize fallback mode without long-term memory."""
        try:
            from memory.stores.short_term import ShortTermMemory
            from memory.stores.feature_memory import FeatureMemory
            from memory.managers.policies import ShortTermPolicy
            import os
            
            self._short_term = ShortTermMemory(
                scope=scope,
                policy=ShortTermPolicy()
            )
            self._features = FeatureMemory(
                scope=scope,
                storage_dir=os.path.join(data_dir, "features")
            )
            self._fallback_mode = True
            self._use_long_term = False
            logger.info(f"MemoryAgent fallback mode: short-term + features only")
        except Exception as e:
            logger.error(f"MemoryAgent fallback also failed: {e}")
            self._enabled = False
    
    def gather_context(
        self, 
        query: str, 
        session: Optional[Dict] = None
    ) -> Optional[AgentContext]:
        """
        Gather memory context for the query.
        
        Retrieves:
        - Short-term conversation history
        - Long-term relevant memories (semantic search) if available
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
            # Check if using full memory manager or fallback mode
            if self._memory is not None:
                # Full mode with MemoryManager
                context = self._memory.get_context(
                    query=query,
                    max_tokens=2000,
                    include_short_term=True,
                    include_features=True,
                    include_long_term=self._use_long_term,
                    long_term_limit=3
                )
                context_str = self._memory._build_combined_context(context)
                
                return AgentContext(
                    content=context_str,
                    metadata={
                        "short_term_count": len(context.get("short_term", [])),
                        "long_term_count": len(context.get("long_term", [])),
                        "features_count": len(context.get("features", {}))
                    },
                    priority=10
                ) if context_str.strip() else None
            else:
                # Fallback mode - use direct access to short-term and features
                context_parts = []
                
                # Get short-term context
                if hasattr(self, '_short_term') and self._short_term:
                    history = self._short_term.get_context()
                    if history:
                        context_parts.append("=== RECENT CONVERSATION ===")
                        for entry in history[-10:]:  # Last 10 turns
                            role = entry.get("role", "unknown")
                            content = entry.get("content", "")
                            context_parts.append(f"{role.title()}: {content}")
                
                # Get feature memory (user facts)
                if hasattr(self, '_features') and self._features:
                    facts_json = self._features.to_prompt()
                    if facts_json and facts_json != "{}":
                        context_parts.append("\n=== USER FACTS ===")
                        context_parts.append(facts_json)
                
                if not context_parts:
                    return None
                
                return AgentContext(
                    content="\n".join(context_parts),
                    metadata={"fallback_mode": True},
                    priority=10
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
            if self._memory is not None:
                # Full mode with MemoryManager
                self._memory.add_turn(
                    user_message=user_input,
                    assistant_response=response,
                    extract_facts=True
                )
                
                if self._memory.needs_management():
                    self._memory.manage_short_term()
            else:
                # Fallback mode - use direct access
                if hasattr(self, '_short_term') and self._short_term:
                    self._short_term.add_turn(user_input, response)
                    logger.debug("MemoryAgent: Turn saved to short-term (fallback mode)")
                    
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
