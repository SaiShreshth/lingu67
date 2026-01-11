"""
Memory Core - Main MemoryManager orchestrating all memory types.

Combines short-term, long-term, and feature memory into a unified interface.
"""

import os
import sys
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

# Add parent for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.utils.scopes import MemoryScope, parse_scope
from memory.managers.policies import ShortTermPolicy
from memory.stores.short_term import ShortTermMemory
from memory.stores.long_term import LongTermMemory
from memory.stores.feature_memory import FeatureMemory
from memory.managers.llm_manager import LLMMemoryManager, SimpleMemoryManager
from memory.utils.helpers import count_tokens, TokenBudget

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Unified memory manager orchestrating all memory types.
    
    Memory Types:
    - short_term: Volatile conversation context (fed to LLM)
    - long_term: Persistent storage (all conversations)
    - features: JSON facts with history (fed to LLM)
    
    Example:
        mm = MemoryManager(scope="user:123", llm_client=llm)
        
        # Get context for LLM
        context = mm.get_context("What's my name?")
        
        # After LLM response
        mm.add_turn("What's my name?", "Your name is John")
        
        # Manage memory periodically
        if mm.needs_management():
            mm.manage_short_term()
    """
    
    def __init__(
        self,
        scope: str = "global",
        llm_client = None,
        data_dir: str = "data/memory",
        short_term_policy: Optional[ShortTermPolicy] = None,
        use_llm_management: bool = True
    ):
        """
        Initialize memory manager.
        
        Args:
            scope: Memory scope ("global", "user:id", "session:id")
            llm_client: LLM client for embeddings and management
            data_dir: Directory for persistent storage
            short_term_policy: Policy for short-term memory
            use_llm_management: Use LLM for memory management decisions
        """
        self.scope = parse_scope(scope)
        self.llm_client = llm_client
        self.data_dir = data_dir
        self.use_llm_management = use_llm_management and llm_client is not None
        
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize memory types
        self.short_term = ShortTermMemory(
            scope=str(self.scope),
            policy=short_term_policy or ShortTermPolicy()
        )
        
        self.long_term = LongTermMemory(
            scope=str(self.scope),
            qdrant_path=os.path.join(data_dir, "qdrant"),
            llm_client=llm_client
        )
        
        self.features = FeatureMemory(
            scope=str(self.scope),
            storage_dir=os.path.join(data_dir, "features")
        )
        
        # Initialize memory manager
        if self.use_llm_management:
            self.memory_manager = LLMMemoryManager(llm_client)
        else:
            self.memory_manager = SimpleMemoryManager()
        
        logger.info(f"MemoryManager initialized for scope: {self.scope}")
    
    def add_turn(
        self,
        user_message: str,
        assistant_response: str,
        extract_facts: bool = True
    ):
        """
        Add a conversation turn to all memories.
        
        Args:
            user_message: User's message
            assistant_response: Assistant's response
            extract_facts: Whether to extract facts for feature memory
        """
        # Add to short-term
        self.short_term.add(user_message, "user")
        self.short_term.add(assistant_response, "assistant")
        
        # Add to long-term
        self.long_term.store_conversation(user_message, assistant_response)
        
        # Extract facts if LLM available
        if extract_facts and self.use_llm_management:
            self._extract_and_store_facts(user_message, assistant_response)
        
        logger.debug("Added conversation turn to all memories")
    
    def get_context(
        self,
        query: str,
        max_tokens: int = 3000,
        include_short_term: bool = True,
        include_features: bool = True,
        include_long_term: bool = True,
        long_term_limit: int = 3
    ) -> Dict[str, Any]:
        """
        Get combined context for LLM prompt.
        
        Args:
            query: Current user query
            max_tokens: Maximum total tokens
            include_short_term: Include short-term context
            include_features: Include feature memory
            include_long_term: Include long-term retrieval
            long_term_limit: Max long-term memories to retrieve
            
        Returns:
            Dict with context components and combined string
        """
        budget = TokenBudget(max_tokens)
        context = {
            "short_term": "",
            "features": "",
            "long_term": "",
            "combined": "",
            "token_usage": {}
        }
        
        # 1. Feature memory (highest priority, usually small)
        if include_features and len(self.features) > 0:
            features_json = self.features.to_prompt(include_history=True)
            features_tokens = count_tokens(features_json)
            
            if budget.allocate("features", features_tokens):
                context["features"] = features_json
        
        # 2. Short-term memory (recent context)
        if include_short_term:
            remaining = budget.remaining()
            stm_context = self.short_term.get_context_string(max_tokens=remaining)
            stm_tokens = count_tokens(stm_context)
            
            if budget.allocate("short_term", stm_tokens):
                context["short_term"] = stm_context
        
        # 3. Long-term retrieval (relevant past)
        if include_long_term:
            remaining = budget.remaining()
            if remaining > 100:  # Need at least some space
                ltm_context = self.long_term.get_relevant_context(
                    query=query,
                    max_tokens=remaining,
                    limit=long_term_limit
                )
                ltm_tokens = count_tokens(ltm_context)
                
                if budget.allocate("long_term", ltm_tokens):
                    context["long_term"] = ltm_context
        
        # Build combined context
        context["combined"] = self._build_combined_context(context)
        context["token_usage"] = {
            "total": budget.used_tokens,
            "max": max_tokens,
            "breakdown": budget.allocations
        }
        
        return context
    
    def get_context_string(self, query: str, max_tokens: int = 3000) -> str:
        """Get combined context as a single string."""
        return self.get_context(query, max_tokens)["combined"]
    
    def _build_combined_context(self, context: Dict) -> str:
        """Build combined context string."""
        parts = []
        
        if context["features"]:
            parts.append(f"=== USER PROFILE ===\n{context['features']}")
        
        if context["long_term"]:
            parts.append(f"=== RELEVANT MEMORIES ===\n{context['long_term']}")
        
        if context["short_term"]:
            parts.append(f"=== RECENT CONVERSATION ===\n{context['short_term']}")
        
        return "\n\n".join(parts)
    
    def _extract_and_store_facts(self, user_message: str, assistant_response: str):
        """Extract facts from conversation and store in feature memory."""
        try:
            facts = self.memory_manager.extract_facts(user_message, assistant_response)
            
            for fact in facts:
                key = fact.get("key")
                value = fact.get("value")
                confidence = fact.get("confidence", 0.8)
                
                if key and value:
                    self.features.set(key, value, source="conversation", confidence=confidence)
                    logger.debug(f"Extracted fact: {key} = {value}")
                    
        except Exception as e:
            logger.debug(f"Fact extraction skipped: {e}")
    
    def needs_management(self) -> bool:
        """Check if short-term memory needs management."""
        return self.short_term.needs_management()
    
    def manage_short_term(self):
        """
        Run LLM-driven (or rule-based) short-term memory management.
        
        Applies KEEP/DELETE/COMPRESS/PROMOTE actions.
        """
        if not self.short_term.needs_management():
            logger.debug("Management not needed")
            return
        
        entries = self.short_term.get_entries_for_management()
        
        actions = self.memory_manager.get_management_actions(
            entries=entries,
            total_tokens=self.short_term.total_tokens,
            max_tokens=self.short_term.policy.max_tokens
        )
        
        # Process PROMOTE actions first (to feature memory)
        for action in actions:
            if action.get("action") == "PROMOTE":
                fact_key = action.get("fact_key")
                fact_value = action.get("fact_value")
                if fact_key and fact_value:
                    self.features.set(fact_key, fact_value, source="promoted")
        
        # Apply remaining actions to short-term
        self.short_term.apply_management_actions(actions)
        
        logger.info(f"Applied {len(actions)} management actions")
    
    def set_fact(self, key: str, value: Any, source: str = "manual"):
        """Manually set a fact in feature memory."""
        self.features.set(key, value, source=source)
    
    def get_fact(self, key: str, default: Any = None) -> Any:
        """Get a fact from feature memory."""
        return self.features.get(key, default)
    
    def search_long_term(self, query: str, limit: int = 5) -> List[Dict]:
        """Search long-term memory."""
        return self.long_term.search(query, limit=limit)
    
    def get_stats(self) -> Dict:
        """Get memory statistics."""
        return {
            "scope": str(self.scope),
            "short_term": self.short_term.get_stats(),
            "long_term": self.long_term.get_stats(),
            "features": {
                "fact_count": len(self.features),
                "facts": list(self.features.keys())
            }
        }
    
    def clear_session(self):
        """Clear short-term memory (for session reset)."""
        self.short_term.clear()
        logger.info("Session cleared (short-term memory)")
    
    def clear_all(self):
        """Clear all memories (use with caution!)."""
        self.short_term.clear()
        self.long_term.clear()
        self.features.clear()
        logger.warning("All memories cleared!")


# Factory for scope-based instances
_managers: Dict[str, MemoryManager] = {}


def get_memory_manager(
    scope: str = "global",
    llm_client = None,
    data_dir: str = "data/memory"
) -> MemoryManager:
    """
    Get or create a MemoryManager for a scope.
    
    Args:
        scope: Memory scope
        llm_client: LLM client (optional)
        data_dir: Data directory
        
    Returns:
        MemoryManager instance
    """
    if scope not in _managers:
        _managers[scope] = MemoryManager(
            scope=scope,
            llm_client=llm_client,
            data_dir=data_dir
        )
    return _managers[scope]


def clear_manager_cache():
    """Clear the manager cache."""
    _managers.clear()
