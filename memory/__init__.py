"""
LLM Memory Framework
A general-purpose memory system for LLM applications.

Memory Types:
- ShortTermMemory: Volatile context for current conversation
- LongTermMemory: Persistent storage for all interactions  
- FeatureMemory: Structured JSON facts with history tracking

Usage:
    from memory import MemoryManager, MemoryScope
    
    mm = MemoryManager(scope="user:123", llm_client=llm)
    context = mm.get_context(user_message)
    mm.add_turn(user_message, assistant_response)
"""

from memory.core import MemoryManager, get_memory_manager
from memory.utils.scopes import MemoryScope
from memory.managers.policies import ShortTermPolicy
from memory.stores.short_term import ShortTermMemory
from memory.stores.long_term import LongTermMemory
from memory.stores.feature_memory import FeatureMemory

__all__ = [
    "MemoryManager",
    "get_memory_manager",
    "MemoryScope",
    "ShortTermPolicy",
    "ShortTermMemory",
    "LongTermMemory",
    "FeatureMemory",
]

__version__ = "1.0.0"
