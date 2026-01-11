"""
Intent Router - Classifies user intent and routes to appropriate agents.

Uses LLM or rule-based classification to determine query intent.
"""

import re
import logging
from enum import Enum, auto
from typing import Optional, List, Set
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class Intent(Enum):
    """Possible user intents."""
    CHAT = auto()           # General conversation
    QUESTION = auto()       # Asking a question (may need RAG)
    FILE_QUERY = auto()     # Query about uploaded files
    FILE_UPLOAD = auto()    # User wants to upload a file
    FILE_LIST = auto()      # User wants to list files
    FILE_DELETE = auto()    # User wants to delete a file
    PROFILE_UPDATE = auto() # User providing personal info
    MEMORY_RECALL = auto()  # User asking about past conversations
    SYSTEM = auto()         # System commands (clear, help, etc.)


@dataclass
class RoutingDecision:
    """Result of intent routing."""
    intent: Intent
    confidence: float  # 0.0 to 1.0
    agents_to_use: Set[str]  # Which agents to activate
    metadata: Optional[dict] = None


class IntentRouter:
    """
    Routes user queries to appropriate agents based on intent.
    
    Uses rule-based classification for speed and predictability.
    Can be enhanced with LLM classification for complex cases.
    """
    
    # Keywords for intent detection
    FILE_KEYWORDS = {
        "upload", "uploaded", "file", "document", "pdf", "txt", 
        "csv", "read", "ingested", "load"
    }
    FILE_LIST_KEYWORDS = {"list files", "show files", "what files", "my files"}
    FILE_DELETE_KEYWORDS = {"delete", "remove file", "remove document"}
    MEMORY_KEYWORDS = {
        "remember", "recalled", "told you", "said earlier", 
        "mentioned", "last time", "before", "previously"
    }
    PROFILE_KEYWORDS = {
        "my name is", "i am called", "i live in", "i work as",
        "my job is", "i like", "i prefer"
    }
    SYSTEM_KEYWORDS = {
        "clear", "reset", "help", "exit", "quit", "settings"
    }
    QUESTION_STARTERS = {"what", "who", "when", "where", "why", "how", "is", "are", "can", "do", "does"}
    
    def __init__(self, llm_adapter=None):
        """
        Initialize the router.
        
        Args:
            llm_adapter: Optional LLM adapter for complex classification
        """
        self.llm = llm_adapter
        logger.info("IntentRouter initialized")
    
    def route(self, query: str, context: Optional[dict] = None) -> RoutingDecision:
        """
        Route a query to the appropriate intent and agents.
        
        Args:
            query: User's query
            context: Optional context (e.g., previous intents)
            
        Returns:
            RoutingDecision with intent and agents to use
        """
        query_lower = query.lower().strip()
        
        # Check for system commands first
        if self._is_system_command(query_lower):
            return RoutingDecision(
                intent=Intent.SYSTEM,
                confidence=1.0,
                agents_to_use=set()
            )
        
        # Check for file operations
        if self._is_file_list(query_lower):
            return RoutingDecision(
                intent=Intent.FILE_LIST,
                confidence=0.9,
                agents_to_use={"file"}
            )
        
        if self._is_file_delete(query_lower):
            return RoutingDecision(
                intent=Intent.FILE_DELETE,
                confidence=0.9,
                agents_to_use={"file"}
            )
        
        if self._is_file_query(query_lower):
            return RoutingDecision(
                intent=Intent.FILE_QUERY,
                confidence=0.8,
                agents_to_use={"file", "memory", "profile"}
            )
        
        # Check for profile updates
        if self._is_profile_update(query_lower):
            return RoutingDecision(
                intent=Intent.PROFILE_UPDATE,
                confidence=0.8,
                agents_to_use={"memory", "profile"}
            )
        
        # Check for memory recall
        if self._is_memory_recall(query_lower):
            return RoutingDecision(
                intent=Intent.MEMORY_RECALL,
                confidence=0.7,
                agents_to_use={"memory", "rag", "profile"}
            )
        
        # Check for questions
        if self._is_question(query_lower):
            return RoutingDecision(
                intent=Intent.QUESTION,
                confidence=0.6,
                agents_to_use={"memory", "rag", "profile", "file"}
            )
        
        # Default: general chat
        return RoutingDecision(
            intent=Intent.CHAT,
            confidence=0.5,
            agents_to_use={"memory", "profile"}
        )
    
    def _is_system_command(self, query: str) -> bool:
        """Check if query is a system command."""
        return any(kw in query for kw in self.SYSTEM_KEYWORDS)
    
    def _is_file_list(self, query: str) -> bool:
        """Check if user wants to list files."""
        return any(kw in query for kw in self.FILE_LIST_KEYWORDS)
    
    def _is_file_delete(self, query: str) -> bool:
        """Check if user wants to delete a file."""
        return any(kw in query for kw in self.FILE_DELETE_KEYWORDS)
    
    def _is_file_query(self, query: str) -> bool:
        """Check if query is about files."""
        return any(kw in query for kw in self.FILE_KEYWORDS)
    
    def _is_profile_update(self, query: str) -> bool:
        """Check if user is providing personal info."""
        return any(kw in query for kw in self.PROFILE_KEYWORDS)
    
    def _is_memory_recall(self, query: str) -> bool:
        """Check if user is asking about past conversations."""
        return any(kw in query for kw in self.MEMORY_KEYWORDS)
    
    def _is_question(self, query: str) -> bool:
        """Check if query is a question."""
        first_word = query.split()[0] if query.split() else ""
        return first_word in self.QUESTION_STARTERS or query.endswith("?")
