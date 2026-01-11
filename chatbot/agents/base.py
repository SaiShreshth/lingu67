"""
Base Agent - Abstract interface for all agents.

All specialized agents inherit from this base class.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AgentContext:
    """Context gathered by an agent."""
    content: str  # The context text to include in LLM prompt
    metadata: Optional[Dict[str, Any]] = None  # Additional metadata
    priority: int = 0  # Higher priority = included first


class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    
    Agents are specialized workers that:
    1. Gather context relevant to a query
    2. Post-process after LLM response (e.g., save to memory)
    """
    
    def __init__(self, name: str):
        """
        Initialize the agent.
        
        Args:
            name: Unique name for this agent
        """
        self.name = name
        self._enabled = True
        logger.debug(f"Agent '{name}' initialized")
    
    @property
    def enabled(self) -> bool:
        """Check if agent is enabled."""
        return self._enabled
    
    def enable(self):
        """Enable the agent."""
        self._enabled = True
        
    def disable(self):
        """Disable the agent."""
        self._enabled = False
    
    @abstractmethod
    def gather_context(
        self, 
        query: str, 
        session: Optional[Dict] = None
    ) -> Optional[AgentContext]:
        """
        Gather context relevant to the query.
        
        Args:
            query: The user's query
            session: Optional session data
            
        Returns:
            AgentContext with relevant information, or None if nothing relevant
        """
        pass
    
    @abstractmethod
    def post_process(
        self, 
        user_input: str, 
        response: str,
        session: Optional[Dict] = None
    ) -> None:
        """
        Process after LLM generates a response.
        
        Use this for saving to memory, updating state, etc.
        
        Args:
            user_input: The original user input
            response: The LLM's response
            session: Optional session data
        """
        pass
    
    def __repr__(self) -> str:
        status = "enabled" if self._enabled else "disabled"
        return f"<{self.__class__.__name__} name='{self.name}' {status}>"
