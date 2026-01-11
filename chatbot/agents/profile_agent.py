"""
Profile Agent - Handles user profile management.

Manages persistent facts about the user (name, preferences, etc.)
"""

import os
import json
import logging
from typing import Dict, Any, Optional

from chatbot.agents.base import BaseAgent, AgentContext

logger = logging.getLogger(__name__)


class ProfileAgent(BaseAgent):
    """
    Agent for user profile management.
    
    Provides:
    - Profile loading and saving
    - Fact extraction from conversations
    - Profile context for LLM
    """
    
    def __init__(self, profile_path: Optional[str] = None):
        """
        Initialize profile agent.
        
        Args:
            profile_path: Path to profile JSON file
        """
        super().__init__(name="profile")
        
        if profile_path is None:
            from config import PROFILE_PATH
            profile_path = PROFILE_PATH
        
        self.profile_path = profile_path
        self._profile: Dict[str, Any] = {}
        self._load_profile()
        
        logger.info(f"ProfileAgent initialized with {len(self._profile)} facts")
    
    def _load_profile(self):
        """Load profile from disk."""
        if os.path.exists(self.profile_path):
            try:
                with open(self.profile_path, 'r') as f:
                    self._profile = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load profile: {e}")
                self._profile = {}
    
    def _save_profile(self):
        """Save profile to disk."""
        try:
            os.makedirs(os.path.dirname(self.profile_path), exist_ok=True)
            with open(self.profile_path, 'w') as f:
                json.dump(self._profile, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save profile: {e}")
    
    def gather_context(
        self, 
        query: str, 
        session: Optional[Dict] = None
    ) -> Optional[AgentContext]:
        """
        Gather profile context.
        
        Args:
            query: The user's query
            session: Optional session data
            
        Returns:
            AgentContext with user profile information
        """
        if not self._enabled or not self._profile:
            return None
        
        # Format profile as context
        profile_str = "User Profile:\n"
        for key, value in self._profile.items():
            profile_str += f"- {key}: {value}\n"
        
        return AgentContext(
            content=profile_str.strip(),
            metadata={"fact_count": len(self._profile)},
            priority=15  # Very high - personalization is important
        )
    
    def post_process(
        self, 
        user_input: str, 
        response: str,
        session: Optional[Dict] = None
    ) -> None:
        """
        Extract and save facts from conversation.
        
        Args:
            user_input: The user's message
            response: The assistant's response
            session: Optional session data with LLM client
        """
        if not self._enabled:
            return
        
        # Simple heuristic extraction (can be enhanced with LLM)
        extracted = self._extract_facts_simple(user_input)
        
        if extracted:
            self._profile.update(extracted)
            self._save_profile()
            logger.debug(f"ProfileAgent: Extracted {len(extracted)} facts")
    
    def _extract_facts_simple(self, text: str) -> Dict[str, str]:
        """
        Simple rule-based fact extraction.
        
        Args:
            text: User's message
            
        Returns:
            Dict of extracted facts
        """
        facts = {}
        text_lower = text.lower()
        
        # Name extraction
        if "my name is" in text_lower:
            idx = text_lower.find("my name is") + len("my name is")
            name = text[idx:].strip().split()[0] if idx < len(text) else ""
            if name:
                facts["user_name"] = name.strip(".,!?")
        
        if "i am called" in text_lower:
            idx = text_lower.find("i am called") + len("i am called")
            name = text[idx:].strip().split()[0] if idx < len(text) else ""
            if name:
                facts["user_name"] = name.strip(".,!?")
        
        # Location extraction
        if "i live in" in text_lower:
            idx = text_lower.find("i live in") + len("i live in")
            location = text[idx:].strip().split(".")[0]
            if location:
                facts["location"] = location.strip()
        
        # Job extraction
        if "i work as" in text_lower:
            idx = text_lower.find("i work as") + len("i work as")
            job = text[idx:].strip().split(".")[0]
            if job:
                facts["occupation"] = job.strip()
        
        if "my job is" in text_lower:
            idx = text_lower.find("my job is") + len("my job is")
            job = text[idx:].strip().split(".")[0]
            if job:
                facts["occupation"] = job.strip()
        
        return facts
    
    def set_fact(self, key: str, value: Any) -> None:
        """Manually set a profile fact."""
        self._profile[key] = value
        self._save_profile()
    
    def get_fact(self, key: str, default: Any = None) -> Any:
        """Get a profile fact."""
        return self._profile.get(key, default)
    
    def get_profile(self) -> Dict[str, Any]:
        """Get the entire profile."""
        return self._profile.copy()
    
    def clear_profile(self) -> None:
        """Clear all profile data."""
        self._profile = {}
        self._save_profile()
        logger.info("ProfileAgent: Profile cleared")
