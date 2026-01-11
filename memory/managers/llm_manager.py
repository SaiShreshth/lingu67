"""
LLM Memory Manager - LLM-driven memory management decisions.

Uses the LLM to decide what to keep, delete, compress, or promote
from short-term to feature memory.
"""

import logging
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

from memory.utils.helpers import extract_json_from_text

logger = logging.getLogger(__name__)


# Prompt templates for memory management
MEMORY_MANAGEMENT_PROMPT = """You are a memory management system for an AI assistant.

Analyze the short-term memory entries below and decide what action to take for each:

**Current Short-Term Memory:**
{entries_text}

**Statistics:**
- Total entries: {entry_count}
- Total tokens: {total_tokens} / {max_tokens} max
- Token budget remaining: {remaining_tokens}

**Your task:**
For each memory entry, decide one of these actions:
- **KEEP**: Important for ongoing conversation context
- **DELETE**: No longer relevant or duplicated
- **COMPRESS**: Can be summarized to save tokens (provide compressed version)
- **PROMOTE**: Contains a user fact → extract to Feature Memory

**Rules:**
1. Always KEEP the most recent {min_keep} entries
2. DELETE entries that are:
   - Already covered by more recent context
   - Greetings/pleasantries older than 5 minutes
   - Resolved questions/topics
3. COMPRESS entries that are:
   - Long explanations that can be summarized
   - Multiple similar messages
4. PROMOTE entries containing:
   - User's name, preferences, facts about them
   - Important information to remember long-term

Respond with valid JSON only:
```json
{{
  "actions": [
    {{"id": "entry_id", "action": "KEEP"}},
    {{"id": "entry_id", "action": "DELETE", "reason": "why"}},
    {{"id": "entry_id", "action": "COMPRESS", "compressed": "shorter version"}},
    {{"id": "entry_id", "action": "PROMOTE", "fact_key": "key", "fact_value": "value"}}
  ]
}}
```"""


FACT_EXTRACTION_PROMPT = """Extract any user facts from this conversation turn.

**User said:** {user_message}
**Assistant replied:** {assistant_response}

Extract factual information about the user (name, preferences, occupation, etc.)
Only extract NEW or CHANGED facts, not general statements.

Respond with JSON:
```json
{{
  "facts": [
    {{"key": "name", "value": "John", "confidence": 0.95}},
    {{"key": "favorite_color", "value": "blue", "confidence": 0.8}}
  ]
}}
```

If no facts found, respond with: {{"facts": []}}"""


class LLMMemoryManager:
    """
    Uses LLM to manage short-term memory.
    
    Features:
    - Decides what to keep/delete/compress
    - Extracts facts for feature memory
    - Respects token budgets
    
    Example:
        manager = LLMMemoryManager(llm_client)
        actions = manager.get_management_actions(short_term_memory)
        short_term_memory.apply_management_actions(actions)
    """
    
    def __init__(
        self, 
        llm_client,
        min_keep: int = 3,
        max_tokens_for_management: int = 1000
    ):
        """
        Initialize LLM memory manager.
        
        Args:
            llm_client: LLM client with complete() method
            min_keep: Minimum entries to always keep
            max_tokens_for_management: Max tokens for management prompt
        """
        self.llm = llm_client
        self.min_keep = min_keep
        self.max_tokens_for_management = max_tokens_for_management
    
    def get_management_actions(
        self,
        entries: List[Dict],
        total_tokens: int,
        max_tokens: int
    ) -> List[Dict]:
        """
        Get LLM-decided actions for memory management.
        
        Args:
            entries: List of entry dicts from short_term.get_entries_for_management()
            total_tokens: Current total tokens
            max_tokens: Maximum allowed tokens
            
        Returns:
            List of action dicts
        """
        if not entries:
            return []
        
        # Format entries for prompt
        entries_text = self._format_entries(entries)
        
        prompt = MEMORY_MANAGEMENT_PROMPT.format(
            entries_text=entries_text,
            entry_count=len(entries),
            total_tokens=total_tokens,
            max_tokens=max_tokens,
            remaining_tokens=max_tokens - total_tokens,
            min_keep=self.min_keep
        )
        
        try:
            response = self.llm.complete(
                prompt,
                max_tokens=self.max_tokens_for_management,
                temperature=0.3
            )
            
            actions = self._parse_actions(response)
            logger.debug(f"LLM returned {len(actions)} actions")
            return actions
            
        except Exception as e:
            logger.error(f"LLM management error: {e}")
            return []
    
    def extract_facts(
        self,
        user_message: str,
        assistant_response: str
    ) -> List[Dict]:
        """
        Extract facts from a conversation turn.
        
        Args:
            user_message: User's message
            assistant_response: Assistant's response
            
        Returns:
            List of {"key": "...", "value": "...", "confidence": ...}
        """
        prompt = FACT_EXTRACTION_PROMPT.format(
            user_message=user_message[:500],
            assistant_response=assistant_response[:500]
        )
        
        try:
            response = self.llm.complete(prompt, max_tokens=200, temperature=0)
            
            data = extract_json_from_text(response)
            if data and "facts" in data:
                return data["facts"]
            return []
            
        except Exception as e:
            logger.debug(f"Fact extraction error: {e}")
            return []
    
    def should_compress(self, content: str, threshold_tokens: int = 100) -> bool:
        """Check if content should be compressed."""
        from memory.utils.helpers import count_tokens
        return count_tokens(content) > threshold_tokens
    
    def compress_content(self, content: str, max_length: int = 200) -> str:
        """
        Use LLM to compress content.
        
        Args:
            content: Content to compress
            max_length: Target max length
            
        Returns:
            Compressed content
        """
        prompt = f"""Summarize this in under {max_length} characters, keeping key facts:

{content}

Summary:"""
        
        try:
            response = self.llm.complete(prompt, max_tokens=100, temperature=0)
            return response.strip()[:max_length]
        except Exception as e:
            logger.error(f"Compression error: {e}")
            # Fallback: simple truncation
            return content[:max_length] + "..."
    
    def _format_entries(self, entries: List[Dict]) -> str:
        """Format entries for prompt."""
        lines = []
        for e in entries:
            age = e.get("age_minutes", 0)
            compressed = " [COMPRESSED]" if e.get("compressed") else ""
            lines.append(
                f"ID: {e['id']}\n"
                f"Role: {e['role']}{compressed}\n"
                f"Content: {e['content']}\n"
                f"Tokens: {e['tokens']} | Age: {age}min | Accesses: {e.get('access_count', 1)}\n"
            )
        return "\n---\n".join(lines)
    
    def _parse_actions(self, response: str) -> List[Dict]:
        """Parse LLM response into actions."""
        data = extract_json_from_text(response)
        
        if data and "actions" in data:
            actions = data["actions"]
            # Validate actions
            valid_actions = []
            for action in actions:
                if "id" in action and "action" in action:
                    action["action"] = action["action"].upper()
                    if action["action"] in ["KEEP", "DELETE", "COMPRESS", "PROMOTE"]:
                        valid_actions.append(action)
            return valid_actions
        
        return []


class SimpleMemoryManager:
    """
    Simple rule-based memory manager (no LLM required).
    
    Use this when LLM is not available or for faster management.
    """
    
    def __init__(self, min_keep: int = 5):
        self.min_keep = min_keep
    
    def get_management_actions(
        self,
        entries: List[Dict],
        total_tokens: int,
        max_tokens: int
    ) -> List[Dict]:
        """
        Get rule-based actions for memory management.
        
        Rules:
        1. Keep most recent min_keep entries
        2. Delete entries older than 2 hours
        3. Delete greetings after 10 minutes
        """
        if len(entries) <= self.min_keep:
            return []
        
        actions = []
        
        # Keep recent entries
        keep_ids = set(e["id"] for e in entries[-self.min_keep:])
        
        for entry in entries:
            if entry["id"] in keep_ids:
                continue
            
            age = entry.get("age_minutes", 0)
            content_lower = entry.get("content", "").lower()
            
            # Delete very old entries
            if age > 120:
                actions.append({
                    "id": entry["id"],
                    "action": "DELETE",
                    "reason": "older than 2 hours"
                })
            # Delete old greetings
            elif age > 10 and any(g in content_lower for g in 
                                  ["hello", "hi", "hey", "thanks", "bye"]):
                actions.append({
                    "id": entry["id"],
                    "action": "DELETE",
                    "reason": "old greeting"
                })
        
        return actions
