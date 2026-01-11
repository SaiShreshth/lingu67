"""
Short-Term Memory - Volatile context memory for current conversation.

Subject to retention, decay, and compression policies.
LLM decides what to keep, delete, or compress.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
import json

from memory.utils.scopes import MemoryScope, parse_scope
from memory.managers.policies import ShortTermPolicy, PolicyEnforcer
from memory.utils.helpers import count_tokens, generate_id, truncate_to_tokens

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """A single entry in short-term memory."""
    
    id: str
    content: str
    role: str  # "user", "assistant", "system"
    created_at: datetime
    accessed_at: datetime
    access_count: int = 1
    confidence: float = 1.0
    tokens: int = 0
    compressed: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.tokens == 0:
            self.tokens = count_tokens(self.content)
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "content": self.content,
            "role": self.role,
            "created_at": self.created_at.isoformat(),
            "accessed_at": self.accessed_at.isoformat(),
            "access_count": self.access_count,
            "confidence": self.confidence,
            "tokens": self.tokens,
            "compressed": self.compressed,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> "MemoryEntry":
        return cls(
            id=d["id"],
            content=d["content"],
            role=d["role"],
            created_at=datetime.fromisoformat(d["created_at"]),
            accessed_at=datetime.fromisoformat(d["accessed_at"]),
            access_count=d.get("access_count", 1),
            confidence=d.get("confidence", 1.0),
            tokens=d.get("tokens", 0),
            compressed=d.get("compressed", False),
            metadata=d.get("metadata", {})
        )
    
    def touch(self):
        """Mark as accessed."""
        self.accessed_at = datetime.now()
        self.access_count += 1


class ShortTermMemory:
    """
    Volatile short-term memory for conversation context.
    
    Features:
    - Stores recent conversation turns
    - Subject to token limits and policies
    - Supports compression and pruning
    - Fed to LLM with each prompt
    
    Example:
        stm = ShortTermMemory(scope="user:123")
        stm.add("Hello!", "user")
        stm.add("Hi there!", "assistant")
        
        context = stm.get_context()  # Returns messages for LLM
    """
    
    def __init__(
        self,
        scope: str = "global",
        policy: Optional[ShortTermPolicy] = None
    ):
        self.scope = parse_scope(scope)
        self.policy = policy or ShortTermPolicy()
        self.enforcer = PolicyEnforcer(self.policy)
        
        self.entries: List[MemoryEntry] = []
        self._total_tokens = 0
    
    def add(
        self,
        content: str,
        role: str = "user",
        confidence: float = 1.0,
        metadata: Optional[Dict] = None
    ) -> MemoryEntry:
        """
        Add a new entry to short-term memory.
        
        Args:
            content: Message content
            role: "user", "assistant", or "system"
            confidence: Initial confidence (0-1)
            metadata: Additional metadata
            
        Returns:
            The created entry
        """
        now = datetime.now()
        
        entry = MemoryEntry(
            id=generate_id(),
            content=content,
            role=role,
            created_at=now,
            accessed_at=now,
            confidence=confidence,
            metadata=metadata or {}
        )
        
        self.entries.append(entry)
        self._total_tokens += entry.tokens
        
        self.enforcer.record_turn()
        
        # Auto-prune if over capacity
        if self.policy.is_over_capacity(self._total_tokens):
            self._auto_prune()
        
        logger.debug(f"Added entry: {entry.id} ({entry.tokens} tokens)")
        return entry
    
    def add_turn(self, user_message: str, assistant_response: str) -> tuple:
        """Add a complete conversation turn."""
        user_entry = self.add(user_message, "user")
        assistant_entry = self.add(assistant_response, "assistant")
        return user_entry, assistant_entry
    
    def get_entries(
        self,
        limit: Optional[int] = None,
        include_compressed: bool = True
    ) -> List[MemoryEntry]:
        """
        Get memory entries.
        
        Args:
            limit: Max entries to return (most recent)
            include_compressed: Include compressed entries
            
        Returns:
            List of entries
        """
        entries = self.entries
        
        if not include_compressed:
            entries = [e for e in entries if not e.compressed]
        
        if limit:
            entries = entries[-limit:]
        
        return entries
    
    def get_context(self, max_tokens: Optional[int] = None) -> List[Dict]:
        """
        Get messages formatted for LLM context.
        
        Args:
            max_tokens: Maximum tokens to include
            
        Returns:
            List of {"role": "...", "content": "..."} dicts
        """
        max_tokens = max_tokens or self.policy.max_tokens
        
        messages = []
        token_count = 0
        
        # Include entries from most recent, respecting token limit
        for entry in reversed(self.entries):
            if token_count + entry.tokens > max_tokens:
                break
            
            messages.insert(0, {
                "role": entry.role,
                "content": entry.content
            })
            token_count += entry.tokens
            
            # Touch the entry
            entry.touch()
        
        return messages
    
    def get_context_string(self, max_tokens: Optional[int] = None) -> str:
        """Get context as formatted string."""
        messages = self.get_context(max_tokens)
        
        lines = []
        for msg in messages:
            role = msg["role"].capitalize()
            lines.append(f"{role}: {msg['content']}")
        
        return "\n\n".join(lines)
    
    @property
    def total_tokens(self) -> int:
        """Get total token count."""
        return self._total_tokens
    
    @property
    def entry_count(self) -> int:
        """Get number of entries."""
        return len(self.entries)
    
    def needs_management(self) -> bool:
        """Check if LLM management should run."""
        return (
            self.enforcer.should_manage() or
            self.policy.needs_compression(self._total_tokens)
        )
    
    def _auto_prune(self):
        """Automatically prune oldest entries when over capacity."""
        while (self._total_tokens > self.policy.max_tokens and 
               len(self.entries) > self.policy.min_entries):
            
            removed = self.entries.pop(0)
            self._total_tokens -= removed.tokens
            logger.debug(f"Auto-pruned: {removed.id}")
    
    def delete(self, entry_id: str) -> bool:
        """Delete an entry by ID."""
        for i, entry in enumerate(self.entries):
            if entry.id == entry_id:
                removed = self.entries.pop(i)
                self._total_tokens -= removed.tokens
                logger.debug(f"Deleted: {entry_id}")
                return True
        return False
    
    def compress_entry(self, entry_id: str, compressed_content: str) -> bool:
        """
        Replace an entry with compressed version.
        
        Args:
            entry_id: Entry to compress
            compressed_content: Compressed version
            
        Returns:
            True if successful
        """
        for entry in self.entries:
            if entry.id == entry_id:
                old_tokens = entry.tokens
                entry.content = compressed_content
                entry.tokens = count_tokens(compressed_content)
                entry.compressed = True
                
                self._total_tokens = self._total_tokens - old_tokens + entry.tokens
                logger.debug(f"Compressed: {entry_id} ({old_tokens} → {entry.tokens} tokens)")
                return True
        return False
    
    def apply_management_actions(self, actions: List[Dict]):
        """
        Apply LLM-decided management actions.
        
        Args:
            actions: List of {"id": "...", "action": "KEEP/DELETE/COMPRESS", ...}
        """
        for action in actions:
            entry_id = action.get("id")
            action_type = action.get("action", "KEEP").upper()
            
            if action_type == "DELETE":
                self.delete(entry_id)
            
            elif action_type == "COMPRESS":
                compressed = action.get("compressed", "")
                if compressed:
                    self.compress_entry(entry_id, compressed)
        
        self.enforcer.reset_manage_counter()
    
    def get_entries_for_management(self) -> List[Dict]:
        """
        Get entries formatted for LLM management prompt.
        
        Returns:
            List of entry summaries for LLM
        """
        return [
            {
                "id": e.id,
                "role": e.role,
                "content": e.content[:500] + "..." if len(e.content) > 500 else e.content,
                "tokens": e.tokens,
                "age_minutes": (datetime.now() - e.created_at).seconds // 60,
                "access_count": e.access_count,
                "compressed": e.compressed
            }
            for e in self.entries
        ]
    
    def get_stats(self) -> Dict:
        """Get memory statistics."""
        return {
            "entry_count": len(self.entries),
            "total_tokens": self._total_tokens,
            "max_tokens": self.policy.max_tokens,
            "usage_percent": round(self._total_tokens / self.policy.max_tokens * 100, 1),
            "compressed_count": sum(1 for e in self.entries if e.compressed),
            "needs_management": self.needs_management()
        }
    
    def clear(self):
        """Clear all entries."""
        self.entries.clear()
        self._total_tokens = 0
        self.enforcer.reset_manage_counter()
    
    def to_dict(self) -> Dict:
        """Serialize to dict."""
        return {
            "scope": str(self.scope),
            "entries": [e.to_dict() for e in self.entries],
            "total_tokens": self._total_tokens
        }
    
    def from_dict(self, data: Dict):
        """Load from dict."""
        self.entries = [MemoryEntry.from_dict(e) for e in data.get("entries", [])]
        self._total_tokens = sum(e.tokens for e in self.entries)
    
    def __len__(self) -> int:
        return len(self.entries)
