"""
Feature Memory - Structured JSON facts with history tracking.

Stores concise facts about users/topics that are sent with each LLM prompt.
Tracks historical changes to facts (episodic-style history).
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field, asdict

from memory.utils.scopes import MemoryScope, parse_scope
from memory.utils.helpers import generate_id, safe_json_loads

logger = logging.getLogger(__name__)


@dataclass
class FactValue:
    """A single value in fact history."""
    value: Any
    since: str  # ISO timestamp
    until: Optional[str] = None  # ISO timestamp, None if current
    source: Optional[str] = None  # Where this fact came from
    confidence: float = 1.0
    
    def is_current(self) -> bool:
        return self.until is None
    
    def to_dict(self) -> Dict:
        d = {"value": self.value, "since": self.since}
        if self.until:
            d["until"] = self.until
        if self.source:
            d["source"] = self.source
        if self.confidence != 1.0:
            d["confidence"] = self.confidence
        return d
    
    @classmethod
    def from_dict(cls, d: Dict) -> "FactValue":
        return cls(
            value=d["value"],
            since=d["since"],
            until=d.get("until"),
            source=d.get("source"),
            confidence=d.get("confidence", 1.0)
        )


@dataclass
class Fact:
    """A fact with current value and history."""
    key: str
    current: Any
    history: List[FactValue] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "current": self.current,
            "history": [h.to_dict() for h in self.history]
        }
    
    @classmethod
    def from_dict(cls, key: str, d: Dict) -> "Fact":
        return cls(
            key=key,
            current=d["current"],
            history=[FactValue.from_dict(h) for h in d.get("history", [])]
        )
    
    def get_history_summary(self, max_items: int = 3) -> str:
        """Get brief history summary."""
        if len(self.history) <= 1:
            return ""
        
        recent = self.history[-max_items:]
        parts = []
        for h in recent[:-1]:  # Exclude current
            parts.append(f"{h.value} (until {h.until[:10] if h.until else '?'})")
        
        if parts:
            return f"Previously: {' → '.join(parts)}"
        return ""


class FeatureMemory:
    """
    Structured JSON fact storage with history tracking.
    
    Features:
    - Stores facts in JSON format (small, concise)
    - Tracks historical changes to facts
    - Scoped per user/global/session
    - Optimized for inclusion in LLM prompts
    
    Example:
        fm = FeatureMemory(scope="user:123", storage_dir="data/features")
        fm.set("name", "John", source="user_stated")
        fm.update("favorite_color", "blue")  # Will track history if changed
        
        prompt_json = fm.to_prompt()  # Compact JSON for LLM
    """
    
    def __init__(
        self,
        scope: str = "global",
        storage_dir: str = "data/feature_memory"
    ):
        self.scope = parse_scope(scope)
        self.storage_dir = storage_dir
        self.facts: Dict[str, Fact] = {}
        
        os.makedirs(storage_dir, exist_ok=True)
        self._load()
    
    def _get_filepath(self) -> str:
        """Get storage file path for this scope."""
        filename = f"{self.scope.get_storage_key()}_features.json"
        return os.path.join(self.storage_dir, filename)
    
    def _load(self):
        """Load facts from disk."""
        filepath = self._get_filepath()
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for key, fact_data in data.get("facts", {}).items():
                    self.facts[key] = Fact.from_dict(key, fact_data)
                
                logger.debug(f"Loaded {len(self.facts)} facts for scope {self.scope}")
            except Exception as e:
                logger.error(f"Error loading feature memory: {e}")
                self.facts = {}
    
    def _save(self):
        """Save facts to disk."""
        filepath = self._get_filepath()
        try:
            data = {
                "scope": str(self.scope),
                "facts": {k: v.to_dict() for k, v in self.facts.items()},
                "updated_at": datetime.now().isoformat()
            }
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving feature memory: {e}")
    
    def set(
        self,
        key: str,
        value: Any,
        source: Optional[str] = None,
        confidence: float = 1.0
    ):
        """
        Set a fact value.
        
        If the key exists with a different value, the old value is archived.
        
        Args:
            key: Fact key (e.g., "name", "favorite_color")
            value: Fact value
            source: Source of this fact (e.g., "user_stated", "inferred")
            confidence: Confidence in this fact (0-1)
        """
        now = datetime.now().isoformat()
        
        if key in self.facts:
            existing = self.facts[key]
            
            # If value hasn't changed, just update confidence
            if existing.current == value:
                # Update confidence of current history entry
                if existing.history and existing.history[-1].is_current():
                    existing.history[-1].confidence = max(
                        existing.history[-1].confidence,
                        confidence
                    )
                return
            
            # Value changed - archive old value
            if existing.history and existing.history[-1].is_current():
                existing.history[-1].until = now
            
            # Add new value
            existing.history.append(FactValue(
                value=value,
                since=now,
                source=source,
                confidence=confidence
            ))
            existing.current = value
            
        else:
            # New fact
            self.facts[key] = Fact(
                key=key,
                current=value,
                history=[FactValue(
                    value=value,
                    since=now,
                    source=source,
                    confidence=confidence
                )]
            )
        
        self._save()
        logger.debug(f"Set fact: {key} = {value}")
    
    def update(
        self,
        key: str,
        value: Any,
        source: Optional[str] = None,
        confidence: float = 1.0
    ):
        """Alias for set() - updates or creates a fact."""
        self.set(key, value, source, confidence)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get current value of a fact."""
        if key in self.facts:
            return self.facts[key].current
        return default
    
    def get_with_history(self, key: str) -> Optional[Fact]:
        """Get fact with full history."""
        return self.facts.get(key)
    
    def delete(self, key: str) -> bool:
        """Delete a fact."""
        if key in self.facts:
            del self.facts[key]
            self._save()
            return True
        return False
    
    def keys(self) -> List[str]:
        """Get all fact keys."""
        return list(self.facts.keys())
    
    def items(self) -> List[tuple]:
        """Get all (key, current_value) pairs."""
        return [(k, v.current) for k, v in self.facts.items()]
    
    def to_dict(self) -> Dict[str, Any]:
        """Get current values as simple dict."""
        return {k: v.current for k, v in self.facts.items()}
    
    def to_full_dict(self) -> Dict[str, Dict]:
        """Get full facts with history."""
        return {k: v.to_dict() for k, v in self.facts.items()}
    
    def to_prompt(self, include_history: bool = False, max_tokens: int = 500) -> str:
        """
        Generate compact JSON for LLM prompt.
        
        Args:
            include_history: Include brief history summaries
            max_tokens: Approximate max tokens for output
            
        Returns:
            JSON string suitable for LLM prompt
        """
        if not self.facts:
            return "{}"
        
        if include_history:
            # Include history summaries for facts that changed
            result = {}
            for key, fact in self.facts.items():
                history_summary = fact.get_history_summary()
                if history_summary:
                    result[key] = {
                        "current": fact.current,
                        "note": history_summary
                    }
                else:
                    result[key] = fact.current
            return json.dumps(result, ensure_ascii=False)
        else:
            # Just current values
            return json.dumps(self.to_dict(), ensure_ascii=False)
    
    def to_narrative(self) -> str:
        """
        Generate narrative description of facts for LLM.
        
        Returns:
            Natural language description
        """
        if not self.facts:
            return "No known facts."
        
        lines = []
        for key, fact in self.facts.items():
            # Format key nicely
            nice_key = key.replace("_", " ").title()
            
            # Current value
            line = f"- {nice_key}: {fact.current}"
            
            # Brief history if available
            if len(fact.history) > 1:
                prev = fact.history[-2]
                line += f" (previously: {prev.value})"
            
            lines.append(line)
        
        return "\n".join(lines)
    
    def merge_from(self, other: "FeatureMemory"):
        """Merge facts from another FeatureMemory."""
        for key, fact in other.facts.items():
            if key not in self.facts:
                self.facts[key] = fact
            else:
                # Compare timestamps and keep most recent
                if fact.history and self.facts[key].history:
                    other_latest = fact.history[-1].since
                    our_latest = self.facts[key].history[-1].since
                    if other_latest > our_latest:
                        self.set(key, fact.current)
        self._save()
    
    def clear(self):
        """Clear all facts."""
        self.facts.clear()
        self._save()
    
    def __len__(self) -> int:
        return len(self.facts)
    
    def __contains__(self, key: str) -> bool:
        return key in self.facts
    
    def __getitem__(self, key: str) -> Any:
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any):
        self.set(key, value)
