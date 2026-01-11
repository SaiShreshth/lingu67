"""
Memory Policies - Retention, decay, and compression for short-term memory.
"""

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, timedelta


@dataclass
class ShortTermPolicy:
    """
    Policy configuration for short-term memory management.
    
    These policies ONLY apply to short-term memory.
    Long-term and feature memory are not affected.
    """
    
    # Token limits
    max_tokens: int = 3000
    target_tokens: int = 2000  # Target after compression
    warning_tokens: int = 2500  # Trigger warning
    
    # Entry limits
    max_entries: int = 50
    min_entries: int = 5  # Always keep at least this many
    
    # Age limits
    max_age_minutes: int = 120  # 2 hours
    
    # Compression settings
    compress_threshold: float = 0.8  # Compress at 80% capacity
    similarity_merge_threshold: float = 0.85  # Merge >85% similar
    
    # LLM management
    manage_every_n_turns: int = 5  # Run LLM management every N turns
    auto_manage: bool = True  # Automatically trigger management
    
    # Decay settings (for confidence scoring)
    enable_decay: bool = True
    decay_half_life_minutes: int = 60  # Confidence halves every hour
    access_boost: float = 0.1  # Boost when accessed
    
    def get_compress_token_threshold(self) -> int:
        """Token count that triggers compression."""
        return int(self.max_tokens * self.compress_threshold)
    
    def is_over_capacity(self, current_tokens: int) -> bool:
        """Check if over max capacity."""
        return current_tokens > self.max_tokens
    
    def needs_compression(self, current_tokens: int) -> bool:
        """Check if compression should be triggered."""
        return current_tokens > self.get_compress_token_threshold()
    
    def is_entry_expired(self, created_at: datetime) -> bool:
        """Check if an entry has exceeded max age."""
        age = datetime.now() - created_at
        return age > timedelta(minutes=self.max_age_minutes)
    
    def calculate_decay(self, created_at: datetime, last_accessed: datetime) -> float:
        """
        Calculate decay factor (0.0 - 1.0).
        
        Uses exponential decay based on time since last access.
        """
        if not self.enable_decay:
            return 1.0
        
        now = datetime.now()
        time_since_access = (now - last_accessed).total_seconds() / 60  # minutes
        
        # Exponential decay: factor = 0.5 ^ (t / half_life)
        import math
        decay_factor = math.pow(0.5, time_since_access / self.decay_half_life_minutes)
        
        return max(0.0, min(1.0, decay_factor))


@dataclass
class RetentionRule:
    """Rule for what to retain in short-term memory."""
    
    min_confidence: float = 0.3  # Min confidence to keep
    min_importance: float = 0.2  # Min importance score
    keep_recent_n: int = 3  # Always keep N most recent
    keep_if_referenced: bool = True  # Keep if referenced in conversation


@dataclass 
class CompressionRule:
    """Rule for how to compress memories."""
    
    max_compressed_length: int = 200  # Max chars after compression
    preserve_key_facts: bool = True  # Keep names, numbers, dates
    merge_similar: bool = True  # Merge highly similar entries
    summarize_old: bool = True  # Summarize entries older than threshold


class PolicyEnforcer:
    """Enforces policies on memory entries."""
    
    def __init__(self, policy: ShortTermPolicy):
        self.policy = policy
        self.turns_since_manage = 0
    
    def should_manage(self) -> bool:
        """Check if LLM management should run."""
        if not self.policy.auto_manage:
            return False
        return self.turns_since_manage >= self.policy.manage_every_n_turns
    
    def record_turn(self):
        """Record a conversation turn."""
        self.turns_since_manage += 1
    
    def reset_manage_counter(self):
        """Reset after management runs."""
        self.turns_since_manage = 0
    
    def get_expired_entries(self, entries: list) -> list:
        """Get entries that have expired based on age."""
        expired = []
        for entry in entries:
            if hasattr(entry, 'created_at') and self.policy.is_entry_expired(entry.created_at):
                expired.append(entry)
        return expired
    
    def calculate_effective_score(
        self, 
        base_confidence: float, 
        created_at: datetime,
        last_accessed: datetime,
        access_count: int = 1
    ) -> float:
        """
        Calculate effective score considering decay and access.
        
        Args:
            base_confidence: Original confidence (0-1)
            created_at: When entry was created
            last_accessed: When entry was last accessed
            access_count: Number of times accessed
            
        Returns:
            Effective score (0-1)
        """
        decay = self.policy.calculate_decay(created_at, last_accessed)
        
        # Access boost (logarithmic to prevent runaway)
        import math
        access_boost = self.policy.access_boost * math.log(access_count + 1)
        
        score = base_confidence * decay + access_boost
        return max(0.0, min(1.0, score))
