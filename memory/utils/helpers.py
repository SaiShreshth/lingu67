"""
Memory Utilities - Token counting, text helpers, and common functions.
"""

import re
import json
from typing import List, Dict, Any, Optional
from datetime import datetime


def count_tokens(text: str) -> int:
    """
    Estimate token count for text.
    
    Uses a simple heuristic: ~4 characters per token for English.
    For more accuracy, use tiktoken with specific model.
    
    Args:
        text: Input text
        
    Returns:
        Estimated token count
    """
    if not text:
        return 0
    
    # Rough estimate: 1 token ≈ 4 characters or 0.75 words
    char_estimate = len(text) // 4
    word_estimate = len(text.split()) * 1.3
    
    return int((char_estimate + word_estimate) / 2)


def count_tokens_messages(messages: List[Dict]) -> int:
    """
    Count tokens in a list of messages.
    
    Args:
        messages: List of {"role": "...", "content": "..."} dicts
        
    Returns:
        Total estimated tokens
    """
    total = 0
    for msg in messages:
        # Role wrapper ~4 tokens
        total += 4
        if "content" in msg:
            total += count_tokens(msg["content"])
    return total


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """
    Truncate text to approximately max_tokens.
    
    Args:
        text: Input text
        max_tokens: Maximum tokens
        
    Returns:
        Truncated text
    """
    current = count_tokens(text)
    if current <= max_tokens:
        return text
    
    # Estimate character limit
    ratio = max_tokens / current
    char_limit = int(len(text) * ratio * 0.9)  # 10% buffer
    
    truncated = text[:char_limit]
    
    # Try to end at word boundary
    last_space = truncated.rfind(' ')
    if last_space > char_limit * 0.8:
        truncated = truncated[:last_space]
    
    return truncated + "..."


def format_timestamp(dt: datetime) -> str:
    """Format datetime for display."""
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def parse_timestamp(ts_str: str) -> datetime:
    """Parse timestamp string to datetime."""
    formats = [
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(ts_str, fmt)
        except ValueError:
            continue
    
    raise ValueError(f"Cannot parse timestamp: {ts_str}")


def safe_json_loads(text: str, default: Any = None) -> Any:
    """Safely load JSON with fallback."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return default


def extract_json_from_text(text: str) -> Optional[Dict]:
    """
    Extract JSON object from text that may contain other content.
    
    Args:
        text: Text potentially containing JSON
        
    Returns:
        Parsed JSON dict or None
    """
    # Find JSON-like pattern
    patterns = [
        r'\{[^{}]*\}',  # Simple object
        r'\{(?:[^{}]|\{[^{}]*\})*\}',  # Nested one level
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in reversed(matches):  # Try last match first
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
    
    return None


def generate_id() -> str:
    """Generate a unique ID."""
    import uuid
    return str(uuid.uuid4())


def hash_content(content: str) -> str:
    """Generate hash for content deduplication."""
    import hashlib
    return hashlib.md5(content.encode()).hexdigest()[:16]


def sanitize_for_storage(text: str) -> str:
    """Sanitize text for safe storage."""
    # Remove null bytes and control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
    return text.strip()


def summarize_text(text: str, max_length: int = 200) -> str:
    """
    Create a simple summary by extracting key sentences.
    
    For better summaries, use LLM.
    
    Args:
        text: Input text
        max_length: Max summary length
        
    Returns:
        Summary text
    """
    if len(text) <= max_length:
        return text
    
    # Split into sentences
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return text[:max_length] + "..."
    
    # Take first sentence(s) up to limit
    summary = ""
    for sentence in sentences:
        if len(summary) + len(sentence) + 2 <= max_length:
            summary += sentence + ". "
        else:
            break
    
    return summary.strip() or (text[:max_length] + "...")


def merge_similar_texts(texts: List[str], threshold: float = 0.85) -> List[str]:
    """
    Merge highly similar texts.
    
    Uses simple word overlap for similarity.
    For better merging, use embeddings.
    
    Args:
        texts: List of texts
        threshold: Similarity threshold (0-1)
        
    Returns:
        Deduplicated list
    """
    if len(texts) <= 1:
        return texts
    
    def word_similarity(t1: str, t2: str) -> float:
        words1 = set(t1.lower().split())
        words2 = set(t2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union)
    
    merged = []
    used = set()
    
    for i, text in enumerate(texts):
        if i in used:
            continue
        
        # Find similar texts
        similar_group = [text]
        for j, other in enumerate(texts[i+1:], start=i+1):
            if j not in used and word_similarity(text, other) >= threshold:
                similar_group.append(other)
                used.add(j)
        
        # Keep longest from similar group
        merged.append(max(similar_group, key=len))
        used.add(i)
    
    return merged


class TokenBudget:
    """Track token budget usage."""
    
    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens
        self.used_tokens = 0
        self.allocations: Dict[str, int] = {}
    
    def allocate(self, name: str, tokens: int) -> bool:
        """
        Try to allocate tokens.
        
        Returns:
            True if allocation succeeded
        """
        if self.used_tokens + tokens > self.max_tokens:
            return False
        
        self.allocations[name] = self.allocations.get(name, 0) + tokens
        self.used_tokens += tokens
        return True
    
    def remaining(self) -> int:
        """Get remaining token budget."""
        return max(0, self.max_tokens - self.used_tokens)
    
    def usage_ratio(self) -> float:
        """Get usage as ratio (0-1)."""
        return self.used_tokens / self.max_tokens if self.max_tokens > 0 else 0
    
    def reset(self):
        """Reset budget."""
        self.used_tokens = 0
        self.allocations.clear()
