"""
Memory Scopes - User/Global/Session scoping for memory isolation.
"""

from enum import Enum
from typing import Optional
from dataclasses import dataclass
import re


class ScopeType(Enum):
    """Types of memory scopes."""
    GLOBAL = "global"
    USER = "user"
    SESSION = "session"


@dataclass
class MemoryScope:
    """
    Memory scope for isolation.
    
    Scopes:
    - "global": Shared across all users
    - "user:{user_id}": Private to a specific user
    - "session:{session_id}": Temporary session (cleared on end)
    """
    
    scope_type: ScopeType
    identifier: Optional[str] = None
    
    # Class-level constants for convenience
    GLOBAL = "global"
    
    def __str__(self) -> str:
        if self.scope_type == ScopeType.GLOBAL:
            return "global"
        return f"{self.scope_type.value}:{self.identifier}"
    
    def __hash__(self):
        return hash(str(self))
    
    def __eq__(self, other):
        if isinstance(other, str):
            return str(self) == other
        if isinstance(other, MemoryScope):
            return str(self) == str(other)
        return False
    
    @classmethod
    def parse(cls, scope_str: str) -> "MemoryScope":
        """
        Parse a scope string.
        
        Args:
            scope_str: "global", "user:123", "session:abc"
            
        Returns:
            MemoryScope instance
        """
        scope_str = scope_str.strip().lower()
        
        if scope_str == "global":
            return cls(scope_type=ScopeType.GLOBAL)
        
        match = re.match(r"^(user|session):(.+)$", scope_str)
        if match:
            scope_type_str, identifier = match.groups()
            scope_type = ScopeType(scope_type_str)
            return cls(scope_type=scope_type, identifier=identifier)
        
        raise ValueError(f"Invalid scope format: {scope_str}. Expected 'global', 'user:id', or 'session:id'")
    
    @classmethod
    def user(cls, user_id: str) -> "MemoryScope":
        """Create a user-scoped memory scope."""
        return cls(scope_type=ScopeType.USER, identifier=str(user_id))
    
    @classmethod
    def session(cls, session_id: str) -> "MemoryScope":
        """Create a session-scoped memory scope."""
        return cls(scope_type=ScopeType.SESSION, identifier=str(session_id))
    
    @classmethod
    def global_scope(cls) -> "MemoryScope":
        """Create a global memory scope."""
        return cls(scope_type=ScopeType.GLOBAL)
    
    def is_global(self) -> bool:
        return self.scope_type == ScopeType.GLOBAL
    
    def is_user(self) -> bool:
        return self.scope_type == ScopeType.USER
    
    def is_session(self) -> bool:
        return self.scope_type == ScopeType.SESSION
    
    def get_storage_key(self) -> str:
        """Get a safe key for file/db storage."""
        if self.scope_type == ScopeType.GLOBAL:
            return "global"
        # Sanitize identifier for filesystem
        safe_id = re.sub(r'[^\w\-]', '_', self.identifier or "unknown")
        return f"{self.scope_type.value}_{safe_id}"


def validate_scope(scope: str) -> bool:
    """Validate a scope string."""
    try:
        MemoryScope.parse(scope)
        return True
    except ValueError:
        return False


def parse_scope(scope_input) -> MemoryScope:
    """
    Parse various scope inputs into MemoryScope.
    
    Args:
        scope_input: str, MemoryScope, or None (defaults to global)
        
    Returns:
        MemoryScope instance
    """
    if scope_input is None:
        return MemoryScope.global_scope()
    
    if isinstance(scope_input, MemoryScope):
        return scope_input
    
    if isinstance(scope_input, str):
        return MemoryScope.parse(scope_input)
    
    raise TypeError(f"Cannot parse scope from type {type(scope_input)}")
