"""
Context Manager - Manages context window for LLM prompts.

Handles context budget allocation and prioritization.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from chatbot.agents.base import AgentContext

logger = logging.getLogger(__name__)


@dataclass
class ContextBudget:
    """Budget allocation for different context sources."""
    total_tokens: int = 3000
    system_tokens: int = 500
    agent_tokens: int = 2000
    query_tokens: int = 500


@dataclass 
class ComposedContext:
    """Final composed context ready for LLM."""
    system_prompt: str
    agent_contexts: List[AgentContext]
    estimated_tokens: int
    
    def to_prompt(self) -> str:
        """Convert to a single prompt string."""
        parts = [self.system_prompt]
        
        # Sort agent contexts by priority (descending)
        sorted_contexts = sorted(
            self.agent_contexts, 
            key=lambda x: x.priority, 
            reverse=True
        )
        
        for ctx in sorted_contexts:
            if ctx.content:
                parts.append(ctx.content)
        
        return "\n\n".join(parts)


class ContextManager:
    """
    Manages context composition for LLM prompts.
    
    Responsibilities:
    - Allocate token budget to agents
    - Prioritize context based on relevance
    - Truncate if necessary
    - Compose final system prompt
    """
    
    SYSTEM_TEMPLATE = """You are a helpful AI assistant with memory capabilities.
You can remember past conversations, access uploaded files, and maintain user preferences.

{profile_context}
{memory_context}
{file_context}

INSTRUCTIONS:
- Use the context above to provide personalized, informed responses
- If you don't know something, say so rather than making things up
- Reference specific memories or files when relevant
- Be helpful, concise, and accurate"""
    
    def __init__(self, budget: Optional[ContextBudget] = None):
        """
        Initialize context manager.
        
        Args:
            budget: Token budget allocation
        """
        self.budget = budget or ContextBudget()
        logger.info(f"ContextManager initialized (budget: {self.budget.total_tokens} tokens)")
    
    def compose(
        self, 
        agent_contexts: List[AgentContext],
        query: str
    ) -> ComposedContext:
        """
        Compose final context from agent outputs.
        
        Args:
            agent_contexts: List of contexts from agents
            query: User's query
            
        Returns:
            ComposedContext ready for LLM
        """
        # Filter out None and empty contexts
        valid_contexts = [
            ctx for ctx in agent_contexts 
            if ctx and ctx.content and ctx.content.strip()
        ]
        
        # Sort by priority
        sorted_contexts = sorted(
            valid_contexts, 
            key=lambda x: x.priority, 
            reverse=True
        )
        
        # Estimate tokens and truncate if needed
        total_tokens = 0
        included_contexts = []
        
        for ctx in sorted_contexts:
            ctx_tokens = self._estimate_tokens(ctx.content)
            
            if total_tokens + ctx_tokens <= self.budget.agent_tokens:
                included_contexts.append(ctx)
                total_tokens += ctx_tokens
            else:
                # Try to include truncated version
                remaining = self.budget.agent_tokens - total_tokens
                if remaining > 100:  # Worth including something
                    truncated_content = self._truncate(ctx.content, remaining)
                    truncated_ctx = AgentContext(
                        content=truncated_content,
                        metadata=ctx.metadata,
                        priority=ctx.priority
                    )
                    included_contexts.append(truncated_ctx)
                    total_tokens += self._estimate_tokens(truncated_content)
                break
        
        # Build system prompt
        system_prompt = self._build_system_prompt(included_contexts)
        
        return ComposedContext(
            system_prompt=system_prompt,
            agent_contexts=included_contexts,
            estimated_tokens=total_tokens + self._estimate_tokens(system_prompt)
        )
    
    def _build_system_prompt(self, contexts: List[AgentContext]) -> str:
        """Build the system prompt from contexts."""
        # Categorize contexts by agent name
        profile_parts = []
        memory_parts = []
        file_parts = []
        other_parts = []
        
        for ctx in contexts:
            agent_name = ctx.metadata.get("agent", "") if ctx.metadata else ""
            content = ctx.content
            
            if "profile" in agent_name.lower() or "Profile" in content:
                profile_parts.append(content)
            elif "memory" in agent_name.lower() or "Memory" in content:
                memory_parts.append(content)
            elif "file" in agent_name.lower() or "File" in content:
                file_parts.append(content)
            else:
                other_parts.append(content)
        
        return self.SYSTEM_TEMPLATE.format(
            profile_context="\n".join(profile_parts) if profile_parts else "",
            memory_context="\n".join(memory_parts) if memory_parts else "",
            file_context="\n".join(file_parts + other_parts) if file_parts or other_parts else ""
        )
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (1 token ≈ 4 chars)."""
        return len(text) // 4
    
    def _truncate(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit token budget."""
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "...[truncated]"
