"""
Chat Orchestrator - Central coordinator for the chatbot.

The brain of the system that routes queries, gathers context,
generates responses, and manages post-processing.
"""

import logging
from typing import Dict, List, Optional, Any, Generator
from dataclasses import dataclass

from chatbot.orchestrator.router import IntentRouter, Intent, RoutingDecision
from chatbot.orchestrator.context import ContextManager, ComposedContext
from chatbot.agents.base import BaseAgent, AgentContext
from chatbot.adapters.llm_adapter import LLMAdapter, ChatMessage, LLMResponse

logger = logging.getLogger(__name__)


@dataclass
class ChatSession:
    """Represents a chat session with state."""
    session_id: str
    conversation_id: Optional[str] = None
    history: List[Dict[str, str]] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.history is None:
            self.history = []
        if self.metadata is None:
            self.metadata = {}
    
    def add_turn(self, user: str, assistant: str):
        """Add a conversation turn."""
        self.history.append({"role": "user", "content": user})
        self.history.append({"role": "assistant", "content": assistant})
    
    def get_recent_history(self, n: int = 10) -> List[Dict[str, str]]:
        """Get recent conversation history."""
        return self.history[-n:] if self.history else []


@dataclass
class ChatResponse:
    """Response from the orchestrator."""
    content: str
    intent: Intent
    agents_used: List[str]
    metadata: Optional[Dict[str, Any]] = None


class ChatOrchestrator:
    """
    Central orchestrator for the chatbot.
    
    Coordinates:
    - Intent routing
    - Agent context gathering
    - LLM response generation
    - Post-processing (memory saving, etc.)
    """
    
    def __init__(
        self,
        llm_adapter: Optional[LLMAdapter] = None,
        agents: Optional[Dict[str, BaseAgent]] = None,
        router: Optional[IntentRouter] = None,
        context_manager: Optional[ContextManager] = None
    ):
        """
        Initialize the orchestrator.
        
        Args:
            llm_adapter: LLM adapter for generating responses
            agents: Dict of agent name -> agent instance
            router: Intent router
            context_manager: Context manager
        """
        # Initialize LLM adapter
        if llm_adapter is None:
            from chatbot.adapters.llm_adapter import get_llm_adapter
            llm_adapter = get_llm_adapter()
        self.llm = llm_adapter
        
        # Initialize agents (lazy load defaults)
        self.agents: Dict[str, BaseAgent] = agents or {}
        if not self.agents:
            self._init_default_agents()
        
        # Initialize router and context manager
        self.router = router or IntentRouter()
        self.context_manager = context_manager or ContextManager()
        
        logger.info(f"ChatOrchestrator initialized with {len(self.agents)} agents")
    
    def _init_default_agents(self):
        """Initialize default agents."""
        self.agents = {}
        
        # Initialize each agent individually to avoid one failure blocking all
        try:
            from chatbot.agents.memory_agent import MemoryAgent
            # Use separate data_dir to avoid Qdrant conflicts with model_server
            self.agents["memory"] = MemoryAgent(data_dir="data/chatbot_memory")
            logger.info("MemoryAgent initialized")
        except Exception as e:
            logger.warning(f"MemoryAgent failed: {e}")
        
        try:
            from chatbot.agents.profile_agent import ProfileAgent
            self.agents["profile"] = ProfileAgent()
            logger.info("ProfileAgent initialized")
        except Exception as e:
            logger.warning(f"ProfileAgent failed: {e}")
        
        try:
            from chatbot.agents.file_agent import FileAgent
            self.agents["file"] = FileAgent()
            logger.info("FileAgent initialized")
        except Exception as e:
            logger.warning(f"FileAgent failed: {e}")
        
        try:
            from chatbot.agents.rag_agent import RAGAgent
            # Use separate collection to avoid conflicts
            self.agents["rag"] = RAGAgent(collection="chatbot_rag")
            logger.info("RAGAgent initialized")
        except Exception as e:
            logger.warning(f"RAGAgent failed: {e}")
        
        logger.info(f"Initialized {len(self.agents)} agents: {list(self.agents.keys())}")
    
    def process(
        self, 
        query: str, 
        session: Optional[ChatSession] = None,
        stream: bool = False
    ) -> ChatResponse | Generator[str, None, ChatResponse]:
        """
        Process a user query and generate a response.
        
        Args:
            query: User's query
            session: Optional chat session
            stream: If True, yield tokens as generated
            
        Returns:
            ChatResponse or generator yielding tokens
        """
        if session is None:
            session = ChatSession(session_id="default")
        
        # 1. Route the query
        routing = self.router.route(query)
        logger.debug(f"Routed to intent: {routing.intent.name}")
        
        # 2. Handle system commands
        if routing.intent == Intent.SYSTEM:
            return self._handle_system_command(query, session)
        
        # 3. Gather context from agents
        agent_contexts = self._gather_contexts(query, routing, session)
        
        # 4. Compose final context
        composed = self.context_manager.compose(agent_contexts, query)
        
        # 5. Generate response
        if stream:
            return self._generate_streaming(query, composed, routing, session)
        else:
            response = self._generate_response(query, composed, session)
            
            # 6. Post-process
            self._post_process(query, response, routing, session)
            
            return ChatResponse(
                content=response,
                intent=routing.intent,
                agents_used=list(routing.agents_to_use),
                metadata={"tokens": composed.estimated_tokens}
            )
    
    def _gather_contexts(
        self, 
        query: str, 
        routing: RoutingDecision,
        session: ChatSession
    ) -> List[AgentContext]:
        """Gather context from relevant agents."""
        contexts = []
        
        logger.info(f"=== Gathering context for: '{query[:50]}...' ===")
        logger.info(f"Agents to use: {routing.agents_to_use}")
        
        for agent_name in routing.agents_to_use:
            agent = self.agents.get(agent_name)
            if agent and agent.enabled:
                try:
                    ctx = agent.gather_context(query, session.metadata)
                    if ctx:
                        # Tag with agent name
                        if ctx.metadata is None:
                            ctx.metadata = {}
                        ctx.metadata["agent"] = agent_name
                        contexts.append(ctx)
                        
                        # Log what we got
                        content_preview = ctx.content[:200] if ctx.content else "(empty)"
                        logger.info(f"  [{agent_name}] ✓ Content ({len(ctx.content)} chars): {content_preview}...")
                    else:
                        logger.info(f"  [{agent_name}] - No context returned")
                except Exception as e:
                    logger.error(f"  [{agent_name}] ✗ Failed: {e}")
            else:
                if not agent:
                    logger.warning(f"  [{agent_name}] Agent not found")
                elif not agent.enabled:
                    logger.info(f"  [{agent_name}] Agent disabled")
        
        logger.info(f"=== Total contexts gathered: {len(contexts)} ===")
        return contexts
    
    def _generate_response(
        self, 
        query: str, 
        composed: ComposedContext,
        session: ChatSession
    ) -> str:
        """Generate response using LLM."""
        messages = [
            ChatMessage(role="system", content=composed.to_prompt())
        ]
        
        # Add recent history
        for turn in session.get_recent_history(6):
            messages.append(ChatMessage(
                role=turn["role"],
                content=turn["content"]
            ))
        
        # Add current query
        messages.append(ChatMessage(role="user", content=query))
        
        # Generate
        result = self.llm.chat(messages, max_tokens=512, stream=False)
        return result.content
    
    def _generate_streaming(
        self, 
        query: str, 
        composed: ComposedContext,
        routing: RoutingDecision,
        session: ChatSession
    ) -> Generator[str, None, ChatResponse]:
        """Generate streaming response."""
        messages = [
            ChatMessage(role="system", content=composed.to_prompt())
        ]
        
        for turn in session.get_recent_history(6):
            messages.append(ChatMessage(
                role=turn["role"],
                content=turn["content"]
            ))
        
        messages.append(ChatMessage(role="user", content=query))
        
        # Stream tokens
        full_response = ""
        for token in self.llm.chat(messages, max_tokens=512, stream=True):
            full_response += token
            yield token
        
        # Post-process after streaming
        self._post_process(query, full_response, routing, session)
        
        return ChatResponse(
            content=full_response,
            intent=routing.intent,
            agents_used=list(routing.agents_to_use)
        )
    
    def _post_process(
        self, 
        query: str, 
        response: str,
        routing: RoutingDecision,
        session: ChatSession
    ):
        """Run post-processing on all active agents."""
        # Update session history
        session.add_turn(query, response)
        
        # Call post_process on each agent
        for agent_name in routing.agents_to_use:
            agent = self.agents.get(agent_name)
            if agent and agent.enabled:
                try:
                    agent.post_process(query, response, session.metadata)
                except Exception as e:
                    logger.error(f"Agent '{agent_name}' post_process failed: {e}")
    
    def _handle_system_command(
        self, 
        query: str, 
        session: ChatSession
    ) -> ChatResponse:
        """Handle system commands."""
        query_lower = query.lower().strip()
        
        if "clear" in query_lower or "reset" in query_lower:
            # Clear session
            session.history = []
            if "memory" in self.agents:
                self.agents["memory"].clear_session()
            return ChatResponse(
                content="Session cleared. Starting fresh!",
                intent=Intent.SYSTEM,
                agents_used=[]
            )
        
        if "help" in query_lower:
            return ChatResponse(
                content=self._get_help_text(),
                intent=Intent.SYSTEM,
                agents_used=[]
            )
        
        return ChatResponse(
            content="Command not recognized. Try 'help' for options.",
            intent=Intent.SYSTEM,
            agents_used=[]
        )
    
    def _get_help_text(self) -> str:
        """Get help text."""
        return """**Available Commands:**
- `clear` or `reset` - Clear conversation history
- `help` - Show this message

**Features:**
- I remember our conversations
- Upload files and I can answer questions about them
- I learn your preferences over time

Just chat naturally and I'll help!"""
    
    # Convenience methods
    def chat(self, message: str, session_id: str = "default") -> str:
        """Simple chat interface."""
        session = ChatSession(session_id=session_id)
        result = self.process(message, session)
        return result.content
    
    def ingest_file(self, file_path: str, name: Optional[str] = None) -> Dict:
        """Ingest a file."""
        file_agent = self.agents.get("file")
        if file_agent:
            return file_agent.ingest_file(file_path, name)
        return {"error": "File agent not available"}
    
    def list_files(self) -> List[Dict]:
        """List ingested files."""
        file_agent = self.agents.get("file")
        if file_agent:
            return file_agent.list_files()
        return []
