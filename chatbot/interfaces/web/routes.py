"""
API Routes - Flask blueprint with all API endpoints.

Provides REST API for the chatbot web interface.
"""

import os
import logging
import tempfile
from flask import Blueprint, request, jsonify, render_template_string, Response, stream_with_context

from chatbot.orchestrator.core import ChatSession

logger = logging.getLogger(__name__)

bp = Blueprint("api", __name__)


# Session storage (in production, use Redis or similar)
_sessions = {}


def _get_session(session_id: str) -> ChatSession:
    """Get or create a session."""
    if session_id not in _sessions:
        _sessions[session_id] = ChatSession(session_id=session_id)
    return _sessions[session_id]


@bp.route("/")
def index():
    """Serve the main chat interface."""
    from chatbot.interfaces.web.templates import HTML_TEMPLATE
    return render_template_string(HTML_TEMPLATE)


@bp.route("/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy"})


@bp.route("/chat", methods=["POST"])
def chat():
    """
    Handle chat message.
    
    Request JSON:
        - message: User's message
        - session_id: Optional session ID
        - history: Optional conversation history
        
    Response JSON:
        - response: AI response
        - intent: Detected intent
        - agents_used: List of agents used
    """
    from chatbot.interfaces.web.app import get_orchestrator
    
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "Missing 'message' field"}), 400
    
    message = data["message"]
    session_id = data.get("session_id", "web_default")
    
    try:
        orchestrator = get_orchestrator()
        session = _get_session(session_id)
        
        # Add any provided history to session
        if "history" in data:
            for turn in data["history"]:
                if turn.get("role") == "user":
                    session.history.append(turn)
                elif turn.get("role") == "assistant":
                    session.history.append(turn)
        
        result = orchestrator.process(
            query=message,
            session=session,
            stream=False
        )
        
        return jsonify({
            "response": result.content,
            "intent": result.intent.name,
            "agents_used": result.agents_used
        })
        
    except Exception as e:
        logger.exception("Chat error")
        return jsonify({"error": str(e)}), 500


@bp.route("/chat/stream", methods=["POST"])
def chat_stream():
    """
    Handle streaming chat message.
    
    Returns Server-Sent Events with tokens.
    """
    from chatbot.interfaces.web.app import get_orchestrator
    
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "Missing 'message' field"}), 400
    
    message = data["message"]
    session_id = data.get("session_id", "web_default")
    
    def generate():
        try:
            orchestrator = get_orchestrator()
            session = _get_session(session_id)
            
            for token in orchestrator.process(
                query=message,
                session=session,
                stream=True
            ):
                yield token
                
        except Exception as e:
            logger.exception("Stream error")
            yield f"[Error: {str(e)}]"
    
    return Response(
        stream_with_context(generate()),
        content_type="text/plain"
    )


@bp.route("/files", methods=["GET"])
def list_files():
    """List uploaded files."""
    from chatbot.interfaces.web.app import get_orchestrator
    
    try:
        orchestrator = get_orchestrator()
        files = orchestrator.list_files()
        return jsonify({"files": files})
    except Exception as e:
        logger.exception("List files error")
        return jsonify({"error": str(e)}), 500


@bp.route("/files/upload", methods=["POST"])
def upload_file():
    """
    Upload and ingest a file.
    
    Expects multipart form with 'file' field.
    """
    from chatbot.interfaces.web.app import get_orchestrator
    
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    
    try:
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            file.save(tmp.name)
            temp_path = tmp.name
        
        # Ingest
        orchestrator = get_orchestrator()
        result = orchestrator.ingest_file(temp_path, file.filename)
        
        # Cleanup
        os.unlink(temp_path)
        
        return jsonify({
            "success": True,
            "filename": file.filename,
            "chunks": result.get("chunks", 0)
        })
        
    except Exception as e:
        logger.exception("Upload error")
        return jsonify({"error": str(e)}), 500


@bp.route("/files/<filename>", methods=["DELETE"])
def delete_file(filename: str):
    """Delete an uploaded file."""
    from chatbot.interfaces.web.app import get_orchestrator
    
    try:
        orchestrator = get_orchestrator()
        file_agent = orchestrator.agents.get("file")
        
        if file_agent:
            success = file_agent.delete_file(filename)
            return jsonify({"success": success})
        
        return jsonify({"error": "File agent not available"}), 500
        
    except Exception as e:
        logger.exception("Delete error")
        return jsonify({"error": str(e)}), 500


@bp.route("/session/clear", methods=["POST"])
def clear_session():
    """Clear current session."""
    data = request.get_json() or {}
    session_id = data.get("session_id", "web_default")
    
    if session_id in _sessions:
        del _sessions[session_id]
    
    return jsonify({"success": True})
