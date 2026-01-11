"""
Flask Web App - Web interface for the chatbot.

Factory pattern for Flask app creation.
"""

import os
import sys
import logging
from flask import Flask
from flask_cors import CORS

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))

logger = logging.getLogger(__name__)


def create_app(test_config=None):
    """
    Flask application factory.
    
    Args:
        test_config: Optional test configuration
        
    Returns:
        Configured Flask app
    """
    app = Flask(__name__, template_folder="templates")
    CORS(app)
    
    # Load config
    if test_config is None:
        app.config.from_mapping(
            SECRET_KEY=os.environ.get("SECRET_KEY", "dev-key-change-me"),
            MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 16MB max upload
        )
    else:
        app.config.from_mapping(test_config)
    
    # Ensure upload folder exists
    os.makedirs(app.config.get("UPLOAD_FOLDER", "data/uploads"), exist_ok=True)
    
    # Register blueprints
    from chatbot.interfaces.web.routes import bp
    app.register_blueprint(bp)
    
    logger.info("Flask app created")
    return app


# Global orchestrator instance (lazy loaded)
_orchestrator = None


def get_orchestrator():
    """Get or create the shared orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        from chatbot.orchestrator.core import ChatOrchestrator
        _orchestrator = ChatOrchestrator()
        logger.info("Orchestrator initialized for web app")
    return _orchestrator


def main():
    """Run the development server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    app = create_app()
    print("\n🚀 Starting Lingu67 Web Interface...")
    print("   http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False)


if __name__ == "__main__":
    main()
