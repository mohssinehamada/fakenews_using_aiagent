"""
Flask application module for fake news detection API.
"""

from flask import Flask
from typing import Dict, Any

def create_app(config: Dict[str, Any] = None) -> Flask:
    """
    Create and configure Flask application.
    
    Args:
        config (Dict[str, Any]): Application configuration
        
    Returns:
        Flask: Configured Flask application
    """
    app = Flask(__name__)
    
    if config:
        app.config.update(config)
    
    # Register routes
    from .routes import setup_routes
    setup_routes(app)
    
    return app

def init_models(app: Flask) -> None:
    """
    Initialize ML models for the application.
    
    Args:
        app (Flask): Flask application instance
    """
    # TODO: Initialize and attach models to app context
    pass 