"""
API module for fake news detection.
Exposes prediction endpoints via Flask.
"""

from .app import create_app
from .routes import setup_routes

__all__ = ['create_app', 'setup_routes'] 