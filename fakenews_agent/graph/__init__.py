"""
Graph module for fake news detection.
Handles graph attention network for claim verification.
"""

from .builder import GraphBuilder
from .trainer import GraphTrainer

__all__ = ['GraphBuilder', 'GraphTrainer'] 