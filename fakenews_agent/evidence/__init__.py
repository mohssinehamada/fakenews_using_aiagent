"""
Evidence retrieval module for fake news detection.
Handles retrieval of supporting evidence for claims.
"""

from .retriever import EvidenceRetriever
from .indexer import EvidenceIndexer

__all__ = ['EvidenceRetriever', 'EvidenceIndexer'] 