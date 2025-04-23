"""
Preprocessing module for fake news detection.
Handles claim segmentation, text cleaning, and feature extraction.
"""

from .claim_segmenter import ClaimSegmenter
from .text_processor import TextProcessor

__all__ = ['ClaimSegmenter', 'TextProcessor'] 