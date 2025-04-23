"""
Fusion module for fake news detection.
Handles fusion of multiple prediction sources.
"""

from .classifier import FusionClassifier
from .aggregator import PredictionAggregator

__all__ = ['FusionClassifier', 'PredictionAggregator'] 