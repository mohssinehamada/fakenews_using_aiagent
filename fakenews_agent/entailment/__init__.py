"""
Entailment module for fake news detection.
Handles textual entailment using RoBERTa model.
"""

from .model import EntailmentModel
from .predictor import EntailmentPredictor

__all__ = ['EntailmentModel', 'EntailmentPredictor'] 