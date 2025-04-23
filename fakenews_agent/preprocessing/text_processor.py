"""
Text processing module for cleaning and normalizing text.
"""

from typing import List, Dict, Any
import re
import spacy
from collections import Counter
import string

class TextProcessor:
    """Process and clean text for fake news detection."""
    
    def __init__(self):
        """Initialize text processor with cleaning rules."""
        self.nlp = spacy.load('en_core_web_sm')
        self.punctuation = set(string.punctuation)
        self.stop_words = self.nlp.Defaults.stop_words
        
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize input text.
        
        Args:
            text (str): Raw input text
            
        Returns:
            str: Cleaned and normalized text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'[\w\.-]+@[\w\.-]+', '', text)
        
        # Replace newlines and tabs with spaces
        text = re.sub(r'[\n\t]+', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Normalize quotes
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r'[''']', "'", text)
        
        return text
        
    def extract_features(self, text: str) -> Dict[str, Any]:
        """
        Extract linguistic features from text.
        
        Args:
            text (str): Input text
            
        Returns:
            Dict[str, Any]: Dictionary of extracted features
        """
        doc = self.nlp(text)
        
        # Basic statistics
        word_count = len([token for token in doc if not token.is_punct])
        sentence_count = len(list(doc.sents))
        avg_word_length = sum(len(token.text) for token in doc if not token.is_punct) / word_count if word_count > 0 else 0
        
        # POS tag distribution
        pos_dist = Counter(token.pos_ for token in doc)
        
        # Named entity counts
        ner_counts = Counter(ent.label_ for ent in doc.ents)
        
        # Dependency relation patterns
        dep_patterns = Counter(f"{token.dep_}_{token.head.pos_}" for token in doc)
        
        # Lexical diversity (unique words / total words)
        unique_words = len(set(token.text.lower() for token in doc if not token.is_punct and not token.is_stop))
        lexical_diversity = unique_words / word_count if word_count > 0 else 0
        
        # Sentiment analysis (using spaCy's built-in sentiment analyzer)
        sentiment_scores = {
            'positive': len([token for token in doc if token.sentiment > 0]),
            'negative': len([token for token in doc if token.sentiment < 0]),
            'neutral': len([token for token in doc if token.sentiment == 0])
        }
        
        return {
            'basic_stats': {
                'word_count': word_count,
                'sentence_count': sentence_count,
                'avg_word_length': avg_word_length,
                'lexical_diversity': lexical_diversity
            },
            'pos_distribution': dict(pos_dist),
            'named_entities': dict(ner_counts),
            'dependency_patterns': dict(dep_patterns),
            'sentiment': sentiment_scores
        }
        
    def get_important_phrases(self, text: str, top_k: int = 5) -> List[str]:
        """
        Extract important phrases using noun chunks and named entities.
        
        Args:
            text (str): Input text
            top_k (int): Number of phrases to return
            
        Returns:
            List[str]: List of important phrases
        """
        doc = self.nlp(text)
        phrases = []
        
        # Add named entities
        phrases.extend([ent.text for ent in doc.ents])
        
        # Add noun chunks
        phrases.extend([chunk.text for chunk in doc.noun_chunks])
        
        # Remove duplicates and sort by length (prefer longer phrases)
        phrases = sorted(set(phrases), key=len, reverse=True)
        
        return phrases[:top_k] 