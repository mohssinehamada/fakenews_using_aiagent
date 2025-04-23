"""
Claim segmentation module for extracting individual claims from text.
"""

import spacy
from typing import List, Dict, Any
import re

class ClaimSegmenter:
    """Segment text into individual claims using spaCy and custom rules."""
    
    CLAIM_INDICATORS = [
        "is", "are", "was", "were", "will be",
        "claims", "stated", "says", "said",
        "according to", "reported", "confirms",
        "shows", "proves", "demonstrates"
    ]
    
    def __init__(self):
        """Initialize the claim segmenter with spaCy model."""
        self.nlp = spacy.load('en_core_web_sm')
        
    def segment_claims(self, text: str) -> List[str]:
        """
        Segment input text into individual claims.
        
        Args:
            text (str): Input text to segment
            
        Returns:
            List[str]: List of extracted claims
        """
        # Process text with spaCy
        doc = self.nlp(text)
        
        # Extract sentences
        sentences = [sent.text.strip() for sent in doc.sents]
        
        # Filter for claims
        claims = []
        for sentence in sentences:
            if self._is_claim(sentence):
                # Normalize claim
                claim = self._normalize_claim(sentence)
                claims.append(claim)
        
        return claims
    
    def _is_claim(self, sentence: str) -> bool:
        """Check if a sentence contains a claim."""
        # Convert to lowercase for matching
        sentence_lower = sentence.lower()
        
        # Check for claim indicators
        for indicator in self.CLAIM_INDICATORS:
            if indicator in sentence_lower:
                return True
                
        # Check for subject-verb-object structure using spaCy
        doc = self.nlp(sentence)
        has_subject = False
        has_verb = False
        
        for token in doc:
            if token.dep_ in ['nsubj', 'nsubjpass']:
                has_subject = True
            if token.pos_ == 'VERB':
                has_verb = True
                
        return has_subject and has_verb
    
    def _normalize_claim(self, claim: str) -> str:
        """Normalize claim text."""
        # Remove extra whitespace
        claim = re.sub(r'\s+', ' ', claim.strip())
        
        # Remove quotes if the entire claim is quoted
        if claim.startswith('"') and claim.endswith('"'):
            claim = claim[1:-1].strip()
            
        return claim
        
    def extract_entities(self, claim: str) -> Dict[str, Any]:
        """
        Extract named entities from a claim.
        
        Args:
            claim (str): Individual claim text
            
        Returns:
            Dict[str, Any]: Dictionary of extracted entities and their types
        """
        doc = self.nlp(claim)
        entities = {}
        
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            entities[ent.label_].append({
                'text': ent.text,
                'start': ent.start_char,
                'end': ent.end_char,
                'label': ent.label_
            })
            
        return {
            'entities': entities,
            'noun_chunks': [chunk.text for chunk in doc.noun_chunks],
            'verbs': [token.text for token in doc if token.pos_ == 'VERB']
        } 