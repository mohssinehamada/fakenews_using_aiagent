"""Tests for text processing module."""

import pytest
from fakenews_agent.preprocessing.text_processor import TextProcessor
from tests.data.sample_texts import DIRTY_TEXTS

@pytest.fixture
def processor():
    """Create a TextProcessor instance for testing."""
    return TextProcessor()

def test_clean_urls(processor):
    """Test cleaning of URLs from text."""
    cleaned = processor.clean_text(DIRTY_TEXTS['urls'])
    assert 'http' not in cleaned
    assert 'example.com' not in cleaned
    assert 'test.org' not in cleaned

def test_clean_emails(processor):
    """Test cleaning of email addresses from text."""
    cleaned = processor.clean_text(DIRTY_TEXTS['emails'])
    assert '@' not in cleaned
    assert 'test@example.com' not in cleaned
    assert 'support@test.org' not in cleaned

def test_clean_whitespace(processor):
    """Test cleaning of excessive whitespace."""
    cleaned = processor.clean_text(DIRTY_TEXTS['whitespace'])
    assert '    ' not in cleaned
    assert '\n' not in cleaned
    assert '\t' not in cleaned
    # Words should be separated by single spaces
    assert 'has multiple spaces' in cleaned.lower()

def test_clean_quotes(processor):
    """Test normalization of quotation marks."""
    cleaned = processor.clean_text(DIRTY_TEXTS['quotes'])
    # All quotes should be normalized to standard quotes
    assert '"' in cleaned
    assert '"' not in cleaned
    assert '"' not in cleaned
    assert ''' not in cleaned
    assert "'" in cleaned

def test_clean_mixed(processor):
    """Test cleaning of text with multiple issues."""
    cleaned = processor.clean_text(DIRTY_TEXTS['mixed'])
    assert 'http' not in cleaned
    assert '@' not in cleaned
    assert '    ' not in cleaned
    assert '\n' not in cleaned
    assert 'breakthrough' in cleaned.lower()

def test_extract_features(processor):
    """Test feature extraction from text."""
    features = processor.extract_features(DIRTY_TEXTS['mixed'])
    
    # Check structure
    assert 'basic_stats' in features
    assert 'pos_distribution' in features
    assert 'named_entities' in features
    assert 'dependency_patterns' in features
    assert 'sentiment' in features
    
    # Check basic stats
    stats = features['basic_stats']
    assert stats['word_count'] > 0
    assert stats['sentence_count'] > 0
    assert 0 <= stats['lexical_diversity'] <= 1
    
    # Check sentiment
    sentiment = features['sentiment']
    assert isinstance(sentiment['positive'], int)
    assert isinstance(sentiment['negative'], int)
    assert isinstance(sentiment['neutral'], int)

def test_get_important_phrases(processor):
    """Test extraction of important phrases."""
    phrases = processor.get_important_phrases(DIRTY_TEXTS['mixed'], top_k=3)
    assert len(phrases) <= 3
    assert all(isinstance(phrase, str) for phrase in phrases)
    assert all(len(phrase.split()) >= 1 for phrase in phrases) 