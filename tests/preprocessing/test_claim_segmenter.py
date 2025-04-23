"""Tests for claim segmentation module."""

import pytest
from fakenews_agent.preprocessing.claim_segmenter import ClaimSegmenter
from tests.data.sample_texts import CLAIM_TEXTS

@pytest.fixture
def segmenter():
    """Create a ClaimSegmenter instance for testing."""
    return ClaimSegmenter()

def test_segment_simple_claim(segmenter):
    """Test segmentation of a simple claim."""
    claims = segmenter.segment_claims(CLAIM_TEXTS['simple'])
    assert len(claims) == 1
    assert "drinking coffee reduces the risk of heart disease" in claims[0].lower()

def test_segment_multiple_claims(segmenter):
    """Test segmentation of multiple claims."""
    claims = segmenter.segment_claims(CLAIM_TEXTS['multiple'])
    assert len(claims) >= 3
    assert any("climate change is accelerating" in claim.lower() for claim in claims)
    assert any("sea levels are rising" in claim.lower() for claim in claims)

def test_segment_no_claims(segmenter):
    """Test segmentation of text without claims."""
    claims = segmenter.segment_claims(CLAIM_TEXTS['no_claims'])
    assert len(claims) == 0

def test_segment_complex_text(segmenter):
    """Test segmentation of complex text with multiple claims and non-claims."""
    claims = segmenter.segment_claims(CLAIM_TEXTS['complex'])
    assert len(claims) >= 2
    assert any("ai systems can now predict weather patterns" in claim.lower() for claim in claims)

def test_extract_entities(segmenter):
    """Test entity extraction from claims."""
    entities = segmenter.extract_entities(CLAIM_TEXTS['simple'])
    
    # Check structure
    assert 'entities' in entities
    assert 'noun_chunks' in entities
    assert 'verbs' in entities
    
    # Check content
    assert len(entities['noun_chunks']) > 0
    assert len(entities['verbs']) > 0
    
    # Check specific entities
    all_entities = [ent['text'].lower() for entities_list in entities['entities'].values() 
                   for ent in entities_list]
    assert any("heart disease" in ent for ent in all_entities) 