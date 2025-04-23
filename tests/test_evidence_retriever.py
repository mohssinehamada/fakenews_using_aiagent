import pytest
import numpy as np
from unittest.mock import Mock, patch
import torch
from elasticsearch import Elasticsearch
from transformers import RobertaModel, RobertaTokenizer

from evidence.retriever import EvidenceRetriever

@pytest.fixture
def mock_es():
    mock = Mock(spec=Elasticsearch)
    mock.search.return_value = {
        'hits': {
            'hits': [
                {
                    '_source': {'text': 'Sample evidence 1', 'url': 'http://example1.com'},
                    '_score': 0.8
                },
                {
                    '_source': {'text': 'Sample evidence 2', 'url': 'http://example2.com'},
                    '_score': 0.6
                }
            ]
        }
    }
    return mock

@pytest.fixture
def mock_roberta():
    mock = Mock(spec=RobertaModel)
    mock.return_value = (
        torch.tensor([[[0.1, 0.2, 0.3]]]),  # Last hidden state
        torch.tensor([[0.4, 0.5, 0.6]])      # Pooler output
    )
    return mock

@pytest.fixture
def retriever(mock_es, mock_roberta):
    with patch('transformers.RobertaTokenizer.from_pretrained') as mock_tokenizer:
        mock_tokenizer.return_value = Mock(spec=RobertaTokenizer)
        mock_tokenizer.return_value.encode.return_value = torch.tensor([[1, 2, 3]])
        return EvidenceRetriever(
            es_client=mock_es,
            model_name='roberta-base',
            device='cpu'
        )

def test_init(mock_es):
    """Test initialization of EvidenceRetriever."""
    retriever = EvidenceRetriever(es_client=mock_es, model_name='roberta-base')
    assert isinstance(retriever.es_client, Elasticsearch)
    assert retriever.model_name == 'roberta-base'
    assert retriever.device == 'cuda' if torch.cuda.is_available() else 'cpu'

@pytest.mark.unit
def test_generate_embedding(retriever):
    """Test embedding generation for input text."""
    text = "Test input text"
    embedding = retriever.generate_embedding(text)
    
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (768,)  # RoBERTa base output dimension
    assert not np.any(np.isnan(embedding))

@pytest.mark.integration
def test_retrieve_evidence(retriever):
    """Test evidence retrieval from Elasticsearch."""
    query = "test query"
    results = retriever.retrieve_evidence(query, k=2)
    
    assert isinstance(results, list)
    assert len(results) == 2
    assert all(isinstance(r, dict) for r in results)
    assert all('text' in r and 'url' in r and 'score' in r for r in results)

@pytest.mark.unit
def test_rank_evidence(retriever):
    """Test evidence ranking functionality."""
    query = "test query"
    evidence = [
        {'text': 'Sample evidence 1', 'url': 'http://example1.com'},
        {'text': 'Sample evidence 2', 'url': 'http://example2.com'}
    ]
    
    ranked_evidence = retriever.rank_evidence(query, evidence)
    
    assert isinstance(ranked_evidence, list)
    assert len(ranked_evidence) == len(evidence)
    assert all(0 <= r['score'] <= 1 for r in ranked_evidence)
    # Check if results are sorted by score in descending order
    scores = [r['score'] for r in ranked_evidence]
    assert scores == sorted(scores, reverse=True)

@pytest.mark.integration
def test_faiss_integration(retriever):
    """Test FAISS index integration for semantic search."""
    # Mock FAISS index
    with patch('faiss.IndexFlatL2') as mock_faiss:
        mock_faiss.return_value.search.return_value = (
            np.array([[0.8, 0.6]]),  # Distances
            np.array([[0, 1]])       # Indices
        )
        
        query = "test query"
        evidence = [
            {'text': 'Sample evidence 1', 'url': 'http://example1.com'},
            {'text': 'Sample evidence 2', 'url': 'http://example2.com'}
        ]
        
        results = retriever.semantic_search(query, evidence, k=2)
        
        assert isinstance(results, list)
        assert len(results) == 2
        assert all('score' in r for r in results)
        assert all(isinstance(r['score'], float) for r in results)

@pytest.mark.unit
def test_error_handling(retriever):
    """Test error handling in the retriever."""
    # Test with empty query
    with pytest.raises(ValueError):
        retriever.retrieve_evidence("")
    
    # Test with invalid k value
    with pytest.raises(ValueError):
        retriever.retrieve_evidence("test", k=0)
    
    # Test Elasticsearch error handling
    retriever.es_client.search.side_effect = Exception("ES Error")
    with pytest.raises(Exception):
        retriever.retrieve_evidence("test")

@pytest.mark.unit
def test_score_normalization(retriever):
    """Test score normalization functionality."""
    scores = np.array([0.5, 0.8, 0.2])
    normalized = retriever._normalize_scores(scores)
    
    assert isinstance(normalized, np.ndarray)
    assert normalized.shape == scores.shape
    assert np.all((normalized >= 0) & (normalized <= 1))
    assert np.isclose(normalized.max(), 1.0)
    assert np.isclose(normalized.min(), 0.0)

@pytest.mark.integration
def test_api_response_structure(retriever):
    """Test the structure of API responses."""
    query = "test query"
    results = retriever.retrieve_evidence(query)
    
    for result in results:
        assert isinstance(result, dict)
        assert set(result.keys()) >= {'text', 'url', 'score'}
        assert isinstance(result['text'], str)
        assert isinstance(result['url'], str)
        assert isinstance(result['score'], float)
        assert 0 <= result['score'] <= 1 