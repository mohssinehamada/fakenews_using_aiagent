"""Tests for evidence retriever component."""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from elasticsearch import Elasticsearch
import faiss

from fakenews_agent.evidence.retriever import EvidenceRetriever

@pytest.fixture
def mock_es():
    """Mock Elasticsearch client."""
    es = Mock(spec=Elasticsearch)
    es.search.return_value = {
        'hits': {
            'hits': [
                {
                    '_id': 'doc1',
                    '_score': 0.8,
                    '_source': {
                        'text': 'Sample evidence 1',
                        'source': 'source1',
                        'date': '2024-01-01',
                        'embedding': [0.1] * 768,
                        'metadata': {'type': 'news'}
                    }
                },
                {
                    '_id': 'doc2',
                    '_score': 0.6,
                    '_source': {
                        'text': 'Sample evidence 2',
                        'source': 'source2',
                        'date': '2024-01-02',
                        'embedding': [0.2] * 768,
                        'metadata': {'type': 'article'}
                    }
                }
            ]
        }
    }
    return es

@pytest.fixture
def mock_roberta():
    """Mock RoBERTa model and tokenizer."""
    with patch('transformers.RobertaTokenizer.from_pretrained') as mock_tokenizer, \
         patch('transformers.RobertaModel.from_pretrained') as mock_model:
        
        # Mock tokenizer
        tokenizer = Mock()
        tokenizer.return_value = {
            'input_ids': np.zeros((1, 10)),
            'attention_mask': np.ones((1, 10))
        }
        mock_tokenizer.return_value = tokenizer
        
        # Mock model
        model = Mock()
        model.last_hidden_state = np.random.rand(1, 10, 768)
        outputs = Mock()
        outputs.last_hidden_state = model.last_hidden_state
        model.return_value = outputs
        mock_model.return_value = model
        
        yield mock_tokenizer, mock_model

@pytest.fixture
def retriever(mock_es, mock_roberta):
    """Initialize evidence retriever with mocks."""
    return EvidenceRetriever(es_host="mock:9200", use_faiss=True)

def test_init(mock_es, mock_roberta):
    """Test retriever initialization."""
    retriever = EvidenceRetriever(es_host="mock:9200", use_faiss=True)
    assert retriever.es is not None
    assert retriever.index_name == "evidence"
    assert retriever.use_faiss is True

def test_generate_embedding(retriever):
    """Test embedding generation."""
    text = "Test claim"
    embedding = retriever._generate_embedding(text)
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (768,)  # RoBERTa base dimension

@pytest.mark.parametrize("use_faiss", [True, False])
def test_retrieve_evidence(retriever, use_faiss):
    """Test evidence retrieval with and without FAISS."""
    retriever.use_faiss = use_faiss
    claim = "Test claim"
    results = retriever.retrieve_evidence(claim, top_k=2)
    
    assert len(results) == 2
    assert all(isinstance(r, dict) for r in results)
    assert all(k in results[0] for k in ['id', 'text', 'source', 'date', 'score', 'metadata'])
    assert results[0]['score'] >= results[1]['score']  # Check ranking

def test_rank_evidence(retriever):
    """Test evidence ranking."""
    claim = "Test claim"
    evidence_list = [
        "First piece of evidence",
        "Second piece of evidence",
        "Third piece of evidence"
    ]
    
    ranked = retriever.rank_evidence(claim, evidence_list)
    
    assert len(ranked) == len(evidence_list)
    assert all(isinstance(r, dict) for r in ranked)
    assert all(k in ranked[0] for k in ['text', 'score'])
    assert all(r['score'] <= 1.0 for r in ranked)  # Cosine similarity bounds
    assert all(r['score'] >= -1.0 for r in ranked)
    
    # Check sorting
    scores = [r['score'] for r in ranked]
    assert scores == sorted(scores, reverse=True)

def test_faiss_integration(retriever):
    """Test FAISS index loading and search."""
    # Mock FAISS index
    dim = 768
    index = faiss.IndexFlatL2(dim)
    vectors = np.random.rand(10, dim).astype('float32')
    index.add(vectors)
    
    retriever.faiss_index = index
    retriever.id_to_evidence = {i: f'doc{i}' for i in range(10)}
    
    claim = "Test claim"
    results = retriever.retrieve_evidence(claim, top_k=5)
    
    assert len(results) > 0
    assert all(isinstance(r['score'], float) for r in results)

@pytest.mark.parametrize("error_type", ['connection', 'timeout', 'not_found'])
def test_error_handling(retriever, error_type):
    """Test error handling for various failure scenarios."""
    from elasticsearch import ConnectionError, NotFoundError
    from elasticsearch.exceptions import ConnectionTimeout
    
    errors = {
        'connection': ConnectionError("Connection failed"),
        'timeout': ConnectionTimeout("Request timed out"),
        'not_found': NotFoundError("Index not found")
    }
    
    retriever.es.search.side_effect = errors[error_type]
    
    with pytest.raises((ConnectionError, ConnectionTimeout, NotFoundError)):
        retriever.retrieve_evidence("Test claim") 