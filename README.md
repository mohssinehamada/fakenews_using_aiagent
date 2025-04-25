# Fake News Detection System

A machine learning-based system for detecting fake news using evidence retrieval and analysis.

## Features

- Evidence indexing and retrieval using Elasticsearch
- Semantic search using FAISS
- Text embedding using RoBERTa
- Evidence ranking and scoring
- Comprehensive test suite

## Requirements

- Python 3.8+
- Elasticsearch 8.0+
- FAISS
- PyTorch
- Transformers

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/fakenews_using_aiagent.git
cd fakenews_using_aiagent
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
pip install -r tests/requirements-test.txt
```

4. Set up Elasticsearch:
- Install and start Elasticsearch
- Create the evidence index (this will be done automatically on first run)

## Usage

The system is currently in development. The following components are available:

1. Evidence Indexer:
```python
from fakenews_agent.evidence.indexer import EvidenceIndexer

indexer = EvidenceIndexer()
evidence_id = indexer.index_evidence({
    'text': 'Your evidence text',
    'source': 'Source name',
    'date': '2024-01-01',
    'metadata': {'type': 'news'}
})
```

2. Evidence Retriever:
```python
from fakenews_agent.evidence.retriever import EvidenceRetriever

retriever = EvidenceRetriever()
results = retriever.retrieve_evidence('Your claim', top_k=5)
```

## Testing

Run the test suite:
```bash
pytest
```


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 
