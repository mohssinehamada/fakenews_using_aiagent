"""
Evidence indexing module for storing and managing evidence.
"""

from typing import List, Dict, Any, Optional
from elasticsearch import Elasticsearch
import faiss
import numpy as np
import torch
from transformers import RobertaModel, RobertaTokenizer
import json
import hashlib

class EvidenceIndexer:
    """Index and manage evidence for retrieval."""
    
    def __init__(self, es_host: str = "localhost:9200", index_name: str = "evidence",
                 embedding_dim: int = 768, use_faiss: bool = True):
        """
        Initialize evidence indexer.
        
        Args:
            es_host (str): Elasticsearch host and port
            index_name (str): Name of the evidence index
            embedding_dim (int): Dimension of evidence embeddings
            use_faiss (bool): Whether to use FAISS for vector similarity search
        """
        # Initialize Elasticsearch
        self.es = Elasticsearch([es_host])
        self.index_name = index_name
        
        # Create Elasticsearch index if it doesn't exist
        if not self.es.indices.exists(index=index_name):
            self.es.indices.create(
                index=index_name,
                body={
                    "mappings": {
                        "properties": {
                            "text": {"type": "text"},
                            "source": {"type": "keyword"},
                            "date": {"type": "date"},
                            "embedding": {"type": "dense_vector", "dims": embedding_dim},
                            "metadata": {"type": "object"}
                        }
                    }
                }
            )
        
        # Initialize FAISS
        self.use_faiss = use_faiss
        if use_faiss:
            self.faiss_index = faiss.IndexFlatL2(embedding_dim)
            self.id_to_evidence = {}  # Map FAISS IDs to evidence
        
        # Initialize RoBERTa for embeddings
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.model = RobertaModel.from_pretrained('roberta-base')
        self.model.eval()
        
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text using RoBERTa."""
        inputs = self.tokenizer(text, return_tensors="pt", 
                              max_length=512, truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use CLS token embedding
            embedding = outputs.last_hidden_state[:, 0, :].numpy()
        return embedding.flatten()
        
    def _generate_id(self, evidence: Dict[str, Any]) -> str:
        """Generate unique ID for evidence."""
        content = json.dumps(evidence, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()
        
    def index_evidence(self, evidence: Dict[str, Any]) -> str:
        """
        Index a new piece of evidence.
        
        Args:
            evidence (Dict[str, Any]): Evidence to index with fields:
                - text: str, the evidence text
                - source: str, source of the evidence
                - date: str, date of the evidence (ISO format)
                - metadata: Dict, additional metadata
            
        Returns:
            str: ID of the indexed evidence
        """
        # Generate embedding
        embedding = self._generate_embedding(evidence['text'])
        evidence['embedding'] = embedding.tolist()
        
        # Generate ID
        evidence_id = self._generate_id(evidence)
        
        # Index in Elasticsearch
        self.es.index(
            index=self.index_name,
            id=evidence_id,
            body=evidence
        )
        
        # Index in FAISS if enabled
        if self.use_faiss:
            faiss_id = len(self.id_to_evidence)
            self.faiss_index.add(embedding.reshape(1, -1))
            self.id_to_evidence[faiss_id] = evidence_id
            
        return evidence_id
        
    def update_evidence(self, evidence_id: str, evidence: Dict[str, Any]) -> bool:
        """
        Update existing evidence.
        
        Args:
            evidence_id (str): ID of evidence to update
            evidence (Dict[str, Any]): Updated evidence
            
        Returns:
            bool: Success status
        """
        try:
            # Check if evidence exists
            if not self.es.exists(index=self.index_name, id=evidence_id):
                return False
                
            # Generate new embedding if text changed
            if 'text' in evidence:
                embedding = self._generate_embedding(evidence['text'])
                evidence['embedding'] = embedding.tolist()
            
            # Update in Elasticsearch
            self.es.update(
                index=self.index_name,
                id=evidence_id,
                body={"doc": evidence}
            )
            
            # Update FAISS if enabled and text changed
            if self.use_faiss and 'text' in evidence:
                # For simplicity, we'll rebuild the FAISS index
                # In production, you might want a more efficient update strategy
                self._rebuild_faiss_index()
                
            return True
            
        except Exception as e:
            print(f"Error updating evidence: {e}")
            return False
            
    def _rebuild_faiss_index(self):
        """Rebuild FAISS index from Elasticsearch data."""
        if not self.use_faiss:
            return
            
        # Clear existing index
        self.faiss_index = faiss.IndexFlatL2(self.faiss_index.d)
        self.id_to_evidence.clear()
        
        # Fetch all documents from Elasticsearch
        results = self.es.search(
            index=self.index_name,
            body={"query": {"match_all": {}}, "size": 10000}  # Adjust size as needed
        )
        
        # Rebuild FAISS index
        embeddings = []
        for i, hit in enumerate(results['hits']['hits']):
            embedding = np.array(hit['_source']['embedding'], dtype=np.float32)
            embeddings.append(embedding)
            self.id_to_evidence[i] = hit['_id']
            
        if embeddings:
            self.faiss_index.add(np.vstack(embeddings)) 