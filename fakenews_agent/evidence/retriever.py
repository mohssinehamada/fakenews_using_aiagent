"""
Evidence retrieval module using Elasticsearch/FAISS.
"""

from typing import List, Dict, Any, Optional
from elasticsearch import Elasticsearch
import faiss
import numpy as np
import torch
from transformers import RobertaModel, RobertaTokenizer

class EvidenceRetriever:
    """Retrieve relevant evidence for claims using vector similarity search."""
    
    def __init__(self, es_host: str = "localhost:9200", index_name: str = "evidence",
                 use_faiss: bool = True):
        """
        Initialize evidence retriever.
        
        Args:
            es_host (str): Elasticsearch host and port
            index_name (str): Name of the evidence index
            use_faiss (bool): Whether to use FAISS for vector similarity
        """
        # Initialize Elasticsearch
        self.es = Elasticsearch([es_host])
        self.index_name = index_name
        self.use_faiss = use_faiss
        
        # Initialize RoBERTa for embeddings
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.model = RobertaModel.from_pretrained('roberta-base')
        self.model.eval()
        
        # Load FAISS index if available
        if use_faiss:
            self._load_faiss_index()
            
    def _load_faiss_index(self):
        """Load or initialize FAISS index."""
        # Get a sample document to determine embedding dimension
        result = self.es.search(
            index=self.index_name,
            body={"query": {"match_all": {}}, "size": 1}
        )
        
        if result['hits']['hits']:
            embedding = result['hits']['hits'][0]['_source']['embedding']
            dim = len(embedding)
            self.faiss_index = faiss.IndexFlatL2(dim)
            self.id_to_evidence = {}
            
            # Load all embeddings into FAISS
            results = self.es.search(
                index=self.index_name,
                body={"query": {"match_all": {}}, "size": 10000}
            )
            
            embeddings = []
            for i, hit in enumerate(results['hits']['hits']):
                embedding = np.array(hit['_source']['embedding'], dtype=np.float32)
                embeddings.append(embedding)
                self.id_to_evidence[i] = hit['_id']
                
            if embeddings:
                self.faiss_index.add(np.vstack(embeddings))
                
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text using RoBERTa."""
        inputs = self.tokenizer(text, return_tensors="pt", 
                              max_length=512, truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use CLS token embedding
            embedding = outputs.last_hidden_state[:, 0, :].numpy()
        return embedding.flatten()
        
    def retrieve_evidence(self, claim: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve top-k most relevant evidence for a claim.
        
        Args:
            claim (str): Input claim
            top_k (int): Number of evidence to retrieve
            
        Returns:
            List[Dict[str, Any]]: List of retrieved evidence with scores
        """
        results = []
        
        # Generate claim embedding
        claim_embedding = self._generate_embedding(claim)
        
        # Semantic search using FAISS
        if self.use_faiss and self.faiss_index.ntotal > 0:
            distances, indices = self.faiss_index.search(
                claim_embedding.reshape(1, -1), 
                min(top_k, self.faiss_index.ntotal)
            )
            
            # Get evidence details from Elasticsearch
            for i, idx in enumerate(indices[0]):
                es_id = self.id_to_evidence[idx]
                es_doc = self.es.get(index=self.index_name, id=es_id)
                
                results.append({
                    'id': es_id,
                    'text': es_doc['_source']['text'],
                    'source': es_doc['_source'].get('source'),
                    'date': es_doc['_source'].get('date'),
                    'score': float(1 / (1 + distances[0][i])),  # Convert distance to similarity score
                    'metadata': es_doc['_source'].get('metadata', {})
                })
                
        # Keyword search using Elasticsearch
        es_results = self.es.search(
            index=self.index_name,
            body={
                "query": {
                    "multi_match": {
                        "query": claim,
                        "fields": ["text^3", "source"],  # Boost text field
                        "fuzziness": "AUTO"
                    }
                },
                "size": top_k
            }
        )
        
        # Combine results from Elasticsearch
        for hit in es_results['hits']['hits']:
            # Check if this evidence was already found by FAISS
            if not any(r['id'] == hit['_id'] for r in results):
                results.append({
                    'id': hit['_id'],
                    'text': hit['_source']['text'],
                    'source': hit['_source'].get('source'),
                    'date': hit['_source'].get('date'),
                    'score': hit['_score'],
                    'metadata': hit['_source'].get('metadata', {})
                })
                
        # Sort by score and return top_k
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
        
    def rank_evidence(self, claim: str, evidence_list: List[str]) -> List[Dict[str, Any]]:
        """
        Rank retrieved evidence by relevance to claim.
        
        Args:
            claim (str): Input claim
            evidence_list (List[str]): List of evidence to rank
            
        Returns:
            List[Dict[str, Any]]: Ranked evidence with scores
        """
        claim_embedding = self._generate_embedding(claim)
        ranked_evidence = []
        
        for evidence in evidence_list:
            # Generate evidence embedding
            evidence_embedding = self._generate_embedding(evidence)
            
            # Calculate cosine similarity
            similarity = np.dot(claim_embedding, evidence_embedding) / (
                np.linalg.norm(claim_embedding) * np.linalg.norm(evidence_embedding)
            )
            
            ranked_evidence.append({
                'text': evidence,
                'score': float(similarity)
            })
            
        # Sort by similarity score
        ranked_evidence.sort(key=lambda x: x['score'], reverse=True)
        return ranked_evidence 