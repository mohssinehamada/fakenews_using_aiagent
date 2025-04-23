"""Configuration settings for the fake news detection system."""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
API_CONFIG = {
    'host': os.getenv('API_HOST', 'localhost'),
    'port': int(os.getenv('API_PORT', 5000)),
    'debug': os.getenv('API_DEBUG', 'False').lower() == 'true',
    'workers': int(os.getenv('API_WORKERS', 4))
}

# Model Configuration
MODEL_CONFIG = {
    'roberta_model': os.getenv('ROBERTA_MODEL', 'roberta-base'),
    'spacy_model': os.getenv('SPACY_MODEL', 'en_core_web_sm'),
    'gat_hidden_dim': int(os.getenv('GAT_HIDDEN_DIM', 64)),
    'fusion_hidden_dim': int(os.getenv('FUSION_HIDDEN_DIM', 64))
}

# Evidence Retrieval Configuration
EVIDENCE_CONFIG = {
    'elasticsearch': {
        'host': os.getenv('ES_HOST', 'localhost'),
        'port': int(os.getenv('ES_PORT', 9200)),
        'index_name': os.getenv('ES_INDEX', 'evidence'),
        'timeout': int(os.getenv('ES_TIMEOUT', 30))
    },
    'faiss': {
        'dimension': int(os.getenv('FAISS_DIM', 768)),  # RoBERTa embedding dimension
        'index_type': os.getenv('FAISS_INDEX_TYPE', 'Flat'),
        'metric': os.getenv('FAISS_METRIC', 'L2')
    }
}

# Preprocessing Configuration
PREPROCESSING_CONFIG = {
    'max_sequence_length': int(os.getenv('MAX_SEQ_LENGTH', 512)),
    'claim_batch_size': int(os.getenv('CLAIM_BATCH_SIZE', 32)),
    'min_claim_length': int(os.getenv('MIN_CLAIM_LENGTH', 10))
}

# Training Configuration
TRAINING_CONFIG = {
    'batch_size': int(os.getenv('BATCH_SIZE', 16)),
    'learning_rate': float(os.getenv('LEARNING_RATE', 2e-5)),
    'num_epochs': int(os.getenv('NUM_EPOCHS', 10)),
    'warmup_steps': int(os.getenv('WARMUP_STEPS', 500)),
    'max_grad_norm': float(os.getenv('MAX_GRAD_NORM', 1.0)),
    'weight_decay': float(os.getenv('WEIGHT_DECAY', 0.01))
}

# Paths Configuration
PATHS = {
    'data_dir': os.getenv('DATA_DIR', 'data'),
    'model_dir': os.getenv('MODEL_DIR', 'models'),
    'cache_dir': os.getenv('CACHE_DIR', 'cache'),
    'log_dir': os.getenv('LOG_DIR', 'logs')
}

# Create directories if they don't exist
for directory in PATHS.values():
    os.makedirs(directory, exist_ok=True) 