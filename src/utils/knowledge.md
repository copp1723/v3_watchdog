# Term Normalization System

## Overview
The term normalization system uses both rule-based and semantic similarity matching to standardize terms across the application.

## Features

### Embedding-Based Matching
- Uses sentence-transformers for semantic similarity
- Model: all-MiniLM-L6-v2
- Similarity threshold: 0.85
- Caches embeddings for performance

### Fallback Mechanism
- Falls back to fuzzy string matching if embeddings unavailable
- Uses rapidfuzz with token set ratio
- Fuzzy threshold: 0.8 (75% for personnel titles)

### Supported Categories
- Lead Sources (e.g., CarGurus, AutoTrader)
- Personnel Titles (e.g., SalesRep)
- Vehicle Types (e.g., SUV)

### Best Practices
- Check titles before names in personnel_titles category
- Use lower fuzzy thresholds (75%) for personnel titles
- Preprocess compound words (e.g., "car gurus" -> "cargurus")
- Handle URL variations in lead sources
- Normalize names with pattern matching first, then fallback to first/last extraction

## Usage

### Basic Usage
```python
from src.utils.term_normalizer import normalize

# Normalize DataFrame columns
df = normalize(df, columns=['LeadSource', 'VehicleType'])
```

### Custom Normalization
```python
from src.utils.term_normalizer import TermNormalizer

# Create custom normalizer
normalizer = TermNormalizer(
    similarity_threshold=0.9,  # More strict matching
    embedding_model='all-MiniLM-L6-v2'
)

# Normalize specific terms
normalized = normalizer.normalize_term('cargurus.com', 'lead_sources')
```

## Configuration
- Rules defined in config/normalization_rules.yml
- Supports both exact matches and semantic similarity
- Configurable thresholds per category
- Pattern-based name normalization