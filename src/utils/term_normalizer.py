"""
Term normalization module for Watchdog AI.

Provides functionality for normalizing common term variations based on YAML rules
and semantic similarity using embeddings.
"""

import yaml
import os
import pandas as pd
import logging
import sentry_sdk
import json
import hashlib
from typing import Dict, List, Any, Optional, Union, Tuple
from functools import lru_cache
from rapidfuzz import fuzz
import numpy as np
from sentence_transformers import SentenceTransformer
from redis import Redis
from datetime import timedelta

# Configure Redis client
REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.environ.get('REDIS_PORT', 6379))
REDIS_DB = int(os.environ.get('REDIS_DB', 0))
REDIS_PASSWORD = os.environ.get('REDIS_PASSWORD', None)
REDIS_MAX_ENTRIES = int(os.environ.get('REDIS_NORMALIZER_MAX_ENTRIES', 1000))
REDIS_CACHE_TTL = int(os.environ.get('REDIS_NORMALIZER_TTL', 86400))  # 1 day default
REDIS_CACHE_ENABLED = os.environ.get('REDIS_CACHE_ENABLED', 'true').lower() == 'true'

logger = logging.getLogger(__name__)

# Create Redis client
redis_client = None
try:
    if REDIS_CACHE_ENABLED:
        redis_client = Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            password=REDIS_PASSWORD,
            socket_timeout=5,
            decode_responses=True
        )
        redis_client.ping()  # Test connection
        logger.info("Redis connection established for term normalization cache")
except Exception as e:
    redis_client = None
    logger.warning(f"Failed to connect to Redis for term normalization cache: {str(e)}")
    logger.warning("Term normalization will use in-memory LRU cache only")

class TermNormalizer:
    """
    Normalizes common term variations based on rules and semantic similarity.
    """
    
    def __init__(self, config_path="config/normalization_rules.yml", 
                 embedding_model="all-MiniLM-L6-v2",
                 default_columns=None,
                 similarity_threshold=0.85,
                 use_redis_cache=True):
        """
        Initialize the term normalizer with rules and embeddings.
        
        Args:
            config_path: Path to the YAML rules file
            embedding_model: Name of the sentence-transformers model to use
            default_columns: List of default columns to normalize
            similarity_threshold: Threshold for semantic similarity matching
            use_redis_cache: Whether to use Redis for caching normalization results
        """
        # Load rules
        self.config_path = config_path
        self.fuzzy_threshold, self.rules = self._load_normalization_rules()
        
        # Initialize embedding model
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
            self.use_embeddings = True
        except Exception as e:
            logger.warning(f"Failed to load embedding model: {e}. Falling back to fuzzy matching.")
            self.embedding_model = None
            self.use_embeddings = False
        
        # Set thresholds
        self.similarity_threshold = similarity_threshold
        
        # Cache for embeddings
        self._embedding_cache = {}
        
        # Default columns
        if default_columns is None:
            self.default_columns = list(self.rules.keys())
        else:
            self.default_columns = default_columns
        
        # Initialize Redis cache if available and enabled
        self.use_redis_cache = use_redis_cache and REDIS_CACHE_ENABLED and redis_client is not None
        self.redis_client = redis_client if self.use_redis_cache else None
        self.cache_prefix = "watchdog:term_normalizer:"
        self.cache_stats = {"hits": 0, "misses": 0, "total": 0}
        
        if self.use_redis_cache:
            # Check cache size and prune if needed
            self._check_and_prune_cache()
            logger.info("Redis cache enabled for term normalization")
            
        logger.info(f"Initialized TermNormalizer with {len(self.rules)} rules")
        if self.use_embeddings:
            logger.info(f"Using embedding model: {embedding_model}")
    
    def _load_normalization_rules(self) -> Tuple[float, Dict]:
        """Load rules from YAML file."""
        try:
            with open(self.config_path, "r") as f:
                rules = yaml.safe_load(f)
            fuzzy_threshold = rules.get("fuzzy_threshold", 0.8)
            # Remove fuzzy_threshold key to leave only category definitions
            categories = {k: v for k, v in rules.items() if k != "fuzzy_threshold"}
            return fuzzy_threshold, categories
        except Exception as e:
            logger.error(f"Error loading rules: {e}")
            return 0.8, {}
            
    def _check_and_prune_cache(self):
        """Check Redis cache size and prune if needed."""
        if not self.redis_client:
            return
            
        try:
            # Get all cache keys
            pattern = f"{self.cache_prefix}*"
            keys = self.redis_client.keys(pattern)
            
            # Check if we need to prune
            if len(keys) > REDIS_MAX_ENTRIES:
                logger.info(f"Pruning Redis term normalization cache ({len(keys)} entries > {REDIS_MAX_ENTRIES} max)")
                
                # Randomly remove oldest entries until we're under the limit
                # We're using TTL-based random selection with zipf distribution to favor keeping newer entries
                keys_to_delete = len(keys) - REDIS_MAX_ENTRIES
                
                # Get all keys with TTL
                key_ttls = []
                for key in keys:
                    ttl = self.redis_client.ttl(key)
                    key_ttls.append((key, ttl))
                
                # Sort by TTL (ascending)
                key_ttls.sort(key=lambda x: x[1] if x[1] > 0 else 2147483647)
                
                # Delete oldest keys
                for i in range(min(keys_to_delete, len(key_ttls))):
                    self.redis_client.delete(key_ttls[i][0])
                    
                logger.info(f"Pruned {min(keys_to_delete, len(key_ttls))} oldest entries from Redis cache")
        except Exception as e:
            logger.error(f"Error while pruning Redis cache: {e}")
    
    def _get_cache_key(self, term: str, category: str) -> str:
        """Generate a Redis cache key for a term and category."""
        # Use hash to ensure valid key characters
        term_hash = hashlib.md5(term.lower().strip().encode()).hexdigest()
        return f"{self.cache_prefix}{category}:{term_hash}"
    
    def _check_cache(self, term: str, category: str) -> Optional[str]:
        """Check if a normalized term is in the Redis cache."""
        if not self.redis_client:
            return None
            
        try:
            # Increment total counter
            self.cache_stats["total"] += 1
            
            # Get cache key
            cache_key = self._get_cache_key(term, category)
            
            # Check Redis
            cached_value = self.redis_client.get(cache_key)
            
            if cached_value:
                # Cache hit
                self.cache_stats["hits"] += 1
                # Log every 100 hits
                if self.cache_stats["hits"] % 100 == 0:
                    hit_rate = self.cache_stats["hits"] / self.cache_stats["total"]
                    logger.info(f"Term normalizer cache hit rate: {hit_rate:.2f} ({self.cache_stats['hits']}/{self.cache_stats['total']})")
                return cached_value
            else:
                # Cache miss
                self.cache_stats["misses"] += 1
                return None
        except Exception as e:
            logger.warning(f"Error checking Redis cache: {e}")
            return None
    
    def _update_cache(self, term: str, category: str, normalized: str) -> None:
        """Store a normalized term in the Redis cache."""
        if not self.redis_client:
            return
            
        try:
            # Get cache key
            cache_key = self._get_cache_key(term, category)
            
            # Store in Redis with TTL
            self.redis_client.set(
                cache_key, 
                normalized,
                ex=REDIS_CACHE_TTL
            )
        except Exception as e:
            logger.warning(f"Error updating Redis cache: {e}")
    
    @lru_cache(maxsize=1024)
    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for a text string with caching."""
        if not self.use_embeddings:
            return None
            
        try:
            if text in self._embedding_cache:
                return self._embedding_cache[text]
            
            embedding = self.embedding_model.encode([text])[0]
            self._embedding_cache[text] = embedding
            return embedding
            
        except Exception as e:
            logger.warning(f"Error generating embedding for '{text}': {e}")
            return None
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for similarity comparison."""
        # Convert to string and normalize
        text = str(text).lower().strip()
        
        # Remove common punctuation and normalize spaces
        text = text.replace('.', ' ').replace('-', ' ').replace('_', ' ')
        text = text.replace('/', ' ').replace('\\', ' ').replace('&', ' and ')
        text = text.replace(',', ' ').replace('(', ' ').replace(')', ' ')
        
        # Remove common URL parts
        text = text.replace('.com', '').replace('www.', '')
        text = text.replace('http://', '').replace('https://', '')
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        # Remove common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        words = text.split()
        words = [w for w in words if w not in stop_words]
        text = ' '.join(words)
        
        # Join compound words
        text = text.replace('car gurus', 'cargurus')
        text = text.replace('auto trader', 'autotrader')
        text = text.replace('sales representative', 'salesrep')
        text = text.replace('sales person', 'salesrep')
        text = text.replace('sales rep', 'salesrep')
        text = text.replace('sales associate', 'salesrep')
        text = text.replace('sales consultant', 'salesrep')
        text = text.replace('sales personnel', 'salesrep')
        
        return text
    
    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two texts using embeddings or fuzzy matching."""
        # Preprocess inputs
        text1 = self._preprocess_text(text1)
        text2 = self._preprocess_text(text2)
        
        # Check for exact match first
        if text1 == text2:
            return 1.0
            
        # Check for substring match
        if text1 in text2 or text2 in text1:
            return 0.9
            
        # Check for URL variations
        text1_clean = text1.replace('www.', '').replace('.com', '')
        text2_clean = text2.replace('www.', '').replace('.com', '')
        if text1_clean == text2_clean:
            return 0.95
            
        # Check for fuzzy match first as it's faster
        fuzzy_sim = fuzz.token_set_ratio(text1, text2) / 100.0
        if fuzzy_sim > 0.85:
            return fuzzy_sim
            
        # Try partial token matching
        text1_tokens = set(text1.split())
        text2_tokens = set(text2.split())
        if text1_tokens & text2_tokens:  # If there are common tokens
            return max(fuzzy_sim, 0.85)
            
        # Try fuzzy token matching
        token_sim = max(
            fuzz.token_set_ratio(t1, t2) / 100.0
            for t1 in text1_tokens
            for t2 in text2_tokens
        )
        if token_sim > 0.85:
            return token_sim
            
        if self.use_embeddings:
            try:
                emb1 = self._get_embedding(text1)
                emb2 = self._get_embedding(text2)
                if emb1 is not None and emb2 is not None:
                    # Compute cosine similarity
                    sim = float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
                    # Boost similarity for partial matches
                    if text1_clean in text2_clean or text2_clean in text1_clean:
                        sim = max(sim, 0.85)
                    # Boost similarity for token matches
                    if text1_tokens & text2_tokens:
                        sim = max(sim, 0.85)
                    # Boost similarity for known patterns
                    if ('car' in text1 and 'guru' in text2) or ('car' in text2 and 'guru' in text1):
                        sim = max(sim, 0.85)
                    if ('auto' in text1 and 'trader' in text2) or ('auto' in text2 and 'trader' in text1):
                        sim = max(sim, 0.85)
                    if ('sales' in text1 and 'rep' in text2) or ('sales' in text2 and 'rep' in text1):
                        sim = max(sim, 0.85)
                    return sim
            except Exception as e:
                logger.warning(f"Error computing embedding similarity: {e}")
        
        return fuzzy_sim
    
    def _normalize_name(self, name: str) -> str:
        """Normalize a person's name."""
        # Convert to string and normalize
        name = str(name).lower().strip()
        
        # Remove punctuation
        name = name.replace('.', ' ').replace('-', ' ').replace('_', ' ')
        
        # Normalize whitespace
        name = ' '.join(name.split())
        
        # Check name patterns from rules
        if 'name_patterns' in self.rules:
            best_score = 0
            best_match = None
            
            for pattern_group in self.rules['name_patterns']:
                for pattern in pattern_group:
                    score = fuzz.token_set_ratio(name, pattern)
                    if score > best_score:
                        best_score = score
                        best_match = pattern_group[0]
            
            if best_score > 75:  # Lower threshold for better matching
                return best_match
        
        # Extract first and last name
        parts = name.split()
        if len(parts) >= 2:
            # Handle abbreviated first names
            first = parts[0]
            if len(first) == 1:
                first = f"{first}."
            # Use first and last name only
            return f"{first} {parts[-1]}"
        
        return name
    
    def normalize_term(self, term: str, category: str) -> str:
        """
        Normalize a term using rules and semantic similarity.
        
        Args:
            term: Term to normalize
            category: Category for rule lookup
            
        Returns:
            Normalized term
        """
        # Handle empty or None values
        if pd.isna(term) or term == '':
            return term
            
        # Convert to string and normalize
        term_lower = self._preprocess_text(term)
        
        # Check Redis cache first if enabled
        if self.use_redis_cache:
            cached_result = self._check_cache(term_lower, category)
            if cached_result:
                return cached_result
        
        # Special handling for personnel titles
        if category == 'personnel_titles':
            # Check if it's a title first
            for canonical, synonyms in self.rules[category].items():
                if term_lower == self._preprocess_text(canonical):
                    return canonical
                for syn in synonyms:
                    if term_lower == self._preprocess_text(syn):
                        return canonical
                # Try fuzzy matching for titles
                for syn in [canonical] + synonyms:
                    if self._compute_similarity(term_lower, syn) > 0.85:
                        return canonical
                # Try fuzzy matching with lower threshold for titles
                for syn in [canonical] + synonyms:
                    if fuzz.token_set_ratio(term_lower, syn) > 75:  # Lower threshold for titles
                        return canonical
            
            # If not a title, check if it's a name
            if any(x in str(term).lower() for x in ['name', 'rep', 'person']):
                return self._normalize_name(term)
            # Check if it's a name pattern
            if 'name_patterns' in self.rules:
                for pattern_group in self.rules['name_patterns']:
                    for pattern in pattern_group:
                        if fuzz.token_set_ratio(term_lower, pattern) > 75:
                            return self._normalize_name(term)
        
        # Check rules
        if category in self.rules:
            # Exact match check
            for canonical, synonyms in self.rules[category].items():
                if term_lower == self._preprocess_text(canonical):
                    if self.use_redis_cache:
                        self._update_cache(term_lower, category, canonical)
                    return canonical
                for syn in synonyms:
                    if term_lower == self._preprocess_text(syn):
                        if self.use_redis_cache:
                            self._update_cache(term_lower, category, canonical)
                        return canonical
            
            # Similarity match
            best_match = None
            best_score = 0
            
            for canonical, synonyms in self.rules[category].items():
                # Check canonical term
                score = self._compute_similarity(term_lower, canonical)
                if score > best_score:
                    best_score = score
                    best_match = canonical
                
                # Check synonyms
                for syn in synonyms:
                    score = self._compute_similarity(term_lower, syn)
                    if score > best_score:
                        best_score = score
                        best_match = canonical
            
            if best_score >= self.similarity_threshold:
                if self.use_redis_cache:
                    self._update_cache(term_lower, category, best_match)
                return best_match
        
        # No match found - return original term
        if self.use_redis_cache:
            self._update_cache(term_lower, category, str(term))
        return str(term)
    
    def normalize_column(self, df: pd.DataFrame, column: str, category: str = None, inplace: bool = False) -> pd.DataFrame:
        """
        Normalize values in a specific column.
        
        Args:
            df: DataFrame to process
            column: Column name to normalize
            category: Category for rule lookup (defaults to column name)
            inplace: Whether to modify the original DataFrame
            
        Returns:
            DataFrame with normalized values
        """
        if not inplace:
            df = df.copy()
            
        if column not in df.columns:
            logger.warning(f"Column '{column}' not found in DataFrame")
            return df
            
        # Use column name as category if not specified
        if category is None:
            category = column.lower()
        
        # Track normalization
        sentry_sdk.set_tag("normalization_rules_version", getattr(self, 'rules_version', None))
        sentry_sdk.set_tag("normalization_step", "column_value")
        
        # Normalize values
        df[column] = df[column].apply(lambda x: self.normalize_term(x, category))
        
        # Log results
        unique_values = df[column].dropna().unique()
        logger.info(f"Normalized '{column}' column: {len(unique_values)} unique values")
        
        return df
    
    def normalize_dataframe(self, df: pd.DataFrame, columns: Optional[Dict[str, str]] = None,
                          inplace: bool = False) -> pd.DataFrame:
        """
        Normalize multiple columns in a DataFrame.
        
        Args:
            df: DataFrame to process
            columns: Dict mapping column names to categories, or list of columns
            inplace: Whether to modify the original DataFrame
            
        Returns:
            DataFrame with normalized values
        """
        if not inplace:
            df = df.copy()
            
        if not columns:
            # Default column mappings
            columns = {
                'LeadSource': 'lead_sources',
                'Lead_Source': 'lead_sources',
                'lead_source': 'lead_sources',
                'Source': 'lead_sources',
                'SalesRep': 'personnel_titles',
                'Sales_Rep': 'personnel_titles',
                'sales_rep': 'personnel_titles',
                'Salesperson': 'personnel_titles',
                'VehicleType': 'vehicle_types',
                'Vehicle_Type': 'vehicle_types',
                'vehicle_type': 'vehicle_types'
            }
            
            # Filter to columns that exist in DataFrame
            columns = {col: cat for col, cat in columns.items() if col in df.columns}
            
        if isinstance(columns, (list, tuple)):
            # Convert list to dict using column names as categories
            columns = {col: col.lower() for col in columns}
            
        if not columns:
            logger.warning("No normalizable columns found")
            return df
            
        # Normalize each column
        for column, category in columns.items():
            df = self.normalize_column(df, column, category, inplace=True)
            
        return df

# Create global instance
normalizer = TermNormalizer()

def normalize(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Normalize DataFrame using global normalizer instance."""
    return normalizer.normalize(df, columns)

# Backward compatibility
normalize_terms = normalize
