"""
Unit tests for Redis caching in term normalizer.
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock, call

from src.utils.term_normalizer import TermNormalizer, redis_client

# Sample data for testing
SAMPLE_DATA = pd.DataFrame({
    'LeadSource': ['Website', 'CarGurus', 'Walk-in', 'Facebook'],
    'SalesRep': ['John Doe', 'Jane Smith', 'J. Doe', 'Robert Johnson']
})

@pytest.fixture
def mock_redis():
    """Create mock Redis client."""
    mock = MagicMock()
    # Mock successful ping
    mock.ping.return_value = True
    return mock

@pytest.fixture
def normalizer_with_mock_redis(mock_redis):
    """Create term normalizer with mock Redis client."""
    with patch('src.utils.term_normalizer.redis_client', mock_redis):
        normalizer = TermNormalizer(use_redis_cache=True)
        normalizer.redis_client = mock_redis
        normalizer.use_redis_cache = True
        yield normalizer

def test_term_normalizer_init_redis_disabled():
    """Test term normalizer initialization with Redis disabled."""
    normalizer = TermNormalizer(use_redis_cache=False)
    assert normalizer.use_redis_cache is False
    assert normalizer.redis_client is None

def test_term_normalizer_init_redis_enabled(mock_redis):
    """Test term normalizer initialization with Redis enabled."""
    with patch('src.utils.term_normalizer.redis_client', mock_redis):
        with patch('src.utils.term_normalizer.REDIS_CACHE_ENABLED', True):
            normalizer = TermNormalizer(use_redis_cache=True)
            assert normalizer.use_redis_cache is True
            assert normalizer.redis_client is not None
            assert normalizer.cache_prefix == "watchdog:term_normalizer:"
            assert normalizer.cache_stats == {"hits": 0, "misses": 0, "total": 0}

def test_normalize_term_cache_hit(normalizer_with_mock_redis):
    """Test normalize_term with Redis cache hit."""
    # Setup mock Redis to return a cached value
    normalizer = normalizer_with_mock_redis
    mock_redis = normalizer.redis_client
    mock_redis.get.return_value = "Cached Value"
    
    # Call normalize_term
    result = normalizer.normalize_term("test term", "LeadSource")
    
    # Verify cache was checked
    assert normalizer.cache_stats["total"] == 1
    assert normalizer.cache_stats["hits"] == 1
    
    # Verify result is cache value
    assert result == "Cached Value"
    
    # Verify Redis get was called with correct key
    mock_redis.get.assert_called_once()
    
    # The exact key will depend on the hash implementation
    assert mock_redis.get.call_args[0][0].startswith(normalizer.cache_prefix)

def test_normalize_term_cache_miss(normalizer_with_mock_redis):
    """Test normalize_term with Redis cache miss."""
    # Setup mock Redis to return None (cache miss)
    normalizer = normalizer_with_mock_redis
    mock_redis = normalizer.redis_client
    mock_redis.get.return_value = None
    
    # Mock the _compute_similarity method to always return a high value
    with patch.object(normalizer, '_compute_similarity', return_value=0.9):
        # Add a simple rule for testing
        normalizer.rules = {
            "LeadSource": {
                "Website": ["web", "site", "online"],
                "CarGurus": ["car gurus", "cargurus"],
                "Walk-in": ["walkin", "walk in"]
            }
        }
        
        # Call normalize_term
        result = normalizer.normalize_term("online store", "LeadSource")
    
    # Verify cache was checked
    assert normalizer.cache_stats["total"] == 1
    assert normalizer.cache_stats["misses"] == 1
    assert normalizer.cache_stats["hits"] == 0
    
    # Verify result is from normalization logic
    assert result == "Website"
    
    # Verify Redis get was called
    mock_redis.get.assert_called_once()
    
    # Verify Redis set was called to update cache
    mock_redis.set.assert_called_once()
    # First arg should be the key
    assert mock_redis.set.call_args[0][0].startswith(normalizer.cache_prefix)
    # Second arg should be the normalized value
    assert mock_redis.set.call_args[0][1] == "Website"
    # Third arg should be TTL
    assert "ex" in mock_redis.set.call_args[1]

def test_normalize_column_with_cache(normalizer_with_mock_redis):
    """Test normalizing a column with Redis cache."""
    normalizer = normalizer_with_mock_redis
    mock_redis = normalizer.redis_client
    
    # Setup cache behavior
    cache_values = {
        # These keys are mock patterns - actual keys would be hashed
        "watchdog:term_normalizer:LeadSource:website": "Website",
        "watchdog:term_normalizer:LeadSource:cargurus": "CarGurus"
    }
    
    def mock_get(key):
        # Simple mock of Redis get that returns values from our dict
        # This is a simplified version that doesn't account for hashing
        for k, v in cache_values.items():
            if k in key:
                return v
        return None
    
    # Set up mock Redis get to use our function
    mock_redis.get.side_effect = mock_get
    
    # Mock _compute_similarity to always return high value for testing
    with patch.object(normalizer, '_compute_similarity', return_value=0.9):
        # Add rules for testing
        normalizer.rules = {
            "LeadSource": {
                "Website": ["web", "site", "online"],
                "CarGurus": ["car gurus", "cargurus"],
                "Walk-in": ["walkin", "walk in"]
            }
        }
        
        # Normalize column
        df = pd.DataFrame({
            'LeadSource': ['website', 'cargurus', 'walk in', 'facebook']
        })
        
        result_df = normalizer.normalize_column(df, "LeadSource")
    
    # Verify cache stats
    assert normalizer.cache_stats["total"] == 4  # One for each row
    
    # Verify result has normalized values
    assert result_df["LeadSource"].tolist() == ["Website", "CarGurus", "Walk-in", "facebook"]
    
    # Verify Redis set was called to update cache for items not in cache
    # Should be called for "walk in" and "facebook"
    assert mock_redis.set.call_count == 2

def test_cache_pruning(normalizer_with_mock_redis):
    """Test Redis cache pruning logic."""
    normalizer = normalizer_with_mock_redis
    mock_redis = normalizer.redis_client
    
    # Setup mock Redis to return a list of keys
    keys = [f"watchdog:term_normalizer:key{i}" for i in range(1100)]
    mock_redis.keys.return_value = keys
    
    # Setup TTL values for sorting
    ttl_values = {key: i % 500 for i, key in enumerate(keys)}
    mock_redis.ttl.side_effect = lambda key: ttl_values.get(key, 0)
    
    # Call pruning method
    with patch('src.utils.term_normalizer.REDIS_MAX_ENTRIES', 1000):
        normalizer._check_and_prune_cache()
    
    # Verify keys method was called
    mock_redis.keys.assert_called_once_with("watchdog:term_normalizer:*")
    
    # Verify TTL was checked for each key
    assert mock_redis.ttl.call_count == 1100
    
    # Verify delete was called for 100 keys (1100 - 1000)
    assert mock_redis.delete.call_count == 100