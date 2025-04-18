"""
Unit tests for Redis caching in insight generator.
"""

import pytest
import pandas as pd
import json
from unittest.mock import patch, MagicMock, call
from datetime import datetime, timedelta

from src.insights.insight_generator import InsightGenerator, redis_client

# Sample data for testing
SAMPLE_DATA = pd.DataFrame({
    'SaleDate': ['2023-01-01', '2023-01-02', '2023-01-03'],
    'SalePrice': [25000, 30000, 28000],
    'LeadSource': ['Website', 'CarGurus', 'Walk-in'],
    'VIN': ['VIN001', 'VIN002', 'VIN003'],
    'TotalGross': [2500, 3000, 2800]
})

# Sample insight result
SAMPLE_INSIGHT = {
    "summary": "CarGurus had the highest sale price of $30,000",
    "title": "Highest Sale Price Analysis",
    "is_error": False,
    "is_direct_calculation": True,
    "timestamp": datetime.now().isoformat()
}

@pytest.fixture
def mock_redis():
    """Create mock Redis client."""
    mock = MagicMock()
    # Mock successful ping
    mock.ping.return_value = True
    return mock

@pytest.fixture
def generator_with_mock_redis(mock_redis):
    """Create insight generator with mock Redis client."""
    with patch('src.insights.insight_generator.redis_client', mock_redis):
        generator = InsightGenerator(use_redis_cache=True)
        generator.redis_client = mock_redis
        generator.use_redis_cache = True
        
        # Mock intent manager to avoid actual generation
        mock_intent_manager = MagicMock()
        mock_intent_manager.generate_insight.return_value = SAMPLE_INSIGHT
        generator.intent_manager = mock_intent_manager
        
        yield generator

def test_insight_generator_init_redis_disabled():
    """Test insight generator initialization with Redis disabled."""
    generator = InsightGenerator(use_redis_cache=False)
    assert generator.use_redis_cache is False
    assert generator.redis_client is None

def test_insight_generator_init_redis_enabled(mock_redis):
    """Test insight generator initialization with Redis enabled."""
    with patch('src.insights.insight_generator.redis_client', mock_redis):
        with patch('src.insights.insight_generator.REDIS_CACHE_ENABLED', True):
            generator = InsightGenerator(use_redis_cache=True)
            assert generator.use_redis_cache is True
            assert generator.redis_client is not None
            assert generator.cache_prefix == "watchdog:insights:"
            assert generator.cache_stats == {"hits": 0, "misses": 0, "total": 0}

def test_should_cache_insight():
    """Test the logic that determines if an insight should be cached."""
    generator = InsightGenerator(use_redis_cache=False)
    
    # Test "hot" insights that should be cached
    hot_prompts = [
        "Show me the top performers",
        "What are the best lead sources?",
        "Which rep had the highest sales last month?",
        "Show me the lowest gross profit vehicles",
        "What was the average sale price?",
        "Give me a summary of this year's performance",
        "Compare Q1 performance to last year"
    ]
    
    for prompt in hot_prompts:
        assert generator._should_cache_insight(prompt) is True, f"Should cache: {prompt}"
    
    # Test "cool" insights that shouldn't be cached
    cool_prompts = [
        "Help me understand this data",
        "What does this mean?",
        "Tell me about my dealership",
        "Show me all sales",
        "Generate a custom report"
    ]
    
    for prompt in cool_prompts:
        assert generator._should_cache_insight(prompt) is False, f"Should not cache: {prompt}"

def test_generate_insight_cache_hit(generator_with_mock_redis):
    """Test generate_insight with Redis cache hit."""
    generator = generator_with_mock_redis
    mock_redis = generator.redis_client
    
    # Setup cache hit
    cached_result = {
        "summary": "Cached result from Redis",
        "is_error": False,
        "timestamp": "2023-04-17T12:00:00Z",
        "is_direct_calculation": True
    }
    mock_redis.get.return_value = json.dumps(cached_result)
    
    # Call generate_insight with a "hot" prompt
    result = generator.generate_insight(
        "Show me the top performers",
        SAMPLE_DATA,
        async_mode=False
    )
    
    # Verify cache was checked
    assert generator.cache_stats["total"] == 1
    assert generator.cache_stats["hits"] == 1
    
    # Verify result matches cached value with cache metadata added
    assert result["summary"] == cached_result["summary"]
    assert result["cached"] is True
    assert "cache_timestamp" in result
    
    # Verify intent manager was not called (used cache instead)
    generator.intent_manager.generate_insight.assert_not_called()

def test_generate_insight_cache_miss(generator_with_mock_redis):
    """Test generate_insight with Redis cache miss."""
    generator = generator_with_mock_redis
    mock_redis = generator.redis_client
    
    # Setup cache miss
    mock_redis.get.return_value = None
    
    # Call generate_insight with a hot prompt
    prompt = "Show me the highest sale price"
    result = generator.generate_insight(
        prompt,
        SAMPLE_DATA,
        async_mode=False
    )
    
    # Verify cache was checked
    assert generator.cache_stats["total"] == 1
    assert generator.cache_stats["misses"] == 1
    
    # Verify result comes from intent manager
    assert result["summary"] == SAMPLE_INSIGHT["summary"]
    
    # Verify intent manager was called
    generator.intent_manager.generate_insight.assert_called_once_with(prompt, SAMPLE_DATA)
    
    # Verify cache was updated
    mock_redis.set.assert_called_once()
    # First arg should be the key
    assert mock_redis.set.call_args[0][0].startswith(generator.cache_prefix)
    # Second arg should be serialized result
    assert isinstance(mock_redis.set.call_args[0][1], str)
    # TTL should be set
    assert mock_redis.set.call_args[1].get("ex") is not None

def test_dataframe_signature():
    """Test that DataFrame signatures are consistent and unique."""
    generator = InsightGenerator(use_redis_cache=False)
    
    # Create similar but different DataFrames
    df1 = pd.DataFrame({
        'SaleDate': ['2023-01-01', '2023-01-02'],
        'SalePrice': [25000, 30000],
        'LeadSource': ['Website', 'CarGurus']
    })
    
    df2 = pd.DataFrame({
        'SaleDate': ['2023-01-01', '2023-01-02'],
        'SalePrice': [25000, 30000],
        'LeadSource': ['Website', 'CarGurus']
    })
    
    df3 = pd.DataFrame({
        'SaleDate': ['2023-01-01', '2023-01-02'],
        'SalePrice': [25000, 30001],  # One value changed
        'LeadSource': ['Website', 'CarGurus']
    })
    
    df4 = pd.DataFrame({
        'SaleDate': ['2023-01-01', '2023-01-02'],
        'SalePrice': [25000, 30000],
        'LeadSource': ['Website', 'CarGurus'],
        'Extra': [1, 2]  # Extra column
    })
    
    # Same DataFrames should have the same signature
    assert generator._get_dataframe_signature(df1) == generator._get_dataframe_signature(df2)
    
    # Different DataFrames should have different signatures
    assert generator._get_dataframe_signature(df1) != generator._get_dataframe_signature(df3)
    assert generator._get_dataframe_signature(df1) != generator._get_dataframe_signature(df4)

def test_cache_key_generation():
    """Test cache key generation for insights."""
    generator = InsightGenerator(use_redis_cache=False)
    
    # Test that same prompt + data gets same key
    prompt1 = "Show me the top performers"
    prompt2 = "  show me the TOP performers  "  # Different spacing and case
    df_sig = "test_signature"
    
    key1 = generator._get_cache_key(prompt1, df_sig)
    key2 = generator._get_cache_key(prompt2, df_sig)
    
    # Keys should match due to normalization
    assert key1 == key2
    
    # Different prompts should get different keys
    prompt3 = "Show me the worst performers"
    key3 = generator._get_cache_key(prompt3, df_sig)
    assert key1 != key3
    
    # Different data should get different keys
    df_sig2 = "different_signature"
    key4 = generator._get_cache_key(prompt1, df_sig2)
    assert key1 != key4