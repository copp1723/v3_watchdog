import unittest
import pytest
from unittest.mock import patch, mock_open, MagicMock, call
import json
from src.llm_engine import LLMEngine, redis_client

# Sample columns for testing
SAMPLE_COLUMNS = ["lead_source", "profit", "sold_price", "vehicle_year", "vehicle_make"]

# Sample column mapping response
SAMPLE_MAPPING = {
    "mapping": {
        "VehicleInformation": {
            "VIN": {"column": None, "confidence": 0.00},
            "VehicleYear": {"column": "vehicle_year", "confidence": 0.99},
            "VehicleMake": {"column": "vehicle_make", "confidence": 0.99}
        },
        "TransactionInformation": {
            "SalePrice": {"column": "sold_price", "confidence": 0.97},
            "TotalGross": {"column": "profit", "confidence": 0.98}
        },
        "SalesProcessInformation": {
            "LeadSource": {"column": "lead_source", "confidence": 1.00}
        }
    },
    "clarifications": [],
    "unmapped_columns": []
}

class TestLLMEngine(unittest.TestCase):

    @patch("builtins.open", new_callable=mock_open, read_data="Template content with {{columns}}")
    @patch("src.llm_engine.LLMEngine.get_llm_response")
    def test_map_columns_jeopardy_success(self, mock_get_llm_response, mock_file_open):
        """Test map_columns_jeopardy successfully reads template, calls LLM, and returns parsed response."""
        # Arrange
        engine = LLMEngine(use_mock=False) # Use mock=False to test the actual method logic
        test_columns = ["col1", "col2", "customer name"]
        expected_prompt = f"Template content with {str(test_columns)}"
        mock_response_dict = {
            "mapping": {"CustomerInformation": {"CustomerFirstName": {"column": "customer name", "confidence": 0.9}}},
            "clarifications": [],
            "unmapped_columns": [{"column": "col1", "notes": "unmapped"}]
        }
        mock_get_llm_response.return_value = mock_response_dict

        # Act
        result = engine.map_columns_jeopardy(test_columns)

        # Assert
        mock_file_open.assert_called_once_with("src/insights/prompts/column_mapping.tpl")
        mock_get_llm_response.assert_called_once_with(prompt=expected_prompt)
        self.assertEqual(result, mock_response_dict)

    @patch("builtins.open", side_effect=FileNotFoundError("Template not found"))
    @patch("src.llm_engine.LLMEngine.get_llm_response") # Mock to prevent actual call
    @patch("streamlit.error") # Mock streamlit error display
    def test_map_columns_jeopardy_file_not_found(self, mock_st_error, mock_get_llm_response, mock_file_open):
        """Test map_columns_jeopardy handles FileNotFoundError correctly."""
        # Arrange
        engine = LLMEngine(use_mock=False)
        test_columns = ["col1", "col2"]
        expected_error_result = {
            "mapping": {},
            "clarifications": [],
            "unmapped_columns": [
                {"column": "col1", "potential_category": None, "notes": "Template file missing"},
                {"column": "col2", "potential_category": None, "notes": "Template file missing"}
            ]
        }

        # Act
        result = engine.map_columns_jeopardy(test_columns)

        # Assert
        mock_file_open.assert_called_once_with("src/insights/prompts/column_mapping.tpl")
        mock_st_error.assert_called_once_with("Column mapping prompt template 'src/insights/prompts/column_mapping.tpl' not found.")
        mock_get_llm_response.assert_not_called()
        self.assertEqual(result, expected_error_result)

    @patch("builtins.open", new_callable=mock_open, read_data="Template content with {{columns}}")
    @patch("src.llm_engine.LLMEngine.get_llm_response", side_effect=Exception("LLM API error"))
    @patch("streamlit.error") # Mock streamlit error display
    def test_map_columns_jeopardy_llm_exception(self, mock_st_error, mock_get_llm_response, mock_file_open):
        """Test map_columns_jeopardy handles exceptions during LLM call."""
        # Arrange
        engine = LLMEngine(use_mock=False)
        test_columns = ["col1"]
        expected_error_result = {
            "mapping": {},
            "clarifications": [],
            "unmapped_columns": [
                {"column": "col1", "potential_category": None, "notes": "Error: LLM API error"}
            ]
        }

        # Act
        result = engine.map_columns_jeopardy(test_columns)

        # Assert
        mock_file_open.assert_called_once_with("src/insights/prompts/column_mapping.tpl")
        mock_get_llm_response.assert_called_once() # Called but raised exception
        mock_st_error.assert_called_once_with("Error during column mapping: LLM API error")
        self.assertEqual(result, expected_error_result)

if __name__ == '__main__':
    unittest.main() 


# Redis Caching Tests using pytest
@pytest.fixture
def mock_redis():
    """Create mock Redis client."""
    mock = MagicMock()
    # Mock successful ping
    mock.ping.return_value = True
    return mock

@pytest.fixture
def llm_engine_with_mock_redis(mock_redis):
    """Create LLM engine with mock Redis client."""
    with patch('src.llm_engine.redis_client', mock_redis):
        with patch('src.llm_engine.REDIS_CACHE_ENABLED', True):
            engine = LLMEngine(use_mock=True, use_redis_cache=True)
            engine.redis_client = mock_redis
            yield engine

def test_generate_columns_cache_key(llm_engine_with_mock_redis):
    """Test cache key generation."""
    engine = llm_engine_with_mock_redis
    
    # Test key generation with normal columns
    key1 = engine._generate_columns_cache_key(["a", "b", "c"])
    key2 = engine._generate_columns_cache_key(["a", "b", "c"])
    
    # Keys should be the same for same column set
    assert key1 == key2
    
    # Test different order, same columns
    key3 = engine._generate_columns_cache_key(["c", "a", "b"])
    assert key1 == key3  # Should generate same key regardless of order
    
    # Test different column set
    key4 = engine._generate_columns_cache_key(["d", "e", "f"])
    assert key1 != key4  # Should be different for different columns

def test_map_columns_cache_hit(llm_engine_with_mock_redis):
    """Test column mapping with Redis cache hit."""
    engine = llm_engine_with_mock_redis
    mock_redis = engine.redis_client
    
    # Setup cache hit
    cache_key = engine._generate_columns_cache_key(SAMPLE_COLUMNS)
    mock_redis.get.return_value = json.dumps(SAMPLE_MAPPING)
    
    # Mock the get_llm_response to ensure it's not called
    with patch.object(engine, 'get_llm_response') as mock_llm:
        # Call the function
        result = engine.map_columns_jeopardy(SAMPLE_COLUMNS)
        
        # Verify result matches the mock data
        assert result == SAMPLE_MAPPING
        
        # Verify Redis get was called
        mock_redis.get.assert_called_once()
        # Check that the key pattern is as expected
        assert mock_redis.get.call_args[0][0] == cache_key
        
        # Verify LLM was not called (cache hit)
        mock_llm.assert_not_called()
        
        # Verify cache stats
        assert engine.cache_stats["total"] == 1
        assert engine.cache_stats["hits"] == 1
        assert engine.cache_stats["misses"] == 0

def test_map_columns_cache_miss(llm_engine_with_mock_redis):
    """Test column mapping with Redis cache miss."""
    engine = llm_engine_with_mock_redis
    mock_redis = engine.redis_client
    
    # Setup cache miss
    mock_redis.get.return_value = None
    
    # Setup LLM response
    with patch.object(engine, 'get_llm_response', return_value=SAMPLE_MAPPING):
        # Mock file open to avoid file not found error
        with patch('builtins.open', MagicMock()):
            # Call the function
            result = engine.map_columns_jeopardy(SAMPLE_COLUMNS)
        
        # Verify result matches the expected response
        assert result == SAMPLE_MAPPING
        
        # Verify Redis get and set were called
        mock_redis.get.assert_called_once()
        mock_redis.set.assert_called_once()
        
        # Check the arguments to redis.set
        assert mock_redis.set.call_args[0][1] == json.dumps(SAMPLE_MAPPING)
        assert "ex" in mock_redis.set.call_args[1]  # Check TTL was set
        
        # Verify cache stats
        assert engine.cache_stats["total"] == 1
        assert engine.cache_stats["hits"] == 0
        assert engine.cache_stats["misses"] == 1

def test_cache_stats(llm_engine_with_mock_redis):
    """Test cache statistics."""
    engine = llm_engine_with_mock_redis
    mock_redis = engine.redis_client
    
    # Setup redis info mock
    mock_redis.info.return_value = {
        "used_memory_human": "1.2M",
        "redis_version": "6.2.7"
    }
    
    # Setup keys mock
    mock_redis.keys.return_value = ["key1", "key2", "key3"]
    
    # Manually set cache stats
    engine.cache_stats = {"hits": 10, "misses": 5, "total": 15}
    
    # Get stats
    stats = engine.get_cache_stats()
    
    # Verify base stats
    assert stats["hits"] == 10
    assert stats["misses"] == 5
    assert stats["total"] == 15
    assert stats["hit_rate"] == 10/15
    
    # Verify Redis info
    assert stats["cache_size"] == 3
    assert stats["redis_memory_used"] == "1.2M"
    assert stats["redis_version"] == "6.2.7"

def test_clear_cache(llm_engine_with_mock_redis):
    """Test clearing the cache."""
    engine = llm_engine_with_mock_redis
    mock_redis = engine.redis_client
    
    # Setup keys to delete
    cache_keys = [f"{engine.cache_prefix}key1", f"{engine.cache_prefix}key2"]
    mock_redis.keys.return_value = cache_keys
    mock_redis.delete.return_value = len(cache_keys)  # Number of keys deleted
    
    # Set some stats
    engine.cache_stats = {"hits": 10, "misses": 5, "total": 15}
    
    # Clear cache
    result = engine.clear_column_mapping_cache()
    
    # Verify result
    assert result is True
    
    # Verify Redis operations
    mock_redis.keys.assert_called_once_with(f"{engine.cache_prefix}*")
    mock_redis.delete.assert_called_once_with(*cache_keys)
    
    # Verify stats were reset
    assert engine.cache_stats == {"hits": 0, "misses": 0, "total": 0}

def test_redis_failure_fallback(llm_engine_with_mock_redis):
    """Test fallback to LLM when Redis fails."""
    engine = llm_engine_with_mock_redis
    mock_redis = engine.redis_client
    
    # Setup Redis to raise an exception on both get and set
    mock_redis.get.side_effect = Exception("Redis connection error")
    mock_redis.set.side_effect = Exception("Redis connection error")
    
    # Setup LLM response
    with patch.object(engine, 'get_llm_response', return_value=SAMPLE_MAPPING):
        # Mock file open to avoid file not found error
        with patch('builtins.open', MagicMock()):
            # Call the function
            result = engine.map_columns_jeopardy(SAMPLE_COLUMNS)
        
        # Verify result matches the expected response despite Redis failure
        assert result == SAMPLE_MAPPING
        
        # Verify Redis get was called
        mock_redis.get.assert_called_once()
        
        # Verify Redis set was called but failed silently
        mock_redis.set.assert_called_once()