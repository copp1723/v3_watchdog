"""
Tests for Redis-based caching functionality.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import json
import hashlib

from src.utils.cache import DataFrameCache
from src.utils.data_io import load_data


class TestRedisCache:
    """Test suite for Redis caching implementation."""
    
    def test_cache_instantiation(self):
        """Test that the cache can be instantiated."""
        cache = DataFrameCache(host="localhost", port=6379)
        assert cache is not None
        assert cache.host == "localhost"
        assert cache.port == 6379
        
    def test_create_key(self):
        """Test that cache keys are created correctly."""
        cache = DataFrameCache()
        
        # Test with different content, same rules
        key1 = cache.create_key(b"content1", "1.0")
        key2 = cache.create_key(b"content2", "1.0")
        assert key1 != key2
        
        # Test with same content, different rules
        key3 = cache.create_key(b"content1", "1.0")
        key4 = cache.create_key(b"content1", "2.0")
        assert key3 != key4
        
        # Test with same content and rules
        key5 = cache.create_key(b"content1", "1.0")
        assert key1 == key5
        
    def test_cache_set_get(self):
        """Test setting and getting from cache."""
        # Create test DataFrame
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['a', 'b', 'c']
        })
        
        # Mock Redis client
        with patch('redis.Redis') as mock_redis_class:
            mock_client = MagicMock()
            mock_redis_class.return_value = mock_client
            mock_client.ping.return_value = True
            mock_client.setex.return_value = True
            
            # Set up the mock to return serialized data
            serialized_df = json.dumps({
                'data': {'A': [1, 2, 3], 'B': ['a', 'b', 'c']},
                'dtypes': {'A': 'int64', 'B': 'object'},
                'row_count': 3,
                'column_count': 2,
                'cached_at': '2023-01-01T00:00:00'
            })
            mock_client.get.return_value = serialized_df
            
            # Initialize cache with mock
            cache = DataFrameCache()
            assert cache.is_available
            
            # Test setting value
            key = "test_key"
            result = cache.set(key, df)
            assert result is True
            
            # Verify Redis setex was called
            mock_client.setex.assert_called_once()
            # Extract serialized data from call args
            args = mock_client.setex.call_args[0]
            assert args[0] == key
            assert isinstance(args[1], int)  # TTL
            
            # Test getting value
            retrieved = cache.get(key)
            assert retrieved is not None
            assert isinstance(retrieved, pd.DataFrame)
            assert len(retrieved) == 3
            assert list(retrieved.columns) == ['A', 'B']
        
    @patch('redis.Redis')
    def test_cache_miss(self, mock_redis):
        """Test behavior on cache miss."""
        # Setup mock
        mock_client = MagicMock()
        mock_redis.return_value = mock_client
        mock_client.ping.return_value = True
        mock_client.get.return_value = None
        
        # Initialize cache with mock
        cache = DataFrameCache()
        
        # Test getting non-existent value
        key = "nonexistent_key"
        result = cache.get(key)
        assert result is None
        mock_client.get.assert_called_once_with(key)
        
    @patch('redis.Redis')
    def test_cache_invalidation(self, mock_redis):
        """Test cache invalidation."""
        # Setup mock
        mock_client = MagicMock()
        mock_redis.return_value = mock_client
        mock_client.ping.return_value = True
        mock_client.delete.return_value = 1
        
        # Initialize cache with mock
        cache = DataFrameCache()
        
        # Test invalidating a key
        key = "test_key"
        result = cache.invalidate(key)
        assert result is True
        mock_client.delete.assert_called_once_with(key)
        
    def test_cache_connection_failure(self):
        """Test behavior when Redis connection fails."""
        # Mock Redis to raise exception on initialization
        with patch('redis.Redis', side_effect=Exception("Connection refused")):
            cache = DataFrameCache(host="nonexistent", port=1234)
            assert not cache.is_available
            
            # Test operations with unavailable cache
            df = pd.DataFrame({'A': [1, 2, 3]})
            assert cache.set("key", df) is False
            assert cache.get("key") is None
            assert cache.invalidate("key") is False
            assert cache.invalidate_all() == 0


@pytest.mark.skip("Skipping integration tests due to Streamlit caching issues in test environment")
class TestRedisCacheIntegration:
    """Integration tests for Redis caching with data_io functions."""
    
    @pytest.fixture
    def mock_streamlit_uploaded_file(self):
        """Create a mock Streamlit uploaded file."""
        class MockFile:
            def __init__(self, content, name):
                self.content = content
                self.name = name
                self._position = 0
                
            def read(self):
                return self.content
                
            def getvalue(self):
                return self.content
                
            def seek(self, position):
                self._position = position
                
        # Create a simple CSV content
        csv_content = """LeadSource,TotalGross,VIN,SaleDate,SalePrice
WebSource,1000,12345678901234567,2023-01-01,20000
Facebook,1500,23456789012345678,2023-01-02,25000
Google,2000,34567890123456789,2023-01-03,30000
"""
        return MockFile(csv_content.encode('utf-8'), "test_file.csv")
    
    @patch('src.utils.data_io.get_cache')
    @patch('streamlit.cache_data')
    def test_load_data_cache_hit(self, mock_cache_data, mock_get_cache, mock_streamlit_uploaded_file):
        """Test that load_data uses Redis cache when available."""
        # Create cache behavior
        mock_cache = MagicMock()
        mock_get_cache.return_value = mock_cache
        mock_cache.is_available = True
        
        # Setup cache hit
        df = pd.DataFrame({
            'LeadSource': ['WebSource', 'Facebook', 'Google'],
            'TotalGross': [1000, 1500, 2000],
            'VIN': ['12345678901234567', '23456789012345678', '34567890123456789'],
            'SaleDate': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'SalePrice': [20000, 25000, 30000]
        })
        mock_cache.get.return_value = df
        mock_cache.create_key.return_value = "test_cache_key"
        
        # Make cache_data decorator a no-op
        mock_cache_data.side_effect = lambda f: f
        
        # Mock the required modules/functions
        with patch('src.utils.encryption', create=True):
            # Import here to allow mocking
            from src.utils.data_io import load_data
            result = load_data(mock_streamlit_uploaded_file)
            
            # Verify cache was checked
            mock_cache.create_key.assert_called_once()
            mock_cache.get.assert_called_once()
            
            # Verify result is from cache
            assert result is df
        
    @patch('src.utils.data_io.get_cache')
    @patch('streamlit.cache_data')
    def test_load_data_cache_miss(self, mock_cache_data, mock_get_cache, mock_streamlit_uploaded_file):
        """Test that load_data caches result on cache miss."""
        # Create cache behavior
        mock_cache = MagicMock()
        mock_get_cache.return_value = mock_cache
        mock_cache.is_available = True
        mock_cache.get.return_value = None
        mock_cache.create_key.return_value = "test_cache_key"
        
        # Make cache_data decorator a no-op
        mock_cache_data.side_effect = lambda f: f
        
        # Patch normalize functions to simplify test
        with patch('src.utils.data_io.normalize_columns', return_value=pd.DataFrame({
                'LeadSource': ['WebSource', 'Facebook', 'Google'],
                'TotalGross': [1000, 1500, 2000],
                'VIN': ['12345678901234567', '23456789012345678', '34567890123456789'],
                'SaleDate': ['2023-01-01', '2023-01-02', '2023-01-03'],
                'SalePrice': [20000, 25000, 30000]
            })), \
            patch('src.utils.data_io.normalize', return_value=pd.DataFrame({
                'LeadSource': ['WebSource', 'Facebook', 'Google'],
                'TotalGross': [1000, 1500, 2000],
                'VIN': ['12345678901234567', '23456789012345678', '34567890123456789'],
                'SaleDate': ['2023-01-01', '2023-01-02', '2023-01-03'],
                'SalePrice': [20000, 25000, 30000]
            })), \
            patch('src.utils.data_io.validate_schema', return_value=pd.DataFrame({
                'LeadSource': ['WebSource', 'Facebook', 'Google'],
                'TotalGross': [1000, 1500, 2000],
                'VIN': ['12345678901234567', '23456789012345678', '34567890123456789'],
                'SaleDate': ['2023-01-01', '2023-01-02', '2023-01-03'],
                'SalePrice': [20000, 25000, 30000]
            })), \
            patch('src.utils.encryption', create=True), \
            patch('src.utils.data_io.hashlib'), \
            patch('builtins.open', create=True), \
            patch('os.makedirs'):
            
            # Mock the encryption module
            import sys
            encryption_mock = MagicMock()
            encryption_mock.encrypt_bytes.return_value = b"encrypted"
            sys.modules['src.utils.encryption'] = encryption_mock
            
            # Import here to allow mocking
            from src.utils.data_io import load_data
            result = load_data(mock_streamlit_uploaded_file)
            
            # Verify cache was checked and then set
            mock_cache.create_key.assert_called_once()
            mock_cache.get.assert_called_once()
            mock_cache.set.assert_called_once()
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 3