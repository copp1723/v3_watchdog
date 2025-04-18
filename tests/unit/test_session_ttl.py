import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from src.utils.session import record_action

@pytest.fixture
def mock_redis():
    """Fixture providing a mocked Redis client."""
    mock = MagicMock()
    mock.rpush = MagicMock(return_value=True)
    mock.expire = MagicMock(return_value=True)
    return mock

@pytest.mark.parametrize("ttl_setting,expected_ttl", [
    ("31536000", 31536000),  # 365 days
    ("86400", 86400),        # 1 day
    (None, 31536000)         # Default
])
def test_session_ttl_enforcement(mock_redis, monkeypatch, ttl_setting, expected_ttl):
    """Test that session actions get correct TTL based on environment."""
    if ttl_setting is not None:
        monkeypatch.setenv("SESSION_TTL_SECONDS", ttl_setting)
    
    # Call with test data
    result = record_action("test_session", "test_action", {}, redis_client=mock_redis)
    
    # Verify behavior
    assert result is True
    mock_redis.rpush.assert_called_once()
    mock_redis.expire.assert_called_once_with("session:test_session", expected_ttl)

def test_session_ttl_failure_handling(mock_redis):
    """Test that TTL failures are properly handled."""
    # Configure mock to fail on expire
    mock_redis.expire.side_effect = Exception("Redis error")
    
    # Call with test data
    result = record_action("test_session", "test_action", {}, redis_client=mock_redis)
    
    # Verify behavior
    assert result is False
    mock_redis.rpush.assert_called_once()
    mock_redis.expire.assert_called_once() 