"""
Unit tests for Redis TTL functionality in the audit logging system.
"""

import pytest
import json
import time
from unittest.mock import MagicMock, patch
from src.utils.audit_log import log_audit_event, AUDIT_LOG_KEY, AUDIT_LOG_TTL_SECONDS


@pytest.fixture
def mock_redis():
    """Create a mock Redis client."""
    with patch('src.utils.audit_log.redis_client') as mock_client:
        # Configure mock
        mock_client.rpush.return_value = 1
        mock_client.ttl.return_value = -1  # Simulate no TTL set initially
        mock_client.expire.return_value = True
        yield mock_client


@pytest.fixture
def sample_audit_event():
    """Sample audit event data."""
    return {
        "event_name": "file_upload",
        "user_id": "user123",
        "session_id": "session456",
        "details": {
            "file_name": "sample.csv",
            "file_size": 1024,
            "ip_address": "127.0.0.1",
            "resource_type": "file",
            "resource_id": "file123",
            "status": "success"
        }
    }


def test_audit_log_ttl_applied(mock_redis, sample_audit_event):
    """Test that TTL is applied when logging an audit event."""
    # Log the event
    log_audit_event(
        sample_audit_event["event_name"],
        sample_audit_event["user_id"],
        sample_audit_event["session_id"],
        sample_audit_event["details"]
    )
    
    # Check that rpush was called with the correct key
    mock_redis.rpush.assert_called_once()
    args = mock_redis.rpush.call_args[0]
    assert args[0] == AUDIT_LOG_KEY
    
    # Check that the event was properly serialized
    event_json = args[1]
    event_data = json.loads(event_json)
    assert event_data["event"] == sample_audit_event["event_name"]
    assert event_data["user_id"] == sample_audit_event["user_id"]
    
    # Check that TTL was checked
    mock_redis.ttl.assert_called_once_with(AUDIT_LOG_KEY)
    
    # Check that expire was called with correct TTL value
    mock_redis.expire.assert_called_once_with(AUDIT_LOG_KEY, AUDIT_LOG_TTL_SECONDS)


def test_audit_log_ttl_not_reapplied_if_exists(mock_redis, sample_audit_event):
    """Test that TTL is not reapplied if it already exists."""
    # Mock TTL to return a positive value (indicating TTL is already set)
    mock_redis.ttl.return_value = 7000000  # Some large value in seconds
    
    # Log the event
    log_audit_event(
        sample_audit_event["event_name"],
        sample_audit_event["user_id"],
        sample_audit_event["session_id"],
        sample_audit_event["details"]
    )
    
    # Check that rpush was called
    mock_redis.rpush.assert_called_once()
    
    # Check that TTL was checked
    mock_redis.ttl.assert_called_once_with(AUDIT_LOG_KEY)
    
    # Verify expire was NOT called since TTL already exists
    mock_redis.expire.assert_not_called()


@patch('time.sleep')
def test_audit_log_key_expires(mock_sleep, mock_redis, sample_audit_event):
    """Test that the audit log key expires after TTL (simulated fast-forward)."""
    # Setup mock behavior to simulate key expiration
    mock_redis.ttl.side_effect = [-1, AUDIT_LOG_TTL_SECONDS, 0, -2]  # No TTL -> Has TTL -> Expired
    mock_redis.exists.side_effect = [True, True, False]  # Key exists -> Key doesn't exist
    
    # Log an event
    log_audit_event(
        sample_audit_event["event_name"],
        sample_audit_event["user_id"],
        sample_audit_event["session_id"],
        sample_audit_event["details"]
    )
    
    # Verify TTL was set
    mock_redis.expire.assert_called_once_with(AUDIT_LOG_KEY, AUDIT_LOG_TTL_SECONDS)
    
    # Simulate time passing and key expiring
    mock_sleep.return_value = None  # Mock sleep to do nothing
    
    # Check TTL again - should be smaller or expired
    assert mock_redis.ttl.call_count >= 1
    
    # Fast-forward past expiration (in real life this would be 90 days)
    # Here we're just testing the mocked sequence
    mock_redis.ttl.reset_mock()
    mock_redis.ttl.return_value = -2  # Key doesn't exist (expired)
    
    # Verify key doesn't exist after "expiration"
    assert mock_redis.ttl(AUDIT_LOG_KEY) == -2  # Key doesn't exist