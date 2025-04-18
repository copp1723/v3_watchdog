"""
Unit tests for the insight feedback system.
"""

import pytest
from unittest.mock import Mock, patch
import json
from datetime import datetime
from src.insights.feedback import FeedbackManager

@pytest.fixture
def mock_redis():
    """Create a mock Redis client."""
    redis = Mock()
    redis.ping.return_value = True
    redis.rpush.return_value = True
    redis.lrange.return_value = []
    return redis

@pytest.fixture
def feedback_manager(mock_redis):
    """Create a FeedbackManager with mocked Redis."""
    with patch('src.insights.feedback.redis.Redis', return_value=mock_redis):
        manager = FeedbackManager()
        manager.redis = mock_redis
        return manager

@pytest.fixture
def sample_feedback():
    """Create sample feedback data."""
    return {
        "insight_id": "test_insight_123",
        "feedback_type": "helpful",
        "user_id": "user_123",
        "session_id": "session_456",
        "comment": "Great insight!",
        "metadata": {"source": "test"}
    }

def test_record_feedback_success(feedback_manager, sample_feedback, mock_redis):
    """Test successful feedback recording."""
    result = feedback_manager.record_feedback(
        insight_id=sample_feedback["insight_id"],
        feedback_type=sample_feedback["feedback_type"],
        user_id=sample_feedback["user_id"],
        session_id=sample_feedback["session_id"],
        comment=sample_feedback["comment"],
        metadata=sample_feedback["metadata"]
    )
    
    assert result is True
    mock_redis.rpush.assert_called_once()
    
    # Verify the stored data structure
    stored_data = json.loads(mock_redis.rpush.call_args[0][1])
    assert stored_data["insight_id"] == sample_feedback["insight_id"]
    assert stored_data["feedback_type"] == sample_feedback["feedback_type"]
    assert "timestamp" in stored_data

@patch('src.insights.feedback.sentry_sdk.capture_exception')
def test_record_feedback_redis_error(mock_capture_exception, feedback_manager, sample_feedback, mock_redis):
    """Test feedback recording with Redis error."""
    mock_redis.rpush.side_effect = Exception("Redis error")
    
    result = feedback_manager.record_feedback(
        insight_id=sample_feedback["insight_id"],
        feedback_type=sample_feedback["feedback_type"],
        user_id=sample_feedback["user_id"],
        session_id=sample_feedback["session_id"]
    )
    
    assert result is False
    mock_capture_exception.assert_called_once()

def test_get_feedback_with_filters(feedback_manager, mock_redis):
    """Test feedback retrieval with filters."""
    # Mock stored feedback
    stored_feedback = [
        {
            "insight_id": "insight_1",
            "session_id": "session_1",
            "feedback_type": "helpful",
            "timestamp": datetime.now().isoformat()
        },
        {
            "insight_id": "insight_2",
            "session_id": "session_1",
            "feedback_type": "not_helpful",
            "timestamp": datetime.now().isoformat()
        }
    ]
    mock_redis.lrange.return_value = [json.dumps(f) for f in stored_feedback]
    
    # Test filtering by insight_id
    results = feedback_manager.get_feedback(insight_id="insight_1")
    assert len(results) == 1
    assert results[0]["insight_id"] == "insight_1"
    
    # Test filtering by session_id
    results = feedback_manager.get_feedback(session_id="session_1")
    assert len(results) == 2

def test_get_feedback_stats(feedback_manager, mock_redis):
    """Test feedback statistics calculation."""
    # Mock stored feedback
    stored_feedback = [
        {"feedback_type": "helpful"},
        {"feedback_type": "helpful"},
        {"feedback_type": "not_helpful"}
    ]
    mock_redis.lrange.return_value = [json.dumps(f) for f in stored_feedback]
    
    stats = feedback_manager.get_feedback_stats()
    
    assert stats["total_feedback"] == 3
    assert stats["counts"]["helpful"] == 2
    assert stats["counts"]["not_helpful"] == 1
    assert stats["percentages"]["helpful"] == pytest.approx(66.67, rel=0.01)
    assert stats["percentages"]["not_helpful"] == pytest.approx(33.33, rel=0.01)

def test_fallback_to_memory_storage():
    """Test fallback to in-memory storage when Redis is unavailable."""
    with patch('src.insights.feedback.redis.Redis') as mock_redis_class:
        mock_redis_class.side_effect = Exception("Redis unavailable")
        
        manager = FeedbackManager()
        assert manager.storage_available is False
        
        # Record should still work
        result = manager.record_feedback(
            insight_id="test",
            feedback_type="helpful",
            user_id="user",
            session_id="session"
        )
        assert result is True
        
        # Should be stored in memory
        feedback = manager.get_feedback()
        assert len(feedback) == 1
        assert feedback[0]["insight_id"] == "test"

@patch('src.insights.feedback.log_audit_event')
def test_audit_logging(mock_audit_log, feedback_manager, sample_feedback):
    """Test audit logging of feedback events."""
    feedback_manager.record_feedback(
        insight_id=sample_feedback["insight_id"],
        feedback_type=sample_feedback["feedback_type"],
        user_id=sample_feedback["user_id"],
        session_id=sample_feedback["session_id"]
    )
    
    mock_audit_log.assert_called_once_with(
        event_name="insight_feedback_recorded",
        user_id=sample_feedback["user_id"],
        session_id=sample_feedback["session_id"],
        details={
            "insight_id": sample_feedback["insight_id"],
            "feedback_type": sample_feedback["feedback_type"]
        }
    )