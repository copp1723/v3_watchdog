"""
Tests for the Digest System Module.

This module contains tests for the DigestSystem class and related components.
"""

import os
import json
import pytest
import tempfile
import shutil
from datetime import datetime, timedelta
from typing import Dict, Any

from src.digest_system import (
    DigestSystem,
    DigestFrequency,
    DigestType,
    DigestFormat,
    DigestRecipient,
    DigestDelivery
)

@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def digest_system(temp_data_dir):
    """Create a DigestSystem instance with a temporary data directory."""
    return DigestSystem(data_dir=temp_data_dir)

def test_add_recipient(digest_system):
    """Test adding a new recipient."""
    # Add a recipient
    result = digest_system.add_recipient(
        user_id="user1",
        name="Test User",
        email="test@example.com",
        slack_id="U123456",
        frequency=DigestFrequency.DAILY,
        digest_types=[DigestType.SALES_SUMMARY, DigestType.PERFORMANCE_OVERVIEW],
        preferred_format=DigestFormat.SLACK
    )
    
    # Check that the operation was successful
    assert result is True
    
    # Check that the recipient was added
    recipient = digest_system.get_recipient("user1")
    assert recipient is not None
    assert recipient.user_id == "user1"
    assert recipient.name == "Test User"
    assert recipient.email == "test@example.com"
    assert recipient.slack_id == "U123456"
    assert recipient.frequency == DigestFrequency.DAILY
    assert DigestType.SALES_SUMMARY in recipient.digest_types
    assert DigestType.PERFORMANCE_OVERVIEW in recipient.digest_types
    assert recipient.preferred_format == DigestFormat.SLACK

def test_update_recipient(digest_system):
    """Test updating an existing recipient."""
    # Add a recipient
    digest_system.add_recipient(
        user_id="user1",
        name="Test User",
        email="test@example.com"
    )
    
    # Update the recipient
    result = digest_system.update_recipient(
        user_id="user1",
        name="Updated User",
        email="updated@example.com",
        slack_id="U654321",
        frequency=DigestFrequency.WEEKLY,
        digest_types=[DigestType.TREND_ALERTS],
        preferred_format=DigestFormat.EMAIL
    )
    
    # Check that the operation was successful
    assert result is True
    
    # Check that the recipient was updated
    recipient = digest_system.get_recipient("user1")
    assert recipient is not None
    assert recipient.name == "Updated User"
    assert recipient.email == "updated@example.com"
    assert recipient.slack_id == "U654321"
    assert recipient.frequency == DigestFrequency.WEEKLY
    assert DigestType.TREND_ALERTS in recipient.digest_types
    assert recipient.preferred_format == DigestFormat.EMAIL

def test_remove_recipient(digest_system):
    """Test removing a recipient."""
    # Add a recipient
    digest_system.add_recipient(
        user_id="user1",
        name="Test User",
        email="test@example.com"
    )
    
    # Remove the recipient
    result = digest_system.remove_recipient("user1")
    
    # Check that the operation was successful
    assert result is True
    
    # Check that the recipient was removed
    recipient = digest_system.get_recipient("user1")
    assert recipient is None

def test_generate_digest(digest_system):
    """Test generating a digest."""
    # Generate a sales summary digest
    digest_id = digest_system.generate_digest(DigestType.SALES_SUMMARY, {})
    
    # Check that the digest was generated
    assert digest_id is not None
    assert digest_id in digest_system.digests
    
    # Check the digest content
    digest = digest_system.digests[digest_id]
    assert digest["digest_type"] == DigestType.SALES_SUMMARY
    assert "content" in digest
    assert "generated_at" in digest
    assert "data" in digest

def test_deliver_digest(digest_system):
    """Test delivering a digest."""
    # Add a recipient
    digest_system.add_recipient(
        user_id="user1",
        name="Test User",
        email="test@example.com",
        slack_id="U123456",
        preferred_format=DigestFormat.SLACK
    )
    
    # Generate a digest
    digest_id = digest_system.generate_digest(DigestType.SALES_SUMMARY, {})
    
    # Deliver the digest
    result = digest_system.deliver_digest(digest_id, "user1")
    
    # Check that the operation was successful
    assert result is True
    
    # Check that a delivery record was created
    deliveries = [d for d in digest_system.deliveries if d.recipient_id == "user1" and d.digest_type == DigestType.SALES_SUMMARY]
    assert len(deliveries) == 1
    
    # Check the delivery record
    delivery = deliveries[0]
    assert delivery.status == "delivered"
    assert delivery.format == DigestFormat.SLACK
    
    # Check that the recipient's last_delivered timestamp was updated
    recipient = digest_system.get_recipient("user1")
    assert recipient.last_delivered is not None

def test_record_feedback(digest_system):
    """Test recording feedback for a digest delivery."""
    # Add a recipient
    digest_system.add_recipient(
        user_id="user1",
        name="Test User",
        email="test@example.com"
    )
    
    # Generate and deliver a digest
    digest_id = digest_system.generate_digest(DigestType.SALES_SUMMARY, {})
    digest_system.deliver_digest(digest_id, "user1")
    
    # Get the delivery ID
    delivery = digest_system.deliveries[0]
    
    # Record feedback
    feedback = {
        "thumbs_up": True,
        "comment": "Great insights!"
    }
    result = digest_system.record_feedback(delivery.delivery_id, feedback)
    
    # Check that the operation was successful
    assert result is True
    
    # Check that the feedback was recorded
    delivery = digest_system.deliveries[0]
    assert "thumbs_up" in delivery.engagement
    assert delivery.engagement["comment"] == "Great insights!"
    
    # Check that the recipient's feedback was updated
    recipient = digest_system.get_recipient("user1")
    assert delivery.delivery_id in recipient.feedback
    assert recipient.feedback[delivery.delivery_id]["thumbs_up"] is True

def test_get_delivery_stats(digest_system):
    """Test getting delivery statistics."""
    # Add recipients
    digest_system.add_recipient(
        user_id="user1",
        name="User 1",
        email="user1@example.com",
        preferred_format=DigestFormat.SLACK
    )
    digest_system.add_recipient(
        user_id="user2",
        name="User 2",
        email="user2@example.com",
        preferred_format=DigestFormat.EMAIL
    )
    
    # Generate and deliver digests
    digest_id1 = digest_system.generate_digest(DigestType.SALES_SUMMARY, {})
    digest_id2 = digest_system.generate_digest(DigestType.PERFORMANCE_OVERVIEW, {})
    
    digest_system.deliver_digest(digest_id1, "user1")
    digest_system.deliver_digest(digest_id2, "user2")
    
    # Record some feedback
    delivery1 = digest_system.deliveries[0]
    delivery2 = digest_system.deliveries[1]
    
    digest_system.record_feedback(delivery1.delivery_id, {"thumbs_up": True})
    digest_system.record_feedback(delivery2.delivery_id, {"thumbs_down": True})
    
    # Get delivery statistics
    stats = digest_system.get_delivery_stats()
    
    # Check the statistics
    assert stats["total_deliveries"] == 2
    assert stats["successful_deliveries"] == 2
    assert stats["failed_deliveries"] == 0
    
    assert stats["deliveries_by_format"][DigestFormat.SLACK] == 1
    assert stats["deliveries_by_format"][DigestFormat.EMAIL] == 1
    
    assert stats["deliveries_by_type"][DigestType.SALES_SUMMARY] == 1
    assert stats["deliveries_by_type"][DigestType.PERFORMANCE_OVERVIEW] == 1
    
    assert stats["deliveries_by_recipient"]["user1"] == 1
    assert stats["deliveries_by_recipient"]["user2"] == 1
    
    assert stats["feedback_stats"]["thumbs_up"] == 1
    assert stats["feedback_stats"]["thumbs_down"] == 1

def test_scheduler(digest_system):
    """Test the digest scheduler."""
    # Add a recipient
    digest_system.add_recipient(
        user_id="user1",
        name="Test User",
        email="test@example.com",
        frequency=DigestFrequency.DAILY
    )
    
    # Start the scheduler with a short interval
    digest_system.start_scheduler(interval=1)
    
    # Wait a bit for the scheduler to run
    import time
    time.sleep(2)
    
    # Stop the scheduler
    digest_system.stop_scheduler()
    
    # Check that a digest was delivered
    deliveries = [d for d in digest_system.deliveries if d.recipient_id == "user1"]
    assert len(deliveries) > 0
    
    # Check that the recipient's last_delivered timestamp was updated
    recipient = digest_system.get_recipient("user1")
    assert recipient.last_delivered is not None

def test_persistence(digest_system, temp_data_dir):
    """Test that data is persisted correctly."""
    # Add a recipient
    digest_system.add_recipient(
        user_id="user1",
        name="Test User",
        email="test@example.com"
    )
    
    # Generate and deliver a digest
    digest_id = digest_system.generate_digest(DigestType.SALES_SUMMARY, {})
    digest_system.deliver_digest(digest_id, "user1")
    
    # Record feedback
    delivery = digest_system.deliveries[0]
    digest_system.record_feedback(delivery.delivery_id, {"thumbs_up": True})
    
    # Create a new DigestSystem instance with the same data directory
    new_digest_system = DigestSystem(data_dir=temp_data_dir)
    
    # Check that the data was loaded correctly
    recipient = new_digest_system.get_recipient("user1")
    assert recipient is not None
    assert recipient.name == "Test User"
    assert recipient.email == "test@example.com"
    
    assert len(new_digest_system.deliveries) == 1
    assert new_digest_system.deliveries[0].recipient_id == "user1"
    assert new_digest_system.deliveries[0].digest_type == DigestType.SALES_SUMMARY
    
    assert digest_id in new_digest_system.digests
    assert new_digest_system.digests[digest_id]["digest_type"] == DigestType.SALES_SUMMARY 