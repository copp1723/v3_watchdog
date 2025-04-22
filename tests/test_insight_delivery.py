"""
Tests for the insight delivery system.
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
import json

from src.insights.insight_delivery_manager import (
    InsightDeliveryManager,
    DeliveryStatus,
    DeliveryRecord
)

@pytest.fixture
def sample_insights():
    """Create sample insights for testing."""
    return [
        {
            "title": "Sales Performance",
            "summary": "Strong sales growth in Q1",
            "metrics": {
                "Total Sales": 125000,
                "Growth Rate": 15.5
            },
            "recommendations": [
                "Focus on high-margin products",
                "Expand successful product lines"
            ]
        },
        {
            "title": "Inventory Health",
            "summary": "Aging inventory requires attention",
            "metrics": {
                "Aging Units": 25,
                "Days Supply": 75
            },
            "recommendations": [
                "Review pricing strategy",
                "Consider promotional campaign"
            ]
        }
    ]

@pytest.fixture
def sample_kpis():
    """Create sample KPIs for testing."""
    return [
        {
            "label": "Total Revenue",
            "value": 1250000,
            "trend": 15
        },
        {
            "label": "Gross Margin",
            "value": 22.5,
            "trend": -2
        },
        {
            "label": "Units Sold",
            "value": 125,
            "trend": 5
        }
    ]

@pytest.fixture
def mock_notification_service():
    """Create a mock notification service."""
    service = Mock()
    service.queue = Mock()
    service.queue.add_email.return_value = "MSG123"
    return service

@pytest.fixture
def delivery_manager(mock_notification_service):
    """Create a delivery manager with mocked dependencies."""
    return InsightDeliveryManager(notification_service=mock_notification_service)

def test_schedule_daily_summary(delivery_manager, sample_insights):
    """Test scheduling a daily summary."""
    delivery_id = delivery_manager.schedule_daily_summary(
        "user@example.com",
        sample_insights
    )
    
    assert delivery_id in delivery_manager.delivery_records
    record = delivery_manager.delivery_records[delivery_id]
    assert record.delivery_type == "daily"
    assert record.recipient == "user@example.com"
    assert record.insights == sample_insights
    assert record.status == DeliveryStatus.PENDING

def test_schedule_weekly_executive(delivery_manager, sample_insights, sample_kpis):
    """Test scheduling a weekly executive summary."""
    delivery_id = delivery_manager.schedule_weekly_executive(
        "exec@example.com",
        sample_insights,
        sample_kpis,
        "Strong performance this week",
        "Jan 1 - Jan 7, 2024"
    )
    
    assert delivery_id in delivery_manager.delivery_records
    record = delivery_manager.delivery_records[delivery_id]
    assert record.delivery_type == "weekly"
    assert record.recipient == "exec@example.com"
    assert record.insights["insights"] == sample_insights
    assert record.insights["kpis"] == sample_kpis
    assert record.status == DeliveryStatus.PENDING

def test_send_alert(delivery_manager):
    """Test sending an alert."""
    alert = {
        "title": "Critical Issue",
        "description": "Immediate attention required",
        "metrics": [
            {"label": "Affected Units", "value": 50},
            {"label": "Impact", "value": "$25,000"}
        ],
        "actions": [
            "Review affected inventory",
            "Contact affected customers"
        ]
    }
    
    delivery_id = delivery_manager.send_alert("manager@example.com", alert)
    
    assert delivery_id in delivery_manager.delivery_records
    record = delivery_manager.delivery_records[delivery_id]
    assert record.delivery_type == "alert"
    assert record.recipient == "manager@example.com"
    assert record.insights == alert
    assert record.status == DeliveryStatus.PENDING

def test_delivery_status_tracking(delivery_manager, sample_insights):
    """Test delivery status tracking."""
    delivery_id = delivery_manager.schedule_daily_summary(
        "user@example.com",
        sample_insights
    )
    
    record = delivery_manager.delivery_records[delivery_id]
    
    # Test status transitions
    record.add_attempt(DeliveryStatus.IN_PROGRESS)
    assert record.status == DeliveryStatus.IN_PROGRESS
    
    record.add_attempt(DeliveryStatus.RETRYING, error="Temporary error")
    assert record.status == DeliveryStatus.RETRYING
    
    record.add_attempt(DeliveryStatus.DELIVERED, delivery_id="MSG123")
    assert record.status == DeliveryStatus.DELIVERED
    assert record.completed_at is not None

def test_get_delivery_status(delivery_manager, sample_insights):
    """Test getting delivery status."""
    delivery_id = delivery_manager.schedule_daily_summary(
        "user@example.com",
        sample_insights
    )
    
    # Add some attempts
    record = delivery_manager.delivery_records[delivery_id]
    record.add_attempt(DeliveryStatus.IN_PROGRESS)
    record.add_attempt(DeliveryStatus.DELIVERED, delivery_id="MSG123")
    
    # Get status
    status = delivery_manager.get_delivery_status(delivery_id)
    
    assert status["delivery_id"] == delivery_id
    assert status["status"] == DeliveryStatus.DELIVERED
    assert status["recipient"] == "user@example.com"
    assert status["type"] == "daily"
    assert len(status["attempts"]) == 2
    assert status["completed_at"] is not None

def test_delivery_retry_logic(delivery_manager, sample_insights):
    """Test delivery retry logic."""
    delivery_id = delivery_manager.schedule_daily_summary(
        "user@example.com",
        sample_insights
    )
    
    record = delivery_manager.delivery_records[delivery_id]
    
    # Simulate retries
    record.add_attempt(DeliveryStatus.IN_PROGRESS)
    record.add_attempt(DeliveryStatus.RETRYING, error="First failure")
    record.add_attempt(DeliveryStatus.RETRYING, error="Second failure")
    record.add_attempt(DeliveryStatus.FAILED, error="Final failure")
    
    assert record.attempt_count == 4
    assert record.status == DeliveryStatus.FAILED
    assert record.completed_at is not None

def test_unknown_delivery_status(delivery_manager):
    """Test getting status for unknown delivery ID."""
    status = delivery_manager.get_delivery_status("UNKNOWN")
    
    assert status["status"] == "unknown"
    assert "error" in status

def test_delivery_worker_shutdown(delivery_manager):
    """Test delivery worker shutdown."""
    delivery_manager.shutdown()
    assert delivery_manager.stop_event.is_set()
    assert not delivery_manager.worker_thread.is_alive()