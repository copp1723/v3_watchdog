"""
Tests for the event system components.
"""

import pytest
from unittest.mock import Mock, patch
import pandas as pd
from datetime import datetime, timedelta

from src.watchdog_ai.insights.event_emitter import (
    EventEmitter,
    EventType,
    Event,
    EventHandler
)
from src.watchdog_ai.insights.delivery_handlers import (
    DataNormalizedHandler,
    AlertHandler,
    DeliveryStatusHandler
)
from src.watchdog_ai.insights.data_pipeline import DataPipeline

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame({
        'Gross_Profit': [1000, -500, 2000, -1000],
        'DaysInInventory': [30, 95, 120, 45],
        'VIN': ['VIN1', 'VIN2', 'VIN3', 'VIN4']
    })

@pytest.fixture
def mock_delivery_manager():
    """Create a mock delivery manager."""
    manager = Mock()
    manager.schedule_daily_summary.return_value = "DEL123"
    manager.send_alert.return_value = "ALT456"
    return manager

class TestEventHandler(EventHandler):
    """Test event handler implementation."""
    
    def __init__(self, event_types):
        """Initialize the handler."""
        super().__init__(event_types)
        self.handled_events = []
    
    def handle_event(self, event: Event) -> None:
        """Handle an event."""
        self.handled_events.append(event)

def test_event_emitter_singleton():
    """Test EventEmitter singleton pattern."""
    emitter1 = EventEmitter()
    emitter2 = EventEmitter()
    assert emitter1 is emitter2

def test_event_handler_registration():
    """Test registering event handlers."""
    emitter = EventEmitter()
    handler = TestEventHandler([EventType.DATA_NORMALIZED])
    
    emitter.register_handler(handler)
    assert EventType.DATA_NORMALIZED in emitter.handlers
    assert handler in emitter.handlers[EventType.DATA_NORMALIZED]

def test_event_emission_and_handling():
    """Test emitting and handling events."""
    emitter = EventEmitter()
    handler = TestEventHandler([EventType.DATA_NORMALIZED])
    emitter.register_handler(handler)
    
    event = Event(
        event_type=EventType.DATA_NORMALIZED,
        data={'test': 'data'},
        source='test'
    )
    
    emitter.emit(event)
    
    # Give the worker thread time to process
    import time
    time.sleep(0.1)
    
    assert len(handler.handled_events) == 1
    assert handler.handled_events[0].event_type == EventType.DATA_NORMALIZED

def test_data_normalized_handler(mock_delivery_manager):
    """Test DataNormalizedHandler."""
    handler = DataNormalizedHandler(mock_delivery_manager)
    
    event = Event(
        event_type=EventType.DATA_NORMALIZED,
        data={
            'dealer_id': 'DEALER1',
            'insights': [{'title': 'Test Insight'}]
        },
        source='test'
    )
    
    with patch('src.watchdog_ai.insights.delivery_handlers.get_user_preferences') as mock_prefs:
        mock_prefs.return_value = {
            'types': {'daily_summary': True},
            'email': 'test@example.com'
        }
        
        handler.handle_event(event)
        
        mock_delivery_manager.schedule_daily_summary.assert_called_once()

def test_alert_handler(mock_delivery_manager):
    """Test AlertHandler."""
    handler = AlertHandler(mock_delivery_manager)
    
    event = Event(
        event_type=EventType.ALERT_TRIGGERED,
        data={
            'dealer_id': 'DEALER1',
            'alert': {'title': 'Test Alert'},
            'severity': 'high'
        },
        source='test'
    )
    
    with patch('src.watchdog_ai.insights.delivery_handlers.get_user_preferences') as mock_prefs:
        mock_prefs.return_value = {
            'types': {'critical_alerts': True},
            'channels': {'email': True, 'sms': True},
            'email': 'test@example.com',
            'phone': '1234567890'
        }
        
        handler.handle_event(event)
        
        assert mock_delivery_manager.send_alert.call_count == 2  # Email + SMS

def test_delivery_status_handler():
    """Test DeliveryStatusHandler."""
    handler = DeliveryStatusHandler()
    
    # Emit some events
    success_event = Event(
        event_type=EventType.DELIVERY_COMPLETED,
        data={'delivery_id': 'DEL1'},
        source='test'
    )
    
    failure_event = Event(
        event_type=EventType.DELIVERY_FAILED,
        data={
            'delivery_id': 'DEL2',
            'error': 'Test error',
            'recipient': 'test@example.com'
        },
        source='test'
    )
    
    handler.handle_event(success_event)
    handler.handle_event(failure_event)
    
    metrics = handler.get_status_metrics()
    assert metrics['counts']['completed'] == 1
    assert metrics['counts']['failed'] == 1
    assert metrics['success_rate'] == 50.0

def test_data_pipeline_alerts(sample_data):
    """Test data pipeline alert generation."""
    pipeline = DataPipeline()
    
    # Mock event emitter
    mock_emitter = Mock()
    pipeline.event_emitter = mock_emitter
    
    # Process data
    pipeline.process_data(sample_data, 'DEALER1')
    
    # Verify alerts were emitted
    assert mock_emitter.emit.call_count == 3  # Normalized + 2 alerts
    
    # Check alert types
    alert_calls = [
        call for call in mock_emitter.emit.call_args_list
        if call[0][0].event_type == EventType.ALERT_TRIGGERED
    ]
    assert len(alert_calls) == 2  # Negative gross + aging inventory

def test_data_pipeline_error_handling():
    """Test data pipeline error handling."""
    pipeline = DataPipeline()
    
    # Test with invalid data
    result = pipeline.process_data(None, 'DEALER1')
    assert not result['success']
    assert 'error' in result

def test_event_shutdown():
    """Test clean shutdown of event system."""
    emitter = EventEmitter()
    emitter.shutdown()
    assert emitter.stop_event.is_set()
    assert not emitter.worker_thread.is_alive()