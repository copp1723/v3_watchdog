"""
Tests for the notification dashboard and settings components.
"""

import pytest
from unittest.mock import Mock, patch
import pandas as pd
from datetime import datetime, timedelta

from src.watchdog_ai.ui.pages.delivery_status import DeliveryStatusDashboard
from src.watchdog_ai.ui.pages.notification_settings import NotificationSettings
from src.scheduler.notification_service import NotificationService

@pytest.fixture
def mock_notification_service():
    """Create a mock notification service."""
    service = Mock(spec=NotificationService)
    return service

@pytest.fixture
def mock_metrics_logger():
    """Create a mock metrics logger."""
    logger = Mock()
    logger.get_delivery_metrics.return_value = {
        'delivery_times': [
            {'duration': 1.5},
            {'duration': 2.1}
        ],
        'pdf_sizes': [
            {'size_kb': 150},
            {'size_kb': 200}
        ]
    }
    return logger

@pytest.fixture
def dashboard(mock_notification_service, mock_metrics_logger):
    """Create a dashboard instance with mocked dependencies."""
    return DeliveryStatusDashboard(
        notification_service=mock_notification_service,
        metrics_logger=mock_metrics_logger
    )

@pytest.fixture
def settings(mock_notification_service):
    """Create a settings instance with mocked dependencies."""
    return NotificationSettings(
        notification_service=mock_notification_service
    )

def test_dashboard_initialization(dashboard):
    """Test dashboard initialization."""
    assert dashboard.notification_service is not None
    assert dashboard.metrics_logger is not None

def test_settings_initialization(settings):
    """Test settings initialization."""
    assert settings.notification_service is not None

@patch('streamlit.dataframe')
def test_recent_deliveries_display(mock_dataframe, dashboard):
    """Test displaying recent deliveries."""
    # Setup mock data
    records = [
        {
            "timestamp": datetime.now().isoformat(),
            "recipient": "test@example.com",
            "type": "daily_summary",
            "status": "Delivered",
            "attempt_count": 1
        }
    ]
    
    # Mock the internal method
    dashboard._get_filtered_records = Mock(return_value=records)
    
    # Render dashboard
    dashboard._render_recent_deliveries()
    
    # Verify DataFrame was created
    mock_dataframe.assert_called()
    df = mock_dataframe.call_args[0][0]
    assert len(df) == 1
    assert df.iloc[0]['status'] == 'Delivered'

@patch('streamlit.plotly_chart')
def test_performance_metrics_display(mock_plotly, dashboard, mock_metrics_logger):
    """Test displaying performance metrics."""
    # Render metrics
    dashboard._render_performance_metrics()
    
    # Verify charts were created
    assert mock_plotly.call_count == 2

@patch('streamlit.success')
def test_notification_preferences_save(mock_success, settings):
    """Test saving notification preferences."""
    with patch('streamlit.session_state', {'user': Mock(username='test_user')}):
        # Mock form submission
        with patch('streamlit.form') as mock_form:
            mock_form.return_value.__enter__.return_value = None
            mock_form.return_value.__exit__.return_value = None
            
            # Render settings
            settings._render_delivery_preferences({})
            
            # Verify success message
            mock_success.assert_not_called()  # No submission yet

def test_schedule_calculation(settings):
    """Test delivery schedule calculation."""
    from datetime import time
    
    # Test daily delivery
    next_daily = settings._calculate_next_delivery(
        time(9, 0),  # 9:00 AM
        None,
        'UTC'
    )
    assert isinstance(next_daily, datetime)
    
    # Test weekly delivery
    next_weekly = settings._calculate_next_delivery(
        time(9, 0),  # 9:00 AM
        'Monday',
        'UTC'
    )
    assert isinstance(next_weekly, datetime)
    assert next_weekly.weekday() == 0  # Monday

@patch('streamlit.error')
def test_dashboard_error_handling(mock_error, dashboard):
    """Test dashboard error handling."""
    # Force an error in metrics
    dashboard.metrics_logger.get_delivery_metrics.side_effect = Exception("Test error")
    
    # Render metrics
    dashboard._render_performance_metrics()
    
    # Verify error was displayed
    mock_error.assert_called()

@patch('streamlit.warning')
def test_settings_unauthenticated(mock_warning, settings):
    """Test settings behavior when user is not logged in."""
    # Render settings without user in session
    settings.render()
    
    # Verify warning was displayed
    mock_warning.assert_called_with("Please log in to configure notification settings.")