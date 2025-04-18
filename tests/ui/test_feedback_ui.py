"""
Unit tests for the feedback UI components.
"""

import pytest
import streamlit as st
from unittest.mock import Mock, patch
import pandas as pd
from datetime import datetime

from src.ui.pages.feedback_view import (
    feedback_view,
    _handle_feedback_submission,
    render_feedback_form,
    render_feedback_history
)

@pytest.fixture
def mock_feedback_manager():
    """Create a mock feedback manager."""
    manager = Mock()
    manager.record_feedback.return_value = True
    manager.get_feedback.return_value = [
        {
            "insight_id": "test_insight",
            "feedback_type": "helpful",
            "comment": "Great insight!",
            "timestamp": datetime.now().isoformat()
        }
    ]
    manager.get_feedback_stats.return_value = {
        "total_feedback": 1,
        "counts": {"helpful": 1},
        "percentages": {"helpful": 100.0}
    }
    return manager

@pytest.fixture
def mock_session_state():
    """Create a mock session state."""
    return {
        "user_id": "test_user",
        "session_id": "test_session",
        "insights": [
            {
                "insight_type": "test_insight",
                "summary": "Test insight summary"
            }
        ]
    }

def test_feedback_submission(mock_feedback_manager):
    """Test feedback submission handling."""
    with patch('src.ui.pages.feedback_view.feedback_manager', mock_feedback_manager):
        with patch('streamlit.success') as mock_success:
            _handle_feedback_submission(
                insight_id="test_insight",
                feedback_type="helpful",
                comment="Great insight!",
                session_id="test_session"
            )
            
            mock_feedback_manager.record_feedback.assert_called_once()
            mock_success.assert_called_once()

def test_feedback_submission_error(mock_feedback_manager):
    """Test feedback submission error handling."""
    mock_feedback_manager.record_feedback.side_effect = Exception("Test error")
    
    with patch('src.ui.pages.feedback_view.feedback_manager', mock_feedback_manager):
        with patch('streamlit.error') as mock_error:
            _handle_feedback_submission(
                insight_id="test_insight",
                feedback_type="helpful",
                comment="Great insight!",
                session_id="test_session"
            )
            
            mock_error.assert_called_once()

def test_render_feedback_history(mock_feedback_manager):
    """Test feedback history rendering."""
    with patch('src.ui.pages.feedback_view.feedback_manager', mock_feedback_manager):
        with patch('streamlit.dataframe') as mock_dataframe:
            render_feedback_history()
            
            mock_feedback_manager.get_feedback.assert_called_once()
            mock_dataframe.assert_called_once()

def test_render_feedback_history_empty(mock_feedback_manager):
    """Test feedback history rendering with no entries."""
    mock_feedback_manager.get_feedback.return_value = []
    
    with patch('src.ui.pages.feedback_view.feedback_manager', mock_feedback_manager):
        with patch('streamlit.info') as mock_info:
            render_feedback_history()
            
            mock_info.assert_called_once_with("No feedback entries found.")

def test_feedback_view_initialization():
    """Test feedback view initialization."""
    with patch('streamlit.session_state', {}) as mock_session_state:
        with patch('src.ui.pages.feedback_view._initialize_session_state') as mock_init:
            with patch('streamlit.title'):  # Mock st.title to prevent errors
                feedback_view()
                
                mock_init.assert_called_once()
                assert 'feedback_submitted' in mock_session_state
                assert 'selected_insight' in mock_session_state

def test_sentry_integration(mock_feedback_manager):
    """Test Sentry integration in feedback submission."""
    with patch('src.ui.pages.feedback_view.feedback_manager', mock_feedback_manager):
        with patch('sentry_sdk.capture_message') as mock_capture_message:
            with patch('streamlit.success'):
                _handle_feedback_submission(
                    insight_id="test_insight",
                    feedback_type="helpful",
                    comment="Great insight!",
                    session_id="test_session"
                )
                
                mock_capture_message.assert_called_once()