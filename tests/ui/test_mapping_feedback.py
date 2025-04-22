"""
Tests for the column mapping feedback UI component.
"""

import pytest
import streamlit as st
import pandas as pd
from datetime import datetime
from unittest.mock import Mock, patch

from src.watchdog_ai.ui.components.mapping_feedback import (
    MappingFeedbackUI,
    MappingFeedback,
    MappingSuggestion
)

@pytest.fixture
def mock_lineage():
    """Create mock lineage system."""
    lineage = Mock()
    lineage.store._memory_store = {}
    lineage.store.prefix = "watchdog:lineage:"
    return lineage

@pytest.fixture
def sample_suggestions():
    """Create sample mapping suggestions."""
    return [
        MappingSuggestion(
            original_name="total_gross_profit",
            canonical_name="gross",
            confidence=0.85,
            context={"column_type": "float64"},
            timestamp=datetime.now().isoformat()
        ),
        MappingSuggestion(
            original_name="source_of_lead",
            canonical_name="lead_source",
            confidence=0.92,
            context={"column_type": "object"},
            timestamp=datetime.now().isoformat()
        )
    ]

def test_feedback_ui_initialization(mock_lineage):
    """Test feedback UI initialization."""
    ui = MappingFeedbackUI(mock_lineage)
    assert hasattr(ui, 'lineage')
    assert 'mapping_feedback_submitted' in st.session_state
    assert isinstance(st.session_state.mapping_feedback_submitted, set)

def test_render_mapping_suggestions_empty(mock_lineage):
    """Test rendering with no suggestions."""
    ui = MappingFeedbackUI(mock_lineage)
    
    with patch('streamlit.info') as mock_info:
        ui.render_mapping_suggestions([])
        mock_info.assert_called_once_with("No mapping suggestions to review.")

def test_render_mapping_suggestions(mock_lineage, sample_suggestions):
    """Test rendering mapping suggestions."""
    ui = MappingFeedbackUI(mock_lineage)
    
    with patch('streamlit.expander') as mock_expander:
        with patch('streamlit.write') as mock_write:
            ui.render_mapping_suggestions(sample_suggestions)
            
            # Should create expander for each suggestion
            assert mock_expander.call_count == len(sample_suggestions)
            # Should write suggestion details
            assert mock_write.call_count > 0

def test_handle_feedback_success(mock_lineage):
    """Test successful feedback handling."""
    ui = MappingFeedbackUI(mock_lineage)
    
    feedback = MappingFeedback(
        original_column="total_gross_profit",
        suggested_column="gross",
        is_correct=True,
        correct_mapping=None,
        confidence=0.85,
        user_id="test_user",
        metadata={"column_type": "float64"}
    )
    
    with patch('streamlit.success') as mock_success:
        ui.handle_feedback(feedback)
        
        # Should track in lineage
        mock_lineage.track_column_mapping.assert_called_once()
        # Should show success message
        mock_success.assert_called_once()
        # Should mark as submitted
        assert feedback.original_column in st.session_state.mapping_feedback_submitted

def test_handle_feedback_error(mock_lineage):
    """Test error handling in feedback submission."""
    ui = MappingFeedbackUI(mock_lineage)
    mock_lineage.track_column_mapping.side_effect = Exception("Test error")
    
    feedback = MappingFeedback(
        original_column="total_gross_profit",
        suggested_column="gross",
        is_correct=True,
        correct_mapping=None,
        confidence=0.85,
        user_id="test_user",
        metadata={"column_type": "float64"}
    )
    
    with patch('streamlit.error') as mock_error:
        ui.handle_feedback(feedback)
        mock_error.assert_called_once()

def test_render_feedback_history_empty(mock_lineage):
    """Test rendering empty feedback history."""
    ui = MappingFeedbackUI(mock_lineage)
    
    with patch('streamlit.info') as mock_info:
        ui.render_feedback_history()
        mock_info.assert_called_once_with("No feedback history available.")

def test_render_feedback_history(mock_lineage):
    """Test rendering feedback history with data."""
    ui = MappingFeedbackUI(mock_lineage)
    
    # Add some mock feedback data
    mock_lineage.store._memory_store = {
        "watchdog:lineage:column_mapping:test": [
            {
                "source_id": "total_gross_profit",
                "target_id": "gross",
                "metadata": {
                    "feedback": "correct",
                    "confidence": 0.85
                },
                "timestamp": datetime.now().isoformat()
            }
        ]
    }
    
    with patch('streamlit.dataframe') as mock_dataframe:
        ui.render_feedback_history()
        mock_dataframe.assert_called_once()