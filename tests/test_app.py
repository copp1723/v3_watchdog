"""
Tests for the main application functionality.
"""

import pytest
import pandas as pd
import streamlit as st
from datetime import datetime
from src.app import (
    process_uploaded_file,
    initialize_session_state,
    render_data_validation,
    render_insight_generation
)
from src.utils.errors import ValidationError

# --- Test Data Fixtures ---

@pytest.fixture
def sample_csv_data():
    """Create sample CSV data for testing."""
    return pd.DataFrame({
        'VIN': ['VIN001', 'VIN002', 'VIN003'],
        'Total Gross': [1000.50, -500.25, 2000.75],
        'LeadSource': ['CarGurus', 'Website', 'Walk-in'],
        'Sale_Date': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'SalesRepName': ['Hunter Adams', 'Sarah Lee', 'Mike Jones']
    })

@pytest.fixture
def mock_conversation_manager(mocker):
    """Create a mock conversation manager."""
    mock = mocker.Mock()
    mock.generate_insight.return_value = {
        "summary": "Test insight",
        "value_insights": [],
        "actionable_flags": [],
        "confidence": "high"
    }
    return mock

# --- Session State Tests ---

def test_initialize_session_state():
    """Test session state initialization."""
    initialize_session_state()
    assert "conversation_history" in st.session_state
    assert "validated_data" in st.session_state
    assert "conversation_manager" in st.session_state
    assert isinstance(st.session_state.conversation_history, list)

def test_handle_file_upload_valid_csv(sample_csv_data):
    """Test handling of valid CSV file upload."""
    class MockFile:
        def __init__(self, data):
            self.data = data
            self.name = "test.csv"
        def read(self):
            return self.data
    
    mock_file = MockFile(sample_csv_data.to_csv().encode())
    df, summary = process_uploaded_file(mock_file)
    
    assert df is not None
    assert summary["status"] == "success"
    assert len(df) == len(sample_csv_data)

def test_handle_file_upload_invalid_csv():
    """Test handling of invalid CSV file."""
    class MockFile:
        def __init__(self):
            self.name = "test.csv"
        def read(self):
            return b"invalid,csv,data\n1,2"
    
    mock_file = MockFile()
    df, summary = process_uploaded_file(mock_file)
    
    assert df is None
    assert summary["status"] == "error"
    assert "error" in summary["message"].lower()

def test_handle_file_upload_no_file():
    """Test handling when no file is provided."""
    df, summary = process_uploaded_file(None)
    assert df is None
    assert summary["status"] == "error"
    assert "no file" in summary["message"].lower()

# --- Data Processing Tests ---

def test_process_data_valid_data(sample_csv_data):
    """Test processing of valid data."""
    st.session_state.validated_data = sample_csv_data
    st.session_state.validation_summary = {
        "status": "success",
        "total_rows": len(sample_csv_data),
        "passed_rows": len(sample_csv_data),
        "failed_rows": 0
    }
    
    render_data_validation()
    assert "success" in st.session_state.validation_summary["status"]

def test_process_data_no_data():
    """Test processing when no data is present."""
    st.session_state.validated_data = None
    render_data_validation()
    assert st.session_state.validated_data is None

# --- Insight Generation Tests ---

def test_display_insights_with_mock_manager(mock_conversation_manager):
    """Test insight display with mock conversation manager."""
    st.session_state.conversation_manager = mock_conversation_manager
    st.session_state.validated_data = pd.DataFrame({'test': [1, 2, 3]})
    
    render_insight_generation()
    assert len(st.session_state.conversation_history) == 0  # No prompts yet

def test_display_insights_with_error(mock_conversation_manager):
    """Test insight display when an error occurs."""
    mock_conversation_manager.generate_insight.side_effect = ValidationError("Test error")
    st.session_state.conversation_manager = mock_conversation_manager
    
    render_insight_generation()
    assert "error" in st.session_state.get("error_details", {}).get("type", "")

# --- Integration Tests ---

def test_main_initialization():
    """Test main application initialization."""
    initialize_session_state()
    assert st.session_state.conversation_manager is not None
    assert st.session_state.conversation_history == []
    assert st.session_state.validated_data is None

def test_main_with_data_flow(sample_csv_data):
    """Test main application flow with data."""
    # Initialize
    initialize_session_state()
    
    # Upload file
    class MockFile:
        def __init__(self, data):
            self.data = data
            self.name = "test.csv"
        def read(self):
            return self.data
    
    mock_file = MockFile(sample_csv_data.to_csv().encode())
    df, summary = process_uploaded_file(mock_file)
    
    assert df is not None
    assert summary["status"] == "success"
    
    # Process data
    st.session_state.validated_data = df
    st.session_state.validation_summary = summary
    
    render_data_validation()
    assert st.session_state.validation_summary["status"] == "success"
    
    # Generate insight
    render_insight_generation()
    assert st.session_state.validated_data is not None