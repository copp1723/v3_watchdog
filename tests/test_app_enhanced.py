"""
Enhanced unit tests for the Streamlit app module.
Focuses on session state management, error handling, and UI components.
"""

import pytest
import streamlit as st
import pandas as pd
from unittest.mock import patch, MagicMock
from datetime import datetime

# Import the app module
from ..src.app import (
    initialize_session_state,
    handle_file_upload,
    process_data,
    display_insights,
    main
)

@pytest.fixture
def sample_csv_data():
    """Fixture providing sample CSV data."""
    return pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=5),
        'sales': [1000, 1500, 2000, 1800, 2200],
        'region': ['North', 'South', 'East', 'West', 'Central'],
        'product': ['A', 'B', 'A', 'C', 'B']
    })

@pytest.fixture
def mock_conversation_manager():
    """Fixture providing a mock conversation manager."""
    with patch('src.app.ConversationManager') as mock:
        instance = mock.return_value
        instance.generate_insight.return_value = {
            'summary': 'Test insight summary',
            'timestamp': datetime.now().isoformat(),
            'is_mock': True
        }
        yield instance

def test_initialize_session_state():
    """Test initialization of session state variables."""
    # Clear any existing session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    
    # Initialize session state
    initialize_session_state()
    
    # Check that all required keys are present
    assert 'conversation_history' in st.session_state
    assert isinstance(st.session_state['conversation_history'], list)
    assert len(st.session_state['conversation_history']) == 0
    
    assert 'current_prompt' in st.session_state
    assert st.session_state['current_prompt'] == ''
    
    assert 'regenerate_insight' in st.session_state
    assert st.session_state['regenerate_insight'] is False
    
    assert 'uploaded_file' in st.session_state
    assert st.session_state['uploaded_file'] is None
    
    assert 'processed_data' in st.session_state
    assert st.session_state['processed_data'] is None
    
    assert 'validation_context' in st.session_state
    assert st.session_state['validation_context'] is None

def test_handle_file_upload_valid_csv(sample_csv_data):
    """Test handling of valid CSV file upload."""
    # Create a mock file uploader
    mock_file = MagicMock()
    mock_file.name = "test.csv"
    mock_file.getvalue.return_value = sample_csv_data.to_csv(index=False).encode()
    
    with patch('streamlit.file_uploader', return_value=mock_file):
        handle_file_upload()
        
        assert st.session_state['uploaded_file'] is not None
        assert isinstance(st.session_state['processed_data'], pd.DataFrame)
        assert len(st.session_state['processed_data']) == len(sample_csv_data)

def test_handle_file_upload_invalid_csv():
    """Test handling of invalid CSV file upload."""
    # Create a mock file with invalid CSV content
    mock_file = MagicMock()
    mock_file.name = "test.csv"
    mock_file.getvalue.return_value = b"invalid,csv,content\nwith,wrong,format"
    
    with patch('streamlit.file_uploader', return_value=mock_file):
        handle_file_upload()
        
        assert st.session_state['uploaded_file'] is None
        assert st.session_state['processed_data'] is None

def test_handle_file_upload_no_file():
    """Test handling when no file is uploaded."""
    with patch('streamlit.file_uploader', return_value=None):
        handle_file_upload()
        
        assert st.session_state['uploaded_file'] is None
        assert st.session_state['processed_data'] is None

def test_process_data_valid_data(sample_csv_data):
    """Test processing of valid data."""
    st.session_state['processed_data'] = sample_csv_data
    
    process_data()
    
    assert st.session_state['validation_context'] is not None
    assert 'data_shape' in st.session_state['validation_context']
    assert 'columns' in st.session_state['validation_context']
    assert 'numeric_columns' in st.session_state['validation_context']

def test_process_data_no_data():
    """Test processing when no data is available."""
    st.session_state['processed_data'] = None
    
    process_data()
    
    assert st.session_state['validation_context'] is None

def test_display_insights_with_mock_manager(mock_conversation_manager):
    """Test display of insights using mock conversation manager."""
    # Set up test data
    st.session_state['current_prompt'] = "Test prompt"
    st.session_state['validation_context'] = {
        'data_shape': (100, 10),
        'columns': ['date', 'sales'],
        'numeric_columns': ['sales']
    }
    
    # Mock the streamlit components
    with patch('streamlit.text_input') as mock_input, \
         patch('streamlit.button') as mock_button, \
         patch('streamlit.write') as mock_write:
        
        mock_input.return_value = "Test prompt"
        mock_button.return_value = True
        
        display_insights()
        
        # Verify that the conversation manager was called
        mock_conversation_manager.generate_insight.assert_called_once()
        
        # Verify that the response was written
        mock_write.assert_called()

def test_display_insights_with_error(mock_conversation_manager):
    """Test display of insights when an error occurs."""
    # Set up test data
    st.session_state['current_prompt'] = "Test prompt"
    st.session_state['validation_context'] = {
        'data_shape': (100, 10),
        'columns': ['date', 'sales'],
        'numeric_columns': ['sales']
    }
    
    # Make the conversation manager raise an error
    mock_conversation_manager.generate_insight.side_effect = Exception("Test error")
    
    # Mock the streamlit components
    with patch('streamlit.text_input') as mock_input, \
         patch('streamlit.button') as mock_button, \
         patch('streamlit.error') as mock_error:
        
        mock_input.return_value = "Test prompt"
        mock_button.return_value = True
        
        display_insights()
        
        # Verify that the error was displayed
        mock_error.assert_called()

def test_main_initialization():
    """Test the main function initialization."""
    with patch('streamlit.title') as mock_title, \
         patch('streamlit.sidebar') as mock_sidebar, \
         patch('src.app.initialize_session_state') as mock_init:
        
            main()
            
        # Verify that session state was initialized
        mock_init.assert_called_once()
        
        # Verify that the title was set
        mock_title.assert_called_once()
        
        # Verify that the sidebar was created
        mock_sidebar.assert_called()

def test_main_with_data_flow(sample_csv_data, mock_conversation_manager):
    """Test the main function with a complete data flow."""
    # Set up test data
    st.session_state['processed_data'] = sample_csv_data
    st.session_state['validation_context'] = {
        'data_shape': (100, 10),
        'columns': ['date', 'sales'],
        'numeric_columns': ['sales']
    }
    st.session_state['current_prompt'] = "Test prompt"
    
    # Mock all streamlit components
    with patch('streamlit.title') as mock_title, \
         patch('streamlit.sidebar') as mock_sidebar, \
         patch('streamlit.text_input') as mock_input, \
         patch('streamlit.button') as mock_button, \
         patch('streamlit.write') as mock_write:
        
        mock_input.return_value = "Test prompt"
        mock_button.return_value = True
        
            main()
            
        # Verify that the conversation manager was called
        mock_conversation_manager.generate_insight.assert_called_once()
        
        # Verify that the response was written
        mock_write.assert_called()
