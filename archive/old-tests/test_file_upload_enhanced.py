"""
Unit tests for the enhanced file upload and validation components.
"""

import pytest
import streamlit as st
import pandas as pd
import io
import os
from unittest.mock import patch, MagicMock, call
import tempfile

# Import functions to test
from src.ui.components.data_upload import render_data_upload
from src.validators.validator_service import process_uploaded_file

@pytest.fixture
def mock_session_state():
    """Fixture to mock Streamlit session state."""
    # Create a backup of the real session state
    original_session_state = st.session_state
    
    # Create a mock session state
    mock_state = {}
    
    # Replace with mock
    st.session_state = mock_state
    
    yield mock_state
    
    # Restore original
    st.session_state = original_session_state

@pytest.fixture
def mock_uploaded_file():
    """Fixture providing a mock UploadedFile."""
    # Create a mock CSV file
    csv_content = "LeadSource,Total Gross\nDealer Website,2500.00\nInternet Lead,-150.75\nPhone Call,950.25"
    csv_bytes = csv_content.encode('utf-8')
    
    # Create a mock UploadedFile
    class MockUploadedFile:
        def __init__(self, content, name):
            self.content = content
            self.name = name
            self._io = io.BytesIO(content)
        
        def read(self):
            return self.content
        
        def getvalue(self):
            return self.content
        
        def seek(self, pos):
            self._io.seek(pos)
        
        def __iter__(self):
            return iter(self._io)
    
    return MockUploadedFile(csv_bytes, "test_file.csv")

@pytest.fixture
def mock_large_uploaded_file():
    """Fixture providing a mock large UploadedFile."""
    # Create a large dataset
    df = pd.DataFrame({
        'LeadSource': ['Dealer Website'] * 1000 + ['Internet Lead'] * 1000 + ['Phone Call'] * 1000,
        'Total Gross': [2500.00] * 1000 + [-150.75] * 1000 + [950.25] * 1000
    })
    
    # Convert to CSV bytes
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue().encode('utf-8')
    
    # Create a mock UploadedFile
    class MockLargeUploadedFile:
        def __init__(self, content, name):
            self.content = content
            self.name = name
            self._io = io.BytesIO(content)
        
        def read(self):
            return self.content
        
        def getvalue(self):
            return self.content
        
        def seek(self, pos):
            self._io.seek(pos)
        
        def __iter__(self):
            return iter(self._io)
    
    return MockLargeUploadedFile(csv_bytes, "large_test_file.csv")

@pytest.fixture
def mock_invalid_uploaded_file():
    """Fixture providing a mock invalid file."""
    # Create an invalid file (not CSV)
    invalid_content = b"This is not a valid CSV file."
    
    # Create a mock UploadedFile
    class MockInvalidUploadedFile:
        def __init__(self, content, name):
            self.content = content
            self.name = name
            self._io = io.BytesIO(content)
        
        def read(self):
            return self.content
        
        def getvalue(self):
            return self.content
        
        def seek(self, pos):
            self._io.seek(pos)
        
        def __iter__(self):
            return iter(self._io)
    
    return MockInvalidUploadedFile(invalid_content, "invalid_file.txt")

def test_render_data_upload_simplified():
    """Test rendering the simplified data upload component."""
    with patch('streamlit.file_uploader') as mock_uploader:
        # Mock the return value of file_uploader
        mock_uploader.return_value = None
        
        result = render_data_upload(simplified=True)
        
        # Verify file_uploader was called
        mock_uploader.assert_called_once()
        assert result is None

def test_render_data_upload_with_file():
    """Test rendering the data upload component with a file."""
    mock_file = MagicMock()
    mock_file.name = "test_file.csv"
    
    with patch('streamlit.file_uploader') as mock_uploader:
        # Mock the return value of file_uploader
        mock_uploader.return_value = mock_file
        
        result = render_data_upload(simplified=True)
        
        # Verify file_uploader was called and file was returned
        mock_uploader.assert_called_once()
        assert result == mock_file

def test_process_uploaded_file_with_profile(mock_uploaded_file):
    """Test processing an uploaded file with a validation profile."""
    # Create a mock validation profile
    mock_profile = MagicMock()
    mock_profile.rules = {'missing_values': True}
    mock_profile.thresholds = {'missing_values_max': 10}
    
    # Call the function
    result_df, summary, report = process_uploaded_file(mock_uploaded_file, selected_profile=mock_profile)
    
    # Verify results
    assert isinstance(result_df, pd.DataFrame)
    assert 'LeadSource' in result_df.columns
    assert 'Total Gross' in result_df.columns
    assert isinstance(summary, dict)
    assert isinstance(report, dict)

def test_process_uploaded_file_without_profile(mock_uploaded_file):
    """Test processing an uploaded file without a validation profile."""
    # Call the function
    result_df, summary, report = process_uploaded_file(mock_uploaded_file, selected_profile=None)
    
    # Verify results
    assert isinstance(result_df, pd.DataFrame)
    assert 'LeadSource' in result_df.columns
    assert 'Total Gross' in result_df.columns
    assert isinstance(summary, dict)
    assert isinstance(report, dict)

def test_process_large_uploaded_file(mock_large_uploaded_file):
    """Test processing a large uploaded file."""
    # Call the function
    result_df, summary, report = process_uploaded_file(mock_large_uploaded_file)
    
    # Verify results
    assert isinstance(result_df, pd.DataFrame)
    assert len(result_df) == 3000  # 3000 rows
    assert isinstance(summary, dict)
    assert isinstance(report, dict)

def test_process_invalid_uploaded_file(mock_invalid_uploaded_file):
    """Test processing an invalid uploaded file."""
    with pytest.raises(Exception):
        # Call the function with invalid file
        result_df, summary, report = process_uploaded_file(mock_invalid_uploaded_file)
