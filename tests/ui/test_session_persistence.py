"""
Test session state persistence between components.
"""

import unittest
import pandas as pd
import streamlit as st
from unittest.mock import patch, MagicMock
import pytest

from watchdog_ai.config import SessionKeys
from watchdog_ai.ui.components.chat_interface import ChatInterface
from watchdog_ai.ui.components.data_uploader import render_data_uploader


class TestSessionPersistence(unittest.TestCase):
    """Test session state persistence between components."""
    
    def setUp(self):
        """Set up test environment."""
        # Mock streamlit session_state
        self.session_state_mock = {}
        self.st_patch = patch.object(st, 'session_state', self.session_state_mock)
        self.st_patch.start()
        
    def tearDown(self):
        """Clean up test environment."""
        self.st_patch.stop()
    
    @patch('streamlit.file_uploader')
    @patch('streamlit.success')
    @patch('streamlit.markdown')
    @patch('streamlit.dataframe')
    def test_uploader_sets_session_state(self, mock_dataframe, mock_markdown, mock_success, mock_uploader):
        """Test that the uploader sets session state correctly."""
        # Prepare test data
        test_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        
        # Mock the file uploader to return a file-like object
        mock_file = MagicMock()
        mock_uploader.return_value = mock_file
        
        # Mock pandas read_csv to return our test dataframe
        with patch('pandas.read_csv', return_value=test_df):
            # Call the function under test
            render_data_uploader()
            
            # Assert session state was updated correctly
            self.assertIn(SessionKeys.UPLOADED_DATA, self.session_state_mock)
            pd.testing.assert_frame_equal(self.session_state_mock[SessionKeys.UPLOADED_DATA], test_df)
    
    @patch('streamlit.checkbox', return_value=False)
    @patch('streamlit.warning')
    @patch('streamlit.stop')
    def test_chat_interface_validates_data(self, mock_stop, mock_warning, mock_checkbox):
        """Test that the chat interface validates uploaded data."""
        # Set up a ChatInterface instance
        chat_interface = ChatInterface()
        
        # Test case 1: No data in session state
        self.session_state_mock.clear()
        chat_interface.render_chat_interface()
        mock_warning.assert_called_once()
        mock_stop.assert_called_once()
        
        # Reset mocks
        mock_warning.reset_mock()
        mock_stop.reset_mock()
        
        # Test case 2: Empty DataFrame
        self.session_state_mock[SessionKeys.UPLOADED_DATA] = pd.DataFrame()
        chat_interface.render_chat_interface()
        mock_warning.assert_called_once()
        mock_stop.assert_called_once()
        
        # Reset mocks
        mock_warning.reset_mock()
        mock_stop.reset_mock()
        
        # Test case 3: Valid DataFrame (should not call stop)
        self.session_state_mock[SessionKeys.UPLOADED_DATA] = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        with patch('streamlit.container', return_value=MagicMock()):
            chat_interface.render_chat_interface()
        mock_stop.assert_not_called()

def test_data_persistence():
    """Test that uploaded data persists in session state."""
    # Clear session state
    for key in st.session_state.keys():
        del st.session_state[key]
    
    # Mock file upload
    df = pd.DataFrame({'test': [1, 2, 3]})
    st.session_state[SessionKeys.UPLOADED_DATA] = df
    
    # Call uploader
    render_data_uploader()
    
    # Check data persists
    assert SessionKeys.UPLOADED_DATA in st.session_state
    assert isinstance(st.session_state[SessionKeys.UPLOADED_DATA], pd.DataFrame)
    assert st.session_state[SessionKeys.UPLOADED_DATA].equals(df)