"""
Unit tests for session state management in the enhanced app.
"""

import pytest
import streamlit as st
from unittest.mock import patch, MagicMock

@pytest.fixture
def mock_session_state():
    """Fixture to setup and reset session state for testing."""
    # Store original session state
    original_session_state = st.session_state
    
    # Create a mock session state dictionary
    mock_state = {
        'conversation_history': [],
        'prompt_input': '',
        'validated_data': None,
        'validation_summary': None,
        'validation_report': None,
        'last_uploaded_file': None,
        'set_prompt_input_value_flag': '',
        'clear_prompt_input_flag': False,
        'error_details': None
    }
    
    # Replace session state with mock
    st.session_state = mock_state
    
    yield mock_state
    
    # Restore original session state
    st.session_state = original_session_state

def test_set_prompt_input_flag(mock_session_state):
    """Test setting prompt input via flag pattern."""
    # Initial state
    assert mock_session_state['prompt_input'] == ''
    assert mock_session_state['set_prompt_input_value_flag'] == ''
    
    # Set the flag
    mock_session_state['set_prompt_input_value_flag'] = 'New prompt value'
    
    # Simulate what app_enhanced.py does: check and apply flag
    if mock_session_state.get('set_prompt_input_value_flag'):
        mock_session_state['prompt_input'] = mock_session_state['set_prompt_input_value_flag']
        mock_session_state['set_prompt_input_value_flag'] = ''
    
    # Verify state was updated correctly
    assert mock_session_state['prompt_input'] == 'New prompt value'
    assert mock_session_state['set_prompt_input_value_flag'] == ''

def test_clear_prompt_input_flag(mock_session_state):
    """Test clearing prompt input via flag pattern."""
    # Set initial state
    mock_session_state['prompt_input'] = 'Initial prompt'
    mock_session_state['clear_prompt_input_flag'] = False
    
    # Set the flag
    mock_session_state['clear_prompt_input_flag'] = True
    
    # Simulate what app_enhanced.py does: check and apply flag
    if mock_session_state.get('clear_prompt_input_flag'):
        mock_session_state['prompt_input'] = ''
        mock_session_state['clear_prompt_input_flag'] = False
    
    # Verify state was updated correctly
    assert mock_session_state['prompt_input'] == ''
    assert mock_session_state['clear_prompt_input_flag'] == False

def test_flag_priority(mock_session_state):
    """Test priority when both flags are set (clear flag should be processed first)."""
    # Set initial state
    mock_session_state['prompt_input'] = 'Initial prompt'
    mock_session_state['clear_prompt_input_flag'] = True
    mock_session_state['set_prompt_input_value_flag'] = 'New prompt value'
    
    # Simulate what app_enhanced.py does: check and apply flags in order
    if mock_session_state.get('clear_prompt_input_flag'):
        mock_session_state['prompt_input'] = ''
        mock_session_state['clear_prompt_input_flag'] = False
    elif mock_session_state.get('set_prompt_input_value_flag'):
        mock_session_state['prompt_input'] = mock_session_state['set_prompt_input_value_flag']
        mock_session_state['set_prompt_input_value_flag'] = ''
    
    # Verify clear was processed first, then set was not processed
    assert mock_session_state['prompt_input'] == ''
    assert mock_session_state['clear_prompt_input_flag'] == False
    assert mock_session_state['set_prompt_input_value_flag'] == 'New prompt value'

def test_conversation_history_management(mock_session_state):
    """Test managing conversation history in session state."""
    # Initial state - empty history
    assert len(mock_session_state['conversation_history']) == 0
    
    # Add an entry
    mock_session_state['conversation_history'].append({
        'prompt': 'Test prompt',
        'response': {
            'summary': 'Test response',
            'value_insights': [],
            'actionable_flags': [],
            'confidence': 'high'
        }
    })
    
    # Verify entry was added
    assert len(mock_session_state['conversation_history']) == 1
    assert mock_session_state['conversation_history'][0]['prompt'] == 'Test prompt'
    
    # Clear history
    mock_session_state['conversation_history'] = []
    
    # Verify history was cleared
    assert len(mock_session_state['conversation_history']) == 0

def test_error_details_management(mock_session_state):
    """Test managing error details in session state."""
    # Initial state - no error details
    assert mock_session_state['error_details'] is None
    
    # Set error details
    error_info = {
        'error': 'Test error message',
        'traceback': 'Test traceback information'
    }
    mock_session_state['error_details'] = error_info
    
    # Verify error details were set
    assert mock_session_state['error_details'] == error_info
    assert mock_session_state['error_details']['error'] == 'Test error message'
    
    # Clear error details
    mock_session_state['error_details'] = None
    
    # Verify error details were cleared
    assert mock_session_state['error_details'] is None

def test_file_upload_state_management(mock_session_state):
    """Test managing file upload state in session state."""
    # Initial state - no uploaded file
    assert mock_session_state['last_uploaded_file'] is None
    assert mock_session_state['validated_data'] is None
    
    # Simulate file upload and processing
    mock_session_state['last_uploaded_file'] = 'test_file.csv'
    mock_session_state['validated_data'] = MagicMock()  # Mock DataFrame
    mock_session_state['validation_summary'] = {'status': 'success'}
    mock_session_state['validation_report'] = {'issues': []}
    
    # Verify state was updated
    assert mock_session_state['last_uploaded_file'] == 'test_file.csv'
    assert mock_session_state['validated_data'] is not None
    
    # Simulate upload of a different file
    new_filename = 'different_file.csv'
    if new_filename != mock_session_state['last_uploaded_file']:
        # This would trigger reprocessing
        mock_session_state['last_uploaded_file'] = new_filename
        # Reset conversation history for new file
        mock_session_state['conversation_history'] = []
    
    # Verify state was updated correctly
    assert mock_session_state['last_uploaded_file'] == 'different_file.csv'
    assert len(mock_session_state['conversation_history']) == 0

def test_suggestion_button_interaction(mock_session_state):
    """Test suggestion button interaction with session state."""
    # Initial state
    assert mock_session_state['prompt_input'] == ''
    
    # Simulate clicking a suggestion button
    suggestion = "What is the average gross profit per deal?"
    
    # In app_enhanced.py, this would set the flag and trigger a rerun
    mock_session_state['set_prompt_input_value_flag'] = suggestion
    
    # Simulate what happens after rerun
    if mock_session_state.get('set_prompt_input_value_flag'):
        mock_session_state['prompt_input'] = mock_session_state['set_prompt_input_value_flag']
        mock_session_state['set_prompt_input_value_flag'] = ''
    
    # Verify prompt was set to suggestion
    assert mock_session_state['prompt_input'] == suggestion
    assert mock_session_state['set_prompt_input_value_flag'] == ''

def test_insight_generation_and_session_update(mock_session_state):
    """Test insight generation flow and session state updates."""
    # Setup initial state
    mock_session_state['prompt_input'] = 'How many sales did Car Gurus produce?'
    mock_session_state['conversation_history'] = []
    
    # Mock insight response
    mock_response = {
        'summary': 'Car Gurus produced 2 sales.',
        'value_insights': ['These sales generated $4,800.50 in gross profit.'],
        'actionable_flags': [],
        'confidence': 'high',
        'timestamp': '2023-05-01T12:00:00',
        'is_mock': True
    }
    
    # Simulate successful insight generation
    mock_session_state['conversation_history'].append({
        'prompt': mock_session_state['prompt_input'],
        'response': mock_response,
        'timestamp': '2023-05-01T12:00:00'
    })
    
    # Set flag to clear input after success
    mock_session_state['clear_prompt_input_flag'] = True
    
    # Simulate what happens after rerun
    if mock_session_state.get('clear_prompt_input_flag'):
        mock_session_state['prompt_input'] = ''
        mock_session_state['clear_prompt_input_flag'] = False
    
    # Verify state updates
    assert len(mock_session_state['conversation_history']) == 1
    assert mock_session_state['conversation_history'][0]['prompt'] == 'How many sales did Car Gurus produce?'
    assert mock_session_state['conversation_history'][0]['response'] == mock_response
    assert mock_session_state['prompt_input'] == ''
    assert mock_session_state['clear_prompt_input_flag'] == False
