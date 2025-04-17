"""
Test suite for conversation integration and edge cases.
"""

import pytest
import streamlit as st
from datetime import datetime
from ..src.insight_conversation import ConversationManager
from ..src.app import render_insight_history

@pytest.fixture
def sample_conversation_history():
    """Provide a sample conversation history for testing."""
    now = datetime.now()
    return [
        {
            'prompt': 'Test prompt 1',
            'response': {
                'summary': 'Test summary 1',
                'chart_data': {'type': 'line', 'data': {'x': [1, 2], 'y': [1, 2]}},
                'recommendation': 'Test recommendation 1',
                'risk_flag': False,
                'timestamp': now.isoformat()
            }
        },
        {
            'prompt': 'Test prompt 2',
            'response': {
                'summary': 'Test summary 2',
                'chart_data': {'type': 'bar', 'data': {'x': [1, 2], 'y': [2, 3]}},
                'recommendation': 'Test recommendation 2',
                'risk_flag': True,
                'timestamp': now.isoformat()
            }
        }
    ]

@pytest.fixture
def conversation_manager():
    """Provide a conversation manager instance for testing."""
    return ConversationManager(use_mock=True)

def test_new_insight_button_resets_state():
    """Test that the new insight button properly resets all state."""
    # Set up initial state
    st.session_state.current_prompt = "test prompt"
    st.session_state.regenerate_insight = True
    st.session_state.regenerate_index = 1
    st.session_state.next_prompt = "next prompt"
    
    # Simulate new insight button click
    render_insight_history([])
    
    # Verify state is reset
    assert st.session_state.current_prompt is None
    assert st.session_state.regenerate_insight is False
    assert st.session_state.regenerate_index is None
    assert st.session_state.next_prompt is None

def test_regenerate_insight_with_validation_context(conversation_manager, sample_conversation_history):
    """Test regenerating an insight with validation context."""
    # Set up test data
    st.session_state.conversation_history = sample_conversation_history
    st.session_state.regenerate_insight = True
    st.session_state.regenerate_index = 0
    st.session_state.current_prompt = "regenerate prompt"
    
    validation_context = {
        'validation_summary': {'status': 'warning', 'message': 'Test validation issues'},
        'validation_report': {'issues': ['issue1', 'issue2']}
    }
    
    # Regenerate insight
    response = conversation_manager.regenerate_insight(
        st.session_state.regenerate_index,
        validation_context=validation_context
    )
    
    assert response is not None
    assert 'summary' in response
    assert 'timestamp' in response

def test_rapid_button_clicks(conversation_manager):
    """Test handling of rapid button clicks."""
    # Set up initial state
    st.session_state.current_prompt = "test prompt"
    st.session_state.conversation_history = []
    
    # Simulate rapid button clicks
    for _ in range(3):
        response = conversation_manager.generate_insight(st.session_state.current_prompt)
        assert response is not None
        assert len(st.session_state.conversation_history) == _ + 1

def test_invalid_file_handling():
    """Test handling of invalid file uploads."""
    # Simulate invalid file upload
    with pytest.raises(Exception):
        process_uploaded_file(None, "Default")

def test_duplicate_prompt_handling(conversation_manager):
    """Test handling of duplicate prompts."""
    prompt = "test prompt"
    
    # Generate first insight
    response1 = conversation_manager.generate_insight(prompt)
    assert response1 is not None
    
    # Generate second insight with same prompt
    response2 = conversation_manager.generate_insight(prompt)
    assert response2 is not None
    assert response2 != response1  # Should generate different response

def test_conversation_state_cleanup():
    """Test that conversation state is properly cleaned up."""
    # Set up test state
    st.session_state.conversation_history = [{'prompt': 'test', 'response': {}}]
    st.session_state.current_prompt = "test"
    st.session_state.regenerate_insight = True
    st.session_state.regenerate_index = 0
    
    # Simulate new insight button click
    render_insight_history([])
    
    # Verify state is cleaned up
    assert st.session_state.current_prompt is None
    assert st.session_state.regenerate_insight is False
    assert st.session_state.regenerate_index is None 