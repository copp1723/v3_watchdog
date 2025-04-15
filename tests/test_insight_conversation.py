"""
Unit tests for the insight conversation module.
"""

import pytest
import streamlit as st
import pandas as pd
import json
from src.insight_conversation import ConversationManager

@pytest.fixture
def conversation_manager():
    """Fixture providing a ConversationManager instance with mock responses."""
    return ConversationManager(use_mock=True)

@pytest.fixture
def sample_prompt():
    """Fixture providing a sample prompt."""
    return "How do sales compare to last month?"

@pytest.fixture
def sample_json_prompt():
    """Fixture providing a sample JSON formatted prompt."""
    return json.dumps({
        "query": "How do sales compare to last month?",
        "context": {"entity": "sales", "timeframe": "last month"},
        "previous_insights": []
    })

@pytest.fixture
def sample_conversation_history():
    """Fixture providing a sample conversation history."""
    return [
        {
            "prompt": "Initial sales question",
            "response": {
                "summary": "Sales increased by 10% compared to last month.",
                "timestamp": "2023-05-01T10:00:00Z"
            }
        },
        {
            "prompt": "What factors contributed to the sales increase?",
            "response": {
                "summary": "Key factors include better marketing and improved inventory.",
                "timestamp": "2023-05-01T10:05:00Z"
            }
        }
    ]

def test_conversation_manager_init(conversation_manager):
    """Test initialization of ConversationManager."""
    assert conversation_manager.use_mock is True
    assert conversation_manager.client is None
    assert 'conversation_history' in st.session_state

def test_get_mock_response_simple_prompt(conversation_manager, sample_prompt):
    """Test mock response generation for a simple prompt."""
    response = conversation_manager.get_mock_response(sample_prompt)
    
    assert isinstance(response, dict)
    assert 'summary' in response
    assert 'chart_data' in response
    assert 'recommendation' in response
    assert 'risk_flag' in response

def test_get_mock_response_json_prompt(conversation_manager, sample_json_prompt):
    """Test mock response generation for a JSON formatted prompt."""
    response = conversation_manager.get_mock_response(sample_json_prompt)
    
    assert isinstance(response, dict)
    assert 'summary' in response
    assert 'How do sales compare to last month?' in response['summary']

def test_generate_insight(conversation_manager, sample_prompt):
    """Test generating an insight with a mock response."""
    # Clear history before test
    st.session_state['conversation_history'] = []
    
    response = conversation_manager.generate_insight(sample_prompt)
    
    assert isinstance(response, dict)
    assert len(st.session_state['conversation_history']) == 1
    assert st.session_state['conversation_history'][0]['prompt'] == sample_prompt
    assert st.session_state['conversation_history'][0]['response'] == response

def test_regenerate_insight(conversation_manager, sample_conversation_history):
    """Test regenerating an insight for a previous prompt index."""
    # Set up history
    st.session_state['conversation_history'] = sample_conversation_history.copy()
    original_length = len(st.session_state['conversation_history'])
    
    # Regenerate the second entry (index 1)
    new_response = conversation_manager.regenerate_insight(1)
    
    assert new_response is not None
    assert 'summary' in new_response
    assert len(st.session_state['conversation_history']) == original_length

def test_regenerate_insight_invalid_index(conversation_manager):
    """Test regenerating an insight with an invalid index returns None."""
    st.session_state['conversation_history'] = []
    
    result = conversation_manager.regenerate_insight(5)
    assert result is None
    result_neg = conversation_manager.regenerate_insight(-1)
    assert result_neg is None

def test_render_conversation_history_empty():
    """Test rendering conversation history when empty."""
    from src.insight_conversation import render_conversation_history
    st.session_state['conversation_history'] = []
    
    # This test just ensures it doesn't crash with empty history
    render_conversation_history(st.session_state['conversation_history'])

def test_render_conversation_history_with_data(sample_conversation_history):
    """Test rendering conversation history with data."""
    from src.insight_conversation import render_conversation_history
    st.session_state['conversation_history'] = sample_conversation_history
    
    # This test just ensures it doesn't crash with data
    render_conversation_history(st.session_state['conversation_history']) 