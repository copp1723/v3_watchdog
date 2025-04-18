"""
Enhanced unit tests for the insight conversation module.
Focuses on error handling, edge cases, and robust API integration.
"""

import pytest
import streamlit as st
import pandas as pd
import json
import traceback
from datetime import datetime
from unittest.mock import patch, MagicMock
from ..src.insight_conversation import ConversationManager, _load_system_prompt

@pytest.fixture
def conversation_manager():
    """Fixture providing a ConversationManager instance with mock responses."""
    return ConversationManager(use_mock=True)

@pytest.fixture
def sample_prompt():
    """Fixture providing a sample prompt."""
    return "How do sales compare to last month?"

@pytest.fixture
def sample_validation_context():
    """Fixture providing a sample validation context."""
    return {
        "data_shape": (100, 10),
        "columns": ["date", "sales", "region", "product"],
        "numeric_columns": ["sales"],
        "basic_stats": {
            "sales": {
                "count": 100,
                "mean": 1500.0,
                "std": 200.0,
                "min": 1000.0,
                "max": 2000.0
            }
        }
    }

@pytest.fixture
def sample_conversation_history():
    """Fixture providing a sample conversation history."""
    return [
        {
            "prompt": "Initial sales question",
            "response": {
                "summary": "Sales increased by 10% compared to last month.",
                "timestamp": "2023-05-01T10:00:00Z",
                "is_mock": True
            }
        },
        {
            "prompt": "What factors contributed to the sales increase?",
            "response": {
                "summary": "Key factors include better marketing and improved inventory.",
                "timestamp": "2023-05-01T10:05:00Z",
                "is_mock": True
            }
        }
    ]

def test_conversation_manager_init(conversation_manager):
    """Test initialization of ConversationManager."""
    assert conversation_manager.use_mock is True
    assert conversation_manager.client is None
    assert 'conversation_history' in st.session_state

def test_load_system_prompt_success():
    """Test successful loading of system prompt."""
    with patch("builtins.open", MagicMock()) as mock_open:
        mock_open.return_value.__enter__.return_value.read.return_value = "Test prompt"
        prompt = _load_system_prompt("test_prompt.md")
        assert prompt == "Test prompt"
        mock_open.assert_called_once_with("test_prompt.md", 'r', encoding='utf-8')

def test_load_system_prompt_file_not_found():
    """Test handling of missing system prompt file."""
    with patch("builtins.open", MagicMock()) as mock_open:
        mock_open.side_effect = FileNotFoundError("File not found")
        prompt = _load_system_prompt("nonexistent.md")
        assert "You are a helpful data analyst" in prompt
        assert "JSON format" in prompt

def test_load_system_prompt_other_error():
    """Test handling of other errors when loading system prompt."""
    with patch("builtins.open", MagicMock()) as mock_open:
        mock_open.side_effect = Exception("Other error")
        prompt = _load_system_prompt("test_prompt.md")
        assert "You are a helpful data analyst" in prompt
        assert "JSON format" in prompt

def test_generate_insight_with_empty_prompt(conversation_manager):
    """Test generating insight with an empty prompt."""
    response = conversation_manager.generate_insight("")
    assert "error" in response
    assert "input_validation" in response["error_type"]
    assert "non-empty string" in response["error"]

def test_generate_insight_with_invalid_prompt_type(conversation_manager):
    """Test generating insight with an invalid prompt type."""
    response = conversation_manager.generate_insight(None)
    assert "error" in response
    assert "input_validation" in response["error_type"]
    assert "non-empty string" in response["error"]

def test_generate_insight_with_invalid_validation_context(conversation_manager, sample_prompt):
    """Test generating insight with invalid validation context."""
    response = conversation_manager.generate_insight(sample_prompt, validation_context="not a dict")
    assert "error" not in response  # Should handle gracefully and convert to empty dict
    assert "summary" in response

def test_generate_insight_with_mock_response(conversation_manager, sample_prompt, sample_validation_context):
    """Test generating insight with mock response."""
    response = conversation_manager.generate_insight(sample_prompt, sample_validation_context)
    assert "summary" in response
    assert "timestamp" in response
    assert response["is_mock"] is True
    assert len(st.session_state.conversation_history) == 1
    assert st.session_state.conversation_history[0]["prompt"] == sample_prompt

def test_generate_insight_with_system_prompt_error(conversation_manager, sample_prompt):
    """Test handling of system prompt loading error."""
    with patch("src.insight_conversation._load_system_prompt", MagicMock()) as mock_load:
        mock_load.side_effect = Exception("System prompt error")
        response = conversation_manager.generate_insight(sample_prompt)
        assert "error" in response
        assert "system_prompt" in response["error_type"]
        assert "System prompt error" in response["error"]

def test_generate_insight_with_prompt_generation_error(conversation_manager, sample_prompt):
    """Test handling of prompt generation error."""
    with patch("src.insight_conversation.PromptGenerator", MagicMock()) as mock_generator:
        mock_generator.return_value.generate_prompt.side_effect = Exception("Prompt generation error")
        response = conversation_manager.generate_insight(sample_prompt)
        assert "error" in response
        assert "prompt_generation" in response["error_type"]
        assert "Prompt generation error" in response["error"]

def test_generate_insight_with_api_error(conversation_manager, sample_prompt):
    """Test handling of API error."""
    # Create a non-mock conversation manager
    cm = ConversationManager(use_mock=False)
    
    # Mock the API call to simulate an error
    with patch.object(cm, "_call_openai", MagicMock()) as mock_call:
        mock_call.side_effect = Exception("API error")
        response = cm.generate_insight(sample_prompt)
        assert "error" in response
        assert "api_call" in response["error_type"]
        assert "API error" in response["error"]

def test_generate_insight_with_response_processing_error(conversation_manager, sample_prompt):
    """Test handling of response processing error."""
    with patch.object(conversation_manager, "_generate_mock_response", MagicMock()) as mock_response:
        mock_response.return_value = "Invalid JSON response"
        response = conversation_manager.generate_insight(sample_prompt)
        assert "error" in response
        assert "response_processing" in response["error_type"]

def test_generate_insight_with_unknown_error(conversation_manager, sample_prompt):
    """Test handling of unknown error."""
    with patch.object(conversation_manager, "_generate_mock_response", MagicMock()) as mock_response:
        mock_response.side_effect = Exception("Unknown error")
        response = conversation_manager.generate_insight(sample_prompt)
        assert "error" in response
        assert "unknown" in response["error_type"]
        assert "Unknown error" in response["error"]

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
    from ..src.insight_conversation import render_conversation_history
    st.session_state['conversation_history'] = []
    
    # This test just ensures it doesn't crash with empty history
    render_conversation_history(st.session_state['conversation_history'])

def test_render_conversation_history_with_data(sample_conversation_history):
    """Test rendering conversation history with data."""
    from ..src.insight_conversation import render_conversation_history
    st.session_state['conversation_history'] = sample_conversation_history
    
    # This test just ensures it doesn't crash with data
    render_conversation_history(st.session_state['conversation_history'])
