"""
Unit tests for AgentOps integration.
"""

import pytest
import os
import pandas as pd
from unittest.mock import patch, MagicMock
from src.utils.agentops_config import init_agentops, get_handler, AgentOpsConfig
from src.llm_engine import LLMEngine
from src.insights.intent_manager import IntentManager

@pytest.fixture
def mock_env_vars():
    """Set up test environment variables."""
    os.environ["AGENTOPS_API_KEY"] = "test-key"
    os.environ["OPENAI_API_KEY"] = "test-openai-key"
    yield
    del os.environ["AGENTOPS_API_KEY"]
    del os.environ["OPENAI_API_KEY"]

def test_agentops_initialization(mock_env_vars):
    """Test AgentOps initialization."""
    with patch('agentops.init') as mock_init:
        result = init_agentops()
        assert result is True
        mock_init.assert_called_once_with("test-key", tags=["watchdog-ai"])
        
def test_agentops_config_initialization(mock_env_vars):
    """Test AgentOpsConfig initialization."""
    with patch('agentops.init') as mock_init, \
         patch('agentops.get_handler') as mock_get_handler:
        config = AgentOpsConfig()
        assert config.enabled is True
        mock_init.assert_called_once_with(api_key="test-key", tags=["watchdog-ai"])

def test_agentops_initialization_no_key():
    """Test AgentOps initialization without API key."""
    result = init_agentops()
    assert result is False

def test_get_handler(mock_env_vars):
    """Test getting AgentOps handler with tags."""
    handler_dict = {'tags': {'service': 'watchdog-ai', 'session_id': 'test-session', 'query_type': 'test-query'}}
    with patch('src.utils.agentops_config.os.getenv', return_value='test-key'):
        handler = get_handler(
            session_id="test-session",
            query_type="test-query"
        )
        assert handler is not None
        assert handler['tags']['service'] == 'watchdog-ai'
        assert handler['tags']['session_id'] == 'test-session'
        assert handler['tags']['query_type'] == 'test-query'

def test_llm_engine_integration(mock_env_vars):
    """Test LLM Engine integration with AgentOps."""
    with patch('src.llm_engine.OpenAI'), \
         patch('agentops.track') as mock_track, \
         patch.object(AgentOpsConfig, '__init__', return_value=None), \
         patch.object(AgentOpsConfig, 'enabled', True), \
         patch.object(AgentOpsConfig, 'track', return_value=MagicMock()):
        
        engine = LLMEngine(
            use_mock=False,
            api_key="test-key",
            session_id="test-session"
        )
        
        # Test insight generation
        result = engine.generate_insight(
            "test prompt",
            {"validated_data": pd.DataFrame()}
        )
        
        # Verify AgentOps track method was called
        assert engine.agentops.track.called

def test_intent_manager_integration(mock_env_vars):
    """Test Intent Manager integration with AgentOps."""
    with patch('agentops.track') as mock_track, \
         patch.object(AgentOpsConfig, '__init__', return_value=None), \
         patch.object(AgentOpsConfig, 'enabled', True), \
         patch.object(AgentOpsConfig, 'track', return_value=MagicMock()):
        
        manager = IntentManager(session_id="test-session")
        
        # Test intent detection
        result = manager.generate_insight(
            "test prompt",
            pd.DataFrame()
        )
        
        # Verify AgentOps track method was called
        assert manager.agentops.track.called

def test_error_handling(mock_env_vars):
    """Test error handling with AgentOps."""
    with patch('agentops.init') as mock_init:
        mock_init.side_effect = Exception("Test error")
        result = init_agentops()
        assert result is False

def test_custom_tags(mock_env_vars):
    """Test custom tag handling."""
    with patch('agentops.AgentOpsLangchainCallbackHandler') as mock_handler:
        handler = get_handler(
            session_id="test-session",
            query_type="test-query",
            additional_tags={"custom": "tag"}
        )
        assert handler is not None
        mock_handler.assert_called_once()
        # Verify tags were passed
        args = mock_handler.call_args
        assert "custom" in args[1]["default_tags"]
        
def test_track_context_manager(mock_env_vars):
    """Test the track context manager."""
    with patch('agentops.track') as mock_track:
        mock_context_manager = MagicMock()
        mock_track.return_value = mock_context_manager
        
        config = AgentOpsConfig()
        # Manually set enabled for testing
        config.enabled = True
        
        # Test with all parameters
        with config.track("test_operation", "test-session", "test query"):
            pass
            
        # Verify agentops.track was called with correct tags
        mock_track.assert_called_once()
        # Get the first call arguments
        call_args = mock_track.call_args
        tags = call_args[1]["tags"]
        assert tags["operation_type"] == "test_operation"
        assert tags["service"] == "watchdog-ai"
        assert tags["session_id"] == "test-session"
        assert "query_hash" in tags
        assert tags["query_length"] == 10
        
        # Verify the context manager was used
        mock_context_manager.__enter__.assert_called_once()
        mock_context_manager.__exit__.assert_called_once()