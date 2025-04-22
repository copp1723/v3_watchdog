# tests/test_insights/test_delivery_handlers.py

import pytest
import streamlit as st
from datetime import datetime
from unittest.mock import Mock, patch
from src.watchdog_ai.insights.delivery_handlers import (
    InsightDeliveryHandler,
    StreamlitDeliveryHandler,
    SlackDeliveryHandler,
    EmailDeliveryHandler
)

@pytest.fixture
def sample_insight():
    """Fixture providing a sample insight for testing."""
    return {
        'summary': 'Test insight summary',
        'metrics': {'value': 100},
        'recommendations': ['Test recommendation'],
        'confidence': 'high',
        'chart_data': {
            'type': 'bar',
            'data': {'x': [1, 2, 3], 'y': [4, 5, 6]},
            'title': 'Test Chart'
        }
    }

@pytest.fixture
def mock_streamlit():
    """Mock streamlit functions."""
    with patch('streamlit.markdown') as mock_markdown, \
         patch('streamlit.bar_chart') as mock_bar_chart, \
         patch('streamlit.line_chart') as mock_line_chart:
        yield {
            'markdown': mock_markdown,
            'bar_chart': mock_bar_chart,
            'line_chart': mock_line_chart
        }

class TestInsightDeliveryHandler:
    """Test suite for base InsightDeliveryHandler."""
    
    def test_init(self):
        """Test initialization."""
        handler = InsightDeliveryHandler()
        assert handler.delivery_type == "base"
        assert handler.supported_formats == ["text"]
        
    def test_format_insight_text(self):
        """Test text formatting of insights."""
        handler = InsightDeliveryHandler()
        insight = {
            'summary': 'Test summary',
            'metrics': {'value': 100},
            'recommendations': ['Do this']
        }
        formatted = handler.format_insight(insight, 'text')
        assert 'Test summary' in formatted
        assert '100' in formatted
        assert 'Do this' in formatted

class TestStreamlitDeliveryHandler:
    """Test suite for StreamlitDeliveryHandler."""
    
    def test_init(self):
        """Test initialization."""
        handler = StreamlitDeliveryHandler()
        assert handler.delivery_type == "streamlit"
        assert "markdown" in handler.supported_formats
        assert "interactive" in handler.supported_formats
        
    def test_deliver_markdown(self, sample_insight, mock_streamlit):
        """Test markdown delivery format."""
        handler = StreamlitDeliveryHandler()
        handler.deliver(sample_insight, format='markdown')
        mock_streamlit['markdown'].assert_called()
        assert 'Test insight summary' in mock_streamlit['markdown'].call_args[0][0]
        
    def test_deliver_interactive(self, sample_insight, mock_streamlit):
        """Test interactive delivery format."""
        handler = StreamlitDeliveryHandler()
        handler.deliver(sample_insight, format='interactive')
        # Verify chart rendering
        mock_streamlit['bar_chart'].assert_called_with(sample_insight['chart_data']['data'])

class TestSlackDeliveryHandler:
    """Test suite for SlackDeliveryHandler."""
    
    @pytest.fixture
    def mock_slack_client(self):
        """Mock Slack client."""
        return Mock()
    
    def test_init(self):
        """Test initialization."""
        handler = SlackDeliveryHandler(token="test-token")
        assert handler.delivery_type == "slack"
        assert "blocks" in handler.supported_formats
        
    def test_deliver_blocks(self, sample_insight, mock_slack_client):
        """Test blocks delivery format."""
        handler = SlackDeliveryHandler(token="test-token")
        handler._client = mock_slack_client
        
        handler.deliver(sample_insight, channel="#test", format='blocks')
        
        mock_slack_client.chat_postMessage.assert_called_once()
        call_args = mock_slack_client.chat_postMessage.call_args[1]
        assert call_args['channel'] == "#test"
        assert any('Test insight summary' in str(block) for block in call_args['blocks'])

class TestEmailDeliveryHandler:
    """Test suite for EmailDeliveryHandler."""
    
    @pytest.fixture
    def mock_smtp(self):
        """Mock SMTP client."""
        with patch('smtplib.SMTP') as mock:
            yield mock
    
    def test_init(self):
        """Test initialization."""
        handler = EmailDeliveryHandler(
            smtp_host="localhost",
            smtp_port=587,
            username="test",
            password="test"
        )
        assert handler.delivery_type == "email"
        assert "html" in handler.supported_formats
        
    def test_deliver_html(self, sample_insight, mock_smtp):
        """Test HTML delivery format."""
        handler = EmailDeliveryHandler(
            smtp_host="localhost",
            smtp_port=587,
            username="test",
            password="test"
        )
        
        handler.deliver(
            sample_insight,
            recipients=["test@example.com"],
            format='html'
        )
        
        mock_smtp.return_value.__enter__.return_value.send_message.assert_called_once()
        message = mock_smtp.return_value.__enter__.return_value.send_message.call_args[0][0]
        assert 'Test insight summary' in str(message)
        assert message['To'] == "test@example.com"

def test_delivery_handler_registry():
    """Test that all delivery handlers are properly registered."""
    from src.watchdog_ai.insights.delivery_handlers import get_delivery_handler
    
    # Test getting each type of handler
    streamlit_handler = get_delivery_handler("streamlit")
    assert isinstance(streamlit_handler, StreamlitDeliveryHandler)
    
    slack_handler = get_delivery_handler("slack", token="test-token")
    assert isinstance(slack_handler, SlackDeliveryHandler)
    
    email_handler = get_delivery_handler("email", 
                                       smtp_host="localhost",
                                       smtp_port=587,
                                       username="test",
                                       password="test")
    assert isinstance(email_handler, EmailDeliveryHandler)
    
    # Test invalid handler type
    with pytest.raises(ValueError):
        get_delivery_handler("invalid")

