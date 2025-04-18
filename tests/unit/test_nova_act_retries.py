import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from src.nova_act.core import NovaActClient

@pytest.fixture
def mock_client():
    """Fixture providing a NovaActClient with mocked dependencies."""
    client = NovaActClient(headless=True)
    client._retry_operation = AsyncMock()
    return client

@pytest.mark.asyncio
async def test_navigation_retry(mock_client):
    """Test that navigation failures are automatically retried."""
    mock_page = MagicMock()
    mock_url = "https://example.com"
    
    # Configure mock to fail first attempt, succeed on retry
    mock_client._retry_operation.side_effect = [False, True]
    
    result = await mock_client._handle_navigation(mock_page, mock_url)
    
    assert mock_client._retry_operation.call_count == 1
    assert result is True

@pytest.mark.asyncio
async def test_login_retry(mock_client):
    """Test that login failures are automatically retried."""
    mock_page = MagicMock()
    mock_credentials = {"username": "test", "password": "test"}
    mock_selectors = {"username": "#user", "password": "#pass", "submit": "#login"}
    
    # Configure mock to fail first attempt, succeed on retry
    mock_client._retry_operation.side_effect = [False, True]
    
    result = await mock_client._handle_login(mock_page, mock_credentials, mock_selectors)
    
    assert mock_client._retry_operation.call_count == 1
    assert result is True

@pytest.mark.asyncio
async def test_max_retries_exceeded(mock_client):
    """Test that operations fail after exceeding max retries."""
    mock_page = MagicMock()
    mock_url = "https://example.com"
    
    # Configure mock to always fail
    mock_client._retry_operation.return_value = False
    
    result = await mock_client._handle_navigation(mock_page, mock_url)
    
    assert mock_client._retry_operation.call_count == 1
    assert result is False 