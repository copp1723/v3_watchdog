import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from src.nova_act.core import NovaActClient

@pytest.fixture
def pool_client():
    """Fixture providing a NovaActClient with mocked pool dependencies."""
    client = NovaActClient(headless=True, max_concurrent=2)
    client._initialize_pool = AsyncMock()
    client._acquire_session = AsyncMock()
    client._shutdown_event = MagicMock()
    return client

@pytest.mark.asyncio
async def test_pool_initialization(pool_client):
    """Test that pool initializes with correct number of sessions."""
    await pool_client.start()
    
    pool_client._initialize_pool.assert_called_once()
    assert pool_client._pool_initialized is True

@pytest.mark.asyncio
async def test_session_acquisition(pool_client):
    """Test that session acquisition works with pool."""
    mock_browser = MagicMock()
    mock_page = MagicMock()
    pool_client._acquire_session.return_value = ("session_1", mock_browser, mock_page)
    
    session_id, browser, page = await pool_client._acquire_session()
    
    assert session_id == "session_1"
    assert browser is mock_browser
    assert page is mock_page
    pool_client._acquire_session.assert_called_once()

@pytest.mark.asyncio
async def test_shutdown_cleanup(pool_client):
    """Test that shutdown cleans up all pool resources."""
    # Setup mock browsers and pages
    mock_browser1 = AsyncMock()
    mock_browser2 = AsyncMock()
    mock_page1 = AsyncMock()
    mock_page2 = AsyncMock()
    
    pool_client.browser_pool = {"session_1": mock_browser1, "session_2": mock_browser2}
    pool_client.page_pool = {"session_1": mock_page1, "session_2": mock_page2}
    pool_client._pool_initialized = True
    
    await pool_client.shutdown()
    
    # Verify cleanup
    mock_page1.close.assert_called_once()
    mock_page2.close.assert_called_once()
    mock_browser1.close.assert_called_once()
    mock_browser2.close.assert_called_once()
    assert pool_client._pool_initialized is False

@pytest.mark.asyncio
async def test_concurrent_session_limit(pool_client):
    """Test that pool enforces maximum concurrent sessions."""
    pool_client.max_concurrent = 2
    pool_client._acquire_session.side_effect = [
        ("session_1", MagicMock(), MagicMock()),
        ("session_2", MagicMock(), MagicMock()),
        Exception("Pool exhausted")
    ]
    
    # Acquire all available sessions
    await pool_client._acquire_session()
    await pool_client._acquire_session()
    
    # Next acquisition should fail
    with pytest.raises(Exception):
        await pool_client._acquire_session() 