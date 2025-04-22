"""
Test module for the NovaActConnector class.
"""

import os
import sys
import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.nova_act.core import NovaActConnector
from src.nova_act.enhanced_credentials import EnhancedCredentialManager
from src.nova_act.fallback import NovaFallback

# Test credentials
TEST_CREDENTIALS = {
    "username": "test_user",
    "password": "test_password",
    "url": "https://test.example.com",
    "2fa_method": "sms",
    "2fa_config": {
        "mock_code": "123456"
    }
}

# Mock vendor configs for testing
MOCK_VENDOR_CONFIGS = {
    "test_vendor": {
        "login_url": "https://test.example.com/login",
        "selectors": {
            "username": "#username",
            "password": "#password",
            "submit": "#login-button",
            "login_error": ".error-message"
        },
        "reports": {
            "test_report": {
                "path": ["reports", "test_report"],
                "selectors": {
                    "reports": "#reports-link",
                    "test_report": "#test-report-link"
                },
                "download_selector": "#download-button",
                "file_pattern": r"report_\d+\.csv"
            }
        }
    }
}

@pytest.fixture
def mock_credential_manager():
    """Mock credential manager for testing."""
    manager = MagicMock(spec=EnhancedCredentialManager)
    manager.get_credential = AsyncMock(return_value=TEST_CREDENTIALS)
    return manager

@pytest.fixture
def mock_fallback_handler():
    """Mock fallback handler for testing."""
    handler = MagicMock(spec=NovaFallback)
    handler.register_fallback_action = MagicMock()
    return handler

@pytest.fixture
def connector(mock_credential_manager, mock_fallback_handler):
    """Create a NovaActConnector instance for testing."""
    with patch('src.nova_act.core.VENDOR_CONFIGS', MOCK_VENDOR_CONFIGS):
        connector = NovaActConnector(
            headless=True,
            max_concurrent=1,
            download_dir="/tmp/test_nova_act",
            credential_manager=mock_credential_manager,
            fallback_handler=mock_fallback_handler
        )
        yield connector

@pytest.mark.asyncio
async def test_connector_initialization(connector):
    """Test that the connector initializes correctly."""
    assert connector.headless is True
    assert connector.max_concurrent == 1
    assert connector.download_dir == "/tmp/test_nova_act"
    assert connector.credential_manager is not None
    assert connector.fallback_handler is not None

@pytest.mark.asyncio
async def test_acquire_session(connector):
    """Test that the connector can acquire a session."""
    with patch('src.nova_act.core.async_playwright') as mock_playwright:
        # Mock the necessary objects and methods
        mock_context = AsyncMock()
        mock_page = AsyncMock()
        
        mock_browser = AsyncMock()
        mock_browser.new_context = AsyncMock(return_value=mock_context)
        
        mock_context.new_page = AsyncMock(return_value=mock_page)
        
        mock_playwright_instance = AsyncMock()
        mock_playwright_instance.chromium.launch = AsyncMock(return_value=mock_browser)
        
        mock_playwright.start = AsyncMock(return_value=mock_playwright_instance)
        
        # Start the connector
        await connector.start()
        
        # Call the method under test
        session_id, browser, page = await connector._acquire_session()
        
        # Assertions
        assert session_id is not None
        assert browser is not None
        assert page is not None
        assert mock_playwright_instance.chromium.launch.called
        assert mock_browser.new_context.called
        assert mock_context.new_page.called

@pytest.mark.asyncio
async def test_handle_navigation_success(connector):
    """Test navigation handling with successful response."""
    mock_page = AsyncMock()
    mock_response = AsyncMock()
    mock_response.ok = True
    
    mock_page.goto = AsyncMock(return_value=mock_response)
    mock_page.wait_for_load_state = AsyncMock()
    
    result = await connector._handle_navigation(mock_page, "https://test.example.com")
    
    assert result is True
    mock_page.goto.assert_called_once_with("https://test.example.com", timeout=15000)
    mock_page.wait_for_load_state.assert_called_once_with("networkidle")

@pytest.mark.asyncio
async def test_handle_navigation_failure(connector):
    """Test navigation handling with failed response."""
    mock_page = AsyncMock()
    mock_response = AsyncMock()
    mock_response.ok = False
    mock_response.status = 404
    
    mock_page.goto = AsyncMock(return_value=mock_response)
    
    result = await connector._handle_navigation(mock_page, "https://test.example.com")
    
    assert result is False
    mock_page.goto.assert_called_once_with("https://test.example.com", timeout=15000)

@pytest.mark.asyncio
async def test_handle_login_success(connector):
    """Test login handling with successful login."""
    mock_page = AsyncMock()
    mock_page.fill = AsyncMock()
    mock_page.click = AsyncMock()
    mock_page.wait_for_load_state = AsyncMock()
    mock_page.query_selector = AsyncMock(return_value=None)  # No error element
    
    selectors = {
        "username": "#username",
        "password": "#password",
        "submit": "#login-button",
        "login_error": ".error-message"
    }
    
    credentials = {
        "username": "test_user",
        "password": "test_password"
    }
    
    result = await connector._handle_login(mock_page, credentials, selectors)
    
    assert result is True
    mock_page.fill.assert_any_call("#username", "test_user")
    mock_page.fill.assert_any_call("#password", "test_password")
    mock_page.click.assert_called_once_with("#login-button")
    mock_page.wait_for_load_state.assert_called_once_with("networkidle")
    mock_page.query_selector.assert_called_once_with(".error-message")

@pytest.mark.asyncio
async def test_handle_login_failure(connector):
    """Test login handling with failed login."""
    mock_page = AsyncMock()
    mock_page.fill = AsyncMock()
    mock_page.click = AsyncMock()
    mock_page.wait_for_load_state = AsyncMock()
    
    # Mock error element
    mock_error = AsyncMock()
    mock_error.text_content = AsyncMock(return_value="Invalid credentials")
    mock_page.query_selector = AsyncMock(return_value=mock_error)
    
    selectors = {
        "username": "#username",
        "password": "#password",
        "submit": "#login-button",
        "login_error": ".error-message"
    }
    
    credentials = {
        "username": "test_user",
        "password": "wrong_password"
    }
    
    result = await connector._handle_login(mock_page, credentials, selectors)
    
    assert result is False
    mock_page.fill.assert_any_call("#username", "test_user")
    mock_page.fill.assert_any_call("#password", "wrong_password")
    mock_page.click.assert_called_once_with("#login-button")
    mock_page.wait_for_load_state.assert_called_once_with("networkidle")
    mock_page.query_selector.assert_called_once_with(".error-message")
    mock_error.text_content.assert_called_once()

@pytest.mark.asyncio
async def test_collect_report_success(connector):
    """Test successful report collection."""
    # Mock necessary methods
    connector._get_credentials = AsyncMock(return_value=TEST_CREDENTIALS)
    connector._handle_navigation = AsyncMock(return_value=True)
    connector._handle_login = AsyncMock(return_value=True)
    connector._handle_download = AsyncMock(return_value="/tmp/test_nova_act/report.csv")
    connector._process_downloaded_report = AsyncMock(return_value={"success": True})
    
    # Mock _acquire_session to return mock objects
    mock_browser = AsyncMock()
    mock_page = AsyncMock()
    connector._acquire_session = AsyncMock(return_value=("test_session", mock_browser, mock_page))
    
    # Mock the health_checker and rate_limiter
    with patch('src.nova_act.core.health_checker') as mock_health_checker, \
         patch('src.nova_act.core.rate_limiter') as mock_rate_limiter, \
         patch('src.nova_act.core.metrics_collector') as mock_metrics:
        
        mock_health_checker.check_vendor_health = AsyncMock(return_value={"status": "healthy"})
        mock_rate_limiter.acquire = AsyncMock(return_value=True)
        mock_rate_limiter.release = AsyncMock()
        mock_metrics.record_operation = AsyncMock()
        
        # Call the method under test
        result = await connector.collect_report(
            vendor="test_vendor",
            dealer_id="test_dealer",
            report_type="test_report"
        )
        
        # Assertions
        assert result["success"] is True
        assert "file_path" in result
        assert "duration" in result
        assert "timestamp" in result
        assert "processing" in result
        
        connector._get_credentials.assert_called_once_with("test_vendor", "test_dealer")
        connector._handle_navigation.assert_called_once()
        connector._handle_login.assert_called_once()
        connector._handle_download.assert_called_once()
        connector._process_downloaded_report.assert_called_once_with(
            "/tmp/test_nova_act/report.csv", 
            "test_vendor", 
            "test_dealer", 
            "test_report"
        )