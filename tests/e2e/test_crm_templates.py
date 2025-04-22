"""
Tests for CRM template functionality.
"""

import pytest
import logging
import os
from pathlib import Path
from datetime import datetime

from src.nova_act.core import NovaActConnector
from src.nova_act.exceptions import TemplateError

LOG_FILE = "/tmp/watchdog_nova_act_sync.log"

@pytest.fixture(autouse=True)
def setup_log_file():
    """Set up and clean log file before each test."""
    # Create /tmp directory if it doesn't exist
    os.makedirs("/tmp", exist_ok=True)
    
    # Remove existing log file if it exists
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
    
    # Configure logging
    logging.basicConfig(
        filename=LOG_FILE,
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        force=True  # Force reconfiguration of the root logger
    )
    
    # Add a file handler to ensure logs are written
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(file_handler)
    
    yield LOG_FILE
    
    # Cleanup after test
    logging.getLogger().removeHandler(file_handler)
    file_handler.close()

@pytest.mark.asyncio
async def test_dealersocket_template(setup_log_file):
    """Test DealerSocket template navigation."""
    connector = NovaActConnector()
    
    # Store credentials
    connector.store_credentials(
        vendor="DealerSocket",
        email="test@dealersocket.com",
        password="password123",
        two_fa_method="SMS",
        reports=["Sales Reports"],
        frequency="Daily"
    )
    
    # Execute login
    success = await connector.login()
    assert success is True
    
    # Fetch report
    files = await connector.fetch_report()
    assert len(files) == 1
    assert files[0] == "/tmp/watchdog_sales_reports_123456789.csv"
    
    # Check logs
    with open(LOG_FILE) as f:
        log_content = f.read()
        
    # Verify login steps were logged
    assert "Mock executing: enter username test@dealersocket.com in #username field" in log_content
    assert "Mock 2FA verification via SMS" in log_content
    
    # Verify report steps were logged
    assert "Mock executing: click #reports-menu to open reports section" in log_content
    assert "Mock download completed: /tmp/watchdog_sales_reports_123456789.csv" in log_content

@pytest.mark.asyncio
async def test_vinsolutions_template(setup_log_file):
    """Test VinSolutions template navigation."""
    connector = NovaActConnector()
    
    # Store credentials
    connector.store_credentials(
        vendor="VinSolutions",
        email="test@vinsolutions.com",
        password="password123",
        two_fa_method="Email",
        reports=["Sales Reports"],
        frequency="Weekly"
    )
    
    # Execute login
    success = await connector.login()
    assert success is True
    
    # Fetch report
    files = await connector.fetch_report()
    assert len(files) == 1
    assert files[0] == "/tmp/watchdog_sales_reports_123456789.csv"
    
    # Check logs
    with open(LOG_FILE) as f:
        log_content = f.read()
        
    # Verify login steps were logged
    assert "Mock executing: enter username test@vinsolutions.com in #email field" in log_content
    assert "Mock 2FA verification via Email" in log_content
    
    # Verify report steps were logged
    assert "Mock executing: click #reportingDashboard to open dashboard" in log_content
    assert "Mock download completed: /tmp/watchdog_sales_reports_123456789.csv" in log_content

@pytest.mark.asyncio
async def test_invalid_vendor():
    """Test handling of invalid vendor."""
    connector = NovaActConnector()
    
    # Try to store credentials for invalid vendor
    with pytest.raises(TemplateError) as exc_info:
        connector.store_credentials(
            vendor="InvalidVendor",
            email="test@test.com",
            password="password123"
        )
    
    assert "No template found for vendor" in str(exc_info.value)

@pytest.mark.asyncio
async def test_no_2fa_flow(setup_log_file):
    """Test login flow without 2FA."""
    connector = NovaActConnector()
    
    # Store credentials without 2FA
    connector.store_credentials(
        vendor="DealerSocket",
        email="test@dealersocket.com",
        password="password123",
        two_fa_method="None"
    )
    
    # Execute login
    success = await connector.login()
    assert success is True
    
    # Check logs
    with open(LOG_FILE) as f:
        log_content = f.read()
        
    # Verify login steps were logged but no 2FA
    assert "Mock executing: enter username test@dealersocket.com in #username field" in log_content
    assert "Mock 2FA verification" not in log_content

@pytest.mark.asyncio
async def test_error_indicators():
    """Test error indicator checking."""
    connector = NovaActConnector()
    
    # Store credentials
    connector.store_credentials(
        vendor="DealerSocket",
        email="test@dealersocket.com",
        password="password123"
    )
    
    # Check for errors
    error = await connector.check_error_indicators()
    assert error is None  # Mock implementation always returns None 