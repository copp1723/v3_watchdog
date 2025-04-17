"""
Shared test fixtures and configuration for Nova Act tests.
"""

import pytest
import os
import tempfile
import shutil
from typing import Dict, Any

@pytest.fixture(scope="session")
def test_storage_dir():
    """Create a temporary directory for test storage."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture(scope="session")
def test_data() -> Dict[str, Any]:
    """Provide consistent test data across tests."""
    return {
        "credentials": {
            "username": "test_user",
            "password": "test_pass",
            "url": "https://test.vinsolutions.com",
            "username_selector": "#username",
            "password_selector": "#password",
            "2fa_method": "sms",
            "2fa_config": {
                "phone": "+1234567890"
            }
        },
        "report_configs": {
            "sales": {
                "type": "sales",
                "path": ["reports", "sales"],
                "download_selector": "#downloadReport",
                "file_pattern": "*.csv",
                "format": "csv"
            },
            "leads": {
                "type": "leads",
                "path": ["reports", "leads"],
                "download_selector": "#downloadReport",
                "file_pattern": "*.csv",
                "format": "csv"
            },
            "inventory": {
                "type": "inventory",
                "path": ["reports", "inventory"],
                "download_selector": "#downloadReport",
                "file_pattern": "*.csv",
                "format": "csv"
            }
        },
        "mock_responses": {
            "login_success": {
                "status": "success",
                "message": "Login successful"
            },
            "login_failure": {
                "status": "error",
                "message": "Invalid credentials"
            },
            "2fa_required": {
                "status": "2fa_required",
                "message": "2FA verification required"
            },
            "download_success": {
                "status": "success",
                "file_path": "test_report.csv"
            }
        }
    }

@pytest.fixture(autouse=True)
def mock_env_vars():
    """Automatically set required environment variables for tests."""
    original_env = dict(os.environ)
    
    # Set test environment variables
    os.environ.update({
        "NOVA_ACT_API_KEY": "test_nova_act_key",
        "WATCHDOG_CRED_KEY": "test_encryption_key",
        "USE_MOCK": "true"
    })
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)

@pytest.fixture
def mock_multion():
    """Provide a mock MultiOn client."""
    class MockMultiOn:
        def __init__(self, *args, **kwargs):
            self.sessions = MockSessions()
    
    class MockSessions:
        async def create(self, *args, **kwargs):
            return type("MockSession", (), {"session_id": "test_session_123"})
        
        async def step(self, *args, **kwargs):
            return {"status": "success"}
        
        async def wait_for_download(self, *args, **kwargs):
            return True
        
        async def close(self, *args, **kwargs):
            return True
    
    return MockMultiOn