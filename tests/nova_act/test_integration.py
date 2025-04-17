"""
Integration tests for Nova Act components.
Tests the interaction between Nova Act manager, credential storage, and collectors.
"""

import pytest
import os
import json
from datetime import datetime
from typing import Dict, Any
import asyncio
from unittest.mock import MagicMock, patch
import schedule
import streamlit as st

from src.nova_act.core import NovaActManager
from src.nova_act.credentials import CredentialManager
from src.nova_act.collectors.vinsolutions import VinSolutionsCollector
from src.nova_act.scheduler import NovaScheduler
from src.nova_act.fallback import NovaFallback

# Test data
TEST_CREDENTIALS = {
    "username": "test_user",
    "password": "test_pass",
    "url": "https://test.vinsolutions.com",
    "username_selector": "#username",
    "password_selector": "#password",
    "2fa_method": "sms",
    "2fa_config": {
        "phone": "+1234567890"
    }
}

@pytest.fixture
def encryption_key():
    """Provide a consistent encryption key for tests."""
    return "dGVzdF9rZXlfZm9yX2NyZWRlbnRpYWxfZW5jcnlwdGlvbl8="

@pytest.fixture
def credential_manager(encryption_key):
    """Fixture providing a configured CredentialManager."""
    return CredentialManager(encryption_key=encryption_key)

@pytest.fixture
def nova_manager():
    """Fixture providing a configured NovaActManager."""
    with patch.dict(os.environ, {'NOVA_ACT_API_KEY': 'test_key'}):
        return NovaActManager()

@pytest.fixture
def vinsolutions_collector(nova_manager):
    """Fixture providing a configured VinSolutionsCollector."""
    return VinSolutionsCollector(nova_manager)

@pytest.mark.asyncio
async def test_end_to_end_data_collection(
    credential_manager,
    nova_manager,
    vinsolutions_collector
):
    """Test complete flow from credential storage to data collection."""
    # Store test credentials
    vendor_id = "vinsolutions_test"
    assert credential_manager.store_credentials(vendor_id, TEST_CREDENTIALS)
    
    # Retrieve and verify credentials
    stored_creds = credential_manager.get_credentials(vendor_id)
    assert stored_creds is not None
    assert stored_creds["username"] == TEST_CREDENTIALS["username"]
    
    # Mock MultiOn client responses
    mock_session = MagicMock()
    mock_session.session_id = "test_session_123"
    nova_manager.client.sessions.create = MagicMock(return_value=mock_session)
    nova_manager.client.sessions.step = MagicMock(return_value={"status": "success"})
    nova_manager.client.sessions.wait_for_download = MagicMock(return_value=True)
    nova_manager.client.sessions.close = MagicMock(return_value=True)
    
    # Test sales report collection
    result = await vinsolutions_collector.collect_sales_report(stored_creds)
    assert result["success"] is True
    assert result["vendor"] == "vinsolutions"
    assert "file_path" in result
    assert result["metadata"]["type"] == "sales"
    
    # Verify the correct sequence of API calls
    nova_manager.client.sessions.create.assert_called_once()
    assert nova_manager.client.sessions.step.call_count >= 2  # Login + navigation
    nova_manager.client.sessions.close.assert_called_once()

@pytest.mark.asyncio
async def test_error_handling_and_recovery(
    credential_manager,
    nova_manager,
    vinsolutions_collector
):
    """Test error handling and recovery during data collection."""
    # Store test credentials
    vendor_id = "vinsolutions_error_test"
    credential_manager.store_credentials(vendor_id, TEST_CREDENTIALS)
    
    # Simulate login failure
    nova_manager.client.sessions.create = MagicMock(return_value=MagicMock(session_id="test_session"))
    nova_manager.client.sessions.step = MagicMock(side_effect=Exception("Login failed"))
    
    # Test error handling during collection
    result = await vinsolutions_collector.collect_sales_report(TEST_CREDENTIALS)
    assert result["success"] is False
    assert "error" in result
    assert "Login failed" in result["error"]
    
    # Verify session cleanup was attempted
    nova_manager.client.sessions.close.assert_called_once()

@pytest.mark.asyncio
async def test_2fa_handling(nova_manager):
    """Test handling of different 2FA methods."""
    # Test SMS 2FA
    sms_result = await nova_manager._handle_2fa(
        "test_session",
        "sms",
        {"phone": "+1234567890"}
    )
    assert isinstance(sms_result, dict)
    assert "success" in sms_result
    
    # Test email 2FA
    email_result = await nova_manager._handle_2fa(
        "test_session",
        "email",
        {"email": "test@example.com"}
    )
    assert isinstance(email_result, dict)
    assert "success" in email_result
    
    # Test authenticator 2FA
    auth_result = await nova_manager._handle_2fa(
        "test_session",
        "authenticator",
        {"code": "123456"}
    )
    assert isinstance(auth_result, dict)
    assert "success" in auth_result
    
    # Test unsupported 2FA method
    invalid_result = await nova_manager._handle_2fa(
        "test_session",
        "unsupported",
        {}
    )
    assert invalid_result["success"] is False
    assert "Unsupported 2FA method" in invalid_result["error"]

def test_credential_encryption(credential_manager):
    """Test encryption and decryption of credentials."""
    vendor_id = "test_encryption"
    test_creds = {
        "username": "test_user",
        "password": "very_secret",
        "api_key": "super_secret_key"
    }
    
    # Store credentials
    assert credential_manager.store_credentials(vendor_id, test_creds)
    
    # Read the encrypted file directly
    file_path = os.path.join(credential_manager.storage_path, f"{vendor_id}.enc")
    with open(file_path, "rb") as f:
        encrypted_data = f.read()
    
    # Verify data is actually encrypted
    assert test_creds["password"].encode() not in encrypted_data
    assert test_creds["api_key"].encode() not in encrypted_data
    
    # Verify decryption works
    decrypted_creds = credential_manager.get_credentials(vendor_id)
    assert decrypted_creds == test_creds

def test_credential_updates(credential_manager):
    """Test updating stored credentials."""
    vendor_id = "test_updates"
    initial_creds = {
        "username": "initial_user",
        "password": "initial_pass",
        "url": "https://test.example.com"
    }
    
    # Store initial credentials
    credential_manager.store_credentials(vendor_id, initial_creds)
    
    # Update credentials
    updates = {
        "password": "new_pass",
        "2fa_method": "sms"
    }
    assert credential_manager.update_credentials(vendor_id, updates)
    
    # Verify updates
    updated_creds = credential_manager.get_credentials(vendor_id)
    assert updated_creds["username"] == initial_creds["username"]  # Unchanged
    assert updated_creds["password"] == updates["password"]  # Changed
    assert updated_creds["2fa_method"] == updates["2fa_method"]  # Added
    
    # Test updating non-existent vendor
    assert not credential_manager.update_credentials("nonexistent", updates)

def test_credential_deletion(credential_manager):
    """Test credential deletion."""
    vendor_id = "test_deletion"
    test_creds = {"username": "delete_me", "password": "temp_pass"}
    
    # Store and verify credentials
    credential_manager.store_credentials(vendor_id, test_creds)
    assert credential_manager.get_credentials(vendor_id) is not None
    
    # Delete credentials
    assert credential_manager.delete_credentials(vendor_id)
    assert credential_manager.get_credentials(vendor_id) is None
    
    # Verify file is actually deleted
    file_path = os.path.join(credential_manager.storage_path, f"{vendor_id}.enc")
    assert not os.path.exists(file_path)
    
    # Test deleting non-existent credentials
    assert not credential_manager.delete_credentials("nonexistent")

@pytest.mark.asyncio
async def test_report_navigation(nova_manager, vinsolutions_collector):
    """Test navigation to different report types."""
    # Mock session creation
    mock_session = MagicMock()
    mock_session.session_id = "test_session_nav"
    nova_manager.client.sessions.create = MagicMock(return_value=mock_session)
    
    # Test navigation to each report type
    report_types = ["sales", "leads", "inventory"]
    
    for report_type in report_types:
        # Reset step mock for each test
        nova_manager.client.sessions.step = MagicMock(return_value={"status": "success"})
        
        # Call appropriate collector method
        if report_type == "sales":
            result = await vinsolutions_collector.collect_sales_report(TEST_CREDENTIALS)
        elif report_type == "leads":
            result = await vinsolutions_collector.collect_lead_report(TEST_CREDENTIALS)
        else:
            result = await vinsolutions_collector.collect_inventory_report(TEST_CREDENTIALS)
        
        # Verify navigation sequence
        assert result["success"] is True
        assert result["metadata"]["type"] == report_type
        
        # Verify correct navigation steps were taken
        navigation_calls = [
            call for call in nova_manager.client.sessions.step.call_args_list
            if "click" in str(call)  # Look for navigation clicks
        ]
        assert len(navigation_calls) >= 2  # At least menu + report type click

# --- NEW Integration Tests for Scheduler and Fallback ---

# Mock NovaActClient for scheduler test
class MockNovaActClient:
    def collect_report(self, vendor, config):
        print(f"[MOCK][Test] Collecting report for {vendor}...")
        # Simulate returning a dummy file path
        return f"/tmp/dummy_report_{vendor}.csv"

def test_scheduler_runs_daily():
    """Test that the scheduler correctly schedules a daily job."""
    # Clear existing schedule for clean test state
    schedule.clear()
    
    mock_client = MockNovaActClient()
    mock_config = {"vinsolutions": {"report_path": "/reports"}} # Minimal config
    scheduler = NovaScheduler(mock_client, mock_config)
    
    scheduler.schedule_task("vinsolutions", "daily", "08:00")
    
    # Verify that a job was scheduled
    jobs = schedule.get_jobs()
    assert len(jobs) == 1
    # Check specifics of the job if needed (e.g., interval, time)
    job = jobs[0]
    assert job.interval == 1 # Runs every 1 day
    assert str(job.at_time) == "08:00"
    
    # Clean up schedule
    schedule.clear()

def test_fallback_on_login_failure():
    """Test that the fallback handler sets the error state correctly."""
    
    # Initialize session state if it doesn't exist (needed for testing outside Streamlit run)
    # This is a basic mock; a more robust approach might use pytest-streamlit
    if 'nova_act_error' not in st.session_state:
         st.session_state.nova_act_error = None 
    if 'nova_act_vendor' not in st.session_state:
         st.session_state.nova_act_vendor = None
         
    # Reset state before test
    st.session_state.nova_act_error = None
    st.session_state.nova_act_vendor = None

    fallback = NovaFallback()
    test_vendor = "vinsolutions_fallback_test"
    test_error_message = "CAPTCHA detected during login"
    
    # Simulate calling the handler (no button click in this test)
    fallback.handle_login_friction(test_vendor, test_error_message)
    
    # Assert that the session state was updated
    assert st.session_state.nova_act_error is not None
    assert test_vendor in st.session_state.nova_act_error
    assert test_error_message in st.session_state.nova_act_error
    assert st.session_state.nova_act_vendor == test_vendor
    
    # Clean up state after test (optional, depends on test runner setup)
    st.session_state.nova_act_error = None
    st.session_state.nova_act_vendor = None

if __name__ == "__main__":
    pytest.main([__file__])