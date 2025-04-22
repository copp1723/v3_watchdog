"""
Integration tests for Nova Act connector.

These tests verify that the Nova Act system properly integrates with the Watchdog
data processing pipeline.
"""

import os
import sys
import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
import pandas as pd

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.nova_act.core import NovaActConnector
from src.nova_act.watchdog_upload import upload_to_watchdog

# Mock test data
TEST_DATA = pd.DataFrame({
    'Date': ['2023-01-01', '2023-01-02', '2023-01-03'],
    'VIN': ['VIN123', 'VIN456', 'VIN789'],
    'Make': ['Toyota', 'Honda', 'Ford'],
    'Model': ['Camry', 'Accord', 'Focus'],
    'SalesAmount': [25000, 28000, 22000]
})

# Create a mock CSV file for testing
@pytest.fixture
def test_csv_file():
    """Create a temporary CSV file for testing."""
    test_file_path = '/tmp/test_sales_data.csv'
    TEST_DATA.to_csv(test_file_path, index=False)
    yield test_file_path
    # Clean up the file after the test
    if os.path.exists(test_file_path):
        os.remove(test_file_path)

@pytest.fixture
def mock_stream_session():
    """Mock Streamlit session state."""
    with patch('streamlit.session_state', create=True) as mock_session:
        # Mock session state properties
        mock_session.__getitem__ = MagicMock(return_value=None)
        mock_session.__setitem__ = MagicMock()
        yield mock_session

@pytest.mark.asyncio
async def test_upload_to_watchdog_integration(test_csv_file, mock_stream_session):
    """Test that uploaded files are properly processed by the Watchdog pipeline."""
    with patch('src.nova_act.watchdog_upload.process_uploaded_file') as mock_process:
        # Mock the processing function to return test data
        mock_process.return_value = (
            TEST_DATA,  # DataFrame
            {"status": "success", "message": "File processed successfully"},  # Summary
            {"validation": "passed"},  # Report
            {"detected_schema": "sales"}  # Schema info
        )
        
        # Call the upload function
        result = upload_to_watchdog(test_csv_file)
        
        # Verify the result
        assert result is True
        mock_process.assert_called_once()
        
        # Verify that session state was updated with the processed data
        mock_stream_session.__setitem__.assert_any_call('validated_data', TEST_DATA)
        mock_stream_session.__setitem__.assert_any_call('validation_summary', 
                                                      {"status": "success", "message": "File processed successfully"})

@pytest.mark.asyncio
async def test_nova_act_data_collection_integration():
    """Test that the NovaActConnector can collect and process data."""
    with patch('src.nova_act.core.NovaActConnector.collect_report') as mock_collect, \
         patch('src.nova_act.core.upload_to_watchdog') as mock_upload:
        
        # Mock the collect_report method
        mock_collect.return_value = {
            "success": True,
            "file_path": "/tmp/test_download.csv",
            "duration": 2.5,
            "timestamp": "2023-01-01T12:00:00"
        }
        
        # Mock the upload_to_watchdog function
        mock_upload.return_value = True
        
        # Create a connector instance
        connector = NovaActConnector(
            headless=True,
            download_dir="/tmp"
        )
        
        # Mock the _get_credentials method
        connector._get_credentials = AsyncMock(return_value={
            "username": "test_user",
            "password": "test_password"
        })
        
        # Call the method to collect a report
        result = await connector._process_downloaded_report(
            "/tmp/test_download.csv",
            "test_vendor",
            "test_dealer_id",
            "sales_report"
        )
        
        # Verify the result
        assert result["success"] is True
        mock_upload.assert_called_once_with("/tmp/test_download.csv", auto_cleaning=True)

@pytest.mark.asyncio
async def test_report_scheduling_integration():
    """Test that the scheduler properly integrates with the NovaActConnector."""
    with patch('src.nova_act.enhanced_scheduler.get_scheduler') as mock_get_scheduler:
        mock_scheduler = MagicMock()
        mock_scheduler.schedule_task = MagicMock(return_value="task_123")
        mock_get_scheduler.return_value = mock_scheduler
        
        # Create a connector instance
        connector = NovaActConnector(
            headless=True,
            download_dir="/tmp"
        )
        
        # Schedule a report collection
        task_id = connector.schedule_report_collection(
            vendor="test_vendor",
            dealer_id="test_dealer",
            report_type="sales",
            frequency="daily",
            hour=2,
            minute=0
        )
        
        # Verify the result
        assert task_id == "task_123"
        mock_scheduler.schedule_task.assert_called_once_with(
            task_type="report_collection",
            schedule_type="daily",
            task_params={
                "vendor": "test_vendor",
                "dealer_id": "test_dealer",
                "report_type": "sales"
            },
            hour=2,
            minute=0
        )