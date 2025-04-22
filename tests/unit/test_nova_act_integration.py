"""
Unit tests for the Nova Act integration.

These tests verify that the NovaActConnector and scheduler integration
work correctly for automating data collection.
"""

import pytest
import asyncio
import os
import tempfile
from unittest.mock import MagicMock, AsyncMock, patch

from src.nova_act.core import NovaActConnector
from src.nova_act.scheduler_bridge import NovaActSchedulerBridge
from src.scheduler.report_scheduler import ReportScheduler, ReportFrequency

@pytest.fixture
def mock_connector():
    """Create a mocked NovaActConnector."""
    connector = MagicMock(spec=NovaActConnector)
    connector.start = AsyncMock()
    connector.shutdown = AsyncMock()
    connector.collect_report = AsyncMock(return_value={
        "success": True,
        "file_path": "/tmp/test_report.csv",
        "timestamp": "2023-01-01T12:00:00"
    })
    connector.verify_credentials = AsyncMock(return_value={
        "success": True,
        "valid": True
    })
    connector.schedule_report_collection = MagicMock(return_value="task_123")
    
    return connector

@pytest.fixture
def mock_scheduler():
    """Create a mocked ReportScheduler."""
    scheduler = MagicMock(spec=ReportScheduler)
    scheduler.create_report = MagicMock(return_value="report_123")
    scheduler.get_due_reports = MagicMock(return_value=[])
    scheduler.save_reports = MagicMock()
    
    return scheduler

@pytest.fixture
def bridge(mock_connector, mock_scheduler):
    """Create a NovaActSchedulerBridge with mocked components."""
    return NovaActSchedulerBridge(
        report_scheduler=mock_scheduler,
        connector=mock_connector
    )

@pytest.mark.asyncio
async def test_bridge_start_stop(bridge):
    """Test starting and stopping the scheduler bridge."""
    # Start the bridge
    await bridge.start()
    
    # Assert that connector.start was called
    bridge.connector.start.assert_called_once()
    
    # Assert that the bridge is running
    assert bridge.running is True
    
    # Stop the bridge
    await bridge.stop()
    
    # Assert that connector.shutdown was called
    bridge.connector.shutdown.assert_called_once()
    
    # Assert that the bridge is not running
    assert bridge.running is False

@pytest.mark.asyncio
async def test_schedule_report_collection(bridge):
    """Test scheduling a report collection."""
    # Schedule a report collection
    report_id = bridge.schedule_report_collection(
        vendor_id="test_vendor",
        dealer_id="test_dealer",
        report_type="test_report",
        frequency=ReportFrequency.DAILY,
        hour=2,
        minute=0
    )
    
    # Assert that report_scheduler.create_report was called with the right parameters
    bridge.report_scheduler.create_report.assert_called_once()
    args, kwargs = bridge.report_scheduler.create_report.call_args
    
    # Check that kwargs has the expected parameters
    assert kwargs["name"] == "test_vendor test_report for test_dealer"
    assert kwargs["template"] == "nova_act"
    assert kwargs["frequency"] == ReportFrequency.DAILY
    
    # Check that the report parameters were passed correctly
    params = kwargs["parameters"]
    assert params["vendor_id"] == "test_vendor"
    assert params["dealer_id"] == "test_dealer"
    assert params["report_type"] == "test_report"
    assert params["is_nova_act"] is True
    
    # Assert that the function returns the report ID
    assert report_id == "report_123"

@pytest.mark.asyncio
async def test_trigger_sync_now(bridge):
    """Test triggering an immediate sync."""
    # Trigger a sync
    result = bridge.trigger_sync_now(
        vendor_id="test_vendor",
        dealer_id="test_dealer",
        report_type="test_report"
    )
    
    # Assert that connector.schedule_report_collection was called with the right parameters
    bridge.connector.schedule_report_collection.assert_called_once_with(
        vendor="test_vendor",
        dealer_id="test_dealer",
        report_type="test_report",
        frequency="once"
    )
    
    # Assert that the function returns the expected result
    assert result["task_id"] == "task_123"
    assert result["vendor_id"] == "test_vendor"
    assert result["dealer_id"] == "test_dealer"
    assert result["report_type"] == "test_report"
    assert "triggered_at" in result

@pytest.mark.asyncio
async def test_process_nova_act_report(bridge):
    """Test processing a Nova Act report."""
    # Create a mock report
    report = MagicMock()
    report.report_id = "report_123"
    report.parameters = {
        "vendor_id": "test_vendor",
        "dealer_id": "test_dealer",
        "report_type": "test_report",
        "is_nova_act": True
    }
    report.update_next_run = MagicMock()
    
    # Process the report
    await bridge._process_nova_act_report(report)
    
    # Assert that connector.collect_report was called with the right parameters
    bridge.connector.collect_report.assert_called_once_with(
        vendor="test_vendor",
        dealer_id="test_dealer",
        report_type="test_report"
    )
    
    # Assert that report.update_next_run was called
    report.update_next_run.assert_called_once()
    
    # Assert that report_scheduler.save_reports was called
    bridge.report_scheduler.save_reports.assert_called_once()
    
    # Assert that the sync status was updated
    key = "test_vendor:test_dealer:test_report"
    assert key in bridge.sync_status
    assert bridge.sync_status[key]["success"] is True
    assert bridge.sync_status[key]["vendor_id"] == "test_vendor"
    assert bridge.sync_status[key]["dealer_id"] == "test_dealer"
    assert bridge.sync_status[key]["report_type"] == "test_report"

@pytest.mark.asyncio
async def test_get_sync_status(bridge):
    """Test getting sync status."""
    # Set up some sync status entries
    bridge.sync_status = {
        "vendor1:dealer1:report1": {
            "vendor_id": "vendor1",
            "dealer_id": "dealer1",
            "report_type": "report1",
            "success": True
        },
        "vendor1:dealer2:report1": {
            "vendor_id": "vendor1",
            "dealer_id": "dealer2",
            "report_type": "report1",
            "success": False
        },
        "vendor2:dealer1:report1": {
            "vendor_id": "vendor2",
            "dealer_id": "dealer1",
            "report_type": "report1",
            "success": True
        }
    }
    
    # Test getting all sync status
    all_status = bridge.get_sync_status()
    assert len(all_status) == 3
    
    # Test filtering by vendor
    vendor1_status = bridge.get_sync_status(vendor_id="vendor1")
    assert len(vendor1_status) == 2
    assert all(status["vendor_id"] == "vendor1" for status in vendor1_status.values())
    
    # Test filtering by dealer
    dealer1_status = bridge.get_sync_status(dealer_id="dealer1")
    assert len(dealer1_status) == 2
    assert all(status["dealer_id"] == "dealer1" for status in dealer1_status.values())
    
    # Test filtering by both
    specific_status = bridge.get_sync_status(vendor_id="vendor1", dealer_id="dealer1")
    assert len(specific_status) == 1
    assert list(specific_status.values())[0]["vendor_id"] == "vendor1"
    assert list(specific_status.values())[0]["dealer_id"] == "dealer1"

@pytest.mark.asyncio
async def test_full_report_collection_flow():
    """Test the full report collection flow with real components but mocked browser."""
    # Use a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a sample CSV file to simulate downloaded data
        csv_path = os.path.join(temp_dir, "test_report.csv")
        with open(csv_path, "w") as f:
            f.write("col1,col2,col3\nvalue1,value2,value3\n")
        
        # Mock Playwright and browser operations
        with patch("src.nova_act.core.async_playwright") as mock_playwright, \
             patch("src.nova_act.core.upload_to_watchdog") as mock_upload:
            
            # Mock the upload function
            mock_upload.return_value = True
            
            # Create all the necessary mocks for the browser automation
            mock_download = AsyncMock()
            mock_download.path = AsyncMock(return_value=csv_path)
            mock_download.save_as = AsyncMock()
            mock_download.suggested_filename = "test_report.csv"
            
            mock_page = AsyncMock()
            mock_page.goto = AsyncMock()
            mock_page.fill = AsyncMock()
            mock_page.click = AsyncMock()
            mock_page.wait_for_load_state = AsyncMock()
            mock_page.wait_for_selector = AsyncMock()
            mock_page.query_selector = AsyncMock(return_value=None)
            mock_page.expect_download = AsyncMock()
            mock_page.expect_download.return_value.__aenter__.return_value.value = mock_download
            
            mock_context = AsyncMock()
            mock_context.new_page = AsyncMock(return_value=mock_page)
            
            mock_browser = AsyncMock()
            mock_browser.new_context = AsyncMock(return_value=mock_context)
            
            mock_chromium = AsyncMock()
            mock_chromium.launch = AsyncMock(return_value=mock_browser)
            
            mock_playwright_instance = AsyncMock()
            mock_playwright_instance.chromium = mock_chromium
            mock_playwright_instance.stop = AsyncMock()
            
            mock_playwright.start = AsyncMock(return_value=mock_playwright_instance)
            
            # Create a connector with the mocked Playwright
            connector = NovaActConnector(
                headless=True,
                download_dir=temp_dir
            )
            
            # Initialize the connector
            await connector.start()
            
            # Test collecting a report
            result = await connector.collect_report(
                vendor="dealersocket",
                dealer_id="test_dealer",
                report_type="sales"
            )
            
            # Assert that the result is successful
            assert result["success"] is True
            assert "file_path" in result
            
            # Assert that the upload function was called
            mock_upload.assert_called_once()
            
            # Verify browser automation steps
            mock_page.goto.assert_called()
            mock_page.fill.assert_called()
            mock_page.click.assert_called()
            
            # Clean up
            await connector.shutdown()