"""
Test script for CRM integration functionality.

This script tests basic functionality of the CRM integration components:
1. NovaActAdapter (mock mode)
2. CRMSyncScheduler initialization
3. Job scheduling
4. Data persistence paths

Run this script directly to verify the CRM integration is working properly.
"""

import os
import sys
import json
import time
import shutil
import logging
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger("crm_integration_test")

# Import our integration components
from watchdog_ai.integrations.crm import NovaActAdapter
from watchdog_ai.workers.scheduler import (
    CRMSyncScheduler, 
    hourly_pull_sales_job, 
    daily_push_insights_job
)


def test_nova_act_adapter():
    """Test NovaActAdapter basic functionality in mock mode."""
    logger.info("=== Testing NovaActAdapter (mock mode) ===")
    
    # Create adapter in mock mode
    adapter = NovaActAdapter(
        base_url="https://api.mocknovaact.example",
        api_key=None,  # Ensure mock mode
        mock_data_path=None  # Use default path
    )
    
    # Verify mock mode is enabled
    assert adapter.use_mock_data is True, "Adapter should be in mock mode"
    logger.info("✓ Adapter initialized in mock mode")
    
    # Test authentication
    adapter.authenticate()
    assert adapter.authenticated is True, "Authentication failed"
    assert adapter.auth_token is not None, "Auth token not set"
    logger.info(f"✓ Mock authentication successful, token: {adapter.auth_token}")
    
    # Test pulling sales data
    sales_data = adapter.pull_sales()
    assert isinstance(sales_data, list), "Sales data should be a list"
    assert len(sales_data) > 0, "Sales data should not be empty"
    logger.info(f"✓ Successfully pulled {len(sales_data)} mock sales records")
    
    # Display a sample sales record
    if sales_data:
        logger.info(f"Sample sales record: {json.dumps(sales_data[0], indent=2)}")
    
    # Test pushing insights
    test_insights = [
        {
            "id": "I001",
            "type": "opportunity",
            "title": "Sales Opportunity",
            "description": "Potential upsell opportunity detected",
            "customer_id": "C001",
            "score": 0.85,
            "created_at": datetime.now().isoformat()
        },
        {
            "id": "I002",
            "type": "risk",
            "title": "Churn Risk",
            "description": "Customer showing signs of disengagement",
            "customer_id": "C002",
            "score": 0.65,
            "created_at": datetime.now().isoformat()
        }
    ]
    
    adapter.push_insights(test_insights)
    logger.info(f"✓ Successfully pushed {len(test_insights)} mock insights")
    
    return True


def test_crm_sync_scheduler():
    """Test CRMSyncScheduler initialization and job scheduling."""
    logger.info("=== Testing CRMSyncScheduler ===")
    
    temp_dir = None
    try:
        # Create temporary directories for testing
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)
        data_dir = temp_path / "data/raw"
        insights_dir = temp_path / "data/processed"
        log_dir = temp_path / "data/logs"
        db_path = temp_path / "data/scheduler.sqlite"
        
        # Create directories
        data_dir.mkdir(parents=True, exist_ok=True)
        insights_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create insights directory and sample file
        sample_insights = [
            {
                "id": "I001",
                "type": "opportunity",
                "title": "Test Insight",
                "description": "This is a test insight",
                "customer_id": "C001",
                "score": 0.75,
                "created_at": datetime.now().isoformat()
            }
        ]
        
        with open(insights_dir / "latest.json", "w") as f:
            json.dump(sample_insights, f, indent=2)
        
        logger.info(f"✓ Created test directories in {temp_path}")
        
        # Create adapter for testing
        crm_adapter = NovaActAdapter(
            base_url="https://api.mocknovaact.example",
            mock_data_path=temp_path / "mock_data.json"
        )
        
        # Test scheduler initialization without starting it
        scheduler = CRMSyncScheduler(
            data_dir=data_dir,
            insights_dir=insights_dir,
            log_dir=log_dir,
            db_path=db_path,
            sales_sync_interval=5,  # 5 minutes for testing
            insights_sync_cron="*/5 * * * *",  # Every 5 minutes for testing
            crm_adapter=crm_adapter
        )
        
        assert scheduler.is_running is False, "Scheduler should not be running yet"
        logger.info("✓ Initialized scheduler with test configuration")
        
        # Test direct job functions instead of using the scheduler
        # This avoids serialization issues with APScheduler
        
        # Test sales sync job
        sales_result = hourly_pull_sales_job(crm_adapter, data_dir, log_dir)
        assert sales_result["status"] == "success", f"Sales job failed: {sales_result}"
        logger.info(f"✓ Sales sync job successful: {sales_result['record_count']} records")
        
        # Verify data file was created
        if sales_result.get("data_file"):
            assert Path(sales_result["data_file"]).exists(), "Data file not created"
            logger.info(f"✓ Data file created at: {sales_result['data_file']}")
            
            # Check file contents
            with open(sales_result["data_file"], "r") as f:
                saved_data = json.load(f)
                assert len(saved_data) > 0, "Saved data should not be empty"
                logger.info(f"✓ Data file contains {len(saved_data)} records")
        
        # Test insights sync job
        insights_result = daily_push_insights_job(crm_adapter, insights_dir, log_dir)
        assert insights_result["status"] == "success", f"Insights job failed: {insights_result}"
        logger.info(f"✓ Insights sync job successful")
        
        # Verify archive file was created
        if insights_result.get("archive_file"):
            assert Path(insights_result["archive_file"]).exists(), "Archive file not created"
            logger.info(f"✓ Archive file created at: {insights_result['archive_file']}")
        
        # Now test the scheduler job registration without starting
        scheduler._register_jobs()
        
        # Verify jobs were created
        jobs = scheduler.scheduler.get_jobs()
        assert len(jobs) == 2, "Should have 2 scheduled jobs"
        job_ids = [job.id for job in jobs]
        assert "sales_sync" in job_ids, "Sales sync job missing"
        assert "insights_sync" in job_ids, "Insights sync job missing"
        logger.info(f"✓ Jobs registered successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in CRMSyncScheduler test: {str(e)}")
        logger.error(traceback.format_exc())
        return False
        
    finally:
        # Clean up temporary directory
        if temp_dir and Path(temp_dir).exists():
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
                logger.info(f"✓ Cleaned up temporary test directory")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary directory: {str(e)}")

def main():
    """Run all tests."""
    logger.info("Starting CRM integration tests")
    
    tests = [
        ("NovaActAdapter", test_nova_act_adapter),
        ("CRMSyncScheduler", test_crm_sync_scheduler)
    ]
    
    results = []
    for name, test_func in tests:
        logger.info(f"\nRunning test: {name}")
        try:
            result = test_func()
            status = "PASSED" if result else "FAILED"
        except Exception as e:
            logger.exception(f"Error in test {name}")
            status = "ERROR"
        
        results.append((name, status))
        logger.info(f"Test {name}: {status}\n")
    
    # Print summary
    logger.info("=== Test Summary ===")
    all_passed = True
    for name, status in results:
        logger.info(f"{name}: {status}")
        if status != "PASSED":
            all_passed = False
    
    if all_passed:
        logger.info("\n✅ All tests passed successfully!")
    else:
        logger.error("\n❌ Some tests failed. See logs for details.")

    return all_passed


if __name__ == "__main__":
    success = main()
    # Use exit code to indicate test success/failure
    sys.exit(0 if success else 1)

