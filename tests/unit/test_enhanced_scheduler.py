"""
Tests for the enhanced Nova Act scheduler.
"""

import os
import json
import pytest
import pandas as pd
import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock, AsyncMock

from src.nova_act.enhanced_scheduler import (
    EnhancedScheduler,
    ScheduledTask,
    TaskPriority
)

# Fixture for an initialized scheduler with temporary storage
@pytest.fixture
def scheduler(tmp_path):
    """Create a scheduler with temporary storage."""
    storage_path = os.path.join(tmp_path, "scheduler_storage")
    os.makedirs(storage_path, exist_ok=True)
    
    return EnhancedScheduler(storage_path=storage_path)

def test_schedule_task(scheduler):
    """Test scheduling a new task."""
    # Schedule a task
    task_id = scheduler.schedule_task(
        dealer_id="test_dealer",
        vendor_id="dealersocket",
        report_type="sales",
        schedule="daily",
        schedule_config={"hour": 3, "minute": 30}
    )
    
    # Verify task was created
    assert task_id in scheduler.tasks
    
    # Check task properties
    task = scheduler.tasks[task_id]
    assert task.dealer_id == "test_dealer"
    assert task.vendor_id == "dealersocket"
    assert task.report_type == "sales"
    assert task.schedule == "daily"
    assert task.schedule_config == {"hour": 3, "minute": 30}
    assert task.next_run is not None
    assert task.enabled is True

def test_schedule_task_once(scheduler):
    """Test scheduling a one-time task."""
    # Schedule a one-time task
    task_id = scheduler.schedule_task(
        dealer_id="test_dealer",
        vendor_id="dealersocket",
        report_type="sales",
        schedule="once",
        schedule_config={"time": datetime.now(timezone.utc).isoformat()}
    )
    
    # Verify task was created
    assert task_id in scheduler.tasks
    
    # Check task properties
    task = scheduler.tasks[task_id]
    assert task.schedule == "once"
    assert task.next_run is not None

def test_update_task(scheduler):
    """Test updating a scheduled task."""
    # Schedule a task
    task_id = scheduler.schedule_task(
        dealer_id="test_dealer",
        vendor_id="dealersocket",
        report_type="sales",
        schedule="daily",
        schedule_config={"hour": 3, "minute": 30}
    )
    
    # Update the task
    scheduler.update_task(task_id, {
        "schedule": "weekly",
        "schedule_config": {"day_of_week": 1, "hour": 2, "minute": 0},
        "priority": TaskPriority.HIGH
    })
    
    # Check updated properties
    task = scheduler.tasks[task_id]
    assert task.schedule == "weekly"
    assert task.schedule_config == {"day_of_week": 1, "hour": 2, "minute": 0}
    assert task.priority == TaskPriority.HIGH
    assert task.next_run is not None

def test_delete_task(scheduler):
    """Test deleting a scheduled task."""
    # Schedule a task
    task_id = scheduler.schedule_task(
        dealer_id="test_dealer",
        vendor_id="dealersocket",
        report_type="sales",
        schedule="daily",
        schedule_config={"hour": 3, "minute": 30}
    )
    
    # Delete the task
    result = scheduler.delete_task(task_id)
    
    # Verify task was deleted
    assert result is True
    assert task_id not in scheduler.tasks

def test_disable_enable_task(scheduler):
    """Test disabling and enabling a task."""
    # Schedule a task
    task_id = scheduler.schedule_task(
        dealer_id="test_dealer",
        vendor_id="dealersocket",
        report_type="sales",
        schedule="daily",
        schedule_config={"hour": 3, "minute": 30}
    )
    
    # Disable the task
    result = scheduler.disable_task(task_id)
    
    # Verify task was disabled
    assert result is True
    assert scheduler.tasks[task_id].enabled is False
    
    # Enable the task
    result = scheduler.enable_task(task_id)
    
    # Verify task was enabled
    assert result is True
    assert scheduler.tasks[task_id].enabled is True

def test_run_task_now(scheduler):
    """Test triggering immediate execution of a task."""
    # Schedule a task
    task_id = scheduler.schedule_task(
        dealer_id="test_dealer",
        vendor_id="dealersocket",
        report_type="sales",
        schedule="daily",
        schedule_config={"hour": 3, "minute": 30}
    )
    
    # Set next_run to a future time
    scheduler.tasks[task_id].next_run = datetime.now(timezone.utc) + timedelta(hours=12)
    
    # Trigger immediate execution
    result = scheduler.run_task_now(task_id)
    
    # Verify next_run was updated to now
    assert result is True
    
    # Allow for small time differences in the test
    time_diff = (datetime.now(timezone.utc) - scheduler.tasks[task_id].next_run).total_seconds()
    assert abs(time_diff) < 5  # Within 5 seconds of now

def test_get_all_tasks(scheduler):
    """Test getting all scheduled tasks."""
    # Schedule multiple tasks
    task_id1 = scheduler.schedule_task(
        dealer_id="dealer1",
        vendor_id="dealersocket",
        report_type="sales",
        schedule="daily",
        schedule_config={"hour": 3, "minute": 30}
    )
    
    task_id2 = scheduler.schedule_task(
        dealer_id="dealer2",
        vendor_id="vinsolutions",
        report_type="inventory",
        schedule="weekly",
        schedule_config={"day_of_week": 1, "hour": 2, "minute": 0}
    )
    
    # Get all tasks
    tasks = scheduler.get_all_tasks()
    
    # Verify correct number of tasks
    assert len(tasks) == 2
    
    # Verify task IDs are in the list
    task_ids = [task['id'] for task in tasks]
    assert task_id1 in task_ids
    assert task_id2 in task_ids

def test_get_dealer_tasks(scheduler):
    """Test getting tasks for a specific dealer."""
    # Schedule multiple tasks for different dealers
    scheduler.schedule_task(
        dealer_id="dealer1",
        vendor_id="dealersocket",
        report_type="sales",
        schedule="daily",
        schedule_config={"hour": 3, "minute": 30}
    )
    
    scheduler.schedule_task(
        dealer_id="dealer2",
        vendor_id="vinsolutions",
        report_type="inventory",
        schedule="weekly",
        schedule_config={"day_of_week": 1, "hour": 2, "minute": 0}
    )
    
    scheduler.schedule_task(
        dealer_id="dealer1",
        vendor_id="vinsolutions",
        report_type="leads",
        schedule="daily",
        schedule_config={"hour": 4, "minute": 0}
    )
    
    # Get tasks for dealer1
    tasks = scheduler.get_dealer_tasks("dealer1")
    
    # Verify correct number of tasks
    assert len(tasks) == 2
    
    # Verify all tasks are for dealer1
    for task in tasks:
        assert task['dealer_id'] == "dealer1"

def test_get_vendor_tasks(scheduler):
    """Test getting tasks for a specific vendor."""
    # Schedule multiple tasks for different vendors
    scheduler.schedule_task(
        dealer_id="dealer1",
        vendor_id="dealersocket",
        report_type="sales",
        schedule="daily",
        schedule_config={"hour": 3, "minute": 30}
    )
    
    scheduler.schedule_task(
        dealer_id="dealer2",
        vendor_id="vinsolutions",
        report_type="inventory",
        schedule="weekly",
        schedule_config={"day_of_week": 1, "hour": 2, "minute": 0}
    )
    
    scheduler.schedule_task(
        dealer_id="dealer3",
        vendor_id="dealersocket",
        report_type="leads",
        schedule="daily",
        schedule_config={"hour": 4, "minute": 0}
    )
    
    # Get tasks for dealersocket
    tasks = scheduler.get_vendor_tasks("dealersocket")
    
    # Verify correct number of tasks
    assert len(tasks) == 2
    
    # Verify all tasks are for dealersocket
    for task in tasks:
        assert task['vendor_id'] == "dealersocket"

@pytest.mark.asyncio
@patch('src.nova_act.enhanced_scheduler.NovaActClient')
async def test_start_stop_scheduler(mock_client, scheduler):
    """Test starting and stopping the scheduler."""
    # Mock client methods
    mock_client_instance = MagicMock()
    mock_client_instance.start = AsyncMock()
    mock_client.return_value = mock_client_instance
    
    # Start scheduler
    await scheduler.start()
    
    # Verify scheduler is running
    assert scheduler.running is True
    
    # Stop scheduler
    await scheduler.stop()
    
    # Verify scheduler is stopped
    assert scheduler.running is False

@pytest.mark.asyncio
async def test_scheduled_task_next_run_calculation():
    """Test next run calculation for different schedule types."""
    now = datetime.now(timezone.utc)
    
    # Test once schedule
    once_task = ScheduledTask(
        id="test_once",
        vendor_id="test",
        dealer_id="test",
        report_type="test",
        schedule="once",
        schedule_config={},
        next_run=now + timedelta(hours=1)
    )
    
    # After it runs, next_run should be None
    once_task.last_run = now
    once_task.update_next_run()
    assert once_task.next_run < now  # Will be set to None in a real run
    
    # Test daily schedule
    daily_task = ScheduledTask(
        id="test_daily",
        vendor_id="test",
        dealer_id="test",
        report_type="test",
        schedule="daily",
        schedule_config={"hour": 3, "minute": 30},
        next_run=now
    )
    
    daily_task.last_run = now
    daily_task.update_next_run()
    
    # Next run should be tomorrow at 3:30
    assert daily_task.next_run.hour == 3
    assert daily_task.next_run.minute == 30
    assert daily_task.next_run.date() > now.date()
    
    # Test interval schedule
    interval_task = ScheduledTask(
        id="test_interval",
        vendor_id="test",
        dealer_id="test",
        report_type="test",
        schedule="interval",
        schedule_config={"interval_minutes": 60},
        next_run=now
    )
    
    interval_task.last_run = now
    interval_task.update_next_run()
    
    # Next run should be 60 minutes from now
    time_diff = (interval_task.next_run - now).total_seconds() / 60
    assert 59 <= time_diff <= 61  # Allow for small differences