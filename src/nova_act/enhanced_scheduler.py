"""
Enhanced scheduler for Nova Act data collection.

This module provides functionality to schedule and manage
data collection tasks with robust error handling and retry logic.
"""

import os
import json
import logging
import asyncio
import time
import uuid
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field, asdict

from .logging_config import log_error, log_info, log_warning
from .enhanced_credentials import get_credential_manager, DealerCredential
from .core import NovaActClient
from .task import Task, TaskPriority, TaskStatus

logger = logging.getLogger(__name__)

@dataclass
class ScheduledTask:
    """Represents a scheduled data collection task."""
    id: str
    vendor_id: str
    dealer_id: str
    report_type: str
    schedule: str  # 'once', 'daily', 'weekly', 'monthly', 'interval'
    schedule_config: Dict[str, Any]  # e.g. {'hour': 3, 'minute': 30} or {'interval_minutes': 60}
    next_run: datetime
    last_run: Optional[datetime] = None
    last_status: Optional[str] = None
    error_count: int = 0
    enabled: bool = True
    priority: TaskPriority = TaskPriority.NORMAL
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = asdict(self)
        # Convert datetime objects to ISO strings
        result['next_run'] = self.next_run.isoformat() if self.next_run else None
        result['last_run'] = self.last_run.isoformat() if self.last_run else None
        result['created_at'] = self.created_at.isoformat() if self.created_at else None
        result['updated_at'] = self.updated_at.isoformat() if self.updated_at else None
        # Convert enum to string
        result['priority'] = self.priority.name
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScheduledTask':
        """Create from dictionary representation."""
        # Convert ISO strings to datetime objects
        if 'next_run' in data and data['next_run']:
            data['next_run'] = datetime.fromisoformat(data['next_run'])
        if 'last_run' in data and data['last_run']:
            data['last_run'] = datetime.fromisoformat(data['last_run'])
        if 'created_at' in data and data['created_at']:
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if 'updated_at' in data and data['updated_at']:
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        
        # Convert string to enum
        if 'priority' in data and isinstance(data['priority'], str):
            data['priority'] = TaskPriority[data['priority']]
        
        return cls(**data)
    
    def update_next_run(self):
        """Update the next run time based on schedule configuration."""
        now = datetime.now(timezone.utc)
        
        if self.schedule == 'once':
            # For one-time tasks, next_run is already set
            # If it's passed, set to None to indicate completion
            if self.next_run and self.next_run < now:
                self.next_run = None
                self.enabled = False
        
        elif self.schedule == 'interval':
            # Interval-based: next_run = last_run + interval
            interval_minutes = self.schedule_config.get('interval_minutes', 60)
            
            if self.last_run:
                self.next_run = self.last_run + timedelta(minutes=interval_minutes)
            else:
                # If never run, schedule based on now
                self.next_run = now + timedelta(minutes=interval_minutes)
        
        elif self.schedule == 'daily':
            # Daily: run at specified hour and minute
            hour = self.schedule_config.get('hour', 0)
            minute = self.schedule_config.get('minute', 0)
            
            # Create the next run time
            next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            
            # If the time has already passed today, schedule for tomorrow
            if next_run <= now:
                next_run = next_run + timedelta(days=1)
            
            self.next_run = next_run
        
        elif self.schedule == 'weekly':
            # Weekly: run on specified day at specified hour and minute
            day_of_week = self.schedule_config.get('day_of_week', 0)  # 0=Monday, 6=Sunday
            hour = self.schedule_config.get('hour', 0)
            minute = self.schedule_config.get('minute', 0)
            
            # Calculate days until the next occurrence of day_of_week
            current_day = now.weekday()
            days_ahead = (day_of_week - current_day) % 7
            
            # If it's today but the time has passed, add 7 days
            if days_ahead == 0 and now.hour > hour or (now.hour == hour and now.minute >= minute):
                days_ahead = 7
            
            # Create the next run time
            next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            next_run = next_run + timedelta(days=days_ahead)
            
            self.next_run = next_run
        
        elif self.schedule == 'monthly':
            # Monthly: run on specified day of month at specified hour and minute
            day_of_month = self.schedule_config.get('day_of_month', 1)
            hour = self.schedule_config.get('hour', 0)
            minute = self.schedule_config.get('minute', 0)
            
            # Create the next run time starting from current month
            current_month = now.month
            current_year = now.year
            
            # Try to create a date for the current month
            try:
                next_run = now.replace(
                    year=current_year,
                    month=current_month,
                    day=day_of_month,
                    hour=hour,
                    minute=minute,
                    second=0,
                    microsecond=0
                )
            except ValueError:
                # Handle invalid day (e.g., February 30)
                if current_month == 12:
                    next_month = 1
                    next_year = current_year + 1
                else:
                    next_month = current_month + 1
                    next_year = current_year
                
                next_run = now.replace(
                    year=next_year,
                    month=next_month,
                    day=1,  # Start with day 1
                    hour=hour,
                    minute=minute,
                    second=0,
                    microsecond=0
                )
                # Subtract one day to get the last day of the current month
                next_run = next_run - timedelta(days=1)
            
            # If the time has already passed this month, move to next month
            if next_run <= now:
                if current_month == 12:
                    next_month = 1
                    next_year = current_year + 1
                else:
                    next_month = current_month + 1
                    next_year = current_year
                
                try:
                    next_run = now.replace(
                        year=next_year,
                        month=next_month,
                        day=day_of_month,
                        hour=hour,
                        minute=minute,
                        second=0,
                        microsecond=0
                    )
                except ValueError:
                    # Handle invalid day for next month
                    if next_month == 12:
                        next_next_month = 1
                        next_next_year = next_year + 1
                    else:
                        next_next_month = next_month + 1
                        next_next_year = next_year
                    
                    next_run = now.replace(
                        year=next_next_year,
                        month=next_next_month,
                        day=1,
                        hour=hour,
                        minute=minute,
                        second=0,
                        microsecond=0
                    )
                    # Subtract one day to get the last day of the month
                    next_run = next_run - timedelta(days=1)
            
            self.next_run = next_run
        
        # Mark as updated
        self.updated_at = datetime.now(timezone.utc)


class EnhancedScheduler:
    """
    Enhanced scheduler for Nova Act data collection.
    
    Features:
    - Multiple schedule types (one-time, interval, daily, weekly, monthly)
    - Persistent storage of tasks
    - Automatic retry on failure
    - Task prioritization
    - Concurrent execution
    """
    
    def __init__(self, storage_path: Optional[str] = None, max_workers: int = 5):
        """
        Initialize the scheduler.
        
        Args:
            storage_path: Optional path to store task data
            max_workers: Maximum number of concurrent task executions
        """
        # Set up storage path
        self.storage_path = storage_path or os.path.join(
            os.path.dirname(__file__),
            "scheduler_storage"
        )
        os.makedirs(self.storage_path, exist_ok=True)
        
        # Tasks storage
        self.tasks_file = os.path.join(self.storage_path, "tasks.json")
        self.tasks: Dict[str, ScheduledTask] = {}
        
        # Execution state
        self.running = False
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.running_tasks: Dict[str, asyncio.Task] = {}
        
        # Initialize internal task queue
        self.task_queue = asyncio.Queue()
        
        # Nova client (lazy initialized)
        self.client = None
        
        # Load existing tasks
        self._load_tasks()
    
    def _load_tasks(self):
        """Load tasks from storage."""
        if os.path.exists(self.tasks_file):
            try:
                with open(self.tasks_file, 'r') as f:
                    tasks_data = json.load(f)
                
                for task_id, task_data in tasks_data.items():
                    self.tasks[task_id] = ScheduledTask.from_dict(task_data)
                
                log_info(
                    f"Loaded {len(self.tasks)} tasks from storage",
                    "system",
                    "scheduler_load"
                )
            except Exception as e:
                log_error(
                    e,
                    "system",
                    "scheduler_load"
                )
    
    def _save_tasks(self):
        """Save tasks to storage."""
        try:
            tasks_data = {
                task_id: task.to_dict() 
                for task_id, task in self.tasks.items()
            }
            
            with open(self.tasks_file, 'w') as f:
                json.dump(tasks_data, f, indent=2)
            
            log_info(
                f"Saved {len(self.tasks)} tasks to storage",
                "system",
                "scheduler_save"
            )
        except Exception as e:
            log_error(
                e,
                "system",
                "scheduler_save"
            )
    
    def schedule_task(
        self,
        dealer_id: str,
        vendor_id: str,
        report_type: str,
        schedule: str,
        schedule_config: Dict[str, Any],
        priority: TaskPriority = TaskPriority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Schedule a new data collection task.
        
        Args:
            dealer_id: ID of the dealership
            vendor_id: ID of the vendor system (e.g., 'dealersocket')
            report_type: Type of report to collect (e.g., 'sales', 'inventory')
            schedule: Schedule type ('once', 'daily', 'weekly', 'monthly', 'interval')
            schedule_config: Configuration for the schedule
            priority: Task priority
            metadata: Additional metadata for the task
            
        Returns:
            ID of the scheduled task
        """
        # Generate a unique task ID
        task_id = str(uuid.uuid4())
        
        # Create initial next_run time
        next_run = self._calculate_initial_run_time(schedule, schedule_config)
        
        # Create the task
        task = ScheduledTask(
            id=task_id,
            vendor_id=vendor_id,
            dealer_id=dealer_id,
            report_type=report_type,
            schedule=schedule,
            schedule_config=schedule_config,
            next_run=next_run,
            priority=priority,
            metadata=metadata or {}
        )
        
        # Store the task
        self.tasks[task_id] = task
        self._save_tasks()
        
        log_info(
            f"Scheduled {schedule} task {task_id} for dealer {dealer_id}, vendor {vendor_id}, report {report_type}",
            dealer_id,
            "schedule_task"
        )
        
        return task_id
    
    def _calculate_initial_run_time(self, schedule: str, config: Dict[str, Any]) -> datetime:
        """Calculate the initial run time for a new task."""
        now = datetime.now(timezone.utc)
        
        if schedule == 'once':
            # For one-time tasks, use the specified time or now + 1 minute
            if 'time' in config and isinstance(config['time'], str):
                try:
                    return datetime.fromisoformat(config['time'])
                except ValueError:
                    pass
            return now + timedelta(minutes=1)
        
        elif schedule == 'interval':
            # Interval-based: now + interval
            interval_minutes = config.get('interval_minutes', 60)
            return now + timedelta(minutes=interval_minutes)
        
        elif schedule == 'daily':
            # Daily: run at specified hour and minute
            hour = config.get('hour', 0)
            minute = config.get('minute', 0)
            
            # Create the next run time
            next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            
            # If the time has already passed today, schedule for tomorrow
            if next_run <= now:
                next_run = next_run + timedelta(days=1)
            
            return next_run
        
        elif schedule == 'weekly':
            # Weekly: run on specified day at specified hour and minute
            day_of_week = config.get('day_of_week', 0)  # 0=Monday, 6=Sunday
            hour = config.get('hour', 0)
            minute = config.get('minute', 0)
            
            # Calculate days until the next occurrence of day_of_week
            current_day = now.weekday()
            days_ahead = (day_of_week - current_day) % 7
            
            # If it's today but the time has passed, add 7 days
            if days_ahead == 0 and now.hour > hour or (now.hour == hour and now.minute >= minute):
                days_ahead = 7
            
            # Create the next run time
            next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            next_run = next_run + timedelta(days=days_ahead)
            
            return next_run
        
        elif schedule == 'monthly':
            # Monthly: run on specified day of month at specified hour and minute
            day_of_month = config.get('day_of_month', 1)
            hour = config.get('hour', 0)
            minute = config.get('minute', 0)
            
            # Create the next run time starting from current month
            current_month = now.month
            current_year = now.year
            
            # Try to create a date for the current month
            try:
                next_run = now.replace(
                    year=current_year,
                    month=current_month,
                    day=day_of_month,
                    hour=hour,
                    minute=minute,
                    second=0,
                    microsecond=0
                )
            except ValueError:
                # Handle invalid day (e.g., February 30)
                if current_month == 12:
                    next_month = 1
                    next_year = current_year + 1
                else:
                    next_month = current_month + 1
                    next_year = current_year
                
                next_run = now.replace(
                    year=next_year,
                    month=next_month,
                    day=1,  # Start with day 1
                    hour=hour,
                    minute=minute,
                    second=0,
                    microsecond=0
                )
                # Subtract one day to get the last day of the current month
                next_run = next_run - timedelta(days=1)
            
            # If the time has already passed this month, move to next month
            if next_run <= now:
                if current_month == 12:
                    next_month = 1
                    next_year = current_year + 1
                else:
                    next_month = current_month + 1
                    next_year = current_year
                
                try:
                    next_run = now.replace(
                        year=next_year,
                        month=next_month,
                        day=day_of_month,
                        hour=hour,
                        minute=minute,
                        second=0,
                        microsecond=0
                    )
                except ValueError:
                    # Handle invalid day for next month
                    if next_month == 12:
                        next_next_month = 1
                        next_next_year = next_year + 1
                    else:
                        next_next_month = next_month + 1
                        next_next_year = next_year
                    
                    next_run = now.replace(
                        year=next_next_year,
                        month=next_next_month,
                        day=1,
                        hour=hour,
                        minute=minute,
                        second=0,
                        microsecond=0
                    )
                    # Subtract one day to get the last day of the month
                    next_run = next_run - timedelta(days=1)
            
            return next_run
        
        # Default: run in 1 minute
        return now + timedelta(minutes=1)
    
    def get_task(self, task_id: str) -> Optional[ScheduledTask]:
        """
        Get a task by ID.
        
        Args:
            task_id: ID of the task
            
        Returns:
            ScheduledTask object or None if not found
        """
        return self.tasks.get(task_id)
    
    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """
        Get all scheduled tasks.
        
        Returns:
            List of task dictionaries
        """
        return [task.to_dict() for task in self.tasks.values()]
    
    def get_pending_tasks(self) -> List[Dict[str, Any]]:
        """
        Get tasks that are scheduled to run in the future.
        
        Returns:
            List of task dictionaries
        """
        now = datetime.now(timezone.utc)
        return [
            task.to_dict() for task in self.tasks.values()
            if task.enabled and task.next_run and task.next_run > now
        ]
    
    def get_vendor_tasks(self, vendor_id: str) -> List[Dict[str, Any]]:
        """
        Get all tasks for a specific vendor.
        
        Args:
            vendor_id: ID of the vendor
            
        Returns:
            List of task dictionaries
        """
        return [
            task.to_dict() for task in self.tasks.values()
            if task.vendor_id == vendor_id
        ]
    
    def get_dealer_tasks(self, dealer_id: str) -> List[Dict[str, Any]]:
        """
        Get all tasks for a specific dealer.
        
        Args:
            dealer_id: ID of the dealer
            
        Returns:
            List of task dictionaries
        """
        return [
            task.to_dict() for task in self.tasks.values()
            if task.dealer_id == dealer_id
        ]
    
    def update_task(self, task_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update a scheduled task.
        
        Args:
            task_id: ID of the task to update
            updates: Dictionary of fields to update
            
        Returns:
            True if task was updated, False if not found
        """
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        
        # Update schedule if changed
        schedule_changed = False
        if 'schedule' in updates and updates['schedule'] != task.schedule:
            schedule_changed = True
        elif 'schedule_config' in updates and updates['schedule_config'] != task.schedule_config:
            schedule_changed = True
        
        # Apply updates
        for key, value in updates.items():
            if hasattr(task, key):
                setattr(task, key, value)
        
        # Update next run time if schedule changed
        if schedule_changed:
            task.update_next_run()
        
        # Save tasks
        self._save_tasks()
        
        log_info(
            f"Updated task {task_id} for dealer {task.dealer_id}",
            task.dealer_id,
            "update_task"
        )
        
        return True
    
    def delete_task(self, task_id: str) -> bool:
        """
        Delete a scheduled task.
        
        Args:
            task_id: ID of the task to delete
            
        Returns:
            True if task was deleted, False if not found
        """
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        dealer_id = task.dealer_id
        
        # Delete the task
        del self.tasks[task_id]
        
        # Save tasks
        self._save_tasks()
        
        log_info(
            f"Deleted task {task_id} for dealer {dealer_id}",
            dealer_id,
            "delete_task"
        )
        
        return True
    
    def disable_task(self, task_id: str) -> bool:
        """
        Disable a scheduled task.
        
        Args:
            task_id: ID of the task to disable
            
        Returns:
            True if task was disabled, False if not found
        """
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        task.enabled = False
        task.updated_at = datetime.now(timezone.utc)
        
        # Save tasks
        self._save_tasks()
        
        log_info(
            f"Disabled task {task_id} for dealer {task.dealer_id}",
            task.dealer_id,
            "disable_task"
        )
        
        return True
    
    def enable_task(self, task_id: str) -> bool:
        """
        Enable a scheduled task.
        
        Args:
            task_id: ID of the task to enable
            
        Returns:
            True if task was enabled, False if not found
        """
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        task.enabled = True
        task.updated_at = datetime.now(timezone.utc)
        
        # Update next run time if it's None or in the past
        now = datetime.now(timezone.utc)
        if task.next_run is None or task.next_run < now:
            task.update_next_run()
        
        # Save tasks
        self._save_tasks()
        
        log_info(
            f"Enabled task {task_id} for dealer {task.dealer_id}",
            task.dealer_id,
            "enable_task"
        )
        
        return True
    
    def run_task_now(self, task_id: str) -> bool:
        """
        Trigger immediate execution of a task.
        
        Args:
            task_id: ID of the task to run
            
        Returns:
            True if task was triggered, False if not found
        """
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        
        # Set next_run to now
        task.next_run = datetime.now(timezone.utc)
        task.updated_at = datetime.now(timezone.utc)
        
        # Mark as enabled if it wasn't
        if not task.enabled:
            task.enabled = True
        
        # Save tasks
        self._save_tasks()
        
        log_info(
            f"Triggered immediate execution of task {task_id} for dealer {task.dealer_id}",
            task.dealer_id,
            "run_task_now"
        )
        
        return True
    
    async def _initialize_client(self):
        """Initialize Nova client if not already done."""
        if self.client is None:
            self.client = NovaActClient()
            await self.client.start()
    
    async def start(self):
        """Start the scheduler."""
        if self.running:
            return
        
        self.running = True
        
        # Initialize Nova client
        await self._initialize_client()
        
        # Start scheduler task
        asyncio.create_task(self._scheduler_loop())
        
        # Start task processor
        asyncio.create_task(self._task_processor())
        
        log_info(
            "Scheduler started",
            "system",
            "scheduler_start"
        )
    
    async def stop(self):
        """Stop the scheduler."""
        if not self.running:
            return
        
        self.running = False
        
        # Cancel all running tasks
        for task_id, task in self.running_tasks.items():
            task.cancel()
        
        # Wait for tasks to cancel
        if self.running_tasks:
            await asyncio.gather(*self.running_tasks.values(), return_exceptions=True)
        
        # Save tasks
        self._save_tasks()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        log_info(
            "Scheduler stopped",
            "system",
            "scheduler_stop"
        )
    
    async def _scheduler_loop(self):
        """Main scheduler loop that checks for due tasks."""
        while self.running:
            try:
                now = datetime.now(timezone.utc)
                
                # Find tasks that are due and enabled
                due_tasks = [
                    task for task in self.tasks.values()
                    if task.enabled and task.next_run and task.next_run <= now
                ]
                
                # Sort by priority and then by next_run
                due_tasks.sort(
                    key=lambda t: (t.priority.value, t.next_run or now),
                    reverse=True  # Higher priority first
                )
                
                # Queue due tasks
                for task in due_tasks:
                    # Update task state
                    task.last_run = now
                    task.last_status = "queued"
                    task.update_next_run()
                    
                    # Put in queue
                    await self.task_queue.put(task)
                    
                    log_info(
                        f"Queued task {task.id} for dealer {task.dealer_id}, vendor {task.vendor_id}",
                        task.dealer_id,
                        "scheduler_queue"
                    )
                
                # Save tasks if any were queued
                if due_tasks:
                    self._save_tasks()
                
                # Sleep until next check
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                log_error(
                    e,
                    "system",
                    "scheduler_loop"
                )
                await asyncio.sleep(30)  # Wait longer on error
    
    async def _task_processor(self):
        """Process tasks from the queue."""
        while self.running:
            try:
                # Get next task from queue
                task = await self.task_queue.get()
                
                # Create async task to execute it
                execution_task = asyncio.create_task(
                    self._execute_task(task)
                )
                
                # Register in running tasks
                self.running_tasks[task.id] = execution_task
                
                # Set up callback to remove from running tasks
                def done_callback(future):
                    if task.id in self.running_tasks:
                        del self.running_tasks[task.id]
                
                execution_task.add_done_callback(done_callback)
                
                # Let queue know we processed this task
                self.task_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                log_error(
                    e,
                    "system",
                    "task_processor"
                )
                await asyncio.sleep(5)
    
    async def _execute_task(self, task: ScheduledTask):
        """Execute a scheduled task."""
        try:
            # Update task status
            task.last_status = "running"
            self._save_tasks()
            
            log_info(
                f"Executing task {task.id} for dealer {task.dealer_id}, vendor {task.vendor_id}, report {task.report_type}",
                task.dealer_id,
                "task_execute"
            )
            
            # Get credentials
            cred_manager = get_credential_manager()
            credentials = cred_manager.get_credential(task.dealer_id, task.vendor_id)
            
            if not credentials:
                raise ValueError(f"No credentials found for dealer {task.dealer_id}, vendor {task.vendor_id}")
            
            # Prepare report config based on report type
            report_config = self._get_report_config(task.report_type)
            
            # Execute data collection
            result = await self.client.collect_report(
                task.vendor_id,
                credentials.to_dict(),
                report_config
            )
            
            if not result.get("success", False):
                raise RuntimeError(f"Collection failed: {result.get('error', 'Unknown error')}")
            
            # Update task status
            task.last_status = "success"
            task.error_count = 0
            self._save_tasks()
            
            log_info(
                f"Successfully executed task {task.id} for dealer {task.dealer_id}",
                task.dealer_id,
                "task_success"
            )
            
            # Save the result data to the appropriate location
            await self._process_collected_data(task, result.get("file_path"))
            
        except Exception as e:
            # Update task error count and status
            task.error_count += 1
            task.last_status = "error"
            
            log_error(
                e,
                task.dealer_id,
                "task_error"
            )
            
            # Handle retry if needed
            if task.error_count < 3:  # Retry up to 3 times
                # Schedule retry in 15 minutes
                retry_time = datetime.now(timezone.utc) + timedelta(minutes=15)
                
                # For one-time tasks, update next_run directly
                if task.schedule == "once":
                    task.next_run = retry_time
                else:
                    # Only modify next_run for recurring tasks if the retry is sooner
                    if not task.next_run or retry_time < task.next_run:
                        task.next_run = retry_time
                
                log_info(
                    f"Scheduled retry for task {task.id} at {retry_time.isoformat()}",
                    task.dealer_id,
                    "task_retry"
                )
            else:
                # Too many retries, disable the task
                if task.schedule == "once":
                    # One-time tasks get deleted after too many failures
                    log_warning(
                        f"Deleting one-time task {task.id} after multiple failures",
                        task.dealer_id,
                        "task_delete"
                    )
                    self.delete_task(task.id)
                    return
                else:
                    # Recurring tasks get disabled
                    task.enabled = False
                    log_warning(
                        f"Disabled recurring task {task.id} after multiple failures",
                        task.dealer_id,
                        "task_disable"
                    )
            
            # Save tasks
            self._save_tasks()
    
    def _get_report_config(self, report_type: str) -> Dict[str, Any]:
        """Get report configuration for a specific report type."""
        # Basic configurations
        configs = {
            "sales": {
                "report_type": "sales_performance",
                "date_range": {
                    "start": (datetime.now(timezone.utc) - timedelta(days=30)).isoformat(),
                    "end": datetime.now(timezone.utc).isoformat()
                },
                "selectors": {
                    "report_menu": "#reports-menu",
                    "date_range": "#date-range",
                    "download": "#download-csv"
                }
            },
            "inventory": {
                "report_type": "inventory_report",
                "selectors": {
                    "report_menu": "#reports-menu",
                    "download": "#download-csv"
                }
            },
            "leads": {
                "report_type": "lead_activity",
                "date_range": {
                    "start": (datetime.now(timezone.utc) - timedelta(days=30)).isoformat(),
                    "end": datetime.now(timezone.utc).isoformat()
                },
                "selectors": {
                    "report_menu": "#reports-menu",
                    "date_range": "#date-range",
                    "download": "#download-csv"
                }
            }
        }
        
        # Return the config or a default
        return configs.get(report_type, {"report_type": report_type})
    
    async def _process_collected_data(self, task: ScheduledTask, file_path: Optional[str]):
        """Process collected data and store in the appropriate location."""
        if not file_path or not os.path.exists(file_path):
            log_warning(
                f"No data file found for task {task.id}",
                task.dealer_id,
                "process_data"
            )
            return
        
        try:
            # Determine target storage location
            target_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "data",
                "collected",
                task.vendor_id,
                task.dealer_id,
                task.report_type
            )
            os.makedirs(target_dir, exist_ok=True)
            
            # Create a timestamped filename
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = f"{task.report_type}_{timestamp}.csv"
            target_path = os.path.join(target_dir, filename)
            
            # Copy the file
            import shutil
            shutil.copy2(file_path, target_path)
            
            # Run the data normalization pipeline on this file
            await self._run_normalization_pipeline(task, target_path)
            
            log_info(
                f"Processed and stored data for task {task.id} at {target_path}",
                task.dealer_id,
                "process_data"
            )
            
        except Exception as e:
            log_error(
                e,
                task.dealer_id,
                "process_data"
            )
    
    async def _run_normalization_pipeline(self, task: ScheduledTask, file_path: str):
        """Run the data normalization pipeline."""
        try:
            # Import the necessary modules
            from ..ingestion_pipeline import normalize_and_validate
            
            # Run normalization pipeline
            result = await normalize_and_validate(
                file_path,
                task.vendor_id,
                task.report_type,
                task.dealer_id
            )
            
            if result.get("success", False):
                log_info(
                    f"Successfully normalized data for task {task.id}",
                    task.dealer_id,
                    "normalize_data"
                )
            else:
                log_warning(
                    f"Data normalization issues for task {task.id}: {result.get('message', 'Unknown error')}",
                    task.dealer_id,
                    "normalize_data"
                )
                
        except Exception as e:
            log_error(
                e,
                task.dealer_id,
                "normalize_data"
            )


# Singleton instance
_scheduler = None

def get_scheduler() -> EnhancedScheduler:
    """
    Get the singleton scheduler instance.
    
    Returns:
        The scheduler instance
    """
    global _scheduler
    if _scheduler is None:
        _scheduler = EnhancedScheduler()
    return _scheduler