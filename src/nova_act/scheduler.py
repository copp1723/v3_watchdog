import schedule
import time
import sys
import os
import json
import threading
from datetime import datetime, timedelta
import atexit
import logging
from typing import Dict, List, Callable, Any, Optional, Tuple
import uuid

# Ensure the project root is in the path for sibling imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    # Attempt relative import first
    from .core import NovaActClient
    from .watchdog_upload import upload_to_watchdog
except ImportError:
    # Fallback to absolute import if relative fails (e.g., when run as a script)
    print("[WARN][Scheduler] Relative import failed, trying absolute.")
    from src.nova_act.core import NovaActClient
    from src.nova_act.watchdog_upload import upload_to_watchdog


class NovaScheduler:
    """
    Scheduler component for Nova Act integration that manages scheduled tasks.
    Provides functionality to schedule, cancel, and run tasks at specified times.
    """
    
    def __init__(self, client: NovaActClient, config: dict, storage_dir: str = None):
        """
        Initializes the scheduler.

        Args:
            client (NovaActClient): An instance of the NovaActClient.
            config (dict): Configuration dictionary containing vendor details.
            storage_dir (str, optional): Directory to store task persistence data.
        """
        self.client = client
        self.config = config
        
        # Set up task storage
        if storage_dir is None:
            storage_dir = os.path.join(os.path.dirname(__file__), 'scheduler_storage')
        self.storage_dir = storage_dir
        
        # Create storage directory if it doesn't exist
        if not os.path.exists(self.storage_dir):
            try:
                os.makedirs(self.storage_dir)
                print(f"[INFO][Scheduler] Created storage directory: {self.storage_dir}")
            except OSError as e:
                print(f"[ERROR][Scheduler] Failed to create storage directory: {e}")
        
        # Track scheduled tasks for persistence
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self.logger = logging.getLogger("NovaScheduler")
        
        # Register shutdown handler
        atexit.register(self.shutdown)
        
        # Load any previously scheduled tasks
        self.load_tasks()
        
        print("[INFO][Scheduler] NovaScheduler initialized.")

    def schedule_task(self, 
                     task_func: Callable, 
                     schedule: str = "once", 
                     schedule_time: Optional[datetime] = None,
                     interval: int = 0,
                     task_args: Tuple = (), 
                     task_kwargs: Dict[str, Any] = None) -> str:
        """
        Schedule a task to be executed
        
        Args:
            task_func: The function to execute
            schedule: 'once', 'interval', or 'daily'
            schedule_time: When to execute the task (if None, execute immediately)
            interval: Seconds between executions (for 'interval' schedule)
            task_args: Positional arguments for the task function
            task_kwargs: Keyword arguments for the task function
        
        Returns:
            task_id: Unique identifier for the scheduled task
        """
        if task_kwargs is None:
            task_kwargs = {}
            
        task_id = str(uuid.uuid4())
        
        with self.lock:
            self.tasks[task_id] = {
                'func': task_func,
                'schedule': schedule,
                'schedule_time': schedule_time or datetime.now(),
                'interval': interval,
                'args': task_args,
                'kwargs': task_kwargs,
                'last_run': None,
                'status': 'pending'
            }
            
        self.logger.info(f"Task {task_id} scheduled with {schedule} schedule")
        return task_id
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a scheduled task
        
        Args:
            task_id: The ID of the task to cancel
            
        Returns:
            bool: True if the task was canceled, False if it wasn't found
        """
        with self.lock:
            if task_id in self.tasks:
                del self.tasks[task_id]
                self.logger.info(f"Task {task_id} canceled")
                return True
            return False
    
    def run_pending_tasks(self):
        """Run all pending tasks that are due"""
        now = datetime.now()
        completed_tasks = []
        
        with self.lock:
            for task_id, task in self.tasks.items():
                if task['status'] == 'pending' and task['schedule_time'] <= now:
                    self._execute_task(task_id, task)
                    
                    # Handle different schedule types
                    if task['schedule'] == 'once':
                        completed_tasks.append(task_id)
                    elif task['schedule'] == 'interval':
                        # Schedule next run
                        task['schedule_time'] = now + timedelta(seconds=task['interval'])
                        task['status'] = 'pending'
                    elif task['schedule'] == 'daily':
                        # Schedule for tomorrow
                        tomorrow = now + timedelta(days=1)
                        task['schedule_time'] = datetime.combine(
                            tomorrow.date(),
                            task['schedule_time'].time()
                        )
                        task['status'] = 'pending'
            
            # Remove completed one-time tasks
            for task_id in completed_tasks:
                del self.tasks[task_id]
    
    def _execute_task(self, task_id: str, task: Dict[str, Any]):
        """Execute a task and update its status"""
        try:
            self.logger.info(f"Executing task {task_id}")
            task['func'](*task['args'], **task['kwargs'])
            task['last_run'] = datetime.now()
            task['status'] = 'completed'
            self.logger.info(f"Task {task_id} completed successfully")
        except Exception as e:
            task['status'] = 'failed'
            task['error'] = str(e)
            self.logger.error(f"Task {task_id} failed: {e}")
    
    def start(self):
        """Start the scheduler thread"""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._run_loop)
        self.thread.daemon = True
        self.thread.start()
        self.logger.info("Scheduler started")
    
    def stop(self):
        """Stop the scheduler thread"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5.0)
        self.logger.info("Scheduler stopped")
    
    def _run_loop(self):
        """Main scheduler loop"""
        while self.running:
            try:
                self.run_pending_tasks()
                time.sleep(1)  # Check every second
            except Exception as e:
                self.logger.error(f"Error in scheduler loop: {e}")
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a task
        
        Args:
            task_id: The ID of the task
            
        Returns:
            Dict containing task status and details or None if not found
        """
        with self.lock:
            if task_id in self.tasks:
                return {
                    'id': task_id,
                    'status': self.tasks[task_id]['status'],
                    'schedule_time': self.tasks[task_id]['schedule_time'],
                    'last_run': self.tasks[task_id]['last_run'],
                    'error': self.tasks[task_id].get('error')
                }
        return None
    
    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """
        Get a list of all scheduled tasks
        
        Returns:
            List of task details
        """
        tasks_list = []
        with self.lock:
            for task_id, task in self.tasks.items():
                tasks_list.append({
                    'id': task_id,
                    'status': task['status'],
                    'schedule': task['schedule'],
                    'schedule_time': task['schedule_time'],
                    'last_run': task['last_run']
                })
        return tasks_list

    def _validate_time_format(self, time_str):
        """Validate time string format (HH:MM)."""
        try:
            datetime.strptime(time_str, "%H:%M")
            return True
        except ValueError:
            return False

    def run_collection(self, vendor: str):
        """
        Runs the data collection process for a given vendor and uploads the result.

        Args:
            vendor (str): The name of the vendor.
        """
        print(f"[INFO][Scheduler] Running scheduled collection for vendor: '{vendor}'...")
        if vendor not in self.config:
             print(f"[ERROR][Scheduler] Cannot run collection: Vendor '{vendor}' not found in config.")
             return
        
        # Update last run time for all tasks for this vendor
        now = datetime.now().isoformat()
        task_updated = False
        
        for task_key, task_info in self.tasks.items():
            if task_info['func'] == self.run_collection and task_info['kwargs']['vendor'] == vendor:
                task_info['last_run'] = now
                task_updated = True
                
        if task_updated:
            self.save_tasks()
             
        try:
            # Pass the specific configuration for this vendor to the collect_report method
            file_path = self.client.collect_report(vendor, self.config[vendor])
            if file_path:
                print(f"[INFO][Scheduler] Report collected for '{vendor}': {file_path}. Attempting upload...")
                success = upload_to_watchdog(file_path)
                if success:
                    print(f"[INFO][Scheduler] Successfully uploaded report from '{vendor}' to Watchdog.")
                    self._update_task_status(vendor, "success")
                else:
                    print(f"[ERROR][Scheduler] Failed to upload report from '{vendor}' to Watchdog.")
                    self._update_task_status(vendor, "upload_failed")
            else:
                print(f"[WARN][Scheduler] No report file collected for vendor '{vendor}'.")
                self._update_task_status(vendor, "collection_failed")
        except Exception as e:
             print(f"[ERROR][Scheduler] Error during collection or upload for vendor '{vendor}': {e}")
             self._update_task_status(vendor, "error", str(e))
             # Consider adding more robust error handling/logging here
    
    def _update_task_status(self, vendor, status, message=None):
        """Update task status information for persistence."""
        for task_key, task_info in self.tasks.items():
            if task_info['func'] == self.run_collection and task_info['kwargs']['vendor'] == vendor:
                task_info['last_status'] = status
                if message:
                    task_info['last_message'] = message
                task_info['last_updated'] = datetime.now().isoformat()
        
        # Save the updated status
        self.save_tasks()

    def save_tasks(self):
        """Save scheduled tasks to disk for persistence."""
        tasks_file = os.path.join(self.storage_dir, 'scheduled_tasks.json')
        
        # Convert to a serializable format (can't serialize job objects)
        serializable_tasks = {}
        for task_key, task_info in self.tasks.items():
            serializable_tasks[task_key] = {k: v for k, v in task_info.items() if k != 'func'}
            
        try:
            with open(tasks_file, 'w') as f:
                json.dump(serializable_tasks, f, indent=2)
            print(f"[INFO][Scheduler] Saved {len(serializable_tasks)} task(s) to {tasks_file}")
        except Exception as e:
            print(f"[ERROR][Scheduler] Failed to save tasks: {e}")

    def load_tasks(self):
        """Load scheduled tasks from disk and recreate schedules."""
        tasks_file = os.path.join(self.storage_dir, 'scheduled_tasks.json')
        
        if not os.path.exists(tasks_file):
            print(f"[INFO][Scheduler] No tasks file found at {tasks_file}")
            return
            
        try:
            with open(tasks_file, 'r') as f:
                saved_tasks = json.load(f)
                
            # Recreate schedules for each saved task
            for task_key, task_info in saved_tasks.items():
                vendor = task_info['vendor']
                frequency = task_info['frequency']
                time_of_day = task_info['time_of_day']
                
                # Only reschedule if vendor still exists in config
                if vendor in self.config:
                    print(f"[INFO][Scheduler] Restoring saved task: {vendor} ({frequency} at {time_of_day})")
                    self.schedule_task(self.run_collection, frequency=frequency, task_kwargs={'vendor': vendor})
                else:
                    print(f"[WARN][Scheduler] Vendor '{vendor}' from saved task is not in current config. Skipping.")
                    
            print(f"[INFO][Scheduler] Loaded scheduler with {len(self.tasks)} task(s)")
        except Exception as e:
            print(f"[ERROR][Scheduler] Failed to load tasks: {e}")

    def shutdown(self):
        """Gracefully shut down the scheduler."""
        print("[INFO][Scheduler] Shutting down scheduler...")
        self.running = False
        # Persist any final state
        self.save_tasks()
        print("[INFO][Scheduler] Scheduler shutdown complete.")

# Example Usage (if run directly)
if __name__ == "__main__":
    print("Running NovaScheduler Example...")
    # This example requires a mock or real NovaActClient and config
    # Replace with actual implementation details
    class MockNovaActClient:
        def collect_report(self, vendor, config):
            print(f"[MOCK] Collecting report for {vendor}...")
            # Simulate creating a dummy file
            dummy_path = f"./dummy_report_{vendor}_{int(time.time())}.csv"
            with open(dummy_path, 'w') as f:
                f.write("col1,col2\nval1,val2")
            print(f"[MOCK] Created dummy report: {dummy_path}")
            return dummy_path

    # Dummy config
    mock_config = {
        "vinsolutions": {"username": "user1", "password": "pass1", "report_path": "/reports/daily"},
        "dealersocket": {"username": "user2", "password": "pass2", "report_path": "/exports"}
    }
    
    mock_client = MockNovaActClient()
    scheduler = NovaScheduler(client=mock_client, config=mock_config)
    
    # Schedule tasks
    scheduler.schedule_task(self.run_collection, schedule="daily", task_kwargs={'vendor': "vinsolutions"})
    scheduler.schedule_task(self.run_collection, schedule="weekly", task_kwargs={'vendor': "dealersocket"})
    
    # Show that task persistence works
    print("\nInitial Scheduled Tasks:")
    for task, info in scheduler.tasks.items():
        print(f"  - {task}: {info['vendor']} ({info['schedule']} at {info['schedule_time']})")
    
    # Example of cancelling and rescheduling
    scheduler.cancel_task("vinsolutions")
    scheduler.schedule_task(self.run_collection, schedule="daily", task_kwargs={'vendor': "vinsolutions"})
    
    print("\nAfter Modification Scheduled Tasks:")
    for task, info in scheduler.tasks.items():
        print(f"  - {task}: {info['vendor']} ({info['schedule']} at {info['schedule_time']})")
    
    print("\nStarting background scheduler for 10 seconds...")
    thread = threading.Thread(target=scheduler.start, daemon=True)
    thread.start()
    time.sleep(10)
    scheduler.stop()
    print("Example complete.") 