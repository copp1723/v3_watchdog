"""
Task scheduling module for Nova Act.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

class NovaScheduler:
    """Manages scheduled tasks for Nova Act."""
    
    def __init__(self):
        """Initialize the scheduler."""
        self.tasks = {}
        self.running = False
    
    def schedule_task(self, task_func, schedule: str, task_kwargs: Optional[Dict[str, Any]] = None):
        """Schedule a task to run."""
        task_id = f"{task_func.__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.tasks[task_id] = {
            'func': task_func,
            'schedule': schedule,
            'kwargs': task_kwargs or {},
            'last_run': None,
            'next_run': self._calculate_next_run(schedule)
        }
        logger.info(f"Scheduled task {task_id} to run {schedule}")
        return task_id

    def _calculate_next_run(self, schedule: str) -> datetime:
        """Calculate the next run time based on schedule."""
        now = datetime.now()
        if schedule == 'daily':
            # Run at midnight next day
            return (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        elif schedule == 'weekly':
            # Run at midnight next Monday
            days_ahead = 7 - now.weekday()
            return (now + timedelta(days=days_ahead)).replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            raise ValueError(f"Unsupported schedule: {schedule}")

# Example usage
scheduler = NovaScheduler()

def run_collection(vendor: str):
    """Run data collection for a vendor."""
    logger.info(f"Running collection for {vendor}")
    # Collection logic here

# Schedule tasks
scheduler.schedule_task(run_collection, schedule="daily", task_kwargs={'vendor': "vinsolutions"})
scheduler.schedule_task(run_collection, schedule="weekly", task_kwargs={'vendor': "dealersocket"})