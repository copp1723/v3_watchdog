"""
CRM Sync Scheduler

This module provides scheduling functionality for CRM data synchronization tasks,
including hourly sales data pulls and daily insight pushes.
"""

import os
import sys
import json
import logging
import signal
import time
import traceback
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable

# APScheduler for job scheduling
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger

# Import our CRM adapters
from watchdog_ai.integrations.crm import NovaActAdapter, BaseCRMAdapter

# Configure logging
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_DATA_DIR = Path("data/raw")
DEFAULT_INSIGHTS_DIR = Path("data/processed")
DEFAULT_LOG_DIR = Path("data/logs")
DEFAULT_DB_PATH = Path("data/scheduler.sqlite")

# Environment variable names
ENV_DATA_DIR = "WATCHDOG_DATA_DIR"
ENV_INSIGHTS_DIR = "WATCHDOG_INSIGHTS_DIR"
ENV_LOG_DIR = "WATCHDOG_LOG_DIR"
ENV_DB_PATH = "WATCHDOG_SCHEDULER_DB"
ENV_SALES_INTERVAL = "WATCHDOG_SALES_SYNC_INTERVAL"
ENV_INSIGHTS_CRON = "WATCHDOG_INSIGHTS_SYNC_CRON"


def configure_logging(log_dir: Optional[Union[str, Path]] = None, level: str = "INFO") -> Path:
    """
    Configure logging for the scheduler.
    
    Args:
        log_dir: Directory for log files
        level: Logging level
        
    Returns:
        Path to log file
    """
    log_dir = Path(log_dir or DEFAULT_LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    log_file = log_dir / f"scheduler_{datetime.now().strftime('%Y%m%d')}.log"
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level))
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add file handler
    file_handler = logging.FileHandler(log_file)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    logger.info(f"Logging configured. Log file: {log_file}")
    return log_file


# Standalone job functions for serialization
def hourly_pull_sales_job(crm_adapter: BaseCRMAdapter, data_dir: Path, log_dir: Path) -> Dict[str, Any]:
    """
    Pull sales data from CRM and save to data/raw directory.
    
    This job runs hourly to fetch the latest sales data.
    
    Args:
        crm_adapter: The CRM adapter to use
        data_dir: Directory to save data
        log_dir: Directory to save logs
        
    Returns:
        Dict with job status information
    """
    start_time = datetime.now()
    job_id = f"sales_pull_{start_time.strftime('%Y%m%d%H%M%S')}"
    
    logger.info(f"Starting hourly sales data pull job: {job_id}")
    
    try:
        # Authenticate with CRM
        crm_adapter.authenticate()
        
        # Pull sales data since last run
        # In a more sophisticated implementation, we'd track the last sync time
        # and only pull data since then, but for simplicity, we're pulling all data
        sales_data = crm_adapter.pull_sales()
        
        if not sales_data:
            logger.info("No sales data retrieved")
            return {
                "job_id": job_id,
                "status": "success",
                "message": "No sales data available",
                "record_count": 0,
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "data_file": None
            }
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"sales_{timestamp}.json"
        file_path = data_dir / filename
        
        # Save data to file
        with open(file_path, 'w') as f:
            json.dump(sales_data, f, indent=2)
        
        logger.info(f"Saved {len(sales_data)} sales records to {file_path}")
        
        # Return job status
        return {
            "job_id": job_id,
            "status": "success",
            "record_count": len(sales_data),
            "start_time": start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "data_file": str(file_path)
        }
        
    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        logger.error(f"Error in hourly sales data pull: {error_msg}")
        logger.debug(error_trace)
        
        # Log error details to file
        error_log_path = log_dir / f"error_{job_id}.log"
        with open(error_log_path, 'w') as f:
            f.write(f"Error: {error_msg}\n\nTraceback:\n{error_trace}")
        
        # Return error status
        return {
            "job_id": job_id,
            "status": "error",
            "message": error_msg,
            "start_time": start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "error_log": str(error_log_path)
        }


def daily_push_insights_job(crm_adapter: BaseCRMAdapter, insights_dir: Path, log_dir: Path) -> Dict[str, Any]:
    """
    Push processed insights to CRM.
    
    This job runs daily to push processed insights back to the CRM system.
    It reads the latest insights from data/processed/latest.json
    
    Args:
        crm_adapter: The CRM adapter to use
        insights_dir: Directory with insight files
        log_dir: Directory to save logs
        
    Returns:
        Dict with job status information
    """
    start_time = datetime.now()
    job_id = f"insights_push_{start_time.strftime('%Y%m%d%H%M%S')}"
    
    logger.info(f"Starting daily insights push job: {job_id}")
    
    try:
        # Check for insights file
        insights_file = insights_dir / "latest.json"
        
        if not insights_file.exists():
            logger.warning(f"Insights file not found: {insights_file}")
            return {
                "job_id": job_id,
                "status": "skipped",
                "message": f"Insights file not found: {insights_file}",
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat()
            }
        
        # Load insights data
        with open(insights_file, 'r') as f:
            insights_data = json.load(f)
        
        if not insights_data:
            logger.info("No insights data to push")
            return {
                "job_id": job_id,
                "status": "success",
                "message": "No insights data to push",
                "record_count": 0,
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat()
            }
        
        # Authenticate with CRM
        crm_adapter.authenticate()
        
        # Push insights to CRM
        crm_adapter.push_insights(insights_data)
        
        logger.info(f"Successfully pushed {len(insights_data)} insights to CRM")
        
        # Archive the insights file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        archive_path = insights_dir / f"archive/insights_{timestamp}.json"
        archive_path.parent.mkdir(exist_ok=True)
        
        # Copy insights to archive
        import shutil
        shutil.copy2(insights_file, archive_path)
        
        # Return job status
        return {
            "job_id": job_id,
            "status": "success",
            "record_count": len(insights_data),
            "start_time": start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "insights_file": str(insights_file),
            "archive_file": str(archive_path)
        }
            
    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        logger.error(f"Error in daily insights push: {error_msg}")
        logger.debug(error_trace)
        
        # Log error details to file
        error_log_path = log_dir / f"error_{job_id}.log"
        with open(error_log_path, 'w') as f:
            f.write(f"Error: {error_msg}\n\nTraceback:\n{error_trace}")
        
        # Return error status
        return {
            "job_id": job_id,
            "status": "error",
            "message": error_msg,
            "start_time": start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "error_log": str(error_log_path)
        }


class CRMSyncScheduler:
    """
    Scheduler for CRM data synchronization tasks.
    
    This class manages periodic CRM data synchronization jobs using APScheduler,
    including hourly pulls of sales data and daily pushes of insights.
    """
    
    def __init__(self, 
                data_dir: Optional[Union[str, Path]] = None,
                insights_dir: Optional[Union[str, Path]] = None,
                log_dir: Optional[Union[str, Path]] = None,
                db_path: Optional[Union[str, Path]] = None,
                sales_sync_interval: int = 60,  # minutes
                insights_sync_cron: str = "0 2 * * *",  # 2:00 AM daily
                crm_adapter: Optional[BaseCRMAdapter] = None):
        """
        Initialize the CRM Sync Scheduler.
        
        Args:
            data_dir: Directory for storing raw CRM data. Default data/raw/
            insights_dir: Directory for processed insights. Default data/processed/
            log_dir: Directory for logs. Default data/logs/
            db_path: Path to SQLite DB for job persistence. Default data/scheduler.sqlite
            sales_sync_interval: Interval in minutes for sales data sync. Default 60 (hourly)
            insights_sync_cron: Cron expression for insights sync. Default "0 2 * * *" (2:00 AM daily)
            crm_adapter: CRM adapter to use. Default is NovaActAdapter with env vars
        """
        # Set up paths from arguments or environment variables
        self.data_dir = Path(data_dir or os.environ.get(ENV_DATA_DIR, DEFAULT_DATA_DIR))
        self.insights_dir = Path(insights_dir or os.environ.get(ENV_INSIGHTS_DIR, DEFAULT_INSIGHTS_DIR))
        self.log_dir = Path(log_dir or os.environ.get(ENV_LOG_DIR, DEFAULT_LOG_DIR))
        self.db_path = Path(db_path or os.environ.get(ENV_DB_PATH, DEFAULT_DB_PATH))
        
        # Create directories if they don't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.insights_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure job schedules
        self.sales_sync_interval = int(os.environ.get(ENV_SALES_INTERVAL, sales_sync_interval))
        self.insights_sync_cron = os.environ.get(ENV_INSIGHTS_CRON, insights_sync_cron)
        
        # Set up the CRM adapter
        self.crm_adapter = crm_adapter or NovaActAdapter()
        
        # Initialize scheduler with SQLite job store
        self.scheduler = BackgroundScheduler()
        self.scheduler.add_jobstore(
            SQLAlchemyJobStore(url=f'sqlite:///{self.db_path}'),
            'default'
        )
        
        # Track if scheduler is running
        self.is_running = False
        
        logger.info(f"Initialized CRM Sync Scheduler with data_dir={self.data_dir}, "
                  f"sales_interval={self.sales_sync_interval}min, "
                  f"insights_cron='{self.insights_sync_cron}'")
    
    def start(self):
        """
        Start the scheduler and register jobs.
        """
        if self.is_running:
            logger.warning("Scheduler is already running")
            return
        
        # Register jobs
        self._register_jobs()
        
        # Start the scheduler
        self.scheduler.start()
        self.is_running = True
        
        logger.info("CRM Sync Scheduler started")
    
    def stop(self):
        """
        Stop the scheduler.
        """
        if not self.is_running:
            logger.warning("Scheduler is not running")
            return
        
        # Shutdown the scheduler
        self.scheduler.shutdown()
        self.is_running = False
        
        logger.info("CRM Sync Scheduler stopped")
    
    def _register_jobs(self):
        """
        Register all scheduled jobs.
        """
        # Sales data sync job - runs hourly by default
        self.scheduler.add_job(
            func=hourly_pull_sales_job,
            args=[self.crm_adapter, self.data_dir, self.log_dir],
            trigger=IntervalTrigger(minutes=self.sales_sync_interval),
            id='sales_sync',
            name='Sales Data Sync',
            replace_existing=True
        )
        
        # Insights sync job - runs at 2:00 AM daily by default
        self.scheduler.add_job(
            func=daily_push_insights_job,
            args=[self.crm_adapter, self.insights_dir, self.log_dir],
            trigger=CronTrigger

"""
CRM Sync Scheduler

This module provides scheduling functionality for CRM data synchronization tasks,
including hourly sales data pulls and daily insight pushes.
"""

import os
import sys
import json
import logging
import signal
import time
import traceback
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable

# APScheduler for job scheduling
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger

# Import our CRM adapters
from watchdog_ai.integrations.crm import NovaActAdapter, BaseCRMAdapter

# Configure logging
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_DATA_DIR = Path("data/raw")
DEFAULT_INSIGHTS_DIR = Path("data/processed")
DEFAULT_LOG_DIR = Path("data/logs")
DEFAULT_DB_PATH = Path("data/scheduler.sqlite")

# Environment variable names
ENV_DATA_DIR = "WATCHDOG_DATA_DIR"
ENV_INSIGHTS_DIR = "WATCHDOG_INSIGHTS_DIR"
ENV_LOG_DIR = "WATCHDOG_LOG_DIR"
ENV_DB_PATH = "WATCHDOG_SCHEDULER_DB"
ENV_SALES_INTERVAL = "WATCHDOG_SALES_SYNC_INTERVAL"
ENV_INSIGHTS_CRON = "WATCHDOG_INSIGHTS_SYNC_CRON"


def configure_logging(log_dir: Optional[Union[str, Path]] = None, level: str = "INFO") -> Path:
    """
    Configure logging for the scheduler.
    
    Args:
        log_dir: Directory for log files
        level: Logging level
        
    Returns:
        Path to log file
    """
    log_dir = Path(log_dir or DEFAULT_LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    log_file = log_dir / f"scheduler_{datetime.now().strftime('%Y%m%d')}.log"
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level))
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add file handler
    file_handler = logging.FileHandler(log_file)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    logger.info(f"Logging configured. Log file: {log_file}")
    return log_file


# Standalone job functions for serialization
def hourly_pull_sales_job(crm_adapter: BaseCRMAdapter, data_dir: Path, log_dir: Path) -> Dict[str, Any]:
    """
    Pull sales data from CRM and save to data/raw directory.
    
    This job runs hourly to fetch the latest sales data.
    
    Args:
        crm_adapter: The CRM adapter to use
        data_dir: Directory to save data
        log_dir: Directory to save logs
        
    Returns:
        Dict with job status information
    """
    start_time = datetime.now()
    job_id = f"sales_pull_{start_time.strftime('%Y%m%d%H%M%S')}"
    
    logger.info(f"Starting hourly sales data pull job: {job_id}")
    
    try:
        # Authenticate with CRM
        crm_adapter.authenticate()
        
        # Pull sales data since last run
        # In a more sophisticated implementation, we'd track the last sync time
        # and only pull data since then, but for simplicity, we're pulling all data
        sales_data = crm_adapter.pull_sales()
        
        if not sales_data:
            logger.info("No sales data retrieved")
            return {
                "job_id": job_id,
                "status": "success",
                "message": "No sales data available",
                "record_count": 0,
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "data_file": None
            }
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"sales_{timestamp}.json"
        file_path = data_dir / filename
        
        # Save data to file
        with open(file_path, 'w') as f:
            json.dump(sales_data, f, indent=2)
        
        logger.info(f"Saved {len(sales_data)} sales records to {file_path}")
        
        # Return job status
        return {
            "job_id": job_id,
            "status": "success",
            "record_count": len(sales_data),
            "start_time": start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "data_file": str(file_path)
        }
        
    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        logger.error(f"Error in hourly sales data pull: {error_msg}")
        logger.debug(error_trace)
        
        # Log error details to file
        error_log_path = log_dir / f"error_{job_id}.log"
        with open(error_log_path, 'w') as f:
            f.write(f"Error: {error_msg}\n\nTraceback:\n{error_trace}")
        
        # Return error status
        return {
            "job_id": job_id,
            "status": "error",
            "message": error_msg,
            "start_time": start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "error_log": str(error_log_path)
        }


def daily_push_insights_job(crm_adapter: BaseCRMAdapter, insights_dir: Path, log_dir: Path) -> Dict[str, Any]:
    """
    Push processed insights to CRM.
    
    This job runs daily to push processed insights back to the CRM system.
    It reads the latest insights from data/processed/latest.json
    
    Args:
        crm_adapter: The CRM adapter to use
        insights_dir: Directory with insight files
        log_dir: Directory to save logs
        
    Returns:
        Dict with job status information
    """
    start_time = datetime.now()
    job_id = f"insights_push_{start_time.strftime('%Y%m%d%H%M%S')}"
    
    logger.info(f"Starting daily insights push job: {job_id}")
    
    try:
        # Check for insights file
        insights_file = insights_dir / "latest.json"
        
        if not insights_file.exists():
            logger.warning(f"Insights file not found: {insights_file}")
            return {
                "job_id": job_id,
                "status": "skipped",
                "message": f"Insights file not found: {insights_file}",
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat()
            }
        
        # Load insights data
        with open(insights_file, 'r') as f:
            insights_data = json.load(f)
        
        if not insights_data:
            logger.info("No insights data to push")
            return {
                "job_id": job_id,
                "status": "success",
                "message": "No insights data to push",
                "record_count": 0,
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat()
            }
        
        # Authenticate with CRM
        crm_adapter.authenticate()
        
        # Push insights to CRM
        crm_adapter.push_insights(insights_data)
        
        logger.info(f"Successfully pushed {len(insights_data)} insights to CRM")
        
        # Archive the insights file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        archive_path = insights_dir / f"archive/insights_{timestamp}.json"
        archive_path.parent.mkdir(exist_ok=True)
        
        # Copy insights to archive
        import shutil
        shutil.copy2(insights_file, archive_path)
        
        # Return job status
        return {
            "job_id": job_id,
            "status": "success",
            "record_count": len(insights_data),
            "start_time": start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "insights_file": str(insights_file),
            "archive_file": str(archive_path)
        }
            
    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        logger.error(f"Error in daily insights push: {error_msg}")
        logger.debug(error_trace)
        
        # Log error details to file
        error_log_path = log_dir / f"error_{job_id}.log"
        with open(error_log_path, 'w') as f:
            f.write(f"Error: {error_msg}\n\nTraceback:\n{error_trace}")
        
        # Return error status
        return {
            "job_id": job_id,
            "status": "error",
            "message": error_msg,
            "start_time": start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "error_log": str(error_log_path)
        }


class CRMSyncScheduler:
    """
    Scheduler for CRM data synchronization tasks.
    
    This class manages periodic CRM data synchronization jobs using APScheduler,
    including hourly pulls of sales data and daily pushes of insights.
    """
    
    def __init__(self, 
                data_dir: Optional[Union[str, Path]] = None,
                insights_dir: Optional[Union[str, Path]] = None,
                log_dir: Optional[Union[str, Path]] = None,
                db_path: Optional[Union[str, Path]] = None,
                sales_sync_interval: int = 60,  # minutes
                insights_sync_cron: str = "0 2 * * *",  # 2:00 AM daily
                crm_adapter: Optional[BaseCRMAdapter] = None):
        """
        Initialize the CRM Sync Scheduler.
        
        Args:
            data_dir: Directory for storing raw CRM data. Default data/raw/
            insights_dir: Directory for processed insights. Default data/processed/
            log_dir: Directory for logs. Default data/logs/
            db_path: Path to SQLite DB for job persistence. Default data/scheduler.sqlite
            sales_sync_interval: Interval in minutes for sales data sync. Default 60 (hourly)
            insights_sync_cron: Cron expression for insights sync. Default "0 2 * * *" (2:00 AM daily)
            crm_adapter: CRM adapter to use. Default is NovaActAdapter with env vars
        """
        # Set up paths from arguments or environment variables
        self.data_dir = Path(data_dir or os.environ.get(ENV_DATA_DIR, DEFAULT_DATA_DIR))
        self.insights_dir = Path(insights_dir or os.environ.get(ENV_INSIGHTS_DIR, DEFAULT_INSIGHTS_DIR))
        self.log_dir = Path(log_dir or os.environ.get(ENV_LOG_DIR, DEFAULT_LOG_DIR))
        self.db_path = Path(db_path or os.environ.get(ENV_DB_PATH, DEFAULT_DB_PATH))
        
        # Create directories if they don't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.insights_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure job schedules
        self.sales_sync_interval = int(os.environ.get(ENV_SALES_INTERVAL, sales_sync_interval))
        self.insights_sync_cron = os.environ.get(ENV_INSIGHTS_CRON, insights_sync_cron)
        
        # Set up the CRM adapter
        self.crm_adapter = crm_adapter or NovaActAdapter()
        
        # Initialize scheduler with SQLite job store
        self.scheduler = BackgroundScheduler()
        self.scheduler.add_jobstore(
            SQLAlchemyJobStore(url=f'sqlite:///{self.db_path}'),
            'default'
        )
        
        # Track if scheduler is running
        self.is_running = False
        
        logger.info(f"Initialized CRM Sync Scheduler with data_dir={self.data_dir}, "
                  f"sales_interval={self.sales_sync_interval}min, "
                  f"insights_cron='{self.insights_sync_cron}'")
    
    def start(self):
        """
        Start the scheduler and register jobs.
        """
        if self.is_running:
            logger.warning("Scheduler is already running")
            return
        
        # Register jobs
        self._register_jobs()
        
        # Start the scheduler
        self.scheduler.start()
        self.is_running = True
        
        logger.info("CRM Sync Scheduler started")
    
    def stop(self):
        """
        Stop the scheduler.
        """
        if not self.is_running:
            logger.warning("Scheduler is not running")
            return
        
        # Shutdown the scheduler
        self.scheduler.shutdown()
        self.is_running = False
        
        logger.info("CRM Sync Scheduler stopped")
    
    def _register_jobs(self):
        """
        Register all scheduled jobs.
        """
        # Sales data sync job - runs hourly by default
        self.scheduler.add_job(
            func=hourly_pull_sales_job,
            args=[self.crm_adapter, self.data_dir, self.log_dir],
            trigger=IntervalTrigger(minutes=self.sales_sync_interval),
            id='sales_sync',
            name='Sales Data Sync',
            replace_existing=True
        )
        
        # Insights sync job - runs at 2:00 AM daily by default
        self.scheduler.add_job(
            func=daily_push_insights_job,
            args=[self.crm_adapter, self.insights_dir, self.log_dir],
            trigger=CronTrigger.from_crontab(self.

"""
CRM Sync Scheduler

This module provides scheduling functionality for CRM data synchronization tasks,
including hourly sales data pulls and daily insight pushes.
"""

import os
import sys
import json
import logging
import signal
import time
import traceback
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable

# APScheduler for job scheduling
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger

# Import our CRM adapters
from watchdog_ai.integrations.crm import NovaActAdapter, BaseCRMAdapter

# Configure logging
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_DATA_DIR = Path("data/raw")
DEFAULT_INSIGHTS_DIR = Path("data/processed")
DEFAULT_LOG_DIR = Path("data/logs")
DEFAULT_DB_PATH = Path("data/scheduler.sqlite")

# Environment variable names
ENV_DATA_DIR = "WATCHDOG_DATA_DIR"
ENV_INSIGHTS_DIR = "WATCHDOG_INSIGHTS_DIR"
ENV_LOG_DIR = "WATCHDOG_LOG_DIR"
ENV_DB_PATH = "WATCHDOG_SCHEDULER_DB"
ENV_SALES_INTERVAL = "WATCHDOG_SALES_SYNC_INTERVAL"
ENV_INSIGHTS_CRON = "WATCHDOG_INSIGHTS_SYNC_CRON"


# Standalone job functions for serialization
def hourly_pull_sales_job(crm_adapter: BaseCRMAdapter, data_dir: Path, log_dir: Path) -> Dict[str, Any]:
    """
    Pull sales data from CRM and save to data/raw directory.
    
    This job runs hourly to fetch the latest sales data.
    
    Args:
        crm_adapter: The CRM adapter to use
        data_dir: Directory to save data
        log_dir: Directory to save logs
        
    Returns:
        Dict with job status information
    """
    start_time = datetime.now()
    job_id = f"sales_pull_{start_time.strftime('%Y%m%d%H%M%S')}"
    
    logger.info(f"Starting hourly sales data pull job: {job_id}")
    
    try:
        # Authenticate with CRM
        crm_adapter.authenticate()
        
        # Pull sales data since last run
        # In a more sophisticated implementation, we'd track the last sync time
        # and only pull data since then, but for simplicity, we're pulling all data
        sales_data = crm_adapter.pull_sales()
        
        if not sales_data:
            logger.info("No sales data retrieved")
            return {
                "job_id": job_id,
                "status": "success",
                "message": "No sales data available",
                "record_count": 0,
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "data_file": None
            }
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"sales_{timestamp}.json"
        file_path = data_dir / filename
        
        # Save data to file
        with open(file_path, 'w') as f:
            json.dump(sales_data, f, indent=2)
        
        logger.info(f"Saved {len(sales_data)} sales records to {file_path}")
        
        # Return job status
        return {
            "job_id": job_id,
            "status": "success",
            "record_count": len(sales_data),
            "start_time": start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "data_file": str(file_path)
        }
        
    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        logger.error(f"Error in hourly sales data pull: {error_msg}")
        logger.debug(error_trace)
        
        # Log error details to file
        error_log_path = log_dir / f"error_{job_id}.log"
        with open(error_log_path, 'w') as f:
            f.write(f"Error: {error_msg}\n\nTraceback:\n{error_trace}")
        
        # Return error status
        return {
            "job_id": job_id,
            "status": "error",
            "message": error_msg,
            "start_time": start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "error_log": str(error_log_path)
        }


def daily_push_insights_job(crm_adapter: BaseCRMAdapter, insights_dir: Path, log_dir: Path) -> Dict[str, Any]:
    """
    Push processed insights to CRM.
    
    This job runs daily to push processed insights back to the CRM system.
    It reads the latest insights from data/processed/latest.json
    
    Args:
        crm_adapter: The CRM adapter to use
        insights_dir: Directory with insight files
        log_dir: Directory to save logs
        
    Returns:
        Dict with job status information
    """
    start_time = datetime.now()
    job_id = f"insights_push_{start_time.strftime('%Y%m%d%H%M%S')}"
    
    logger.info(f"Starting daily insights push job: {job_id}")
    
    try:
        # Check for insights file
        insights_file = insights_dir / "latest.json"
        
        if not insights_file.exists():
            logger.warning(f"Insights file not found: {insights_file}")
            return {
                "job_id": job_id,
                "status": "skipped",
                "message": f"Insights file not found: {insights_file}",
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat()
            }
        
        # Load insights data
        with open(insights_file, 'r') as f:
            insights_data = json.load(f)
        
        if not insights_data:
            logger.info("No insights data to push")
            return {
                "job_id": job_id,
                "status": "success",
                "message": "No insights data to push",
                "record_count": 0,
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat()
            }
        
        # Authenticate with CRM
        crm_adapter.authenticate()
        
        # Push insights to CRM
        crm_adapter.push_insights(insights_data)
        
        logger.info(f"Successfully pushed {len(insights_data)} insights to CRM")
        
        # Archive the insights file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        archive_path = insights_dir / f"archive/insights_{timestamp}.json"
        archive_path.parent.mkdir(exist_ok=True)
        
        # Copy insights to archive
        import shutil
        shutil.copy2(insights_file, archive_path)
        
        # Return job status
        return {
            "job_id": job_id,
            "status": "success",
            "record_count": len(insights_data),
            "start_time": start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "insights_file": str(insights_file),
            "archive_file": str(archive_path)
        }
            
    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        logger.error(f"Error in daily insights push: {error_msg}")
        logger.debug(error_trace)
        
        # Log error details to file
        error_log_path = log_dir / f"error_{job_id}.log"
        with open(error_log_path, 'w') as f:
            f.write(f"Error: {error_msg}\n\nTraceback:\n{error_trace}")
        
        # Return error status
        return {
            "job_id": job_id,
            "status": "error",
            "message": error_msg,
            "start_time": start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "error_log": str(error_log_path)
        }


class CRMSyncScheduler:
    """
    Scheduler for CRM data synchronization tasks.
    
    This class manages periodic CRM data synchronization jobs using APScheduler,
    including hourly pulls of sales data and daily pushes of insights.
    """
    
    def __init__(self, 
                data_dir: Optional[Union[str, Path]] = None,
                insights_dir: Optional[Union[str, Path]] = None,
                log_dir: Optional[Union[str, Path]] = None,
                db_path: Optional[Union[str, Path]] = None,
                sales_sync_interval: int = 60,  # minutes
                insights_sync_cron: str = "0 2 * * *",  # 2:00 AM daily
                crm_adapter: Optional[BaseCRMAdapter] = None):
        """
        Initialize the CRM Sync Scheduler.
        
        Args:
            data_dir: Directory for storing raw CRM data. Default data/raw/
            insights_dir: Directory for processed insights. Default data/processed/
            log_dir: Directory for logs. Default data/logs/
            db_path: Path to SQLite DB for job persistence. Default data/scheduler.sqlite
            sales_sync_interval: Interval in minutes for sales data sync. Default 60 (hourly)
            insights_sync_cron: Cron expression for insights sync. Default "0 2 * * *" (2:00 AM daily)
            crm_adapter: CRM adapter to use. Default is NovaActAdapter with env vars
        """
        # Set up paths from arguments or environment variables
        self.data_dir = Path(data_dir or os.environ.get(ENV_DATA_DIR, DEFAULT_DATA_DIR))
        self.insights_dir = Path(insights_dir or os.environ.get(ENV_INSIGHTS_DIR, DEFAULT_INSIGHTS_DIR))
        self.log_dir = Path(log_dir or os.environ.get(ENV_LOG_DIR, DEFAULT_LOG_DIR))
        self.db_path = Path(db_path or os.environ.get(ENV_DB_PATH, DEFAULT_DB_PATH))
        
        # Create directories if they don't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.insights_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure job schedules
        self.sales_sync_interval = int(os.environ.get(ENV_SALES_INTERVAL, sales_sync_interval))
        self.insights_sync_cron = os.environ.get(ENV_INSIGHTS_CRON, insights_sync_cron)
        
        # Set up the CRM adapter
        self.crm_adapter = crm_adapter or NovaActAdapter()
        
        # Initialize scheduler with SQLite job store
        self.scheduler = BackgroundScheduler()
        self.scheduler.add_jobstore(
            SQLAlchemyJobStore(url=f'sqlite:///{self.db_path}'),
            'default'
        )
        
        # Track if scheduler is running
        self.is_running = False
        
        logger.info(f"Initialized CRM Sync Scheduler with data_dir={self.data_dir}, "
                  f"sales_interval={self.sales_sync_interval}min, "
                  f"insights_cron='{self.insights_sync_cron}'")
    
    def start(self):
        """
        Start the scheduler and register jobs.
        """
        if self.is_running:
            logger.warning("Scheduler is already running")
            return
        
        # Register jobs
        self._register_jobs()
        
        # Start the scheduler
        self.scheduler.start()
        self.is_running = True
        
        logger.info("CRM Sync Scheduler started")
    
    def stop(self):
        """
        Stop the scheduler.
        """
        if not self.is_running:
            logger.warning("Scheduler is not running")
            return
        
        # Shutdown the scheduler
        self.scheduler.shutdown()
        self.is_running = False
        
        logger.info("CRM Sync Scheduler stopped")
    
    def _register_jobs(self):
        """
        Register all scheduled jobs.
        """
        # Sales data sync job - runs hourly by default
        self.scheduler.add_job(
            func=hourly_pull_sales_job,
            args=[self.crm_adapter, self.data_dir, self.log_dir],
            trigger=IntervalTrigger(minutes=self.sales_sync_interval),
            id='sales_sync',
            name='Sales Data Sync',
            replace_existing=True
        )
        
        # Insights sync job - runs at 2:00 AM daily by default
        self.scheduler.add_job(
            func=daily_push_insights_job,
            args=[self.crm_adapter, self.insights_dir, self.log_dir],
            trigger=CronTrigger.from_crontab(self.insights_sync_cron),
            id='insights_sync',
            name='Insights Sync',
            replace_existing=True
        )
        
        logger.info(f"Registered scheduled jobs: sales_sync (every {self.sales_sync_interval}min), "
                  f"insights_sync (at {self.insights_sync_cron})")
    
    def run_now(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Run a specific job immediately.
        
        Args:
            job_id: ID of the job to run ('sales_sync' or 'insights_sync')
            
        Returns:
            Dict with job status or None if job not found
        """
        if job_id == 'sales_sync':
            return hourly_pull_sales_job(self.crm_adapter, self.data_dir, self.log_dir)
        elif job_id == 'insights_sync':
            return daily_push_insights_job(self.crm_adapter, self.insights_dir, self.log_dir)
        else:
            logger.error(f"Unknown job ID: {job_id}")
            return None
    
    def list_jobs(self) -> List[Dict[str, Any]]:
        """
        List all scheduled jobs.
        
        Returns:
            List of job information dictionaries
        """
        jobs = []
        for job in self.scheduler.get_jobs():
            next_run = job.next_run_time.isoformat() if job.next_run_time else None
            jobs.append({
                "id": job.id,
                "name": job.name,
                "next_run": next_run,
                "trigger": str(job

"""
CRM Sync Scheduler

This module provides scheduling functionality for CRM data synchronization tasks,
including hourly sales data pulls and daily insight pushes.
"""

import os
import sys
import json
import logging
import signal
import time
import traceback
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable

# APScheduler for job scheduling
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger

# Import our CRM adapters
from watchdog_ai.integrations.crm import NovaActAdapter, BaseCRMAdapter

# Configure logging
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_DATA_DIR = Path("data/raw")
DEFAULT_INSIGHTS_DIR = Path("data/processed")
DEFAULT_LOG_DIR = Path("data/logs")
DEFAULT_DB_PATH = Path("data/scheduler.sqlite")

# Environment variable names
ENV_DATA_DIR = "WATCHDOG_DATA_DIR"
ENV_INSIGHTS_DIR = "WATCHDOG_INSIGHTS_DIR"
ENV_LOG_DIR = "WATCHDOG_LOG_DIR"
ENV_DB_PATH = "WATCHDOG_SCHEDULER_DB"
ENV_SALES_INTERVAL = "WATCHDOG_SALES_SYNC_INTERVAL"
ENV_INSIGHTS_CRON = "WATCHDOG_INSIGHTS_SYNC_CRON"


# Standalone job functions for serialization
def hourly_pull_sales_job(crm_adapter: BaseCRMAdapter, data_dir: Path, log_dir: Path) -> Dict[str, Any]:
    """
    Pull sales data from CRM and save to data/raw directory.
    
    This job runs hourly to fetch the latest sales data.
    
    Args:
        crm_adapter: The CRM adapter to use
        data_dir: Directory to save data
        log_dir: Directory to save logs
        
    Returns:
        Dict with job status information
    """
    start_time = datetime.now()
    job_id = f"sales_pull_{start_time.strftime('%Y%m%d%H%M%S')}"
    
    logger.info(f"Starting hourly sales data pull job: {job_id}")
    
    try:
        # Authenticate with CRM
        crm_adapter.authenticate()
        
        # Pull sales data since last run
        # In a more sophisticated implementation, we'd track the last sync time
        # and only pull data since then, but for simplicity, we're pulling all data
        sales_data = crm_adapter.pull_sales()
        
        if not sales_data:
            logger.info("No sales data retrieved")
            return {
                "job_id": job_id,
                "status": "success",
                "message": "No sales data available",
                "record_count": 0,
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "data_file": None
            }
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"sales_{timestamp}.json"
        file_path = data_dir / filename
        
        # Save data to file
        with open(file_path, 'w') as f:
            json.dump(sales_data, f, indent=2)
        
        logger.info(f"Saved {len(sales_data)} sales records to {file_path}")
        
        # Return job status
        return {
            "job_id": job_id,
            "status": "success",
            "record_count": len(sales_data),
            "start_time": start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "data_file": str(file_path)
        }
        
    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        logger.error(f"Error in hourly sales data pull: {error_msg}")
        logger.debug(error_trace)
        
        # Log error details to file
        error_log_path = log_dir / f"error_{job_id}.log"
        with open(error_log_path, 'w') as f:
            f.write(f"Error: {error_msg}\n\nTraceback:\n{error_trace}")
        
        # Return error status
        return {
            "job_id": job_id,
            "status": "error",
            "message": error_msg,
            "start_time": start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "error_log": str(error_log_path)
        }


def daily_push_insights_job(crm_adapter: BaseCRMAdapter, insights_dir: Path, log_dir: Path) -> Dict[str, Any]:
    """
    Push processed insights to CRM.
    
    This job runs daily to push processed insights back to the CRM system.
    It reads the latest insights from data/processed/latest.json
    
    Args:
        crm_adapter: The CRM adapter to use
        insights_dir: Directory with insight files
        log_dir: Directory to save logs
        
    Returns:
        Dict with job status information
    """
    start_time = datetime.now()
    job_id = f"insights_push_{start_time.strftime('%Y%m%d%H%M%S')}"
    
    logger.info(f"Starting daily insights push job: {job_id}")
    
    try:
        # Check for insights file
        insights_file = insights_dir / "latest.json"
        
        if not insights_file.exists():
            logger.warning(f"Insights file not found: {insights_file}")
            return {
                "job_id": job_id,
                "status": "skipped",
                "message": f"Insights file not found: {insights_file}",
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat()
            }
        
        # Load insights data
        with open(insights_file, 'r') as f:
            insights_data = json.load(f)
        
        if not insights_data:
            logger.info("No insights data to push")
            return {
                "job_id": job_id,
                "status": "success",
                "message": "No insights data to push",
                "record_count": 0,
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat()
            }
        
        # Authenticate with CRM
        crm_adapter.authenticate()
        
        # Push insights to CRM
        crm_adapter.push_insights(insights_data)
        
        logger.info(f"Successfully pushed {len(insights_data)} insights to CRM")
        
        # Archive the insights file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        archive_path = insights_dir / f"archive/insights_{timestamp}.json"
        archive_path.parent.mkdir(exist_ok=True)
        
        # Copy insights to archive
        import shutil
        shutil.copy2(insights_file, archive_path)
        
        # Return job status
        return {
            "job_id": job_id,
            "status": "success",
            "record_count": len(insights_data),
            "start_time": start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "insights_file": str(insights_file),
            "archive_file": str(archive_path)
        }
            
    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        logger.error(f"Error in daily insights push: {error_msg}")
        logger.debug(error_trace)
        
        # Log error details to file
        error_log_path = log_dir / f"error_{job_id}.log"
        with open(error_log_path, 'w') as f:
            f.write(f"Error: {error_msg}\n\nTraceback:\n{error_trace}")
        
        # Return error status
        return {
            "job_id": job_id,
            "status": "error",
            "message": error_msg,
            "start_time": start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "error_log": str(error_log_path)
        }


class CRMSyncScheduler:
    """
    Scheduler for CRM data synchronization tasks.
    
    This class manages periodic CRM data synchronization jobs using APScheduler,
    including hourly pulls of sales data and daily pushes of insights.
    """
    
    def __init__(self, 
                data_dir: Optional[Union[str, Path]] = None,
                insights_dir: Optional[Union[str, Path]] = None,
                log_dir: Optional[Union[str, Path]] = None,
                db_path: Optional[Union[str, Path]] = None,
                sales_sync_interval: int = 60,  # minutes
                insights_sync_cron: str = "0 2 * * *",  # 2:00 AM daily
                crm_adapter: Optional[BaseCRMAdapter] = None):
        """
        Initialize the CRM Sync Scheduler.
        
        Args:
            data_dir: Directory for storing raw CRM data. Default data/raw/
            insights_dir: Directory for processed insights. Default data/processed/
            log_dir: Directory for logs. Default data/logs/
            db_path: Path to SQLite DB for job persistence. Default data/scheduler.sqlite
            sales_sync_interval: Interval in minutes for sales data sync. Default 60 (hourly)
            insights_sync_cron: Cron expression for insights sync. Default "0 2 * * *" (2:00 AM daily)
            crm_adapter: CRM adapter to use. Default is NovaActAdapter with env vars
        """
        # Set up paths from arguments or environment variables
        self.data_dir = Path(data_dir or os.environ.get(ENV_DATA_DIR, DEFAULT_DATA_DIR))
        self.insights_dir = Path(insights_dir or os.environ.get(ENV_INSIGHTS_DIR, DEFAULT_INSIGHTS_DIR))
        self.log_dir = Path(log_dir or os.environ.get(ENV_LOG_DIR, DEFAULT_LOG_DIR))
        self.db_path = Path(db_path or os.environ.get(ENV_DB_PATH, DEFAULT_DB_PATH))
        
        # Create directories if they don't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.insights_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure job schedules
        self.sales_sync_interval = int(os.environ.get(ENV_SALES_INTERVAL, sales_sync_interval))
        self.insights_sync_cron = os.environ.get(ENV_INSIGHTS_CRON, insights_sync_cron)
        
        # Set up the CRM adapter
        self.crm_adapter = crm_adapter or NovaActAdapter()
        
        # Initialize scheduler with SQLite job store
        self.scheduler = BackgroundScheduler()
        self.scheduler.add_jobstore(
            SQLAlchemyJobStore(url=f'sqlite:///{self.db_path}'),
            'default'
        )
        
        # Track if scheduler is running
        self.is_running = False
        
        logger.info(f"Initialized CRM Sync Scheduler with data_dir={self.data_dir}, "
                  f"sales_interval={self.sales_sync_interval}min, "
                  f"insights_cron='{self.insights_sync_cron}'")
    
    def start(self):
        """
        Start the scheduler and register jobs.
        """
        if self.is_running:
            logger.warning("Scheduler is already running")
            return
        
        # Register jobs
        self._register_jobs()
        
        # Start the scheduler
        self.scheduler.start()
        self.is_running = True
        
        logger.info("CRM Sync Scheduler started")
    
    def stop(self):
        """
        Stop the scheduler.
        """
        if not self.is_running:
            logger.warning("Scheduler is not running")
            return
        
        # Shutdown the scheduler
        self.scheduler.shutdown()
        self.is_running = False
        
        logger.info("CRM Sync Scheduler stopped")
    
    def _register_jobs(self):
        """
        Register all scheduled jobs.
        """
        # Sales data sync job - runs hourly by default
        self.scheduler.add_job(
            func=hourly_pull_sales_job,
            args=[self.crm_adapter, self.data_dir, self.log_dir],
            trigger=IntervalTrigger(minutes=self.sales_sync_interval),
            id='sales_sync',
            name='Sales Data Sync',
            replace_existing=True
        )
        
        # Insights sync job - runs at 2:00 AM daily by default
        self.scheduler.add_job(
            func=daily_push_insights_job,
            args=[self.crm_adapter, self.insights_dir, self.log_dir],
            trigger=CronTrigger.from_crontab(self.insights_sync_cron),
            id='insights_sync',
            name='Insights Sync',
            replace_existing=True
        )
        
        logger.info(f"Registered scheduled jobs: sales_sync (every {self.sales_sync_interval}min), "
                  f"insights_sync (at {self.insights_sync_cron})")
    
    def run_now(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Run a specific job immediately.
        
        Args:
            job_id: ID of the job to run ('sales_sync' or 'insights_sync')
            
        Returns:
            Dict with job status or None if job not found
        """
        if job_id == 'sales_sync':
            return hourly_pull_sales_job(self.crm_adapter, self.data_dir, self.log_dir)
        elif job_id == 'insights_sync':
            return daily_push_insights_job(self.crm_adapter, self.insights_dir, self.log_dir)
        else:
            logger.error(f"Unknown job ID: {job_id}")
            return None
    
    def list_jobs(self) -> List[Dict[str, Any]]:
        """
        List all scheduled jobs.
        
        Returns:
            List of job information dictionaries
        """
        jobs = []
        for job in self.scheduler.get_jobs():
            next_run = job.next_run_time.isoformat() if job.next_run_time else None
            jobs.append({
                "id": job.id,
                "name": job.name,
                "next_run":

"""
CRM Sync Scheduler

This module provides scheduling functionality for CRM data synchronization tasks,
including hourly sales data pulls and daily insight pushes.
"""

import os
import sys
import json
import logging
import signal
import time
import traceback
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable

# APScheduler for job scheduling
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger

# Import our CRM adapters
from watchdog_ai.integrations.crm import NovaActAdapter, BaseCRMAdapter

# Configure logging
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_DATA_DIR = Path("data/raw")
DEFAULT_INSIGHTS_DIR = Path("data/processed")
DEFAULT_LOG_DIR = Path("data/logs")
DEFAULT_DB_PATH = Path("data/scheduler.sqlite")

# Environment variable names
ENV_DATA_DIR = "WATCHDOG_DATA_DIR"
ENV_INSIGHTS_DIR = "WATCHDOG_INSIGHTS_DIR"
ENV_LOG_DIR = "WATCHDOG_LOG_DIR"
ENV_DB_PATH = "WATCHDOG_SCHEDULER_DB"
ENV_SALES_INTERVAL = "WATCHDOG_SALES_SYNC_INTERVAL"
ENV_INSIGHTS_CRON = "WATCHDOG_INSIGHTS_SYNC_CRON"


# Standalone job functions for serialization
def hourly_pull_sales_job(crm_adapter: BaseCRMAdapter, data_dir: Path, log_dir: Path) -> Dict[str, Any]:
    """
    Pull sales data from CRM and save to data/raw directory.
    
    This job runs hourly to fetch the latest sales data.
    
    Args:
        crm_adapter: The CRM adapter to use
        data_dir: Directory to save data
        log_dir: Directory to save logs
        
    Returns:
        Dict with job status information
    """
    start_time = datetime.now()
    job_id = f"sales_pull_{start_time.strftime('%Y%m%d%H%M%S')}"
    
    logger.info(f"Starting hourly sales data pull job: {job_id}")
    
    try:
        # Authenticate with CRM
        crm_adapter.authenticate()
        
        # Pull sales data since last run
        # In a more sophisticated implementation, we'd track the last sync time
        # and only pull data since then, but for simplicity, we're pulling all data
        sales_data = crm_adapter.pull_sales()
        
        if not sales_data:
            logger.info("No sales data retrieved")
            return {
                "job_id": job_id,
                "status": "success",
                "message": "No sales data available",
                "record_count": 0,
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "data_file": None
            }
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"sales_{timestamp}.json"
        file_path = data_dir / filename
        
        # Save data to file
        with open(file_path, 'w') as f:
            json.dump(sales_data, f, indent=2)
        
        logger.info(f"Saved {len(sales_data)} sales records to {file_path}")
        
        # Return job status
        return {
            "job_id": job_id,
            "status": "success",
            "record_count": len(sales_data),
            "start_time": start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "data_file": str(file_path)
        }
        
    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        logger.error(f"Error in hourly sales data pull: {error_msg}")
        logger.debug(error_trace)
        
        # Log error details to file
        error_log_path = log_dir / f"error_{job_id}.log"
        with open(error_log_path, 'w') as f:
            f.write(f"Error: {error_msg}\n\nTraceback:\n{error_trace}")
        
        # Return error status
        return {
            "job_id": job_id,
            "status": "error",
            "message": error_msg,
            "start_time": start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "error_log": str(error_log_path)
        }


def daily_push_insights_job(crm_adapter: BaseCRMAdapter, insights_dir: Path, log_dir: Path) -> Dict[str, Any]:
    """
    Push processed insights to CRM.
    
    This job runs daily to push processed insights back to the CRM system.
    It reads the latest insights from data/processed/latest.json
    
    Args:
        crm_adapter: The CRM adapter to use
        insights_dir: Directory with insight files
        log_dir: Directory to save logs
        
    Returns:
        Dict with job status information
    """
    start_time = datetime.now()
    job_id = f"insights_push_{start_time.strftime('%Y%m%d%H%M%S')}"
    
    logger.info(f"Starting daily insights push job: {job_id}")
    
    try:
        # Check for insights file
        insights_file = insights_dir / "latest.json"
        
        if not insights_file.exists():
            logger.warning(f"Insights file not found: {insights_file}")
            return {
                "job_id": job_id,
                "status": "skipped",
                "message": f"Insights file not found: {insights_file}",
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat()
            }
        
        # Load insights data
        with open(insights_file, 'r') as f:
            insights_data = json.load(f)
        
        if not insights_data:
            logger.info("No insights data to push")
            return {
                "job_id": job_id,
                "status": "success",
                "message": "No insights data to push",
                "record_count": 0,
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat()
            }
        
        # Authenticate with CRM
        crm_adapter.authenticate()
        
        # Push insights to CRM
        crm_adapter.push_insights(insights_data)
        
        logger.info(f"Successfully pushed {len(insights_data)} insights to CRM")
        
        # Archive the insights file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        archive_path = insights_dir / f"archive/insights_{timestamp}.json"
        archive_path.parent.mkdir(exist_ok=True)
        
        # Copy insights to archive
        import shutil
        shutil.copy2(insights_file, archive_path)
        
        # Return job status
        return {
            "job_id": job_id,
            "status": "success",
            "record_count": len(insights_data),
            "start_time": start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "insights_file": str(insights_file),
            "archive_file": str(archive_path)
        }
            
    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        logger.error(f"Error in daily insights push: {error_msg}")
        logger.debug(error_trace)
        
        # Log error details to file
        error_log_path = log_dir / f"error_{job_id}.log"
        with open(error_log_path, 'w') as f:
            f.write(f"Error: {error_msg}\n\nTraceback:\n{error_trace}")
        
        # Return error status
        return {
            "job_id": job_id,
            "status": "error",
            "message": error_msg,
            "start_time": start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "error_log": str(error_log_path)
        }


class CRMSyncScheduler:
    """
    Scheduler for CRM data synchronization tasks.
    
    This class manages periodic CRM data synchronization jobs using APScheduler,
    including hourly pulls of sales data and daily pushes of insights.
    """
    
    def __init__(self, 
                data_dir: Optional[Union[str, Path]] = None,
                insights_dir: Optional[Union[str, Path]] = None,
                log_dir: Optional[Union[str, Path]] = None,
                db_path: Optional[Union[str, Path]] = None,
                sales_sync_interval: int = 60,  # minutes
                insights_sync_cron: str = "0 2 * * *",  # 2:00 AM daily
                crm_adapter: Optional[BaseCRMAdapter] = None):
        """
        Initialize the CRM Sync Scheduler.
        
        Args:
            data_dir: Directory for storing raw CRM data. Default data/raw/
            insights_dir: Directory for processed insights. Default data/processed/
            log_dir: Directory for logs. Default data/logs/
            db_path: Path to SQLite DB for job persistence. Default data/scheduler.sqlite
            sales_sync_interval: Interval in minutes for sales data sync. Default 60 (hourly)
            insights_sync_cron: Cron expression for insights sync. Default "0 2 * * *" (2:00 AM daily)
            crm_adapter: CRM adapter to use. Default is NovaActAdapter with env vars
        """
        # Set up paths from arguments or environment variables
        self.data_dir = Path(data_dir or os.environ.get(ENV_DATA_DIR, DEFAULT_DATA_DIR))
        self.insights_dir = Path(insights_dir or os.environ.get(ENV_INSIGHTS_DIR, DEFAULT_INSIGHTS_DIR))
        self.log_dir = Path(log_dir or os.environ.get(ENV_LOG_DIR, DEFAULT_LOG_DIR))
        self.db_path = Path(db_path or os.environ.get(ENV_DB_PATH, DEFAULT_DB_PATH))
        
        # Create directories if they don't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.insights_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure job schedules
        self.sales_sync_interval = int(os.environ.get(ENV_SALES_INTERVAL, sales_sync_interval))
        self.insights_sync_cron = os.environ.get(ENV_INSIGHTS_CRON, insights_sync_cron)
        
        # Set up the CRM adapter
        self.crm_adapter = crm_adapter or NovaActAdapter()
        
        # Initialize scheduler with SQLite job store
        self.scheduler = BackgroundScheduler()
        self.scheduler.add_jobstore(
            SQLAlchemyJobStore(url=f'sqlite:///{self.db_path}'),
            'default'
        )
        
        # Track if scheduler is running
        self.is_running = False
        
        logger.info(f"Initialized CRM Sync Scheduler with data_dir={self.data_dir}, "
                  f"sales_interval={self.sales_sync_interval}min, "
                  f"insights_cron='{self.insights_sync_cron}'")
    
    def start(self):
        """
        Start the scheduler and register jobs.
        """
        if self.is_running:
            logger.warning("Scheduler is already running")
            return
        
        # Register jobs
        self._register_jobs()
        
        # Start the scheduler
        self.scheduler.start()
        self.is_running = True
        
        logger.info("CRM Sync Scheduler started")
    
    def stop(self):
        """
        Stop the scheduler.
        """
        if not self.is_running:
            logger.warning("Scheduler is not running")
            return
        
        # Shutdown the scheduler
        self.scheduler.shutdown()
        self.is_running = False
        
        logger.info("CRM Sync Scheduler stopped")
    
    def _register_jobs(self):
        """
        Register all scheduled jobs.
        """
        # Sales data sync job - runs hourly by default
        self.scheduler.add_job(
            func=hourly_pull_sales_job,
            args=[self.crm_adapter, self.data_dir, self.log_dir],
            trigger=IntervalTrigger(minutes=self.sales_sync_interval),
            id='sales_sync',
            name='Sales Data Sync',
            replace_existing=True
        )
        
        # Insights sync job - runs at 2:00 AM daily by default
        self.scheduler.add_job(
            func=daily_push_insights_job,
            args=[self.crm_adapter, self.insights_dir, self.log_dir],
            trigger=CronTrigger.from_crontab(self.insights_sync_cron),
            id='insights_sync',
            name='Insights Sync',
            replace_existing=True
        )
        
        logger.info(f"Registered scheduled jobs: sales_sync (every {self.sales_sync_interval}min), "
                  f"insights_sync (at {self.insights_sync_cron})")
    
    def run_now(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Run a specific job immediately.
        
        Args:
            job_id: ID of the job to run ('sales_sync' or 'insights_sync')
            
        Returns:
            Dict with job status or None if job not found
        """
        if job_id == 'sales_sync':
            return hourly_pull_sales_job(self.crm_adapter, self.data_dir, self.log_dir)
        elif job_id == 'insights_sync':
            return daily_push_insights_job(self.crm_adapter, self.insights_dir, self.log_dir)
        else:
            logger.error(f"Unknown job ID: {job_id}")
            return None
    
    def list_jobs(self) -> List[Dict[str, Any]]:
        """
        List all scheduled jobs.
        
        Returns:
            List of job information dictionaries
        """
        jobs = []
        for job in self.scheduler.get_jobs():
            next_run = job.next_run_time.isoformat() if job.next_run_time else None
            jobs.append({
                

"""
CRM Sync Scheduler

This module provides scheduling functionality for CRM data synchronization tasks,
including hourly sales data pulls and daily insight pushes.
"""

import os
import sys
import json
import logging
import signal
import time
import traceback
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable

# APScheduler for job scheduling
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger

# Import our CRM adapters
from watchdog_ai.integrations.crm import NovaActAdapter, BaseCRMAdapter

# Configure logging
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_DATA_DIR = Path("data/raw")
DEFAULT_INSIGHTS_DIR = Path("data/processed")
DEFAULT_LOG_DIR = Path("data/logs")
DEFAULT_DB_PATH = Path("data/scheduler.sqlite")

# Environment variable names
ENV_DATA_DIR = "WATCHDOG_DATA_DIR"
ENV_INSIGHTS_DIR = "WATCHDOG_INSIGHTS_DIR"
ENV_LOG_DIR = "WATCHDOG_LOG_DIR"
ENV_DB_PATH = "WATCHDOG_SCHEDULER_DB"
ENV_SALES_INTERVAL = "WATCHDOG_SALES_SYNC_INTERVAL"
ENV_INSIGHTS_CRON = "WATCHDOG_INSIGHTS_SYNC_CRON"


# Standalone job functions for serialization
def hourly_pull_sales_job(crm_adapter: BaseCRMAdapter, data_dir: Path, log_dir: Path) -> Dict[str, Any]:
    """
    Pull sales data from CRM and save to data/raw directory.
    
    This job runs hourly to fetch the latest sales data.
    
    Args:
        crm_adapter: The CRM adapter to use
        data_dir: Directory to save data
        log_dir: Directory to save logs
        
    Returns:
        Dict with job status information
    """
    start_time = datetime.now()
    job_id = f"sales_pull_{start_time.strftime('%Y%m%d%H%M%S')}"
    
    logger.info(f"Starting hourly sales data pull job: {job_id}")
    
    try:
        # Authenticate with CRM
        crm_adapter.authenticate()
        
        # Pull sales data since last run
        # In a more sophisticated implementation, we'd track the last sync time
        # and only pull data since then, but for simplicity, we're pulling all data
        sales_data = crm_adapter.pull_sales()
        
        if not sales_data:
            logger.info("No sales data retrieved")
            return {
                "job_id": job_id,
                "status": "success",
                "message": "No sales data available",
                "record_count": 0,
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "data_file": None
            }
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"sales_{timestamp}.json"
        file_path = data_dir / filename
        
        # Save data to file
        with open(file_path, 'w') as f:
            json.dump(sales_data, f, indent=2)
        
        logger.info(f"Saved {len(sales_data)} sales records to {file_path}")
        
        # Return job status
        return {
            "job_id": job_id,
            "status": "success",
            "record_count": len(sales_data),
            "start_time": start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "data_file": str(file_path)
        }
        
    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        logger.error(f"Error in hourly sales data pull: {error_msg}")
        logger.debug(error_trace)
        
        # Log error details to file
        error_log_path = log_dir / f"error_{job_id}.log"
        with open(error_log_path, 'w') as f:
            f.write(f"Error: {error_msg}\n\nTraceback:\n{error_trace}")
        
        # Return error status
        return {
            "job_id": job_id,
            "status": "error",
            "message": error_msg,
            "start_time": start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "error_log": str(error_log_path)
        }


def daily_push_insights_job(crm_adapter: BaseCRMAdapter, insights_dir: Path, log_dir: Path) -> Dict[str, Any]:
    """
    Push processed insights to CRM.
    
    This job runs daily to push processed insights back to the CRM system.
    It reads the latest insights from data/processed/latest.json
    
    Args:
        crm_adapter: The CRM adapter to use
        insights_dir: Directory with insight files
        log_dir: Directory to save logs
        
    Returns:
        Dict with job status information
    """
    start_time = datetime.now()
    job_id = f"insights_push_{start_time.strftime('%Y%m%d%H%M%S')}"
    
    logger.info(f"Starting daily insights push job: {job_id}")
    
    try:
        # Check for insights file
        insights_file = insights_dir / "latest.json"
        
        if not insights_file.exists():
            logger.warning(f"Insights file not found: {insights_file}")
            return {
                "job_id": job_id,
                "status": "skipped",
                "message": f"Insights file not found: {insights_file}",
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat()
            }
        
        # Load insights data
        with open(insights_file, 'r') as f:
            insights_data = json.load(f)
        
        if not insights_data:
            logger.info("No insights data to push")
            return {
                "job_id": job_id,
                "status": "success",
                "message": "No insights data to push",
                "record_count": 0,
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat()
            }
        
        # Authenticate with CRM
        crm_adapter.authenticate()
        
        # Push insights to CRM
        crm_adapter.push_insights(insights_data)
        
        logger.info(f"Successfully pushed {len(insights_data)} insights to CRM")
        
        # Archive the insights file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        archive_path = insights_dir / f"archive/insights_{timestamp}.json"
        archive_path.parent.mkdir(exist_ok=True)
        
        # Copy insights to archive
        import shutil
        shutil.copy2(insights_file, archive_path)
        
        # Return job status
        return {
            "job_id": job_id,
            "status": "success",
            "record_count": len(insights_data),
            "start_time": start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "insights_file": str(insights_file),
            "archive_file": str(archive_path)
        }
            
    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        logger.error(f"Error in daily insights push: {error_msg}")
        logger.debug(error_trace)
        
        # Log error details to file
        error_log_path = log_dir / f"error_{job_id}.log"
        with open(error_log_path, 'w') as f:
            f.write(f"Error: {error_msg}\n\nTraceback:\n{error_trace}")
        
        # Return error status
        return {
            "job_id": job_id,
            "status": "error",
            "message": error_msg,
            "start_time": start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "error_log": str(error_log_path)
        }


class CRMSyncScheduler:
    """
    Scheduler for CRM data synchronization tasks.
    
    This class manages periodic CRM data synchronization jobs using APScheduler,
    including hourly pulls of sales data and daily pushes of insights.
    """
    
    def __init__(self, 
                data_dir: Optional[Union[str, Path]] = None,
                insights_dir: Optional[Union[str, Path]] = None,
                log_dir: Optional[Union[str, Path]] = None,
                db_path: Optional[Union[str, Path]] = None,
                sales_sync_interval: int = 60,  # minutes
                insights_sync_cron: str = "0 2 * * *",  # 2:00 AM daily
                crm_adapter: Optional[BaseCRMAdapter] = None):
        """
        Initialize the CRM Sync Scheduler.
        
        Args:
            data_dir: Directory for storing raw CRM data. Default data/raw/
            insights_dir: Directory for processed insights. Default data/processed/
            log_dir: Directory for logs. Default data/logs/
            db_path: Path to SQLite DB for job persistence. Default data/scheduler.sqlite
            sales_sync_interval: Interval in minutes for sales data sync. Default 60 (hourly)
            insights_sync_cron: Cron expression for insights sync. Default "0 2 * * *" (2:00 AM daily)
            crm_adapter: CRM adapter to use. Default is NovaActAdapter with env vars
        """
        # Set up paths from arguments or environment variables
        self.data_dir = Path(data_dir or os.environ.get(ENV_DATA_DIR, DEFAULT_DATA_DIR))
        self.insights_dir = Path(insights_dir or os.environ.get(ENV_INSIGHTS_DIR, DEFAULT_INSIGHTS_DIR))
        self.log_dir = Path(log_dir or os.environ.get(ENV_LOG_DIR, DEFAULT_LOG_DIR))
        self.db_path = Path(db_path or os.environ.get(ENV_DB_PATH, DEFAULT_DB_PATH))
        
        # Create directories if they don't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.insights_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure job schedules
        self.sales_sync_interval = int(os.environ.get(ENV_SALES_INTERVAL, sales_sync_interval))
        self.insights_sync_cron = os.environ.get(ENV_INSIGHTS_CRON, insights_sync_cron)
        
        # Set up the CRM adapter
        self.crm_adapter = crm_adapter or NovaActAdapter()
        
        # Initialize scheduler with SQLite job store
        self.scheduler = BackgroundScheduler()
        self.scheduler.add_jobstore(
            SQLAlchemyJobStore(url=f'sqlite:///{self.db_path}'),
            'default'
        )
        
        # Track if scheduler is running
        self.is_running = False
        
        logger.info(f"Initialized CRM Sync Scheduler with data_dir={self.data_dir}, "
                  f"sales_interval={self.sales_sync_interval}min, "
                  f"insights_cron='{self.insights_sync_cron}'")
    
    def start(self):
        """
        Start the scheduler and register jobs.
        """
        if self.is_running:
            logger.warning("Scheduler is already running")
            return
        
        # Register jobs
        self._register_jobs()
        
        # Start the scheduler
        self.scheduler.start()
        self.is_running = True
        
        logger.info("CRM Sync Scheduler started")
    
    def stop(self):
        """
        Stop the scheduler.
        """
        if not self.is_running:
            logger.warning("Scheduler is not running")
            return
        
        # Shutdown the scheduler
        self.scheduler.shutdown()
        self.is_running = False
        
        logger.info("CRM Sync Scheduler stopped")
    
    def _register_jobs(self):
        """
        Register all scheduled jobs.
        """
        # Sales data sync job - runs hourly by default
        self.scheduler.add_job(
            func=hourly_pull_sales_job,
            args=[self.crm_adapter, self.data_dir, self.log_dir],
            trigger=IntervalTrigger(minutes=self.sales_sync_interval),
            id='sales_sync',
            name='Sales Data Sync',
            replace_existing=True
        )
        
        # Insights sync job - runs at 2:00 AM daily by default
        self.scheduler.add_job(
            func=daily_push_insights_job,
            args=[self.crm_adapter, self.insights_dir, self.log_dir],
            trigger=CronTrigger.from_crontab(self.insights_sync_cron),
            id='insights_sync',
            name='Insights Sync',
            replace_existing=True
        )
        
        logger.info(f"Registered scheduled jobs: sales_sync (every {self.sales_sync_interval}min), "
                  f"insights_sync (at {self.insights_sync_cron})")
    
    def run_now(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Run a specific job immediately.
        
        Args:
            job_id: ID of the job to run ('sales_sync' or 'insights_sync')
            
        Returns:
            Dict with job status or None if job not found
        """
        if job_id == 'sales_sync':
            return hourly_pull_sales_job(self.crm_adapter, self.data_dir, self.log_dir)
        elif job_id == 'insights_sync':
            return daily_push_insights_job(self.crm_adapter, self.insights_dir, self.log_dir)
        else:
            logger.error(f"Unknown job ID: {job_id}")
            return None
    
    def list_jobs(self) -> List[Dict[str, Any]]:
        """
        List all scheduled jobs.
        
        Returns:
            List of job information dictionaries
        """
        jobs = []
        for job in self.scheduler.get_jobs():
            next_run = job.next_run_time.isoformat() if job.next_run_time else None
            jobs.append({
                "id": job.

"""
CRM Sync Scheduler

This module provides scheduling functionality for CRM data synchronization tasks,
including hourly sales data pulls and daily insight pushes.
"""

import os
import sys
import json
import logging
import signal
import time
import traceback
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable

# APScheduler for job scheduling
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger

# Import our CRM adapters
from watchdog_ai.integrations.crm import NovaActAdapter, BaseCRMAdapter

# Configure logging
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_DATA_DIR = Path("data/raw")
DEFAULT_INSIGHTS_DIR = Path("data/processed")
DEFAULT_LOG_DIR = Path("data/logs")
DEFAULT_DB_PATH = Path("data/scheduler.sqlite")

# Environment variable names
ENV_DATA_DIR = "WATCHDOG_DATA_DIR"
ENV_INSIGHTS_DIR = "WATCHDOG_INSIGHTS_DIR"
ENV_LOG_DIR = "WATCHDOG_LOG_DIR"
ENV_DB_PATH = "WATCHDOG_SCHEDULER_DB"
ENV_SALES_INTERVAL = "WATCHDOG_SALES_SYNC_INTERVAL"
ENV_INSIGHTS_CRON = "WATCHDOG_INSIGHTS_SYNC_CRON"


# Standalone job functions for serialization
def hourly_pull_sales_job(crm_adapter: BaseCRMAdapter, data_dir: Path, log_dir: Path) -> Dict[str, Any]:
    """
    Pull sales data from CRM and save to data/raw directory.
    
    This job runs hourly to fetch the latest sales data.
    
    Args:
        crm_adapter: The CRM adapter to use
        data_dir: Directory to save data
        log_dir: Directory to save logs
        
    Returns:
        Dict with job status information
    """
    start_time = datetime.now()
    job_id = f"sales_pull_{start_time.strftime('%Y%m%d%H%M%S')}"
    
    logger.info(f"Starting hourly sales data pull job: {job_id}")
    
    try:
        # Authenticate with CRM
        crm_adapter.authenticate()
        
        # Pull sales data since last run
        # In a more sophisticated implementation, we'd track the last sync time
        # and only pull data since then, but for simplicity, we're pulling all data
        sales_data = crm_adapter.pull_sales()
        
        if not sales_data:
            logger.info("No sales data retrieved")
            return {
                "job_id": job_id,
                "status": "success",
                "message": "No sales data available",
                "record_count": 0,
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "data_file": None
            }
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"sales_{timestamp}.json"
        file_path = data_dir / filename
        
        # Save data to file
        with open(file_path, 'w') as f:
            json.dump(sales_data, f, indent=2)
        
        logger.info(f"Saved {len(sales_data)} sales records to {file_path}")
        
        # Return job status
        return {
            "job_id": job_id,
            "status": "success",
            "record_count": len(sales_data),
            "start_time": start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "data_file": str(file_path)
        }
        
    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        logger.error(f"Error in hourly sales data pull: {error_msg}")
        logger.debug(error_trace)
        
        # Log error details to file
        error_log_path = log_dir / f"error_{job_id}.log"
        with open(error_log_path, 'w') as f:
            f.write(f"Error: {error_msg}\n\nTraceback:\n{error_trace}")
        
        # Return error status
        return {
            "job_id": job_id,
            "status": "error",
            "message": error_msg,
            "start_time": start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "error_log": str(error_log_path)
        }


def daily_push_insights_job(crm_adapter: BaseCRMAdapter, insights_dir: Path, log_dir: Path) -> Dict[str, Any]:
    """
    Push processed insights to CRM.
    
    This job runs daily to push processed insights back to the CRM system.
    It reads the latest insights from data/processed/latest.json
    
    Args:
        crm_adapter: The CRM adapter to use
        insights_dir: Directory with insight files
        log_dir: Directory to save logs
        
    Returns:
        Dict with job status information
    """
    start_time = datetime.now()
    job_id = f"insights_push_{start_time.strftime('%Y%m%d%H%M%S')}"
    
    logger.info(f"Starting daily insights push job: {job_id}")
    
    try:
        # Check for insights file
        insights_file = insights_dir / "latest.json"
        
        if not insights_file.exists():
            logger.warning(f"Insights file not found: {insights_file}")
            return {
                "job_id": job_id,
                "status": "skipped",
                "message": f"Insights file not found: {insights_file}",
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat()
            }
        
        # Load insights data
        with open(insights_file, 'r') as f:
            insights_data = json.load(f)
        
        if not insights_data:
            logger.info("No insights data to push")
            return {
                "job_id": job_id,
                "status": "success",
                "message": "No insights data to push",
                "record_count": 0,
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat()
            }
        
        # Authenticate with CRM
        crm_adapter.authenticate()
        
        # Push insights to CRM
        crm_adapter.push_insights(insights_data)
        
        logger.info(f"Successfully pushed {len(insights_data)} insights to CRM")
        
        # Archive the insights file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        archive_path = insights_dir / f"archive/insights_{timestamp}.json"
        archive_path.parent.mkdir(exist_ok=True)
        
        # Copy insights to archive
        import shutil
        shutil.copy2(insights_file, archive_path)
        
        # Return job status
        return {
            "job_id": job_id,
            "status": "success",
            "record_count": len(insights_data),
            "start_time": start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "insights_file": str(insights_file),
            "archive_file": str(archive_path)
        }
            
    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        logger.error(f"Error in daily insights push: {error_msg}")
        logger.debug(error_trace)
        
        # Log error details to file
        error_log_path = log_dir / f"error_{job_id}.log"
        with open(error_log_path, 'w') as f:
            f.write(f"Error: {error_msg}\n\nTraceback:\n{error_trace}")
        
        # Return error status
        return {
            "job_id": job_id,
            "status": "error",
            "message": error_msg,
            "start_time": start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "error_log": str(error_log_path)
        }


class CRMSyncScheduler:
    """
    Scheduler for CRM data synchronization tasks.
    
    This class manages periodic CRM data synchronization jobs using APScheduler,
    including hourly pulls of sales data and daily pushes of insights.
    """
    
    def __init__(self, 
                data_dir: Optional[Union[str, Path]] = None,
                insights_dir: Optional[Union[str, Path]] = None,
                log_dir: Optional[Union[str, Path]] = None,
                db_path: Optional[Union[str, Path]] = None,
                sales_sync_interval: int = 60,  # minutes
                insights_sync_cron: str = "0 2 * * *",  # 2:00 AM daily
                crm_adapter: Optional[BaseCRMAdapter] = None):
        """
        Initialize the CRM Sync Scheduler.
        
        Args:
            data_dir: Directory for storing raw CRM data. Default data/raw/
            insights_dir: Directory for processed insights. Default data/processed/
            log_dir: Directory for logs. Default data/logs/
            db_path: Path to SQLite DB for job persistence. Default data/scheduler.sqlite
            sales_sync_interval: Interval in minutes for sales data sync. Default 60 (hourly)
            insights_sync_cron: Cron expression for insights sync. Default "0 2 * * *" (2:00 AM daily)
            crm_adapter: CRM adapter to use. Default is NovaActAdapter with env vars
        """
        # Set up paths from arguments or environment variables
        self.data_dir = Path(data_dir or os.environ.get(ENV_DATA_DIR, DEFAULT_DATA_DIR))
        self.insights_dir = Path(insights_dir or os.environ.get(ENV_INSIGHTS_DIR, DEFAULT_INSIGHTS_DIR))
        self.log_dir = Path(log_dir or os.environ.get(ENV_LOG_DIR, DEFAULT_LOG_DIR))
        self.db_path = Path(db_path or os.environ.get(ENV_DB_PATH, DEFAULT_DB_PATH))
        
        # Create directories if they don't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.insights_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure job schedules
        self.sales_sync_interval = int(os.environ.get(ENV_SALES_INTERVAL, sales_sync_interval))
        self.insights_sync_cron = os.environ.get(ENV_INSIGHTS_CRON, insights_sync_cron)
        
        # Set up the CRM adapter
        self.crm_adapter = crm_adapter or NovaActAdapter()
        
        # Initialize scheduler with SQLite job store
        self.scheduler = BackgroundScheduler()
        self.scheduler.add_jobstore(
            SQLAlchemyJobStore(url=f'sqlite:///{self.db_path}'),
            'default'
        )
        
        # Track if scheduler is running
        self.is_running = False
        
        logger.info(f"Initialized CRM Sync Scheduler with data_dir={self.data_dir}, "
                  f"sales_interval={self.sales_sync_interval}min, "
                  f"insights_cron='{self.insights_sync_cron}'")
    
    def start(self):
        """
        Start the scheduler and register jobs.
        """
        if self.is_running:
            logger.warning("Scheduler is already running")
            return
        
        # Register jobs
        self._register_jobs()
        
        # Start the scheduler
        self.scheduler.start()
        self.is_running = True
        
        logger.info("CRM Sync Scheduler started")
    
    def stop(self):
        """
        Stop the scheduler.
        """
        if not self.is_running:
            logger.warning("Scheduler is not running")
            return
        
        # Shutdown the scheduler
        self.scheduler.shutdown()
        self.is_running = False
        
        logger.info("CRM Sync Scheduler stopped")
    
    def _register_jobs(self):
        """
        Register all scheduled jobs.
        """
        # Sales data sync job - runs hourly by default
        self.scheduler.add_job(
            func=hourly_pull_sales_job,
            args=[self.crm_adapter, self.data_dir, self.log_dir],
            trigger=IntervalTrigger(minutes=self.sales_sync_interval),
            id='sales_sync',
            name='Sales Data Sync',
            replace_existing=True
        )
        
        # Insights sync job - runs at 2:00 AM daily by default
        self.scheduler.add_job(
            func=daily_push_insights_job,
            args=[self.crm_adapter, self.insights_dir, self.log_dir],
            trigger=CronTrigger.from_crontab(self.insights_sync_cron),
            id='insights_sync',
            name='Insights Sync',
            replace_existing=True
        )
        
        logger.info(f"Registered scheduled jobs: sales_sync (every {self.sales_sync_interval}min), "
                  f"insights_sync (at {self.insights_sync_cron})")
    
    def run_now(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Run a specific job immediately.
        
        Args:
            job_id: ID of the job to run ('sales_sync' or 'insights_sync')
            
        Returns:
            Dict with job status or None if job not found
        """
        if job_id == 'sales_sync':
            return hourly_pull_sales_job(self.crm_adapter, self.data_dir, self.log_dir)
        elif job_id == 'insights_sync':
            return daily_push_insights_job(self.crm_adapter, self.insights_dir, self.log_dir)
        else:
            logger.error(f"Unknown job ID: {job_id}")
            return None
    
    def list_jobs(self) -> List[Dict[str, Any]]:
        """
        List all scheduled jobs.
        
        Returns:
            List of job information dictionaries
        """
        jobs = []
        for job in self.scheduler.get_jobs():
            next_run = job.next_run_time.isoformat() if job.next_run_time else None
            jobs.append({
                "id": job.id,
                "name": job.name

"""
CRM Sync Scheduler

This module provides scheduling functionality for CRM data synchronization tasks,
including hourly sales data pulls and daily insight pushes.
"""

import os
import sys
import json
import logging
import signal
import time
import traceback
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable

# APScheduler for job scheduling
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger

# Import our CRM adapters
from watchdog_ai.integrations.crm import NovaActAdapter, BaseCRMAdapter

# Configure logging
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_DATA_DIR = Path("data/raw")
DEFAULT_INSIGHTS_DIR = Path("data/processed")
DEFAULT_LOG_DIR = Path("data/logs")
DEFAULT_DB_PATH = Path("data/scheduler.sqlite")

# Environment variable names
ENV_DATA_DIR = "WATCHDOG_DATA_DIR"
ENV_INSIGHTS_DIR = "WATCHDOG_INSIGHTS_DIR"
ENV_LOG_DIR = "WATCHDOG_LOG_DIR"
ENV_DB_PATH = "WATCHDOG_SCHEDULER_DB"
ENV_SALES_INTERVAL = "WATCHDOG_SALES_SYNC_INTERVAL"
ENV_INSIGHTS_CRON = "WATCHDOG_INSIGHTS_SYNC_CRON"


# Standalone job functions for serialization
def hourly_pull_sales_job(crm_adapter: BaseCRMAdapter, data_dir: Path, log_dir: Path) -> Dict[str, Any]:
    """
    Pull sales data from CRM and save to data/raw directory.
    
    This job runs hourly to fetch the latest sales data.
    
    Args:
        crm_adapter: The CRM adapter to use
        data_dir: Directory to save data
        log_dir: Directory to save logs
        
    Returns:
        Dict with job status information
    """
    start_time = datetime.now()
    job_id = f"sales_pull_{start_time.strftime('%Y%m%d%H%M%S')}"
    
    logger.info(f"Starting hourly sales data pull job: {job_id}")
    
    try:
        # Authenticate with CRM
        crm_adapter.authenticate()
        
        # Pull sales data since last run
        # In a more sophisticated implementation, we'd track the last sync time
        # and only pull data since then, but for simplicity, we're pulling all data
        sales_data = crm_adapter.pull_sales()
        
        if not sales_data:
            logger.info("No sales data retrieved")
            return {
                "job_id": job_id,
                "status": "success",
                "message": "No sales data available",
                "record_count": 0,
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "data_file": None
            }
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"sales_{timestamp}.json"
        file_path = data_dir / filename
        
        # Save data to file
        with open(file_path, 'w') as f:
            json.dump(sales_data, f, indent=2)
        
        logger.info(f"Saved {len(sales_data)} sales records to {file_path}")
        
        # Return job status
        return {
            "job_id": job_id,
            "status": "success",
            "record_count": len(sales_data),
            "start_time": start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "data_file": str(file_path)
        }
        
    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        logger.error(f"Error in hourly sales data pull: {error_msg}")
        logger.debug(error_trace)
        
        # Log error details to file
        error_log_path = log_dir / f"error_{job_id}.log"
        with open(error_log_path, 'w') as f:
            f.write(f"Error: {error_msg}\n\nTraceback:\n{error_trace}")
        
        # Return error status
        return {
            "job_id": job_id,
            "status": "error",
            "message": error_msg,
            "start_time": start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "error_log": str(error_log_path)
        }


def daily_push_insights_job(crm_adapter: BaseCRMAdapter, insights_dir: Path, log_dir: Path) -> Dict[str, Any]:
    """
    Push processed insights to CRM.
    
    This job runs daily to push processed insights back to the CRM system.
    It reads the latest insights from data/processed/latest.json
    
    Args:
        crm_adapter: The CRM adapter to use
        insights_dir: Directory with insight files
        log_dir: Directory to save logs
        
    Returns:
        Dict with job status information
    """
    start_time = datetime.now()
    job_id = f"insights_push_{start_time.strftime('%Y%m%d%H%M%S')}"
    
    logger.info(f"Starting daily insights push job: {job_id}")
    
    try:
        # Check for insights file
        insights_file = insights_dir / "latest.json"
        
        if not insights_file.exists():
            logger.warning(f"Insights file not found: {insights_file}")
            return {
                "job_id": job_id,
                "status": "skipped",
                "message": f"Insights file not found: {insights_file}",
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat()
            }
        
        # Load insights data
        with open(insights_file, 'r') as f:
            insights_data = json.load(f)
        
        if not insights_data:
            logger.info("No insights data to push")
            return {
                "job_id": job_id,
                "status": "success",
                "message": "No insights data to push",
                "record_count": 0,
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat()
            }
        
        # Authenticate with CRM
        crm_adapter.authenticate()
        
        # Push insights to CRM
        crm_adapter.push_insights(insights_data)
        
        logger.info(f"Successfully pushed {len(insights_data)} insights to CRM")
        
        # Archive the insights file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        archive_path = insights_dir / f"archive/insights_{timestamp}.json"
        archive_path.parent.mkdir(exist_ok=True)
        
        # Copy insights to archive
        import shutil
        shutil.copy2(insights_file, archive_path)
        
        # Return job status
        return {
            "job_id": job_id,
            "status": "success",
            "record_count": len(insights_data),
            "start_time": start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "insights_file": str(insights_file),
            "archive_file": str(archive_path)
        }
            
    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        logger.error(f"Error in daily insights push: {error_msg}")
        logger.debug(error_trace)
        
        # Log error details to file
        error_log_path = log_dir / f"error_{job_id}.log"
        with open(error_log_path, 'w') as f:
            f.write(f"Error: {error_msg}\n\nTraceback:\n{error_trace}")
        
        # Return error status
        return {
            "job_id": job_id,
            "status": "error",
            "message": error_msg,
            "start_time": start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "error_log": str(error_log_path)
        }


class CRMSyncScheduler:
    """
    Scheduler for CRM data synchronization tasks.
    
    This class manages periodic CRM data synchronization jobs using APScheduler,
    including hourly pulls of sales data and daily pushes of insights.
    """
    
    def __init__(self, 
                data_dir: Optional[Union[str, Path]] = None,
                insights_dir: Optional[Union[str, Path]] = None,
                log_dir: Optional[Union[str, Path]] = None,
                db_path: Optional[Union[str, Path]] = None,
                sales_sync_interval: int = 60,  # minutes
                insights_sync_cron: str = "0 2 * * *",  # 2:00 AM daily
                crm_adapter: Optional[BaseCRMAdapter] = None):
        """
        Initialize the CRM Sync Scheduler.
        
        Args:
            data_dir: Directory for storing raw CRM data. Default data/raw/
            insights_dir: Directory for processed insights. Default data/processed/
            log_dir: Directory for logs. Default data/logs/
            db_path: Path to SQLite DB for job persistence. Default data/scheduler.sqlite
            sales_sync_interval: Interval in minutes for sales data sync. Default 60 (hourly)
            insights_sync_cron: Cron expression for insights sync. Default "0 2 * * *" (2:00 AM daily)
            crm_adapter: CRM adapter to use. Default is NovaActAdapter with env vars
        """
        # Set up paths from arguments or environment variables
        self.data_dir = Path(data_dir or os.environ.get(ENV_DATA_DIR, DEFAULT_DATA_DIR))
        self.insights_dir = Path(insights_dir or os.environ.get(ENV_INSIGHTS_DIR, DEFAULT_INSIGHTS_DIR))
        self.log_dir = Path(log_dir or os.environ.get(ENV_LOG_DIR, DEFAULT_LOG_DIR))
        self.db_path = Path(db_path or os.environ.get(ENV_DB_PATH, DEFAULT_DB_PATH))
        
        # Create directories if they don't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.insights_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure job schedules
        self.sales_sync_interval = int(os.environ.get(ENV_SALES_INTERVAL, sales_sync_interval))
        self.insights_sync_cron = os.environ.get(ENV_INSIGHTS_CRON, insights_sync_cron)
        
        # Set up the CRM adapter
        self.crm_adapter = crm_adapter or NovaActAdapter()
        
        # Initialize scheduler with SQLite job store
        self.scheduler = BackgroundScheduler()
        self.scheduler.add_jobstore(
            SQLAlchemyJobStore(url=f'sqlite:///{self.db_path}'),
            'default'
        )
        
        # Track if scheduler is running
        self.is_running = False
        
        logger.info(f"Initialized CRM Sync Scheduler with data_dir={self.data_dir}, "
                  f"sales_interval={self.sales_sync_interval}min, "
                  f"insights_cron='{self.insights_sync_cron}'")
    
    def start(self):
        """
        Start the scheduler and register jobs.
        """
        if self.is_running:
            logger.warning("Scheduler is already running")
            return
        
        # Register jobs
        self._register_jobs()
        
        # Start the scheduler
        self.scheduler.start()
        self.is_running = True
        
        logger.info("CRM Sync Scheduler started")
    
    def stop(self):
        """
        Stop the scheduler.
        """
        if not self.is_running:
            logger.warning("Scheduler is not running")
            return
        
        # Shutdown the scheduler
        self.scheduler.shutdown()
        self.is_running = False
        
        logger.info("CRM Sync Scheduler stopped")
    
    def _register_jobs(self):
        """
        Register all scheduled jobs.
        """
        # Sales data sync job - runs hourly by default
        self.scheduler.add_job(
            func=hourly_pull_sales_job,
            args=[self.crm_adapter, self.data_dir, self.log_dir],
            trigger=IntervalTrigger(minutes=self.sales_sync_interval),
            id='sales_sync',
            name='Sales Data Sync',
            replace_existing=True
        )
        
        # Insights sync job - runs at 2:00 AM daily by default
        self.scheduler.add_job(
            func=daily_push_insights_job,
            args=[self.crm_adapter, self.insights_dir, self.log_dir],
            trigger=CronTrigger.from_crontab(self.insights_sync_cron),
            id='insights_sync',
            name='Insights Sync',
            replace_existing=True
        )
        
        logger.info(f"Registered scheduled jobs: sales_sync (every {self.sales_sync_interval}min), "
                  f"insights_sync (at {self.insights_sync_cron})")
    
    def run_now(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Run a specific job immediately.
        
        Args:
            job_id: ID of the job to run ('sales_sync' or 'insights_sync')
            
        Returns:
            Dict with job status or None if job not found
        """
        if job_id == 'sales_sync':
            return hourly_pull_sales_job(self.crm_adapter, self.data_dir, self.log_dir)
        elif job_id == 'insights_sync':
            return daily_push_insights_job(self.crm_adapter, self.insights_dir, self.log_dir)
        else:
            logger.error(f"Unknown job ID: {job_id}")
            return None
    
    def list_jobs(self) -> List[Dict[str, Any]]:
        """
        List all scheduled jobs.
        
        Returns:
            List of job information dictionaries
        """
        jobs = []
        for job in self.scheduler.get_jobs():
            next_run = job.next_run_time.isoformat() if job.next_run_time else None
            jobs.append({
                "id": job.id,
                "name": job.name,
                "next_run": next_run,
                

"""
CRM Sync Scheduler

This module provides scheduling functionality for CRM data synchronization tasks,
including hourly sales data pulls and daily insight pushes.
"""

import os
import sys
import json
import logging
import signal
import time
import traceback
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable

# APScheduler for job scheduling
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger

# Import our CRM adapters
from watchdog_ai.integrations.crm import NovaActAdapter, BaseCRMAdapter

# Configure logging
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_DATA_DIR = Path("data/raw")
DEFAULT_INSIGHTS_DIR = Path("data/processed")
DEFAULT_LOG_DIR = Path("data/logs")
DEFAULT_DB_PATH = Path("data/scheduler.sqlite")

# Environment variable names
ENV_DATA_DIR = "WATCHDOG_DATA_DIR"
ENV_INSIGHTS_DIR = "WATCHDOG_INSIGHTS_DIR"
ENV_LOG_DIR = "WATCHDOG_LOG_DIR"
ENV_DB_PATH = "WATCHDOG_SCHEDULER_DB"
ENV_SALES_INTERVAL = "WATCHDOG_SALES_SYNC_INTERVAL"
ENV_INSIGHTS_CRON = "WATCHDOG_INSIGHTS_SYNC_CRON"


# Standalone job functions for serialization
def hourly_pull_sales_job(crm_adapter: BaseCRMAdapter, data_dir: Path, log_dir: Path) -> Dict[str, Any]:
    """
    Pull sales data from CRM and save to data/raw directory.
    
    This job runs hourly to fetch the latest sales data.
    
    Args:
        crm_adapter: The CRM adapter to use
        data_dir: Directory to save data
        log_dir: Directory to save logs
        
    Returns:
        Dict with job status information
    """
    start_time = datetime.now()
    job_id = f"sales_pull_{start_time.strftime('%Y%m%d%H%M%S')}"
    
    logger.info(f"Starting hourly sales data pull job: {job_id}")
    
    try:
        # Authenticate with CRM
        crm_adapter.authenticate()
        
        # Pull sales data since last run
        # In a more sophisticated implementation, we'd track the last sync time
        # and only pull data since then, but for simplicity, we're pulling all data
        sales_data = crm_adapter.pull_sales()
        
        if not sales_data:
            logger.info("No sales data retrieved")
            return {
                "job_id": job_id,
                "status": "success",
                "message": "No sales data available",
                "record_count": 0,
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "data_file": None
            }
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"sales_{timestamp}.json"
        file_path = data_dir / filename
        
        # Save data to file
        with open(file_path, 'w') as f:
            json.dump(sales_data, f, indent=2)
        
        logger.info(f"Saved {len(sales_data)} sales records to {file_path}")
        
        # Return job status
        return {
            "job_id": job_id,
            "status": "success",
            "record_count": len(sales_data),
            "start_time": start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "data_file": str(file_path)
        }
        
    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        logger.error(f"Error in hourly sales data pull: {error_msg}")
        logger.debug(error_trace)
        
        # Log error details to file
        error_log_path = log_dir / f"error_{job_id}.log"
        with open(error_log_path, 'w') as f:
            f.write(f"Error: {error_msg}\n\nTraceback:\n{error_trace}")
        
        # Return error status
        return {
            "job_id": job_id,
            "status": "error",
            "message": error_msg,
            "start_time": start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "error_log": str(error_log_path)
        }


def daily_push_insights_job(crm_adapter: BaseCRMAdapter, insights_dir: Path, log_dir: Path) -> Dict[str, Any]:
    """
    Push processed insights to CRM.
    
    This job runs daily to push processed insights back to the CRM system.
    It reads the latest insights from data/processed/latest.json
    
    Args:
        crm_adapter: The CRM adapter to use
        insights_dir: Directory with insight files
        log_dir: Directory to save logs
        
    Returns:
        Dict with job status information
    """
    start_time = datetime.now()
    job_id = f"insights_push_{start_time.strftime('%Y%m%d%H%M%S')}"
    
    logger.info(f"Starting daily insights push job: {job_id}")
    
    try:
        # Check for insights file
        insights_file = insights_dir / "latest.json"
        
        if not insights_file.exists():
            logger.warning(f"Insights file not found: {insights_file}")
            return {
                "job_id": job_id,
                "status": "skipped",
                "message": f"Insights file not found: {insights_file}",
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat()
            }
        
        # Load insights data
        with open(insights_file, 'r') as f:
            insights_data = json.load(f)
        
        if not insights_data:
            logger.info("No insights data to push")
            return {
                "job_id": job_id,
                "status": "success",
                "message": "No insights data to push",
                "record_count": 0,
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat()
            }
        
        # Authenticate with CRM
        crm_adapter.authenticate()
        
        # Push insights to CRM
        crm_adapter.push_insights(insights_data)
        
        logger.info(f"Successfully pushed {len(insights_data)} insights to CRM")
        
        # Archive the insights file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        archive_path = insights_dir / f"archive/insights_{timestamp}.json"
        archive_path.parent.mkdir(exist_ok=True)
        
        # Copy insights to archive
        import shutil
        shutil.copy2(insights_file, archive_path)
        
        # Return job status
        return {
            "job_id": job_id,
            "status": "success",
            "record_count": len(insights_data),
            "start_time": start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "insights_file": str(insights_file),
            "archive_file": str(archive_path)
        }
            
    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        logger.error(f"Error in daily insights push: {error_msg}")
        logger.debug(error_trace)
        
        # Log error details to file
        error_log_path = log_dir / f"error_{job_id}.log"
        with open(error_log_path, 'w') as f:
            f.write(f"Error: {error_msg}\n\nTraceback:\n{error_trace}")
        
        # Return error status
        return {
            "job_id": job_id,
            "status": "error",
            "message": error_msg,
            "start_time": start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "error_log": str(error_log_path)
        }


class CRMSyncScheduler:
    """
    Scheduler for CRM data synchronization tasks.
    
    This class manages periodic CRM data synchronization jobs using APScheduler,
    including hourly pulls of sales data and daily pushes of insights.
    """
    
    def __init__(self, 
                data_dir: Optional[Union[str, Path]] = None,
                insights_dir: Optional[Union[str, Path]] = None,
                log_dir: Optional[Union[str, Path]] = None,
                db_path: Optional[Union[str, Path]] = None,
                sales_sync_interval: int = 60,  # minutes
                insights_sync_cron: str = "0 2 * * *",  # 2:00 AM daily
                crm_adapter: Optional[BaseCRMAdapter] = None):
        """
        Initialize the CRM Sync Scheduler.
        
        Args:
            data_dir: Directory for storing raw CRM data. Default data/raw/
            insights_dir: Directory for processed insights. Default data/processed/
            log_dir: Directory for logs. Default data/logs/
            db_path: Path to SQLite DB for job persistence. Default data/scheduler.sqlite
            sales_sync_interval: Interval in minutes for sales data sync. Default 60 (hourly)
            insights_sync_cron: Cron expression for insights sync. Default "0 2 * * *" (2:00 AM daily)
            crm_adapter: CRM adapter to use. Default is NovaActAdapter with env vars
        """
        # Set up paths from arguments or environment variables
        self.data_dir = Path(data_dir or os.environ.get(ENV_DATA_DIR, DEFAULT_DATA_DIR))
        self.insights_dir = Path(insights_dir or os.environ.get(ENV_INSIGHTS_DIR, DEFAULT_INSIGHTS_DIR))
        self.log_dir = Path(log_dir or os.environ.get(ENV_LOG_DIR, DEFAULT_LOG_DIR))
        self.db_path = Path(db_path or os.environ.get(ENV_DB_PATH, DEFAULT_DB_PATH))
        
        # Create directories if they don't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.insights_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure job schedules
        self.sales_sync_interval = int(os.environ.get(ENV_SALES_INTERVAL, sales_sync_interval))
        self.insights_sync_cron = os.environ.get(ENV_INSIGHTS_CRON, insights_sync_cron)
        
        # Set up the CRM adapter
        self.crm_adapter = crm_adapter or NovaActAdapter()
        
        # Initialize scheduler with SQLite job store
        self.scheduler = BackgroundScheduler()
        self.scheduler.add_jobstore(
            SQLAlchemyJobStore(url=f'sqlite:///{self.db_path}'),
            'default'
        )
        
        # Track if scheduler is running
        self.is_running = False
        
        logger.info(f"Initialized CRM Sync Scheduler with data_dir={self.data_dir}, "
                  f"sales_interval={self.sales_sync_interval}min, "
                  f"insights_cron='{self.insights_sync_cron}'")
    
    def start(self):
        """
        Start the scheduler and register jobs.
        """
        if self.is_running:
            logger.warning("Scheduler is already running")
            return
        
        # Register jobs
        self._register_jobs()
        
        # Start the scheduler
        self.scheduler.start()
        self.is_running = True
        
        logger.info("CRM Sync Scheduler started")
    
    def stop(self):
        """
        Stop the scheduler.
        """
        if not self.is_running:
            logger.warning("Scheduler is not running")
            return
        
        # Shutdown the scheduler
        self.scheduler.shutdown()
        self.is_running = False
        
        logger.info("CRM Sync Scheduler stopped")
    
    def _register_jobs(self):
        """
        Register all scheduled jobs.
        """
        # Sales data sync job - runs hourly by default
        self.scheduler.add_job(
            func=hourly_pull_sales_job,
            args=[self.crm_adapter, self.data_dir, self.log_dir],
            trigger=IntervalTrigger(minutes=self.sales_sync_interval),
            id='sales_sync',
            name='Sales Data Sync',
            replace_existing=True
        )
        
        # Insights sync job - runs at 2:00 AM daily by default
        self.scheduler.add_job(
            func=daily_push_insights_job,
            args=[self.crm_adapter, self.insights_dir, self.log_dir],
            trigger=CronTrigger.from_crontab(self.insights_sync_cron),
            id='insights_sync',
            name='Insights Sync',
            replace_existing=True
        )
        
        logger.info(f"Registered scheduled jobs: sales_sync (every {self.sales_sync_interval}min), "
                  f"insights_sync (at {self.insights_sync_cron})")
    
    def run_now(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Run a specific job immediately.
        
        Args:
            job_id: ID of the job to run ('sales_sync' or 'insights_sync')
            
        Returns:
            Dict with job status or None if job not found
        """
        if job_id == 'sales_sync':
            return hourly_pull_sales_job(self.crm_adapter, self.data_dir, self.log_dir)
        elif job_id == 'insights_sync':
            return daily_push_insights_job(self.crm_adapter, self.insights_dir, self.log_dir)
        else:
            logger.error(f"Unknown job ID: {job_id}")
            return None
    
    def list_jobs(self) -> List[Dict[str, Any]]:
        """
        List all scheduled jobs.
        
        Returns:
            List of job information dictionaries
        """
        jobs = []
        for job in self.scheduler.get_jobs():
            next_run = job.next_run_time.isoformat() if job.next_run_time else None
            jobs.append({
                "id": job.id,
                "name": job.name,
                "next_run": next_run,
                "trigger": str(job.trigger)
            })
        return jobs


def configure_logging(log_dir: Optional[Union[str, Path]] = None, level: str = "INFO") -> Path:
    """
    Configure logging for the scheduler.
    
    Args:
        log_dir: Directory for log files
        level: Logging level
        
    Returns:
        Path to log file
    """
    log_dir = Path(log_dir or DEFAULT_LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    log_file = log_dir / f"scheduler_{datetime.now().strftime('%Y%m%d')}.log"
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level))
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add file handler
    file_handler = logging.FileHandler(log_file)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    logger.info(f"Logging configured. Log file: {log_file}")
    return log_file
    """
    Pull sales data from CRM and save to data/raw directory.
    
    This job runs hourly to fetch the latest sales data.
    
    Args:
        crm_adapter: The CRM adapter to use
        data_dir: Directory to save data
        log_dir: Directory to save logs
        
    Returns:
        Dict with job status information
    """
    start_time = datetime.now()
    job_id = f"sales_pull_{start_time.strftime('%Y%m%d%H%M%S')}"
    
    logger.info(f"Starting hourly sales data pull job: {job_id}")
    
    try:
        # Authenticate with CRM
        crm_adapter.authenticate()
        
        # Pull sales data since last run
        # In a more sophisticated implementation, we'd track the last sync time
        # and only pull data since then, but for simplicity, we're pulling all data
        sales_data = crm_adapter.pull_sales()
        
        if not sales_data:
            logger.info("No sales data retrieved")
            return {
                "job_id": job_id,
                "status": "success",
                "message": "No sales data available",
                "record_count": 0,
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "data_file": None
            }
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"sales_{timestamp}.json"
        file_path = data_dir / filename
        
        # Save data to file
        with open(file_path, 'w') as f:
            json.dump(sales_data, f, indent=2)
        
        logger.info(f"Saved {len(sales_data)} sales records to {file_path}")
        
        # Return job status
        return {
            "job_id": job_id,
            "status": "success",
            "record_count": len(sales_data),
            "start_time": start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "data_file": str(file_path)
        }
        
    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        logger.error(f"Error in hourly sales data pull: {error_msg}")
        logger.debug(error_trace)
        
        # Log error details to file
        error_log_path = log_dir / f"error_{job_id}.log"
        with open(error_log_path, 'w') as f:
            f.write(f"Error: {error_msg}\n\nTraceback:\n{error_trace}")
        
        # Return error status
        return {
            "job_id": job_id,
            "status": "error",
            "message": error_msg,
            "start_time": start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "error_log": str(error_log_path)
        }


def daily_push_insights_job(crm_adapter: BaseCRMAdapter, insights_dir: Path, log_dir: Path) -> Dict[str, Any]:
    """
    Push processed insights to CRM.
    
    This job runs daily to push processed insights back to the CRM system.
    It reads the latest insights from data/processed/latest.json
    
    Args:
        crm_adapter: The CRM adapter to use
        insights_dir: Directory with insight files
        log_dir: Directory to save logs
        
    Returns:
        Dict with job status information
    """
    start_time = datetime.now()
    job_id = f"insights_push_{start_time.strftime('%Y%m%d%H%M%S')}"
    
    logger.info(f"Starting daily insights push job: {job_id}")
    
    try:
        # Check for insights file
        insights_file = insights_dir / "latest.json"
        
        if not insights_file.exists():
            logger.warning(f"Insights file not found: {insights_file}")
            return {
                "job_id": job_id,
                "status": "skipped",
                "message": f"Insights file not found: {insights_file}",
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat()
            }
        
        # Load insights data
        with open(insights_file, 'r') as f:
            insights_data = json.load(f)
        
        if not insights_data:
            logger.info("No insights data to push")
            return {
                "job_id": job_id,
                "status": "success",
                "message": "No insights data to push",
                "record_count": 0,
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat()
            }
        
        # Authenticate with CRM
        crm_adapter.authenticate()
        
        # Push insights to CRM
        crm_adapter.push_insights(insights_data)
        
        logger.info(f"Successfully pushed {len(insights_data)} insights to CRM")
        
        # Archive the insights file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        archive_path = insights_dir / f"archive/insights_{timestamp}.json"
        archive_path.parent.mkdir(exist_ok=True)
        
        # Copy insights to archive
        import shutil
        shutil.copy2(insights_file, archive_path)
        
        # Return job status
        return {
            "job_id": job_id,
            "status": "success",
            "record_count": len(insights_data),
            "start_time": start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "insights_file": str(insights_file),
            "archive_file": str(archive_path)
        }
            
    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        logger.error(f"Error in daily insights push: {error_msg}")
        logger.debug(error_trace)
        
        # Log error details to file
        error_log_path = log_dir / f"error_{job_id}.log"
        with open(error_log_path, 'w') as f:
            f.write(f"Error: {error_msg}\n\nTraceback:\n{error_trace}")
        
        # Return error status
        return {
            "job_id": job_id,
            "status": "error",
            "message": error_msg,
            "start_time": start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "error_log": str(error_log_path)
        }


class CRMSyncScheduler:
    """
    Scheduler for CRM data synchronization tasks.
    
    This class manages periodic CRM data synchronization jobs using APScheduler,
    including hourly pulls of sales data and daily pushes of insights.
    """
    
    def __init__(self, 
                data_dir: Optional[Union[str, Path]] = None,
                insights_dir: Optional[Union[str, Path]] = None,
                log_dir: Optional[Union[str, Path]] = None,
                db_path: Optional[Union[str, Path]] = None,
                sales_sync_interval: int = 60,  # minutes
                insights_sync_cron: str = "0 2 * * *",  # 2:00 AM daily
                crm_adapter: Optional[BaseCRMAdapter] = None):
        """
        Initialize the CRM Sync Scheduler.
        
        Args:
            data_dir: Directory for storing raw CRM data. Default data/raw/
            insights_dir: Directory for processed insights. Default data/processed/
            log_dir: Directory for logs. Default data/logs/
            db_path: Path to SQLite DB for job persistence. Default data/scheduler.sqlite
            sales_sync_interval: Interval in minutes for sales data sync. Default 60 (hourly)
            insights_sync_cron: Cron expression for insights sync. Default "0 2 * * *" (2:00 AM daily)
            crm_adapter: CRM adapter to use. Default is NovaActAdapter with env vars
        """
        # Set up paths from arguments or environment variables
        self.data_dir = Path(data_dir or os.environ.get(ENV_DATA_DIR, DEFAULT_DATA_DIR))
        self.insights_dir = Path(insights_dir or os.environ.get(ENV_INSIGHTS_DIR, DEFAULT_INSIGHTS_DIR))
        self.log_dir = Path(log_dir or os.environ.get(ENV_LOG_DIR, DEFAULT_LOG_DIR))
        self.db_path = Path(db_path or os.environ.get(ENV_DB_PATH, DEFAULT_DB_PATH))
        
        # Create directories if they don't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.insights_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure job schedules
        self.sales_sync_interval = int(os.environ.get(ENV_SALES_INTERVAL, sales_sync_interval))
        self.insights_sync_cron = os.environ.get(ENV_INSIGHTS_CRON, insights_sync_cron)
        
        # Set up the CRM adapter
        self.crm_adapter = crm_adapter or NovaActAdapter()
        
        # Initialize scheduler with SQLite job store
        self.scheduler = BackgroundScheduler()
        self.scheduler.add_jobstore(
            SQLAlchemyJobStore(url=f'sqlite:///{self.db_path}'),
            'default'
        )
        
        # Track if scheduler is running
        self.is_running = False
        
        logger.info(f"Initialized CRM Sync Scheduler with data_dir={self.data_dir}, "
                  f"sales_interval={self.sales_sync_interval}min, "
                  f"insights_cron='{self.insights_sync_cron}'")
    
    def start(self):
        """
        Start the scheduler and register jobs.
        """
        if self.is_running:
            logger.warning("Scheduler is already running")
            return
        
        # Register jobs
        self._register_jobs()
        
        # Start the scheduler
        self.scheduler.start()
        self.is_running = True
        
        logger.info("CRM Sync Scheduler started")
    
    def stop(self):
        """
        Stop the scheduler.
        """
        if not self.is_running:
            logger.warning("Scheduler is not running")
            return
        
        # Shutdown the scheduler
        self.scheduler.shutdown()
        self.is_running = False
        
        logger.info("CRM Sync Scheduler stopped")
    
    def _register_jobs(self):
        """
        Register all scheduled jobs.
        """
        # Sales data sync job - runs hourly by default
        self.scheduler.add_job(
            func=hourly_pull_sales_job,
            args=[self.crm_adapter, self.data_dir, self.log_dir],
            trigger=IntervalTrigger(minutes=self.sales_sync_interval),
            id='sales_sync',
            name='Sales Data Sync',
            replace_existing=True
        )
        
        # Insights sync job - runs at 2:00 AM daily by default
        self.scheduler.add_job(
            func=daily_push_insights_job,
            args=[self.crm_adapter, self.insights_dir, self.log_dir],
            trigger=CronTrigger.from_crontab(self.insights_sync_cron),
            id='insights_sync',
            name='Insights Sync',
            replace_existing=True
        )
        
        logger.info(f"Registered scheduled jobs: sales_sync (every {self.sales_sync_interval}min), "
                  f"insights_sync (at {self.insights_sync_cron})")
    
    def run_now(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Run a specific job immediately.
        
        Args:
            job_id: ID of the job to run ('sales_sync' or 'insights_sync')
            
        Returns:
            Dict with job status or None if job not found
        """
        if job_id == 'sales_sync':
            return hourly_pull_sales_job(self.crm_adapter, self.data_dir, self.log_dir)
        elif job_id == 'insights_sync':
            return daily_push_insights_job(self.crm_adapter, self.insights_dir, self.log_dir)
        else:
            logger.error(f"Unknown job ID: {job_id}")
            return None
    
    def list_jobs(self) -> List[Dict[str, Any]]:
        """
        List all scheduled jobs.
        
        Returns:
            List of job information dictionaries
        """
        jobs = []
        for job in self.scheduler.get_jobs():
            next_run = job.next_run_time.isoformat() if job.next_run_time else None
            jobs.append({
                "id": job.id,
                "name": job

"""
CRM Sync Scheduler

This module provides scheduling functionality for CRM data synchronization tasks,
including hourly sales data pulls and daily insight pushes.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
import traceback
from typing import Dict, Any, Optional, List, Union, Callable

# APScheduler for job scheduling
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger

# Import our CRM adapters
from watchdog_ai.integrations.crm import NovaActAdapter, BaseCRMAdapter

# Configure logging
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_DATA_DIR = Path("data/raw")
DEFAULT_INSIGHTS_DIR = Path("data/processed")
DEFAULT_LOG_DIR = Path("data/logs")
DEFAULT_DB_PATH = Path("data/scheduler.sqlite")

# Environment variable names
ENV_DATA_DIR = "WATCHDOG_DATA_DIR"
ENV_INSIGHTS_DIR = "WATCHDOG_INSIGHTS_DIR"
ENV_LOG_DIR = "WATCHDOG_LOG_DIR"
ENV_DB_PATH = "WATCHDOG_SCHEDULER_DB"
ENV_SALES_INTERVAL = "WATCHDOG_SALES_SYNC_INTERVAL"
ENV_INSIGHTS_CRON = "WATCHDOG_INSIGHTS_SYNC_CRON"


class CRMSyncScheduler:
    """
    Scheduler for CRM data synchronization tasks.
    
    This class manages periodic CRM data synchronization jobs using APScheduler,
    including hourly pulls of sales data and daily pushes of insights.
    """
    
    def __init__(self, 
                data_dir: Optional[Union[str, Path]] = None,
                insights_dir: Optional[Union[str, Path]] = None,
                log_dir: Optional[Union[str, Path]] = None,
                db_path: Optional[Union[str, Path]] = None,
                sales_sync_interval: int = 60,  # minutes
                insights_sync_cron: str = "0 2 * * *",  # 2:00 AM daily
                crm_adapter: Optional[BaseCRMAdapter] = None):
        """
        Initialize the CRM Sync Scheduler.
        
        Args:
            data_dir: Directory for storing raw CRM data. Default data/raw/
            insights_dir: Directory for processed insights. Default data/processed/
            log_dir: Directory for logs. Default data/logs/
            db_path: Path to SQLite DB for job persistence. Default data/scheduler.sqlite
            sales_sync_interval: Interval in minutes for sales data sync. Default 60 (hourly)
            insights_sync_cron: Cron expression for insights sync. Default "0 2 * * *" (2:00 AM daily)
            crm_adapter: CRM adapter to use. Default is NovaActAdapter with env vars
        """
        # Set up paths from arguments or environment variables
        self.data_dir = Path(data_dir or os.environ.get(ENV_DATA_DIR, DEFAULT_DATA_DIR))
        self.insights_dir = Path(insights_dir or os.environ.get(ENV_INSIGHTS_DIR, DEFAULT_INSIGHTS_DIR))
        self.log_dir = Path(log_dir or os.environ.get(ENV_LOG_DIR, DEFAULT_LOG_DIR))
        self.db_path = Path(db_path or os.environ.get(ENV_DB_PATH, DEFAULT_DB_PATH))
        
        # Create directories if they don't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.insights_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure job schedules
        self.sales_sync_interval = int(os.environ.get(ENV_SALES_INTERVAL, sales_sync_interval))
        self.insights_sync_cron = os.environ.get(ENV_INSIGHTS_CRON, insights_sync_cron)
        
        # Set up the CRM adapter
        self.crm_adapter = crm_adapter or NovaActAdapter()
        
        # Initialize scheduler with SQLite job store
        self.scheduler = BackgroundScheduler()
        self.scheduler.add_jobstore(
            SQLAlchemyJobStore(url=f'sqlite:///{self.db_path}'),
            'default'
        )
        
        # Track if scheduler is running
        self.is_running = False
        
        logger.info(f"Initialized CRM Sync Scheduler with data_dir={self.data_dir}, "
                  f"sales_interval={self.sales_sync_interval}min, "
                  f"insights_cron='{self.insights_sync_cron}'")
    
    def start(self):
        """
        Start the scheduler and register jobs.
        """
        if self.is_running:
            logger.warning("Scheduler is already running")
            return
        
        # Register jobs
        self._register_jobs()
        
        # Start the scheduler
        self.scheduler.start()
        self.is_running = True
        
        logger.info("CRM Sync Scheduler started")
    
    def stop(self):
        """
        Stop the scheduler.
        """
        if not self.is_running:
            logger.warning("Scheduler is not running")
            return
        
        # Shutdown the scheduler
        self.scheduler.shutdown()
        self.is_running = False
        
        logger.info("CRM Sync Scheduler stopped")
    
    def _register_jobs(self):
        """
# Standalone job functions for serialization
def hourly_pull_sales_job(crm_adapter: BaseCRMAdapter, data_dir: Path, log_dir: Path) -> Dict[str, Any]:
    """
    Pull sales data from CRM and save to data/raw directory.
    
    This job runs hourly to fetch the latest sales data.
    
    Args:
        crm_adapter: The CRM adapter to use
        data_dir: Directory to save data
        log_dir: Directory to save logs
        
    Returns:
        Dict with job status information
    """
    start_time = datetime.now()
    job_id = f"sales_pull_{start_time.strftime('%Y%m%d%H%M%S')}"
    
    logger.info(f"Starting hourly sales data pull job: {job_id}")
    
    try:
        # Authenticate with CRM
        crm_adapter.authenticate()
        
        # Pull sales data since last run
        # In a more sophisticated implementation, we'd track the last sync time
        # and only pull data since then, but for simplicity, we're pulling all data
        sales_data = crm_adapter.pull_sales()
        
        if not sales_data:
            logger.info("No sales data retrieved")
            return {
                "job_id": job_id,
                "status": "success",
                "message": "No sales data available",
                "record_count": 0,
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "data_file": None
            }
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"sales_{timestamp}.json"
        file_path = data_dir / filename
        
        # Save data to file
        with open(file_path, 'w') as f:
            json.dump(sales_data, f, indent=2)
        
        logger.info(f"Saved {len(sales_data)} sales records to {file_path}")
        
        # Return job status
        return {
            "job_id": job_id,
            "status": "success",
            "record_count": len(sales_data),
            "start_time": start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "data_file": str(file_path)
        }
        
    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        logger.error(f"Error in hourly sales data pull: {error_msg}")
        logger.debug(error_trace)
        
        # Log error details to file
        error_log_path = log_dir / f"error_{job_id}.log"
        with open(error_log_path, 'w') as f:
            f.write(f"Error: {error_msg}\n\nTraceback:\n{error_trace}")
        
        # Return error status
        return {
            "job_id": job_id,
            "status": "error",
            "message": error_msg,
            "start_time": start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "error_log": str(error_log_path)
        }


def daily_push_insights_job(crm_adapter: BaseCRMAdapter, insights_dir: Path, log_dir: Path) -> Dict[str, Any]:
    """
    Push processed insights to CRM.
    
    This job runs daily to push processed insights back to the CRM system.
    It reads the latest insights from data/processed/latest.json
    
    Args:
        crm_adapter: The CRM adapter to use
        insights_dir: Directory with insight files
        log_dir: Directory to save logs
        
    Returns:
        Dict with job status information
    """
    start_time = datetime.now()
    job_id = f"insights_push_{start_time.strftime('%Y%m%d%H%M%S')}"
    
    logger.info(f"Starting daily insights push job: {job_id}")
    
    try:
        # Check for insights file
        insights_file = insights_dir / "latest.json"
        
        if not insights_file.exists():
            logger.warning(f"Insights file not found: {insights_file}")
            return {
                "job_id": job_id,
                "status": "skipped",
                "message": f"Insights file not found: {insights_file}",
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat()
            }
        
        # Load insights data
        with open(insights_file, 'r') as f:
            insights_data = json.load(f)
        
        if not insights_data:
            logger.info("No insights data to push")
            return {
                "job_id": job_id,
                "status": "success",
                "message": "No insights data to push",
                "record_count": 0,
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat()
            }
        
        # Authenticate with CRM
        crm_adapter.authenticate()
        
        # Push insights to CRM
        crm_adapter.push_insights(insights_data)
        
        logger.info(f"Successfully pushed {len(insights_data)} insights to CRM")
        
        # Archive the insights file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        archive_path = insights_dir / f"archive/insights_{timestamp}.json"
        archive_path.parent.mkdir(exist_ok=True)
        
        # Copy insights to archive
        import shutil
        shutil.copy2(insights_file, archive_path)
        
        # Return job status
        return {
            "job_id": job_id,
            "status": "success",
        """
        Run a specific job immediately.
        
        Args:
            job_id: ID of the job to run ('sales_sync' or 'insights_sync')
            
        Returns:
            Dict with job status or None if job not found
        """
        if job_id == 'sales_sync':
            return self.hourly_pull_sales()
        elif job_id == 'insights_sync':
            return self.daily_push_insights()
        else:
            logger.error(f"Unknown job ID: {job_id}")
            return None
    
    def list_jobs(self) -> List[Dict[str, Any]]:
        """
        List all scheduled jobs.
        
        Returns:
            List of job information dictionaries
        """
        jobs = []
        for job in self.scheduler.get_jobs():
            next_run = job.next_run_time.isoformat() if job.next_run_time else None
            jobs.append({
                "id": job.id,
                "name": job.name,
                "next_run": next_run,
                "trigger": str(job.trigger)
            })
        return jobs

        logger.error(f"Error starting scheduler: {str(e)}")
        traceback.print_exc()
        sys.exit(1)
