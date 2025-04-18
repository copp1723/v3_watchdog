"""
Base Scheduler Module for V3 Watchdog AI.

Provides the abstract base class for all scheduler implementations.
"""

import os
import json
import threading
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional, Union, Callable

# Common enums used across the scheduler components
class ReportFrequency(str, Enum):
    """Report scheduling frequency options."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"

class ReportFormat(str, Enum):
    """Report output format options."""
    PDF = "pdf"
    HTML = "html"
    CSV = "csv"
    JSON = "json"

class DeliveryMethod(str, Enum):
    """Report delivery method options."""
    EMAIL = "email"
    DOWNLOAD = "download"
    DASHBOARD = "dashboard"

class Report:
    """Base class for all report objects that can be scheduled."""
    
    def __init__(self, 
                report_id: str,
                name: str,
                format: ReportFormat,
                delivery: DeliveryMethod,
                parameters: Optional[Dict[str, Any]] = None,
                created_by: Optional[str] = None):
        """
        Initialize a report.
        
        Args:
            report_id: Unique identifier for the report
            name: Display name for the report
            format: Output format for the report
            delivery: How to deliver the report
            parameters: Additional parameters for report generation
            created_by: Username of the user who created the report
        """
        self.report_id = report_id
        self.name = name
        self.format = format
        self.delivery = delivery
        self.parameters = parameters or {}
        self.created_by = created_by
        self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "report_id": self.report_id,
            "name": self.name,
            "format": self.format,
            "delivery": self.delivery,
            "parameters": self.parameters,
            "created_by": self.created_by,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Report':
        """Create from dictionary."""
        report = cls(
            report_id=data["report_id"],
            name=data["name"],
            format=data["format"],
            delivery=data["delivery"],
            parameters=data.get("parameters", {}),
            created_by=data.get("created_by")
        )
        report.created_at = data.get("created_at", report.created_at)
        return report


class ScheduledReport(Report):
    """Represents a scheduled report configuration."""
    
    def __init__(self, 
                report_id: str,
                name: str,
                template: str,
                frequency: ReportFrequency,
                format: ReportFormat,
                delivery: DeliveryMethod,
                recipients: Optional[List[str]] = None,
                parameters: Optional[Dict[str, Any]] = None,
                created_by: Optional[str] = None):
        """
        Initialize a scheduled report.
        
        Args:
            report_id: Unique identifier for the report
            name: Display name for the report
            template: Report template to use
            frequency: How often to generate the report
            format: Output format for the report
            delivery: How to deliver the report
            recipients: List of email recipients (if delivery is email)
            parameters: Additional parameters for report generation
            created_by: Username of the user who created the report
        """
        super().__init__(report_id, name, format, delivery, parameters, created_by)
        self.template = template
        self.frequency = frequency
        self.recipients = recipients or []
        self.last_run = None
        self.next_run = self._calculate_next_run()
        self.enabled = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = super().to_dict()
        data.update({
            "template": self.template,
            "frequency": self.frequency,
            "recipients": self.recipients,
            "last_run": self.last_run,
            "next_run": self.next_run,
            "enabled": self.enabled
        })
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScheduledReport':
        """Create from dictionary."""
        report = cls(
            report_id=data["report_id"],
            name=data["name"],
            template=data["template"],
            frequency=data["frequency"],
            format=data["format"],
            delivery=data["delivery"],
            recipients=data.get("recipients", []),
            parameters=data.get("parameters", {}),
            created_by=data.get("created_by")
        )
        report.created_at = data.get("created_at", report.created_at)
        report.last_run = data.get("last_run")
        report.next_run = data.get("next_run", report._calculate_next_run())
        report.enabled = data.get("enabled", True)
        return report
    
    def _calculate_next_run(self) -> str:
        """Calculate the next run time based on frequency."""
        from datetime import timedelta
        
        now = datetime.now()
        
        if self.frequency == ReportFrequency.DAILY:
            next_run = now + timedelta(days=1)
            # Set time to 1:00 AM
            next_run = next_run.replace(hour=1, minute=0, second=0, microsecond=0)
        
        elif self.frequency == ReportFrequency.WEEKLY:
            # Next Monday at 1:00 AM
            days_ahead = 7 - now.weekday()
            if days_ahead == 0:
                days_ahead = 7
            next_run = now + timedelta(days=days_ahead)
            next_run = next_run.replace(hour=1, minute=0, second=0, microsecond=0)
        
        elif self.frequency == ReportFrequency.MONTHLY:
            # First day of next month at 1:00 AM
            year = now.year + (1 if now.month == 12 else 0)
            month = 1 if now.month == 12 else now.month + 1
            next_run = datetime(year, month, 1, 1, 0, 0)
        
        elif self.frequency == ReportFrequency.QUARTERLY:
            # First day of next quarter at 1:00 AM
            current_quarter = (now.month - 1) // 3 + 1
            year = now.year + (1 if current_quarter == 4 else 0)
            month = 1 if current_quarter == 4 else 3 * current_quarter + 1
            next_run = datetime(year, month, 1, 1, 0, 0)
        
        else:
            # Default to tomorrow at 1:00 AM
            next_run = now + timedelta(days=1)
            next_run = next_run.replace(hour=1, minute=0, second=0, microsecond=0)
        
        return next_run.isoformat()
    
    def update_next_run(self) -> None:
        """Update the next run time after a report is generated."""
        self.last_run = datetime.now().isoformat()
        self.next_run = self._calculate_next_run()


class BaseScheduler(ABC):
    """Abstract base class for all scheduler implementations."""
    
    def __init__(self, reports_dir: str = None):
        """
        Initialize the scheduler.
        
        Args:
            reports_dir: Directory to store reports and configurations
        """
        self.reports_dir = reports_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                                      "data", "reports")
        self.config_file = os.path.join(self.reports_dir, "scheduled_reports.json")
        self.reports = {}
        self.scheduler_thread = None
        self.stop_event = threading.Event()
        
        # Create directory if it doesn't exist
        os.makedirs(self.reports_dir, exist_ok=True)
    
    @abstractmethod
    def schedule(self, report: Report, frequency: ReportFrequency) -> str:
        """
        Schedule a report to run at the specified frequency.
        
        Args:
            report: The report to schedule
            frequency: How often to run the report
            
        Returns:
            The ID of the scheduled report
        """
        pass
    
    @abstractmethod
    def get_due_reports(self) -> List[ScheduledReport]:
        """
        Get reports that are due to be generated.
        
        Returns:
            List of reports due for generation
        """
        pass
    
    @abstractmethod
    def run_due(self) -> None:
        """Run all reports that are currently due."""
        pass
    
    def load_reports(self) -> None:
        """Load scheduled reports from the configuration file."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    report_data = json.load(f)
                
                for report_id, data in report_data.items():
                    self.reports[report_id] = ScheduledReport.from_dict(data)
                    
                print(f"Loaded {len(self.reports)} scheduled reports")
            except Exception as e:
                print(f"Error loading scheduled reports: {e}")
    
    def save_reports(self) -> None:
        """Save scheduled reports to the configuration file."""
        try:
            report_data = {report_id: report.to_dict() for report_id, report in self.reports.items()}
            
            with open(self.config_file, 'w') as f:
                json.dump(report_data, f, indent=2)
                
            print(f"Saved {len(self.reports)} scheduled reports")
        except Exception as e:
            print(f"Error saving scheduled reports: {e}")
    
    def start(self, interval: int = 60) -> None:
        """
        Start the scheduler in a background thread.
        
        Args:
            interval: Check interval in seconds
        """
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            print("Scheduler already running")
            return
        
        # Reset stop event
        self.stop_event.clear()
        
        # Create and start the thread
        self.scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            args=(interval,),
            daemon=True
        )
        self.scheduler_thread.start()
        
        print(f"Scheduler started with interval {interval} seconds")
    
    def stop(self) -> None:
        """Stop the scheduler thread."""
        if not self.scheduler_thread or not self.scheduler_thread.is_alive():
            print("Scheduler not running")
            return
        
        # Set stop event
        self.stop_event.set()
        
        # Wait for thread to terminate
        self.scheduler_thread.join(timeout=5)
        
        print("Scheduler stopped")
    
    def _scheduler_loop(self, interval: int) -> None:
        """
        Main loop for the scheduler thread.
        
        Args:
            interval: Check interval in seconds
        """
        while not self.stop_event.is_set():
            try:
                # Run all due reports
                self.run_due()
                
                # Wait for next check
                self.stop_event.wait(interval)
            except Exception as e:
                print(f"Error in scheduler loop: {e}")
                # Wait before retrying
                self.stop_event.wait(interval)