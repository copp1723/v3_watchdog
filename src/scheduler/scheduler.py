"""
Unified Scheduler Module for V3 Watchdog AI.

Provides functionality for scheduling and generating reports at regular intervals,
including saving reports as PDF or sending them via email.
"""

import os
import json
import uuid
import logging
import argparse
import threading
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Callable, Tuple

import pandas as pd
import numpy as np
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger

from .base_scheduler import (
    BaseScheduler, 
    Report,
    ScheduledReport, 
    ReportFrequency, 
    ReportFormat, 
    DeliveryMethod
)
from .notification_service import NotificationService

# Configure logging
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_DATA_DIR = Path("data/reports")
DEFAULT_LOG_DIR = Path("data/logs")
DEFAULT_DB_PATH = Path("data/scheduler.sqlite")


class ReportTemplate:
    """Available report templates."""
    SALES_SUMMARY = "sales_summary"
    INVENTORY_HEALTH = "inventory_health"
    PROFITABILITY = "profitability"
    LEAD_SOURCE = "lead_source"
    CUSTOM = "custom"


class Scheduler(BaseScheduler):
    """
    Unified scheduler for managing report generation and delivery.
    
    This class provides a consolidated implementation for scheduling, generating, 
    and delivering reports based on various frequencies and formats.
    """
    
    def __init__(self, 
                reports_dir: Optional[Union[str, Path]] = None,
                log_dir: Optional[Union[str, Path]] = None,
                db_path: Optional[Union[str, Path]] = None):
        """
        Initialize the unified scheduler.
        
        Args:
            reports_dir: Directory to store reports and configurations. Default data/reports/
            log_dir: Directory for logs. Default data/logs/
            db_path: Path to SQLite DB for job persistence. Default data/scheduler.sqlite
        """
        # Initialize base scheduler
        super().__init__(reports_dir)
        
        # Set up paths from arguments or defaults
        self.reports_dir = Path(reports_dir or DEFAULT_DATA_DIR)
        self.log_dir = Path(log_dir or DEFAULT_LOG_DIR)
        self.db_path = Path(db_path or DEFAULT_DB_PATH)
        
        # Create directories if they don't exist
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize storage
        self.config_file = self.reports_dir / "scheduled_reports.json"
        self.reports = {}
        
        # Initialize notification service
        self.notification_service = NotificationService(self.reports_dir)
        
        # Initialize APScheduler with SQLite job store
        self.scheduler = BackgroundScheduler()
        self.scheduler.add_jobstore(
            SQLAlchemyJobStore(url=f'sqlite:///{self.db_path}'),
            'default'
        )
        
        # Initialize thread management
        self.scheduler_thread = None
        self.stop_event = threading.Event()
        
        # Track if scheduler is running
        self.is_running = False
        
        # Load existing reports
        self.load_reports()
        
        logger.info(f"Initialized Scheduler with reports_dir={self.reports_dir}")
    
    def load_reports(self) -> None:
        """Load scheduled reports from the configuration file."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    report_data = json.load(f)
                
                for report_id, data in report_data.items():
                    self.reports[report_id] = ScheduledReport.from_dict(data)
                    
                logger.info(f"Loaded {len(self.reports)} scheduled reports")
            except Exception as e:
                logger.error(f"Error loading scheduled reports: {e}")
    
    def save_reports(self) -> None:
        """Save scheduled reports to the configuration file."""
        try:
            report_data = {report_id: report.to_dict() for report_id, report in self.reports.items()}
            
            with open(self.config_file, 'w') as f:
                json.dump(report_data, f, indent=2)
                
            logger.info(f"Saved {len(self.reports)} scheduled reports")
        except Exception as e:
            logger.error(f"Error saving scheduled reports: {e}")
    
    def schedule(self, report: ScheduledReport, frequency: ReportFrequency) -> str:
        """
        Schedule a report to run at the specified frequency.
        
        Args:
            report: The report to schedule
            frequency: How often to run the report
            
        Returns:
            The ID of the scheduled report
        """
        # Update the frequency
        report.frequency = frequency
        
        # Recalculate next run time
        report.next_run = report._calculate_next_run()
        
        # Add to reports dictionary
        self.reports[report.report_id] = report
        
        # Register with APScheduler if scheduler is running
        if self.is_running:
            self._register_report_job(report)
        
        # Save to file
        self.save_reports()
        
        logger.info(f"Scheduled report '{report.name}' with frequency {frequency}")
        
        return report.report_id
    
    def create_report(self, 
                     name: str,
                     template: str,
                     frequency: ReportFrequency,
                     format: ReportFormat,
                     delivery: DeliveryMethod,
                     recipients: Optional[List[str]] = None,
                     parameters: Optional[Dict[str, Any]] = None,
                     created_by: Optional[str] = None) -> str:
        """
        Create a new scheduled report.
        
        Args:
            name: Display name for the report
            template: Report template to use
            frequency: How often to generate the report
            format: Output format for the report
            delivery: How to deliver the report
            recipients: List of email recipients (if delivery is email)
            parameters: Additional parameters for report generation
            created_by: Username of the user who created the report
            
        Returns:
            The report ID of the newly created report
        """
        # Generate a unique ID
        report_id = str(uuid.uuid4())
        
        # Create the report
        report = ScheduledReport(
            report_id=report_id,
            name=name,
            template=template,
            frequency=frequency,
            format=format,
            delivery=delivery,
            recipients=recipients,
            parameters=parameters,
            created_by=created_by
        )
        
        # Schedule the report
        return self.schedule(report, frequency)
    
    def update_report(self, 
                     report_id: str,
                     name: Optional[str] = None,
                     template: Optional[str] = None,
                     frequency: Optional[ReportFrequency] = None,
                     format: Optional[ReportFormat] = None,
                     delivery: Optional[DeliveryMethod] = None,
                     recipients: Optional[List[str]] = None,
                     parameters: Optional[Dict[str, Any]] = None,
                     enabled: Optional[bool] = None) -> bool:
        """
        Update an existing scheduled report.
        
        Args:
            report_id: ID of the report to update
            name: Optional new name
            template: Optional new template
            frequency: Optional new frequency
            format: Optional new format
            delivery: Optional new delivery method
            recipients: Optional new recipients
            parameters: Optional new parameters
            enabled: Optional new enabled state
            
        Returns:
            True if report was updated, False if report not found
        """
        if report_id not in self.reports:
            return False
        
        report = self.reports[report_id]
        
        # Update fields if provided
        if name is not None:
            report.name = name
        
        if template is not None:
            report.template = template
        
        if frequency is not None:
            report.frequency = frequency
            # Recalculate next run time
            report.next_run = report._calculate_next_run()
        
        if format is not None:
            report.format = format
        
        if delivery is not None:
            report.delivery = delivery
        
        if recipients is not None:
            report.recipients = recipients
        
        if parameters is not None:
            report.parameters = parameters
        
        if enabled is not None:
            report.enabled = enabled
        
        # Save to file
        self.save_reports()
        
        # Update job in scheduler if running
        if self.is_running and report_id in self.scheduler.get_jobs():
            if enabled is False:
                # Remove job if disabled
                self.scheduler.remove_job(report_id)
            else:
                # Reschedule job with updated parameters
                self._register_report_job(report)
        
        logger.info(f"Updated report {report_id}")
        return True
    
    def delete_report(self, report_id: str) -> bool:
        """
        Delete a scheduled report.
        
        Args:
            report_id: ID of the report to delete
            
        Returns:
            True if report was deleted, False if report not found
        """
        if report_id not in self.reports:
            return False
        
        # Remove from reports dictionary
        del self.reports[report_id]
        
        # Remove from scheduler if running
        if self.is_running and report_id in [job.id for job in self.scheduler.get_jobs()]:
            self.scheduler.remove_job(report_id)
        
        # Save to file
        self.save_reports()
        
        logger.info(f"Deleted report {report_id}")
        return True
    
    def get_report(self, report_id: str) -> Optional[ScheduledReport]:
        """
        Get a report by ID.
        
        Args:
            report_id: ID of the report to get
            
        Returns:
            The report if found, None otherwise
        """
        return self.reports.get(report_id)
    
    def get_all_reports(self) -> List[ScheduledReport]:
        """
        Get all scheduled reports.
        
        Returns:
            List of all scheduled reports
        """
        return list(self.reports.values())
    
    def get_reports_by_user(self, username: str) -> List[ScheduledReport]:
        """
        Get reports created by a specific user.
        
        Args:
            username: Username to filter by
            
        Returns:
            List of reports created by the user
        """
        return [r for r in self.reports.values() if r.created_by == username]
    
    def get_due_reports(self) -> List[ScheduledReport]:
        """
        Get reports that are due to be generated.
        
        Returns:
            List of reports due for generation
        """
        now = datetime.now().isoformat()
        return [r for r in self.reports.values() 
                if r.enabled and r.next_run <= now]
    
    def run_due(self) -> None:
        """Run all reports that are currently due."""
        # Get due reports
        due_reports = self.get_due_reports()
        
        # Process each due report
        for report in due_reports:
            try:
                logger.info(f"Generating report: {report.name} ({report.report_id})")
                self._generate_report(report)
                
                # Update next run time
                report.update_next_run()
                self.save_reports()
            except Exception as e:
                logger.error(f"Error generating report {report.report_id}: {e}")
                logger.debug(traceback.format_exc())
    
    def run_now(self, report_id: str) -> Optional[Dict[str, Any]]:
        """
        Run a specific report immediately.
        
        Args:
            report_id: ID of the report to run
            
        Returns:
            Dict with job status or None if report not found
        """
        report = self.reports.get(report_id)
        if not report:
            logger.error(f"Report not found: {report_id}")
            return None
        
        try:
            logger.info(f"Running report {report_id} immediately")
            self._generate_report(report)
            
            # Update last run time but don't change next scheduled run
            report.last_run = datetime.now().isoformat()
            self.save_reports()
            
            return {
                "status": "success",
                "report_id": report_id,
                "name": report.name,
                "executed_at": report.last_run
            }
        except Exception as e:
            logger.error(f"Error running report {report_id}: {e}")
            logger.debug(traceback.format_exc())
            
            return {
                "status": "error",
                "report_id": report_id,
                "name": report.name,
                "error": str(e),
                "executed_at": datetime.now().isoformat()
            }
    
    def start(self, interval: int = 60) -> None:
        """
        Start the scheduler.
        
        Args:
            interval: Check interval in seconds for polling mode (default: 60)
        """
        if self.is_running:
            logger.warning("Scheduler already running")
            return
        
        # Reset stop event
        self.stop_event.clear()
        
        # Register all jobs
        self._register_jobs()
        
        # Start the APScheduler
        self.scheduler.start()
        
        # Create and start the polling thread for reports not using APScheduler
        self.scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            args=(interval,),
            daemon=True
        )
        self.scheduler_thread.start()
        
        self.is_running = True
        logger.info(f"Scheduler started with poll interval {interval} seconds")
    
    def stop(self) -> None:
        """Stop the scheduler."""
        if not self.is_running:
            logger.warning("Scheduler not running")
            return
        
        # Set stop event for polling thread
        self.stop_event.set()
        
        # Wait for thread to terminate
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5)
        
        # Shutdown APScheduler
        self.scheduler.shutdown()
        
        self.is_running = False
        logger.info("Scheduler stopped")
    
    def list_jobs(self) -> List[Dict[str, Any]]:
        """
        List all scheduled jobs.
        
        Returns:
            List of job information dictionaries
        """
        jobs = []
        
        # Get jobs from APScheduler
        # Get jobs from APScheduler
        if self.is_running:
            for job in self.scheduler.get_jobs():
                next_run = job.next_run_time.isoformat() if job.next_run_time else None
                report = self.reports.get(job.id)
                
                if report:
                    jobs.append({
                        "id": job.id,
                        "name": report.name,
                        "template": report.template,
                        "frequency": report.frequency,
                        "next_run": next_run,
                        "last_run": report.last_run,
                        "enabled": report.enabled,
                        "format": report.format,
                        "delivery": report.delivery
                    })
        
        # Add reports not yet registered with APScheduler
        for report_id, report in self.reports.items():
            if not any(j.get("id") == report_id for j in jobs):
                jobs.append({
                    "id": report_id,
                    "name": report.name,
                    "template": report.template,
                    "frequency": report.frequency,
                    "next_run": report.next_run,
                    "last_run": report.last_run,
                    "enabled": report.enabled,
                    "format": report.format,
                    "delivery": report.delivery
                })
        
        return jobs
    
    def _register_jobs(self) -> None:
        """Register all active reports with APScheduler."""
        for report in self.reports.values():
            if report.enabled:
                self._register_report_job(report)
    
    def _register_report_job(self, report: ScheduledReport) -> None:
        """
        Register a single report with APScheduler.
        
        Args:
            report: Report to register
        """
        trigger = self._build_trigger(report.frequency)
        
        self.scheduler.add_job(
            func=self._generate_report,
            args=[report],
            trigger=trigger,
            id=report.report_id,
            name=report.name,
            replace_existing=True
        )
        
        logger.debug(f"Registered job for report: {report.name}")
    
    def _build_trigger(self, frequency: ReportFrequency) -> Union[CronTrigger, IntervalTrigger]:
        """
        Build appropriate scheduler trigger based on frequency.
        
        Args:
            frequency: Report frequency
            
        Returns:
            APScheduler trigger object
        """
        if frequency == ReportFrequency.DAILY:
            return CronTrigger(hour=1, minute=0)  # 1:00 AM
            
        elif frequency == ReportFrequency.WEEKLY:
            return CronTrigger(day_of_week=0, hour=1, minute=0)  # Monday 1:00 AM
            
        elif frequency == ReportFrequency.MONTHLY:
            return CronTrigger(day=1, hour=1, minute=0)  # 1st of month 1:00 AM
            
        elif frequency == ReportFrequency.QUARTERLY:
            return CronTrigger(
                month='1,4,7,10',  # Jan, Apr, Jul, Oct
                day=1,
                hour=1,
                minute=0
            )
        else:
            # Default to daily at 1:00 AM
            return CronTrigger(hour=1, minute=0)
    
    def _scheduler_loop(self, interval: int) -> None:
        """
        Main loop for the scheduler thread.
        
        Args:
            interval: Sleep interval in seconds
        """
        while not self.stop_event.is_set():
            try:
                # Run all due reports
                self.run_due()
                
                # Sleep until next check
                self.stop_event.wait(interval)
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                logger.debug(traceback.format_exc())
                # Wait before retrying
                self.stop_event.wait(min(interval, 30))
    
    def _generate_report(self, report: ScheduledReport) -> None:
        """
        Generate a report based on its configuration.
        
        Args:
            report: The report to generate
            
        Raises:
            Exception: If report generation or delivery fails
        """
        try:
            # Load data based on template
            df = self._load_data_for_template(report.template, report.parameters)
            
            # Generate report content
            content = self._render_report_content(df, report)
            
            # Format report
            formatted_report = self._format_report(content, report.format)
            
            # Deliver report
            self._deliver_report(formatted_report, report)
            
            # Update report metadata
            report.last_run = datetime.now().isoformat()
            report.next_run = report._calculate_next_run()
            self.save_reports()
            
            logger.info(f"Successfully generated report: {report.name}")
            
        except Exception as e:
            logger.error(f"Error generating report {report.report_id}: {e}")
            logger.debug(traceback.format_exc())
            raise
    
    def _format_report(self, content: Dict[str, Any], format: ReportFormat) -> Union[str, bytes]:
        """
        Format report content based on the specified format.
        
        Args:
            content: Report content dictionary
            format: Desired output format
            
        Returns:
            Formatted report as string or bytes
        """
        if format == ReportFormat.JSON:
            return json.dumps(content, indent=2)
        
        elif format == ReportFormat.CSV:
            return self._format_csv(content)
        
        elif format == ReportFormat.HTML:
            return self._format_html(content)
        
        elif format == ReportFormat.PDF:
            return self._format_pdf(content)
        
        else:
            return self._format_text(content)
    
    def _format_csv(self, content: Dict[str, Any]) -> str:
        """Format report content as CSV."""
        buffer = []
        
        # Add report header
        buffer.extend([
            f"Report: {content.get('title', 'Untitled')}",
            f"Generated: {content.get('generated_at', '')}",
            f"Summary: {content.get('summary', '')}",
            ""
        ])
        
        # Add tables
        for table in content.get("tables", []):
            # Add table header
            buffer.append(f"# {table.get('title', 'Untitled')}")
            
            # Convert to DataFrame and write CSV
            if table.get("data"):
                df = pd.DataFrame(table["data"])
                buffer.append(df.to_csv(index=False))
            
            # Add separator
            buffer.append("")
        
        return "\n".join(buffer)
    
    def _format_html(self, content: Dict[str, Any]) -> str:
        """Format report content as HTML."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{content.get('title', 'Report')}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                .report-summary {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .chart-container {{ margin-bottom: 30px; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 30px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .footer {{ font-size: 12px; color: #777; margin-top: 50px; text-align: center; }}
            </style>
        </head>
        <body>
            <h1>{content.get('title', 'Report')}</h1>
            <p>Generated: {datetime.fromisoformat(content.get('generated_at', datetime.now().isoformat())).strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="report-summary">
                <h2>Summary</h2>
                <p>{content.get('summary', 'No summary available.')}</p>
            </div>
        """
        
        # Add charts
        for chart in content.get("charts", []):
            html += f"""
            <div class="chart-container">
                <h2>{chart.get('title', 'Chart')}</h2>
                <img src="data:image/png;base64,{chart.get('image', '')}" alt="{chart.get('title', 'Chart')}">
            </div>
            """
        
        # Add tables
        for table in content.get("tables", []):
            html += f"""
            <div class="table-container">
                <h2>{table.get('title', 'Table')}</h2>
                <table>
                    <thead>
                        <tr>
            """
            
            # Add headers
            columns = table.get("columns", [])
            for col in columns:
                html += f"<th>{col}</th>"
            
            html += "</tr></thead><tbody>"
            
            # Add rows
            for row in table.get("data", []):
                html += "<tr>"
                for col in columns:
                    value = row.get(col, "")
                    # Format numbers
                    if isinstance(value, (int, float)):
                        if col.lower().endswith(('gross', 'price', 'cost', 'roi')):
                            html += f"<td>${value:,.2f}</td>"
                        else:
                            html += f"<td>{value:,}</td>"
                    else:
                        html += f"<td>{value}</td>"
                html += "</tr>"
            
            html += "</tbody></table></div>"
        
        # Add footer
        html += f"""
            <div class="footer">
                <p>Report ID: {content.get('metadata', {}).get('report_id', 'N/A')}</p>
                <p>Created by: {content.get('metadata', {}).get('created_by', 'System')}</p>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _format_pdf(self, content: Dict[str, Any]) -> bytes:
        """
        Format report content as PDF.
        
        Args:
            content: Report content dictionary
            
        Returns:
            PDF document as bytes
        """
        # First create HTML content
        html_content = self._format_html(content)
        
        try:
            # Try to use weasyprint if available
            from weasyprint import HTML
            pdf_bytes = HTML(string=html_content).write_pdf()
            return pdf_bytes
        except ImportError:
            # Fallback to HTML if weasyprint is not available
            logger.warning("WeasyPrint not installed. Falling back to HTML format.")
            return html_content.encode('utf-8')
    
    def _format_text(self, content: Dict[str, Any]) -> str:
        """Format report content as plain text."""
        lines = [
            f"Report: {content.get('title', 'Untitled')}",
            f"Generated: {content.get('generated_at', datetime.now().isoformat())}",
            "",
            "Summary:",
            content.get('summary', 'No summary available.'),
            "",
            "=" * 80,
            ""
        ]
        
        # Add tables
        for table in content.get("tables", []):
            lines.extend([
                f"Table: {table.get('title', 'Untitled')}",
                "-" * 80
            ])
            
            if table.get("data"):
                df = pd.DataFrame(table["data"])
                lines.append(df.to_string(index=False))
            
            lines.extend(["", "=" * 80, ""])
        
        return "\n".join(lines)
    
    def _deliver_report(self, report_content: Union[str, bytes], report: ScheduledReport) -> None:
        """
        Deliver the report via the configured delivery method.
        
        Args:
            report_content: Formatted report content
            report: Report configuration
        """
        # Create a timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{report.report_id}_{timestamp}"
        
        # Add appropriate extension
        if report.format == ReportFormat.PDF:
            filename += ".pdf"
        elif report.format == ReportFormat.HTML:
            filename += ".html"
        elif report.format == ReportFormat.CSV:
            filename += ".csv"
        elif report.format == ReportFormat.JSON:
            filename += ".json"
        else:
            filename += ".txt"
        
        # Save to the reports directory
        filepath = self.reports_dir / filename
        
        # Save the file
        mode = "wb" if isinstance(report_content, bytes) else "w"
        with open(filepath, mode) as f:
            f.write(report_content)
        
        logger.info(f"Saved report to {filepath}")
        
        # Handle delivery based on method
        if report.delivery == DeliveryMethod.EMAIL:
            if not report.recipients:
                logger.warning(f"Email delivery selected for report {report.report_id} but no recipients specified")
                return
            
            try:
                self.notification_service.send_email(
                    recipients=report.recipients,
                    subject=f"Report: {report.name}",
                    content=report_content,
                    attachment_path=filepath
                )
                logger.info(f"Sent report via email to {', '.join(report.recipients)}")
            except Exception as e:
                logger.error(f"Failed to send report via email: {e}")
                raise
        
        elif report.delivery == DeliveryMethod.DASHBOARD:
            # Save to dashboard location
            dashboard_path = self.reports_dir / "dashboard" / report.template
            dashboard_path.mkdir(parents=True, exist_ok=True)
            
            # Copy to dashboard with latest marker
            latest_path = dashboard_path / f"latest_{report.template}.{filename.split('.')[-1]}"
            
            import shutil
            shutil.copy2(filepath, dashboard_path / filename)
            shutil.copy2(filepath, latest_path)
            
            logger.info(f"Published report to dashboard: {latest_path}")
        
        # For DOWNLOAD delivery method, the file is already saved to the reports directory

    def _load_data_for_template(self, template: str, parameters: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Load appropriate data for a report template.
        
        Args:
            template: Report template to load data for
            parameters: Optional parameters to customize the data load
            
        Returns:
            DataFrame with data for the report
        """
        # Initialize parameters to empty dict if None
        parameters = parameters or {}
        
        # In a real implementation, this would load data from a database
        # For this example, we'll generate sample data based on template
        if template == ReportTemplate.SALES_SUMMARY:
            # Generate sample sales data
            dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
            np.random.seed(42)  # For reproducibility
            
            n_sales = 500
            sales_indices = np.random.choice(len(dates), size=n_sales, replace=True)
            
            df = pd.DataFrame({
                'Sale_Date': dates[sales_indices],
                'VIN': [f'VIN{i:06d}' for i in range(n_sales)],
                'Gross_Profit': np.random.normal(2000, 800, n_sales),
                'LeadSource': np.random.choice(['Website', 'Walk-in', 'Referral', 'Third-party', 'Unknown'], n_sales),
                'VehicleMake': np.random.choice(['Toyota', 'Honda', 'Ford', 'Chevrolet', 'BMW', 'Mercedes'], n_sales),
                'VehicleModel': np.random.choice(['Sedan', 'SUV', 'Truck', 'Coupe'], n_sales)
            })
        
        elif template == ReportTemplate.INVENTORY_HEALTH:
            # Generate sample inventory data
            np.random.seed(42)  # For reproducibility
            
            df = pd.DataFrame({
                'VIN': [f'INV{i:06d}' for i in range(200)],
                'DaysInInventory': np.random.exponential(scale=45, size=200),
                'VehicleMake': np.random.choice(['Toyota', 'Honda', 'Ford', 'Chevrolet', 'BMW', 'Mercedes'], 200),
                'VehicleModel': np.random.choice(['Sedan', 'SUV', 'Truck', 'Coupe'], 200)
            })
        
        elif template == ReportTemplate.PROFITABILITY:
            # Generate sample profitability data
            dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
            np.random.seed(42)  # For reproducibility
            
            n_sales = 500
            sales_indices = np.random.choice(len(dates), size=n_sales, replace=True)
            
            df = pd.DataFrame({
                'Sale_Date': dates[sales_indices],
                'VIN': [f'VIN{i:06d}' for i in range(n_sales)],
                'Front_Gross': np.random.normal(1500, 700, n_sales),
                'Back_Gross': np.random.normal(800, 400, n_sales),
                'Total_Gross': np.random.normal(2300, 900, n_sales),
                'VehicleMake': np.random.choice(['Toyota', 'Honda', 'Ford', 'Chevrolet', 'BMW', 'Mercedes'], n_sales),
                'VehicleModel': np.random.choice(['Sedan', 'SUV', 'Truck', 'Coupe'], n_sales)
            })
        
        elif template == ReportTemplate.LEAD_SOURCE:
            # Generate sample lead source data
            dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
            np.random.seed(42)  # For reproducibility
            
            n_sales = 500
            sales_indices = np.random.choice(len(dates), size=n_sales, replace=True)
            
            df = pd.DataFrame({
                'Sale_Date': dates[sales_indices],
                'VIN': [f'VIN{i:06d}' for i in range(n_sales)],
                'Gross_Profit': np.random.normal(2000, 800, n_sales),
                'LeadSource': np.random.choice(['Website', 'Walk-in', 'Referral', 'Third-party', 'Unknown'], n_sales, 
                                              p=[0.35, 0.25, 0.15, 0.2, 0.05]),  # Custom probabilities
                'LeadCost': np.random.uniform(50, 500, n_sales),
                'VehicleMake': np.random.choice(['Toyota', 'Honda', 'Ford', 'Chevrolet', 'BMW', 'Mercedes'], n_sales)
            })
        
        else:
            # Return an empty DataFrame for unknown templates
            return pd.DataFrame()
        
        # Apply any filtering from parameters
        if parameters:
            # Filter by date range if specified
            if 'date_from' in parameters and 'date_to' in parameters and 'Sale_Date' in df.columns:
                try:
                    date_from = pd.to_datetime(parameters['date_from'])
                    date_to = pd.to_datetime(parameters['date_to'])
                    df = df[(df['Sale_Date'] >= date_from) & (df['Sale_Date'] <= date_to)]
                except (ValueError, KeyError):
                    logger.warning(f"Invalid date parameters: {parameters.get('date_from')} - {parameters.get('date_to')}")
            
            # Filter by make if specified
            if 'make' in parameters and 'VehicleMake' in df.columns:
                df = df[df['VehicleMake'] == parameters['make']]
        
        return df
    
    def _render_report_content(self, df: pd.DataFrame, 
                             report: ScheduledReport) -> Dict[str, Any]:
        """
        Render the content for a report.
        
        Args:
            df: DataFrame with data for the report
            report: Report configuration
            
        Returns:
            Dictionary with report content
        """
        content = {
            "title": report.name,
            "generated_at": datetime.now().isoformat(),
            "template": report.template,
            "summary": "",
            "charts": [],
            "tables": [],
            "metadata": {
                "report_id": report.report_id,
                "frequency": report.frequency,
                "created_by": report.created_by
            }
        }
        
        if df.empty:
            content["summary"] = "No data available for this report."
            return content
        
        # Generate summary and charts based on template
        if report.template == ReportTemplate.SALES_SUMMARY:
            # Add summary
            total_sales = len(df)
            total_gross = df['Gross_Profit'].sum()
            avg_gross = df['Gross_Profit'].mean()
            
            content["summary"] = f"Sales Summary: {total_sales} total sales with average gross of ${avg_gross:.2f} and total gross of ${total_gross:.2f}."
            
            # Add monthly sales table
            if 'Sale_Date' in df.columns:
                df['Month'] = df['Sale_Date'].dt.to_period('M').astype(str)
                monthly_summary = df.groupby('Month').agg(
                    Sales=('VIN', 'count'),
                    AvgGross=('Gross_Profit', 'mean'),
                    TotalGross=('Gross_Profit', 'sum')
                ).reset_index()
                
                content["tables"].append({
                    "title": "Monthly Sales Summary",
                    "columns": ["Month", "Sales", "Avg Gross", "Total Gross"],
                    "data": monthly_summary.to_dict('records')
                })
            
            # Add lead source table
            if 'LeadSource' in df.columns:
                lead_summary = df.groupby('LeadSource').agg(
                    Sales=('VIN', 'count'),
                    AvgGross=('Gross_Profit', 'mean'),
                    TotalGross=('Gross_Profit', 'sum')
                ).reset_index()
                
                content["tables"].append({
                    "title": "Sales by Lead Source",
                    "columns": ["LeadSource", "Sales", "Avg Gross", "Total Gross"],
                    "data": lead_summary.to_dict('records')
                })
            
            # Add sales by make table
            if 'VehicleMake' in df.columns:
                make_summary = df.groupby('VehicleMake').agg(
                    Sales=('VIN', 'count'),
                    AvgGross=('Gross_Profit', 'mean'),
                    TotalGross=('Gross_Profit', 'sum')
                ).reset_index()
                
                content["tables"].append({
                    "title": "Sales by Vehicle Make",
                    "columns": ["VehicleMake", "Sales", "Avg Gross", "Total Gross"],
                    "data": make_summary.to_dict('records')
                })
        
        elif report.template == ReportTemplate.INVENTORY_HEALTH:
            # Add summary
            total_units = len(df)
            avg_days = df['DaysInInventory'].mean()
            aged_90_plus = (df['DaysInInventory'] > 90).sum()
            aged_pct = (aged_90_plus / total_units) * 100
            
            content["summary"] = f"Inventory Health: {total_units} total units with average age of {avg_days:.1f} days. {aged_90_plus} units ({aged_pct:.1f}%) are over 90 days old."
            
            # Add age distribution table
            df['Age Category'] = pd.cut(
                df['DaysInInventory'],
                bins=[0, 30, 60, 90, float('inf')],
                labels=['<30 days', '30-60 days', '61-90 days', '>90 days']
            )
            age_summary = df.groupby('Age Category').agg(
                Units=('VIN', 'count'),
                AvgDays=('DaysInInventory', 'mean')
            ).reset_index()
            
            content["tables"].append({
                "title": "Inventory Age Summary",
                "columns": ["Age Category", "Units", "Avg Days"],
                "data": age_summary.to_dict('records')
            })
            
            # Add make summary
            if 'VehicleMake' in df.columns:
                make_summary = df.groupby('VehicleMake').agg(
                    Units=('VIN', 'count'),
                    AvgDays=('DaysInInventory', 'mean')
                ).reset_index()
                
                content["tables"].append({
                    "title": "Inventory by Make",
                    "columns": ["VehicleMake", "Units", "Avg Days"],
                    "data": make_summary.to_dict('records')
                })
        
        elif report.template == ReportTemplate.PROFITABILITY:
            # Add summary
            total_sales = len(df)
            total_front = df['Front_Gross'].sum()
            total_back = df['Back_Gross'].sum()
            total_gross = df['Total_Gross'].sum()
            avg_front = df['Front_Gross'].mean()
            avg_back = df['Back_Gross'].mean()
            avg_total = df['Total_Gross'].mean()
            
            content["summary"] = (
                f"Profitability Summary: {total_sales} total sales with "
                f"average front gross of ${avg_front:.2f}, back gross of ${avg_back:.2f}, "
                f"and total gross of ${avg_total:.2f}."
            )
            
            # Add monthly summary
            if 'Sale_Date' in df.columns:
                df['Month'] = df['Sale_Date'].dt.to_period('M').astype(str)
                monthly_summary = df.groupby('Month').agg(
                    Sales=('VIN', 'count'),
                    AvgFrontGross=('Front_Gross', 'mean'),
                    TotalFrontGross=('Front_Gross', 'sum'),
                    AvgBackGross=('Back_Gross', 'mean'),
                    TotalBackGross=('Back_Gross', 'sum'),
                    AvgTotalGross=('Total_Gross', 'mean'),
                    TotalGross=('Total_Gross', 'sum')
                ).reset_index()
                
                content["tables"].append({
                    "title": "Monthly Profitability",
                    "columns": [col for col in monthly_summary.columns],
                    "data": monthly_summary.to_dict('records')
                })
            
            # Add make summary
            if 'VehicleMake' in df.columns:
                make_summary = df.groupby('VehicleMake').agg(
                    Sales=('VIN', 'count'),
                    AvgFrontGross=('Front_Gross', 'mean'),
                    TotalFrontGross=('Front_Gross', 'sum'),
                    AvgBackGross=('Back_Gross', 'mean'),
                    TotalBackGross=('Back_Gross', 'sum'),
                    AvgTotalGross=('Total_Gross', 'mean'),
                    TotalGross=('Total_Gross', 'sum')
                ).reset_index()
                
                content["tables"].append({
                    "title": "Profitability by Make",
                    "columns": [col for col in make_summary.columns],
                    "data": make_summary.to_dict('records')
                })
        
        elif report.template == ReportTemplate.LEAD_SOURCE:
            # Add summary
            total_sales = len(df)
            total_gross = df['Gross_Profit'].sum()
            total_cost = df['LeadCost'].sum()
            roi = ((total_gross - total_cost) / total_cost) * 100 if total_cost > 0 else 0
            
            content["summary"] = (
                f"Lead Source Summary: {total_sales} total sales with "
                f"${total_gross:.2f} gross profit and ${total_cost:.2f} lead cost. "
                f"ROI: {roi:.1f}%"
            )
            
            # Add lead source summary
            lead_summary = df.groupby('LeadSource').agg(
                Sales=('VIN', 'count'),
                AvgGross=('Gross_Profit', 'mean'),
                TotalGross=('Gross_Profit', 'sum'),
                AvgCost=('LeadCost', 'mean'),
                TotalCost=('LeadCost', 'sum')
            ).reset_index()
            
            # Calculate ROI per lead source
            lead_summary['ROI'] = ((lead_summary['TotalGross'] - lead_summary['TotalCost']) / 
                                lead_summary['TotalCost'] * 100).fillna(0)
            
            content["tables"].append({
                "title": "Performance by Lead Source",
                "columns": [col for col in lead_summary.columns],
                "data": lead_summary.to_dict('records')
            })
            
            # Add monthly trend
            if 'Sale_Date' in df.columns:
                df['Month'] = df['Sale_Date'].dt.to_period('M').astype(str)
                monthly_summary = df.groupby(['Month', 'LeadSource']).agg(
                    Sales=('VIN', 'count'),
                    TotalGross=('Gross_Profit', 'sum'),
                    TotalCost=('LeadCost', 'sum')
                ).reset_index()
                
                monthly_summary['ROI'] = ((monthly_summary['TotalGross'] - monthly_summary['TotalCost']) / 
                                        monthly_summary['TotalCost'] * 100).fillna(0)
                
                content["tables"].append({
                    "title": "Monthly Lead Source Performance",
                    "columns": [col for col in monthly_summary.columns],
                    "data": monthly_summary.to_dict('records')
                })
        
        return content


# Command-line interface functions
def main():
    """CLI entry point for the scheduler."""
    parser = argparse.ArgumentParser(description="Report Scheduler CLI")
    
    # Global arguments
    parser.add_argument("--reports-dir", help="Custom reports directory")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Start command
    start_parser = subparsers.add_parser("start", help="Start the scheduler")
    start_parser.add_argument("--interval", type=int, default=60, help="Poll interval in seconds")
    
    # Stop command
    subparsers.add_parser("stop", help="Stop the scheduler")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List scheduled reports")
    list_parser.add_argument("--user", help="Filter by username")
    
    # Create command
    create_parser = subparsers.add_parser("create", help="Create a new report")
    create_parser.add_argument("--name", required=True, help="Report name")
    create_parser.add_argument("--template", required=True, choices=[t for t in dir(ReportTemplate) if not t.startswith('_')], 
                             help="Report template")
    create_parser.add_argument("--frequency", required=True, choices=[f.value for f in ReportFrequency], 
                             help="Report frequency")
    create_parser.add_argument("--format", required=True, choices=[f.value for f in ReportFormat], 
                             help="Output format")
    create_parser.add_argument("--delivery", required=True, choices=[d.value for d in DeliveryMethod], 
                             help="Delivery method")
    create_parser.add_argument("--recipients", help="Email recipients (comma-separated)")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run a report immediately")
    run_parser.add_argument("report_id", help="ID of the report to run")
    
    # Update command
    update_parser = subparsers.add_parser("update", help="Update an existing report")
    update_parser.add_argument("report_id", help="ID of the report to update")
    update_parser.add_argument("--name", help="New report name")
    update_parser.add_argument("--template", choices=[t for t in dir(ReportTemplate) if not t.startswith('_')], 
                             help="New report template")
    update_parser.add_argument("--frequency", choices=[f.value for f in ReportFrequency], 
                             help="New report frequency")
    update_parser.add_argument("--format", choices=[f.value for f in ReportFormat], 
                             help="New output format")
    update_parser.add_argument("--delivery", choices=[d.value for d in DeliveryMethod], 
                             help="New delivery method")
    update_parser.add_argument("--recipients", help="New email recipients (comma-separated)")
    update_parser.add_argument("--enabled", type=bool, help="Enable/disable the report")
    
    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a report")
    delete_parser.add_argument("report_id", help="ID of the report to delete")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create scheduler instance
    scheduler = Scheduler(reports_dir=args.reports_dir)
    
    try:
        if args.command == "start":
            scheduler.start(interval=args.interval)
            print(f"Scheduler started with {args.interval}s interval. Press Ctrl+C to stop.")
            
            # Keep running until interrupted
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nStopping scheduler...")
                scheduler.stop()
                
        elif args.command == "stop":
            scheduler.stop()
            print("Scheduler stopped.")
            
        elif args.command == "list":
            jobs = scheduler.list_jobs()
            if args.user:
                jobs = [j for j in jobs if j.get("created_by") == args.user]
                
            if not jobs:
                print("No scheduled reports found.")
                return
                
            # Print job information
            print("\nScheduled Reports:")
            print("-" * 80)
            for job in jobs:
                print(f"ID: {job['id']}")
                print(f"Name: {job['name']}")
                print(f"Template: {job['template']}")
                print(f"Frequency: {job['frequency']}")
                print(f"Next Run: {job['next_run']}")
                print(f"Format: {job['format']}")
                print(f"Delivery: {job['delivery']}")
                print(f"Enabled: {job['enabled']}")
                print("-" * 80)
                
        elif args.command == "create":
            # Parse recipients if provided
            recipients = args.recipients.split(",") if args.recipients else None
            
            # Create the report
            report_id = scheduler.create_report(
                name=args.name,
                template=getattr(ReportTemplate, args.template, args.template),
                frequency=args.frequency,
                format=args.format,
                delivery=args.delivery,
                recipients=recipients
            )
            
            print(f"Created report with ID: {report_id}")
            
        elif args.command == "run":
            result = scheduler.run_now(args.report_id)
            if result:
                print(f"Report execution {'succeeded' if result['status'] == 'success' else 'failed'}")
                for key, value in result.items():
                    print(f"{key}: {value}")
            else:
                print(f"Report {args.report_id} not found.")
        
        elif args.command == "update":
            # Parse recipients if provided
            recipients = args.recipients.split(",") if args.recipients else None
            
            # Build update kwargs
            update_kwargs = {}
            if args.name:
                update_kwargs['name'] = args.name
            if args.template:
                update_kwargs['template'] = getattr(ReportTemplate, args.template, args.template)
            if args.frequency:
                update_kwargs['frequency'] = args.frequency
            if args.format:
                update_kwargs['format'] = args.format
            if args.delivery:
                update_kwargs['delivery'] = args.delivery
            if recipients is not None:
                update_kwargs['recipients'] = recipients
            if args.enabled is not None:
                update_kwargs['enabled'] = args.enabled
                
            # Update the report
            if scheduler.update_report(args.report_id, **update_kwargs):
                print(f"Updated report {args.report_id}")
            else:
                print(f"Report {args.report_id} not found.")
                
        elif args.command == "delete":
            if scheduler.delete_report(args.report_id):
                print(f"Deleted report {args.report_id}")
            else:
                print(f"Report {args.report_id} not found.")
                
        else:
            parser.print_help()
            
    except Exception as e:
        print(f"Error: {e}")
        logger.error(f"CLI error: {e}")
        logger.debug(traceback.format_exc())
        sys.exit(1)


def create_scheduler(reports_dir: Optional[str] = None) -> Scheduler:
    """
    Create and initialize a scheduler instance.
    
    Args:
        reports_dir: Optional custom reports directory
        
    Returns:
        Initialized scheduler instance
    """
    scheduler = Scheduler(reports_dir=reports_dir)
    return scheduler


if __name__ == "__main__":
    main()
