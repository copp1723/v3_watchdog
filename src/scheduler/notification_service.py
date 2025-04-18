"""
Notification Service Module for V3 Watchdog AI.

Provides services for delivering reports via various channels.
"""

import os
from typing import List, Dict, Any, Union, Optional
from datetime import datetime
from .base_scheduler import DeliveryMethod, Report

class NotificationService:
    """Service for delivering report notifications and files."""
    
    def __init__(self, reports_dir: str = None):
        """
        Initialize the notification service.
        
        Args:
            reports_dir: Directory to store reports
        """
        self.reports_dir = reports_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                                      "data", "reports")
        # Create directory if it doesn't exist
        os.makedirs(self.reports_dir, exist_ok=True)
    
    def notify(self, report: Report, report_content: Union[str, bytes]) -> str:
        """
        Deliver a report via the configured delivery method.
        
        Args:
            report: Report configuration
            report_content: Formatted report content
            
        Returns:
            Path to the saved report file
        """
        # Create a timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{report.report_id}_{timestamp}"
        
        # Add appropriate extension
        if hasattr(report, 'format'):
            if report.format == "pdf":
                filename += ".pdf"
            elif report.format == "html":
                filename += ".html"
            elif report.format == "csv":
                filename += ".csv"
            elif report.format == "json":
                filename += ".json"
            else:
                filename += ".txt"
        else:
            filename += ".txt"
        
        # Save to the reports directory
        filepath = os.path.join(self.reports_dir, filename)
        
        # Save the file
        mode = "wb" if isinstance(report_content, bytes) else "w"
        with open(filepath, mode) as f:
            f.write(report_content)
        
        print(f"Saved report to {filepath}")
        
        # Handle different delivery methods
        if hasattr(report, 'delivery') and report.delivery == DeliveryMethod.EMAIL:
            self._send_email(report, filepath)
        
        # Return the file path for reference
        return filepath
    
    def _send_email(self, report: Report, file_path: str) -> None:
        """
        Send an email with the report attached.
        
        Args:
            report: Report configuration
            file_path: Path to the report file
        """
        if not hasattr(report, 'recipients') or not report.recipients:
            print(f"Warning: Email delivery selected for report {report.report_id} but no recipients specified.")
            return
        
        # In a real implementation, you'd use a library like smtplib to send an email
        # For this example, we'll just print what would happen
        recipients = ", ".join(report.recipients)
        filename = os.path.basename(file_path)
        
        print(f"Would send email to {recipients} with report {filename} attached.")
        print(f"Email subject: '{report.name} Report'")
        print(f"Email body: 'Please find attached the {report.name} report.'")
        
        # In a real implementation:
        # 1. Create email message with proper headers, body, and attachment
        # 2. Connect to SMTP server
        # 3. Send email
        # 4. Close connection
    
    def send_slack_notification(self, channel: str, message: str, file_path: Optional[str] = None) -> None:
        """
        Send a notification to a Slack channel.
        
        Args:
            channel: Slack channel name
            message: Message to send
            file_path: Optional path to a file to attach
        """
        # In a real implementation, you'd use the Slack API to send a message
        # For this example, we'll just print what would happen
        print(f"Would send Slack message to channel '{channel}': {message}")
        if file_path:
            print(f"Would attach file: {file_path}")
    
    def send_webhook(self, webhook_url: str, payload: Dict[str, Any]) -> None:
        """
        Send data to a webhook endpoint.
        
        Args:
            webhook_url: URL of the webhook endpoint
            payload: Data to send
        """
        # In a real implementation, you'd use a library like requests to send the data
        # For this example, we'll just print what would happen
        print(f"Would send webhook to {webhook_url} with payload: {payload}")