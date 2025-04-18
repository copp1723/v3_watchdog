"""
Unit tests for the notification service module.
"""

import os
import uuid
import pytest
from unittest.mock import patch, MagicMock

from src.scheduler.notification_service import NotificationService
from src.scheduler.base_scheduler import Report, DeliveryMethod


class TestNotificationService:
    """Tests for the NotificationService class."""
    
    @pytest.fixture
    def temp_dir(self, tmpdir):
        """Fixture to create a temporary directory."""
        return str(tmpdir)
    
    @pytest.fixture
    def service(self, temp_dir):
        """Fixture to create a notification service instance."""
        return NotificationService(reports_dir=temp_dir)
    
    @pytest.fixture
    def report(self):
        """Fixture to create a test report."""
        report = Report(
            report_id=str(uuid.uuid4()),
            name="Test Report",
            format="pdf",
            delivery=DeliveryMethod.EMAIL,
            recipients=["user@example.com"],
            created_by="test_user"
        )
        
        # Add recipients attribute (not in base Report class)
        report.recipients = ["user@example.com"]
        
        return report
    
    def test_init(self, service, temp_dir):
        """Test service initialization."""
        assert service.reports_dir == temp_dir
        assert os.path.exists(temp_dir)
    
    def test_notify_text_content(self, service, report, temp_dir):
        """Test notify with text content."""
        content = "Test report content"
        filepath = service.notify(report, content)
        
        assert os.path.exists(filepath)
        assert filepath.startswith(temp_dir)
        assert report.report_id in filepath
        
        with open(filepath, 'r') as f:
            saved_content = f.read()
            assert saved_content == content
    
    def test_notify_binary_content(self, service, report, temp_dir):
        """Test notify with binary content."""
        content = b"Binary test content"
        filepath = service.notify(report, content)
        
        assert os.path.exists(filepath)
        assert filepath.startswith(temp_dir)
        assert report.report_id in filepath
        
        with open(filepath, 'rb') as f:
            saved_content = f.read()
            assert saved_content == content
    
    @patch('builtins.print')
    def test_send_email(self, mock_print, service, report):
        """Test sending email notification."""
        file_path = "/path/to/report.pdf"
        
        service._send_email(report, file_path)
        
        # Check that appropriate print statements were made
        mock_print.assert_any_call("Would send email to user@example.com with report report.pdf attached.")
    
    @patch('builtins.print')
    def test_send_email_no_recipients(self, mock_print, service, report):
        """Test sending email with no recipients."""
        file_path = "/path/to/report.pdf"
        report.recipients = []
        
        service._send_email(report, file_path)
        
        # Check warning was printed
        mock_print.assert_any_call(f"Warning: Email delivery selected for report {report.report_id} but no recipients specified.")
    
    @patch('builtins.print')
    def test_send_slack_notification(self, mock_print, service):
        """Test sending Slack notification."""
        channel = "#reports"
        message = "New report available"
        file_path = "/path/to/report.pdf"
        
        service.send_slack_notification(channel, message, file_path)
        
        # Check that appropriate print statements were made
        mock_print.assert_any_call(f"Would send Slack message to channel '{channel}': {message}")
        mock_print.assert_any_call(f"Would attach file: {file_path}")
    
    @patch('builtins.print')
    def test_send_webhook(self, mock_print, service):
        """Test sending webhook notification."""
        webhook_url = "https://example.com/webhook"
        payload = {"report_id": "123", "status": "completed"}
        
        service.send_webhook(webhook_url, payload)
        
        # Check that appropriate print statements were made
        mock_print.assert_any_call(f"Would send webhook to {webhook_url} with payload: {payload}")