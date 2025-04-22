"""
Tests for the notification service module.
"""
import os
import json
import time
import tempfile
import unittest
from unittest import mock
from datetime import datetime
from email.mime.multipart import MIMEMultipart

from src.scheduler.notification_service import (
    NotificationService, EmailMessage, NotificationQueue,
    EmailDeliveryStatus
)
from src.scheduler.base_scheduler import Report, ScheduledReport, DeliveryMethod


class TestEmailMessage(unittest.TestCase):
    """Tests for the EmailMessage class."""
    
    def test_create_email(self):
        """Test creating an email message."""
        recipients = ["test@example.com"]
        subject = "Test Subject"
        html_content = "<p>Test content</p>"
        
        email = EmailMessage(
            recipients=recipients,
            subject=subject,
            html_content=html_content
        )
        
        self.assertEqual(email.recipients, recipients)
        self.assertEqual(email.subject, subject)
        self.assertEqual(email.html_content, html_content)
        self.assertEqual(email.status, EmailDeliveryStatus.QUEUED)
        self.assertEqual(email.retry_count, 0)
        self.assertIsNone(email.error)
        self.assertIsNone(email.sent_time)
    
    def test_html_to_text_conversion(self):
        """Test HTML to text conversion."""
        html = """
        <h1>Test Header</h1>
        <p>This is a paragraph.</p>
        <ul>
            <li>Item 1</li>
            <li>Item 2</li>
        </ul>
        """
        
        email = EmailMessage(
            recipients=["test@example.com"],
            subject="Test",
            html_content=html
        )
        
        # Check that text content was generated from HTML
        self.assertIsNotNone(email.text_content)
        self.assertIn("Test Header", email.text_content)
        self.assertIn("This is a paragraph", email.text_content)
        self.assertIn("Item 1", email.text_content)
        self.assertIn("Item 2", email.text_content)
    
    def test_create_mime_message(self):
        """Test MIME message creation."""
        email = EmailMessage(
            recipients=["test@example.com"],
            subject="Test Subject",
            html_content="<p>Test content</p>",
            from_email="sender@example.com",
            from_name="Sender",
            reply_to="reply@example.com"
        )
        
        mime_message = email.create_mime_message()
        
        self.assertIsInstance(mime_message, MIMEMultipart)
        self.assertEqual(mime_message['Subject'], "Test Subject")
        self.assertEqual(mime_message['From'], "Sender <sender@example.com>")
        self.assertEqual(mime_message['To'], "test@example.com")
        self.assertEqual(mime_message['Reply-To'], "reply@example.com")
        
        # Verify message has both text and HTML parts
        self.assertEqual(len(mime_message.get_payload()), 2)


class TestNotificationQueue(unittest.TestCase):
    """Tests for the NotificationQueue class."""
    
    def setUp(self):
        """Set up the test case."""
        self.queue = NotificationQueue(worker_count=1)
    
    def tearDown(self):
        """Clean up the test case."""
        if self.queue.workers:
            self.queue.stop()
    
    def test_add_email(self):
        """Test adding an email to the queue."""
        email = EmailMessage(
            recipients=["test@example.com"],
            subject="Test Subject",
            html_content="<p>Test content</p>"
        )
        
        message_id = self.queue.add_email(email)
        
        self.assertIsNotNone(message_id)
        self.assertIn(message_id, self.queue.email_messages)
        self.assertEqual(self.queue.email_messages[message_id], email)
        self.assertEqual(self.queue.queue.qsize(), 1)
    
    def test_get_email_status(self):
        """Test getting email status."""
        email = EmailMessage(
            recipients=["test@example.com"],
            subject="Test Subject",
            html_content="<p>Test content</p>"
        )
        
        message_id = self.queue.add_email(email)
        status = self.queue.get_email_status(message_id)
        
        self.assertEqual(status["status"], EmailDeliveryStatus.QUEUED)
        self.assertEqual(status["subject"], "Test Subject")
        self.assertEqual(status["recipients"], ["test@example.com"])
        self.assertEqual(status["retry_count"], 0)


class TestNotificationService(unittest.TestCase):
    """Tests for the NotificationService class."""
    
    def setUp(self):
        """Set up the test case."""
        # Create a temporary directory for storing reports and templates
        self.temp_dir = tempfile.mkdtemp()
        self.templates_dir = os.path.join(self.temp_dir, "templates")
        os.makedirs(self.templates_dir, exist_ok=True)
        
        # Initialize notification service with the temporary directory
        self.notification_service = NotificationService(
            reports_dir=self.temp_dir,
            templates_dir=self.templates_dir
        )
    
    def tearDown(self):
        """Clean up the test case."""
        self.notification_service.shutdown()
        
        # Clean up the temporary directory
        for root, dirs, files in os.walk(self.temp_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(self.temp_dir)
    
    def test_create_default_templates(self):
        """Test that default templates are created."""
        # Templates should be created in setUp
        self.assertTrue(os.path.exists(os.path.join(self.templates_dir, "daily_insight.html")))
        self.assertTrue(os.path.exists(os.path.join(self.templates_dir, "weekly_executive.html")))
        self.assertTrue(os.path.exists(os.path.join(self.templates_dir, "alert.html")))
    
    def test_notify_with_file_saving(self):
        """Test saving a report file."""
        report = Report(
            report_id="test-report-123",
            name="Test Report",
            format="html",
            delivery=DeliveryMethod.DOWNLOAD,
            parameters={}
        )
        
        content = "<html><body><h1>Test Report</h1></body></html>"
        
        file_path = self.notification_service.notify(report, content)
        
        self.assertTrue(os.path.exists(file_path))
        self.assertTrue(file_path.startswith(self.temp_dir))
        self.assertTrue(file_path.endswith(".html"))
        
        with open(file_path, 'r') as f:
            saved_content = f.read()
        
        self.assertEqual(saved_content, content)
    
    @mock.patch('smtplib.SMTP')
    def test_send_email(self, mock_smtp):
        """Test sending an email."""
        # Configure the mock
        mock_smtp_instance = mock_smtp.return_value.__enter__.return_value
        
        # Create a test report
        report = ScheduledReport(
            report_id="test-report-456",
            name="Email Test Report",
            template="sales_summary",
            frequency="daily",
            format="pdf",
            delivery=DeliveryMethod.EMAIL,
            recipients=["test@example.com"],
            parameters={}
        )
        
        # Create a test file
        test_content = "Test PDF content"
        file_path = os.path.join(self.temp_dir, "test_report.pdf")
        with open(file_path, 'w') as f:
            f.write(test_content)
        
        # Call the method
        message_id = self.notification_service._send_email(report, file_path)
        
        # Check that the email was queued
        self.assertIsNotNone(message_id)
        self.assertIn(message_id, self.notification_service.queue.email_messages)
        
        # Email is added to the queue but not sent in the test
        # In a real environment, the worker thread would process it
        email = self.notification_service.queue.email_messages[message_id]
        self.assertEqual(email.recipients, ["test@example.com"])
        self.assertTrue(email.subject.startswith("Email Test Report"))
    
    def test_send_insight_email(self):
        """Test sending an insight email."""
        insights = [
            {
                "title": "Test Insight 1",
                "summary": "This is a test insight",
                "metrics": {
                    "test_metric": 100
                }
            },
            {
                "title": "Test Insight 2",
                "summary": "Another test insight",
                "metrics": {
                    "another_metric": 200
                }
            }
        ]
        
        recipients = ["test@example.com"]
        
        message_id = self.notification_service.send_insight_email(
            recipients=recipients,
            insights=insights,
            subject="Test Insights"
        )
        
        self.assertIsNotNone(message_id)
        self.assertIn(message_id, self.notification_service.queue.email_messages)
        
        email = self.notification_service.queue.email_messages[message_id]
        self.assertEqual(email.recipients, recipients)
        self.assertEqual(email.subject, "Test Insights")
        
        # Check that the HTML content includes the insights
        self.assertIn("Test Insight 1", email.html_content)
        self.assertIn("This is a test insight", email.html_content)
        self.assertIn("Test Insight 2", email.html_content)
        self.assertIn("Another test insight", email.html_content)
    
    def test_send_alert_email(self):
        """Test sending an alert email."""
        alert = {
            "title": "Critical Alert",
            "description": "This is a critical test alert",
            "metrics": [
                {"label": "Test Metric", "value": "100"}
            ],
            "recommendations": [
                "Take immediate action",
                "Check system status"
            ]
        }
        
        recipients = ["test@example.com"]
        
        message_id = self.notification_service.send_alert_email(
            recipients=recipients,
            alert=alert,
            subject="Test Alert"
        )
        
        self.assertIsNotNone(message_id)
        self.assertIn(message_id, self.notification_service.queue.email_messages)
        
        email = self.notification_service.queue.email_messages[message_id]
        self.assertEqual(email.recipients, recipients)
        self.assertEqual(email.subject, "ALERT: Critical Alert")
        
        # Check that the HTML content includes the alert information
        self.assertIn("Critical Alert", email.html_content)
        self.assertIn("This is a critical test alert", email.html_content)
        self.assertIn("Take immediate action", email.html_content)
        self.assertIn("Check system status", email.html_content)