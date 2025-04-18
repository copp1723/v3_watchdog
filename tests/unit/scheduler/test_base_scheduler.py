"""
Unit tests for the base scheduler module.
"""

import os
import uuid
import json
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from src.scheduler.base_scheduler import (
    BaseScheduler, 
    Report,
    ScheduledReport, 
    ReportFrequency,
    ReportFormat, 
    DeliveryMethod
)


class TestReport:
    """Tests for the Report class."""
    
    def test_init(self):
        """Test Report initialization."""
        report_id = str(uuid.uuid4())
        report = Report(
            report_id=report_id,
            name="Test Report",
            format=ReportFormat.PDF,
            delivery=DeliveryMethod.EMAIL,
            parameters={"test": True},
            created_by="test_user"
        )
        
        assert report.report_id == report_id
        assert report.name == "Test Report"
        assert report.format == ReportFormat.PDF
        assert report.delivery == DeliveryMethod.EMAIL
        assert report.parameters == {"test": True}
        assert report.created_by == "test_user"
        assert report.created_at is not None
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        report_id = str(uuid.uuid4())
        report = Report(
            report_id=report_id,
            name="Test Report",
            format=ReportFormat.PDF,
            delivery=DeliveryMethod.EMAIL,
            parameters={"test": True},
            created_by="test_user"
        )
        
        data = report.to_dict()
        assert data["report_id"] == report_id
        assert data["name"] == "Test Report"
        assert data["format"] == ReportFormat.PDF
        assert data["delivery"] == DeliveryMethod.EMAIL
        assert data["parameters"] == {"test": True}
        assert data["created_by"] == "test_user"
        assert "created_at" in data
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        report_id = str(uuid.uuid4())
        data = {
            "report_id": report_id,
            "name": "Test Report",
            "format": ReportFormat.PDF,
            "delivery": DeliveryMethod.EMAIL,
            "parameters": {"test": True},
            "created_by": "test_user",
            "created_at": "2023-01-01T00:00:00"
        }
        
        report = Report.from_dict(data)
        assert report.report_id == report_id
        assert report.name == "Test Report"
        assert report.format == ReportFormat.PDF
        assert report.delivery == DeliveryMethod.EMAIL
        assert report.parameters == {"test": True}
        assert report.created_by == "test_user"
        assert report.created_at == "2023-01-01T00:00:00"


class TestScheduledReport:
    """Tests for the ScheduledReport class."""
    
    def test_init(self):
        """Test ScheduledReport initialization."""
        report_id = str(uuid.uuid4())
        report = ScheduledReport(
            report_id=report_id,
            name="Test Report",
            template="sales_summary",
            frequency=ReportFrequency.DAILY,
            format=ReportFormat.PDF,
            delivery=DeliveryMethod.EMAIL,
            recipients=["user@example.com"],
            parameters={"test": True},
            created_by="test_user"
        )
        
        assert report.report_id == report_id
        assert report.name == "Test Report"
        assert report.template == "sales_summary"
        assert report.frequency == ReportFrequency.DAILY
        assert report.format == ReportFormat.PDF
        assert report.delivery == DeliveryMethod.EMAIL
        assert report.recipients == ["user@example.com"]
        assert report.parameters == {"test": True}
        assert report.created_by == "test_user"
        assert report.created_at is not None
        assert report.last_run is None
        assert report.next_run is not None
        assert report.enabled is True
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        report_id = str(uuid.uuid4())
        report = ScheduledReport(
            report_id=report_id,
            name="Test Report",
            template="sales_summary",
            frequency=ReportFrequency.DAILY,
            format=ReportFormat.PDF,
            delivery=DeliveryMethod.EMAIL,
            recipients=["user@example.com"],
            parameters={"test": True},
            created_by="test_user"
        )
        
        data = report.to_dict()
        assert data["report_id"] == report_id
        assert data["name"] == "Test Report"
        assert data["template"] == "sales_summary"
        assert data["frequency"] == ReportFrequency.DAILY
        assert data["format"] == ReportFormat.PDF
        assert data["delivery"] == DeliveryMethod.EMAIL
        assert data["recipients"] == ["user@example.com"]
        assert data["parameters"] == {"test": True}
        assert data["created_by"] == "test_user"
        assert "created_at" in data
        assert "next_run" in data
        assert data["enabled"] is True
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        report_id = str(uuid.uuid4())
        data = {
            "report_id": report_id,
            "name": "Test Report",
            "template": "sales_summary",
            "frequency": ReportFrequency.DAILY,
            "format": ReportFormat.PDF,
            "delivery": DeliveryMethod.EMAIL,
            "recipients": ["user@example.com"],
            "parameters": {"test": True},
            "created_by": "test_user",
            "created_at": "2023-01-01T00:00:00",
            "last_run": "2023-01-01T12:00:00",
            "next_run": "2023-01-02T01:00:00",
            "enabled": True
        }
        
        report = ScheduledReport.from_dict(data)
        assert report.report_id == report_id
        assert report.name == "Test Report"
        assert report.template == "sales_summary"
        assert report.frequency == ReportFrequency.DAILY
        assert report.format == ReportFormat.PDF
        assert report.delivery == DeliveryMethod.EMAIL
        assert report.recipients == ["user@example.com"]
        assert report.parameters == {"test": True}
        assert report.created_by == "test_user"
        assert report.created_at == "2023-01-01T00:00:00"
        assert report.last_run == "2023-01-01T12:00:00"
        assert report.next_run == "2023-01-02T01:00:00"
        assert report.enabled is True
    
    def test_calculate_next_run_daily(self):
        """Test next run calculation for daily frequency."""
        report = ScheduledReport(
            report_id=str(uuid.uuid4()),
            name="Test Report",
            template="sales_summary",
            frequency=ReportFrequency.DAILY,
            format=ReportFormat.PDF,
            delivery=DeliveryMethod.EMAIL
        )
        
        next_run = datetime.fromisoformat(report.next_run)
        today = datetime.now()
        tomorrow = today + timedelta(days=1)
        tomorrow = tomorrow.replace(hour=1, minute=0, second=0, microsecond=0)
        
        assert next_run.date() == tomorrow.date()
        assert next_run.hour == 1
        assert next_run.minute == 0
    
    def test_calculate_next_run_weekly(self):
        """Test next run calculation for weekly frequency."""
        report = ScheduledReport(
            report_id=str(uuid.uuid4()),
            name="Test Report",
            template="sales_summary",
            frequency=ReportFrequency.WEEKLY,
            format=ReportFormat.PDF,
            delivery=DeliveryMethod.EMAIL
        )
        
        next_run = datetime.fromisoformat(report.next_run)
        today = datetime.now()
        days_ahead = 7 - today.weekday()
        if days_ahead == 0:
            days_ahead = 7
        next_monday = today + timedelta(days=days_ahead)
        next_monday = next_monday.replace(hour=1, minute=0, second=0, microsecond=0)
        
        assert next_run.date() == next_monday.date()
        assert next_run.hour == 1
        assert next_run.minute == 0
        assert next_run.weekday() == 0  # Monday
    
    def test_update_next_run(self):
        """Test updating next run time."""
        report = ScheduledReport(
            report_id=str(uuid.uuid4()),
            name="Test Report",
            template="sales_summary",
            frequency=ReportFrequency.DAILY,
            format=ReportFormat.PDF,
            delivery=DeliveryMethod.EMAIL
        )
        
        old_next_run = report.next_run
        report.update_next_run()
        
        assert report.last_run is not None
        assert report.next_run != old_next_run


class MockScheduler(BaseScheduler):
    """Mock implementation of BaseScheduler for testing."""
    
    def schedule(self, report, frequency):
        """Mock implementation."""
        report.frequency = frequency
        report.next_run = report._calculate_next_run()
        self.reports[report.report_id] = report
        return report.report_id
    
    def get_due_reports(self):
        """Mock implementation."""
        now = datetime.now().isoformat()
        return [r for r in self.reports.values() 
                if r.enabled and r.next_run <= now]
    
    def run_due(self):
        """Mock implementation."""
        pass


class TestBaseScheduler:
    """Tests for the BaseScheduler class."""
    
    @pytest.fixture
    def temp_dir(self, tmpdir):
        """Fixture to create a temporary directory."""
        return str(tmpdir)
    
    @pytest.fixture
    def scheduler(self, temp_dir):
        """Fixture to create a scheduler instance."""
        return MockScheduler(reports_dir=temp_dir)
    
    def test_init(self, scheduler, temp_dir):
        """Test scheduler initialization."""
        assert scheduler.reports_dir == temp_dir
        assert scheduler.config_file == os.path.join(temp_dir, "scheduled_reports.json")
        assert scheduler.reports == {}
        assert scheduler.scheduler_thread is None
        assert scheduler.stop_event is not None
    
    def test_load_reports(self, scheduler, temp_dir):
        """Test loading reports from file."""
        # Create a test report
        report_id = str(uuid.uuid4())
        report = ScheduledReport(
            report_id=report_id,
            name="Test Report",
            template="sales_summary",
            frequency=ReportFrequency.DAILY,
            format=ReportFormat.PDF,
            delivery=DeliveryMethod.EMAIL
        )
        
        # Save report to file
        with open(scheduler.config_file, 'w') as f:
            json.dump({report_id: report.to_dict()}, f)
        
        # Load reports
        scheduler.load_reports()
        
        assert report_id in scheduler.reports
        assert scheduler.reports[report_id].name == "Test Report"
    
    def test_save_reports(self, scheduler):
        """Test saving reports to file."""
        # Create a test report
        report_id = str(uuid.uuid4())
        report = ScheduledReport(
            report_id=report_id,
            name="Test Report",
            template="sales_summary",
            frequency=ReportFrequency.DAILY,
            format=ReportFormat.PDF,
            delivery=DeliveryMethod.EMAIL
        )
        
        # Add report to scheduler
        scheduler.reports[report_id] = report
        
        # Save reports
        scheduler.save_reports()
        
        # Check if file exists and contains the report
        assert os.path.exists(scheduler.config_file)
        with open(scheduler.config_file, 'r') as f:
            data = json.load(f)
            assert report_id in data
            assert data[report_id]["name"] == "Test Report"
    
    @patch('threading.Thread')
    def test_start(self, mock_thread, scheduler):
        """Test starting the scheduler thread."""
        scheduler.start()
        
        mock_thread.assert_called_once()
        assert mock_thread.call_args[1]["target"] == scheduler._scheduler_loop
        assert mock_thread.call_args[1]["args"] == (60,)
        assert mock_thread.call_args[1]["daemon"] is True
        
        mock_thread.return_value.start.assert_called_once()
    
    @patch('threading.Thread')
    def test_stop(self, mock_thread, scheduler):
        """Test stopping the scheduler thread."""
        # Mock the scheduler thread
        scheduler.scheduler_thread = MagicMock()
        scheduler.scheduler_thread.is_alive.return_value = True
        
        scheduler.stop()
        
        scheduler.stop_event.is_set.assert_called_once()
        scheduler.scheduler_thread.join.assert_called_once_with(timeout=5)