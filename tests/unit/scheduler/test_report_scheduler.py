"""
Unit tests for the report scheduler module.
"""

import os
import uuid
import json
import pytest
from unittest.mock import patch, MagicMock, ANY
from datetime import datetime

from src.scheduler.report_scheduler import (
    ReportScheduler,
    ReportTemplate
)
from src.scheduler.base_scheduler import (
    ReportFrequency,
    ReportFormat,
    DeliveryMethod,
    ScheduledReport
)


class TestReportScheduler:
    """Tests for the ReportScheduler class."""
    
    @pytest.fixture
    def temp_dir(self, tmpdir):
        """Fixture to create a temporary directory."""
        return str(tmpdir)
    
    @pytest.fixture
    def scheduler(self, temp_dir):
        """Fixture to create a scheduler instance."""
        with patch('src.scheduler.reports.sales_report.SalesReportGenerator'), \
            patch('src.scheduler.reports.inventory_report.InventoryReportGenerator'):
            return ReportScheduler(reports_dir=temp_dir)
    
    def test_init(self, scheduler, temp_dir):
        """Test scheduler initialization."""
        assert scheduler.reports_dir == temp_dir
        assert scheduler.notification_service is not None
        assert len(scheduler.report_generators) > 0
    
    def test_create_report(self, scheduler):
        """Test creating a new report."""
        name = "Test Report"
        template = ReportTemplate.SALES_SUMMARY
        frequency = ReportFrequency.DAILY
        format = ReportFormat.PDF
        delivery = DeliveryMethod.EMAIL
        recipients = ["user@example.com"]
        parameters = {"include_charts": True}
        created_by = "test_user"
        
        report_id = scheduler.create_report(
            name=name,
            template=template,
            frequency=frequency,
            format=format,
            delivery=delivery,
            recipients=recipients,
            parameters=parameters,
            created_by=created_by
        )
        
        assert report_id in scheduler.reports
        report = scheduler.reports[report_id]
        assert report.name == name
        assert report.template == template
        assert report.frequency == frequency
        assert report.format == format
        assert report.delivery == delivery
        assert report.recipients == recipients
        assert report.parameters == parameters
        assert report.created_by == created_by
    
    def test_update_report(self, scheduler):
        """Test updating an existing report."""
        # Create a test report
        report_id = scheduler.create_report(
            name="Original Name",
            template=ReportTemplate.SALES_SUMMARY,
            frequency=ReportFrequency.DAILY,
            format=ReportFormat.PDF,
            delivery=DeliveryMethod.EMAIL,
            created_by="test_user"
        )
        
        # Update the report
        result = scheduler.update_report(
            report_id=report_id,
            name="Updated Name",
            template=ReportTemplate.INVENTORY_HEALTH,
            frequency=ReportFrequency.WEEKLY,
            format=ReportFormat.HTML,
            delivery=DeliveryMethod.DOWNLOAD,
            recipients=["new@example.com"],
            parameters={"include_charts": False},
            enabled=False
        )
        
        assert result is True
        report = scheduler.reports[report_id]
        assert report.name == "Updated Name"
        assert report.template == ReportTemplate.INVENTORY_HEALTH
        assert report.frequency == ReportFrequency.WEEKLY
        assert report.format == ReportFormat.HTML
        assert report.delivery == DeliveryMethod.DOWNLOAD
        assert report.recipients == ["new@example.com"]
        assert report.parameters == {"include_charts": False}
        assert report.enabled is False
    
    def test_update_nonexistent_report(self, scheduler):
        """Test updating a non-existent report."""
        result = scheduler.update_report(
            report_id="nonexistent",
            name="Updated Name"
        )
        
        assert result is False
    
    def test_delete_report(self, scheduler):
        """Test deleting a report."""
        # Create a test report
        report_id = scheduler.create_report(
            name="Test Report",
            template=ReportTemplate.SALES_SUMMARY,
            frequency=ReportFrequency.DAILY,
            format=ReportFormat.PDF,
            delivery=DeliveryMethod.EMAIL,
            created_by="test_user"
        )
        
        # Delete the report
        result = scheduler.delete_report(report_id)
        
        assert result is True
        assert report_id not in scheduler.reports
    
    def test_delete_nonexistent_report(self, scheduler):
        """Test deleting a non-existent report."""
        result = scheduler.delete_report("nonexistent")
        
        assert result is False
    
    def test_get_report(self, scheduler):
        """Test getting a report by ID."""
        # Create a test report
        report_id = scheduler.create_report(
            name="Test Report",
            template=ReportTemplate.SALES_SUMMARY,
            frequency=ReportFrequency.DAILY,
            format=ReportFormat.PDF,
            delivery=DeliveryMethod.EMAIL,
            created_by="test_user"
        )
        
        # Get the report
        report = scheduler.get_report(report_id)
        
        assert report is not None
        assert report.name == "Test Report"
    
    def test_get_nonexistent_report(self, scheduler):
        """Test getting a non-existent report."""
        report = scheduler.get_report("nonexistent")
        
        assert report is None
    
    def test_get_all_reports(self, scheduler):
        """Test getting all reports."""
        # Create test reports
        report1_id = scheduler.create_report(
            name="Report 1",
            template=ReportTemplate.SALES_SUMMARY,
            frequency=ReportFrequency.DAILY,
            format=ReportFormat.PDF,
            delivery=DeliveryMethod.EMAIL,
            created_by="user1"
        )
        
        report2_id = scheduler.create_report(
            name="Report 2",
            template=ReportTemplate.INVENTORY_HEALTH,
            frequency=ReportFrequency.WEEKLY,
            format=ReportFormat.HTML,
            delivery=DeliveryMethod.DOWNLOAD,
            created_by="user2"
        )
        
        # Get all reports
        reports = scheduler.get_all_reports()
        
        assert len(reports) == 2
        assert any(r.name == "Report 1" for r in reports)
        assert any(r.name == "Report 2" for r in reports)
    
    def test_get_reports_by_user(self, scheduler):
        """Test getting reports by user."""
        # Create test reports
        report1_id = scheduler.create_report(
            name="Report 1",
            template=ReportTemplate.SALES_SUMMARY,
            frequency=ReportFrequency.DAILY,
            format=ReportFormat.PDF,
            delivery=DeliveryMethod.EMAIL,
            created_by="user1"
        )
        
        report2_id = scheduler.create_report(
            name="Report 2",
            template=ReportTemplate.INVENTORY_HEALTH,
            frequency=ReportFrequency.WEEKLY,
            format=ReportFormat.HTML,
            delivery=DeliveryMethod.DOWNLOAD,
            created_by="user2"
        )
        
        report3_id = scheduler.create_report(
            name="Report 3",
            template=ReportTemplate.LEAD_SOURCE,
            frequency=ReportFrequency.MONTHLY,
            format=ReportFormat.CSV,
            delivery=DeliveryMethod.DASHBOARD,
            created_by="user1"
        )
        
        # Get reports by user
        user1_reports = scheduler.get_reports_by_user("user1")
        user2_reports = scheduler.get_reports_by_user("user2")
        
        assert len(user1_reports) == 2
        assert any(r.name == "Report 1" for r in user1_reports)
        assert any(r.name == "Report 3" for r in user1_reports)
        
        assert len(user2_reports) == 1
        assert user2_reports[0].name == "Report 2"
    
    @patch.object(ScheduledReport, '_calculate_next_run')
    def test_get_due_reports(self, mock_calculate_next_run, scheduler):
        """Test getting due reports."""
        # Mock the next run calculation to return a past time
        past_time = (datetime.now().replace(hour=0, minute=0, second=0) \
                    .isoformat())
        future_time = (datetime.now().replace(hour=23, minute=59, second=59) \
                      .isoformat())
        
        mock_calculate_next_run.side_effect = [past_time, future_time, past_time]
        
        # Create test reports
        report1_id = scheduler.create_report(
            name="Due Report 1",
            template=ReportTemplate.SALES_SUMMARY,
            frequency=ReportFrequency.DAILY,
            format=ReportFormat.PDF,
            delivery=DeliveryMethod.EMAIL,
            created_by="user1"
        )
        
        report2_id = scheduler.create_report(
            name="Not Due Report",
            template=ReportTemplate.INVENTORY_HEALTH,
            frequency=ReportFrequency.WEEKLY,
            format=ReportFormat.HTML,
            delivery=DeliveryMethod.DOWNLOAD,
            created_by="user2"
        )
        
        report3_id = scheduler.create_report(
            name="Due Report 2",
            template=ReportTemplate.LEAD_SOURCE,
            frequency=ReportFrequency.MONTHLY,
            format=ReportFormat.CSV,
            delivery=DeliveryMethod.DASHBOARD,
            created_by="user1"
        )
        
        # Disable one of the due reports
        scheduler.update_report(report3_id, enabled=False)
        
        # Get due reports
        due_reports = scheduler.get_due_reports()
        
        assert len(due_reports) == 1
        assert due_reports[0].name == "Due Report 1"
    
    @patch.object(ReportScheduler, '_load_data_for_template')
    @patch.object(ReportScheduler, '_format_report')
    def test_generate_report(self, mock_format_report, mock_load_data, scheduler):
        """Test generating a report."""
        # Create a mock report
        report = MagicMock(spec=ScheduledReport)
        report.report_id = str(uuid.uuid4())
        report.name = "Test Report"
        report.template = ReportTemplate.SALES_SUMMARY
        report.format = ReportFormat.PDF
        report.delivery = DeliveryMethod.EMAIL
        report.parameters = {}
        
        # Mock the data loading and content generation
        mock_df = MagicMock()
        mock_load_data.return_value = mock_df
        
        mock_content = "Test content"
        mock_format_report.return_value = mock_content
        
        # Mock the report generator
        mock_generator = MagicMock()
        mock_generator.generate.return_value = {
            "title": "Test Report",
            "generated_at": datetime.now().isoformat(),
            "summary": "Test summary",
            "charts": [],
            "tables": []
        }
        scheduler.report_generators[ReportTemplate.SALES_SUMMARY] = mock_generator
        
        # Mock the notification service
        scheduler.notification_service = MagicMock()
        
        # Generate the report
        scheduler._generate_report(report)
        
        # Check that the appropriate methods were called
        mock_load_data.assert_called_once_with(report.template, report.parameters)
        mock_generator.generate.assert_called_once_with(mock_df, report.parameters)
        mock_format_report.assert_called_once()
        scheduler.notification_service.notify.assert_called_once_with(report, mock_content)
    
    def test_format_report_json(self, scheduler):
        """Test formatting a report as JSON."""
        content = {
            "title": "Test Report",
            "generated_at": datetime.now().isoformat(),
            "summary": "Test summary",
            "charts": [],
            "tables": []
        }
        
        result = scheduler._format_report(content, ReportFormat.JSON)
        
        # Check that the result is valid JSON
        parsed = json.loads(result)
        assert parsed["title"] == "Test Report"
        assert parsed["summary"] == "Test summary"
    
    def test_format_report_html(self, scheduler):
        """Test formatting a report as HTML."""
        content = {
            "title": "Test Report",
            "generated_at": datetime.now().isoformat(),
            "summary": "Test summary",
            "charts": [],
            "tables": []
        }
        
        result = scheduler._format_report(content, ReportFormat.HTML)
        
        # Check that the result is valid HTML
        assert "<!DOCTYPE html>" in result
        assert "<title>Test Report</title>" in result
        assert "Test summary" in result
    
    def test_format_report_csv(self, scheduler):
        """Test formatting a report as CSV."""
        content = {
            "title": "Test Report",
            "generated_at": datetime.now().isoformat(),
            "summary": "Test summary",
            "tables": [{
                "title": "Test Table",
                "columns": ["Column1", "Column2"],
                "data": [
                    {"Column1": "Value1", "Column2": 1},
                    {"Column1": "Value2", "Column2": 2}
                ]
            }]
        }
        
        result = scheduler._format_report(content, ReportFormat.CSV)
        
        # Check that the result is valid CSV
        assert "# Test Table" in result
        assert "Column1,Column2" in result
        assert "Value1,1" in result
        assert "Value2,2" in result