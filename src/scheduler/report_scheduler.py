"""
Report Scheduler Module for V3 Watchdog AI.

Provides functionality for scheduling and generating reports at regular intervals,
including saving reports as PDF or sending them via email.
"""

import os
import json
import uuid
from datetime import datetime, timedelta
import time
import threading
import pandas as pd
import streamlit as st
from typing import Dict, Any, List, Optional, Union, Callable
import json
import base64
from enum import Enum
import plotly.graph_objects as go
import plotly.express as px
import io

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

class ReportTemplate(str, Enum):
    """Available report templates."""
    SALES_SUMMARY = "sales_summary"
    INVENTORY_HEALTH = "inventory_health"
    PROFITABILITY = "profitability"
    LEAD_SOURCE = "lead_source"
    CUSTOM = "custom"

class ScheduledReport:
    """Represents a scheduled report configuration."""
    
    def __init__(self, 
                report_id: str,
                name: str,
                template: ReportTemplate,
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
        self.report_id = report_id
        self.name = name
        self.template = template
        self.frequency = frequency
        self.format = format
        self.delivery = delivery
        self.recipients = recipients or []
        self.parameters = parameters or {}
        self.created_by = created_by
        self.created_at = datetime.now().isoformat()
        self.last_run = None
        self.next_run = self._calculate_next_run()
        self.enabled = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "report_id": self.report_id,
            "name": self.name,
            "template": self.template,
            "frequency": self.frequency,
            "format": self.format,
            "delivery": self.delivery,
            "recipients": self.recipients,
            "parameters": self.parameters,
            "created_by": self.created_by,
            "created_at": self.created_at,
            "last_run": self.last_run,
            "next_run": self.next_run,
            "enabled": self.enabled
        }
    
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


class ReportScheduler:
    """Manages scheduled reports and their execution."""
    
    def __init__(self, reports_dir: str = None):
        """
        Initialize the report scheduler.
        
        Args:
            reports_dir: Directory to store reports and configurations
        """
        self.reports_dir = reports_dir or os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                                    "data", "reports")
        self.config_file = os.path.join(self.reports_dir, "scheduled_reports.json")
        self.reports = {}
        self.scheduler_thread = None
        self.stop_event = threading.Event()
        
        # Create directory if it doesn't exist
        os.makedirs(self.reports_dir, exist_ok=True)
        
        # Load existing reports
        self.load_reports()
    
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
    
    def create_report(self, 
                     name: str,
                     template: ReportTemplate,
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
        
        # Add to reports dictionary
        self.reports[report_id] = report
        
        # Save to file
        self.save_reports()
        
        return report_id
    
    def update_report(self, 
                     report_id: str,
                     name: Optional[str] = None,
                     template: Optional[ReportTemplate] = None,
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
        
        # Save to file
        self.save_reports()
        
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
    
    def run_scheduler(self, interval: int = 60) -> None:
        """
        Run the scheduler in a background thread.
        
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
        
        print(f"Report scheduler started with interval {interval} seconds")
    
    def stop_scheduler(self) -> None:
        """Stop the scheduler thread."""
        if not self.scheduler_thread or not self.scheduler_thread.is_alive():
            print("Scheduler not running")
            return
        
        # Set stop event
        self.stop_event.set()
        
        # Wait for thread to terminate
        self.scheduler_thread.join(timeout=5)
        
        print("Report scheduler stopped")
    
    def _scheduler_loop(self, interval: int) -> None:
        """
        Main loop for the scheduler thread.
        
        Args:
            interval: Check interval in seconds
        """
        while not self.stop_event.is_set():
            try:
                # Get due reports
                due_reports = self.get_due_reports()
                
                # Process each due report
                for report in due_reports:
                    try:
                        print(f"Generating report: {report.name} ({report.report_id})")
                        self._generate_report(report)
                        
                        # Update next run time
                        report.update_next_run()
                        self.save_reports()
                    except Exception as e:
                        print(f"Error generating report {report.report_id}: {e}")
                
                # Wait for next check
                self.stop_event.wait(interval)
            except Exception as e:
                print(f"Error in scheduler loop: {e}")
                # Wait before retrying
                self.stop_event.wait(interval)
    
    def _generate_report(self, report: ScheduledReport) -> None:
        """
        Generate a report based on its configuration.
        
        Args:
            report: The report to generate
        """
        # Load data based on template
        df = self._load_data_for_template(report.template, report.parameters)
        
        # Generate report content
        content = self._render_report_content(df, report)
        
        # Format report
        formatted_report = self._format_report(content, report.format)
        
        # Deliver report
        self._deliver_report(formatted_report, report)
    
    def _load_data_for_template(self, template: ReportTemplate, 
                              parameters: Dict[str, Any]) -> pd.DataFrame:
        """
        Load appropriate data for a report template.
        
        Args:
            template: Report template
            parameters: Template parameters
            
        Returns:
            DataFrame with data for the report
        """
        # In a real implementation, this would load data from a database
        # For this example, we'll generate sample data
        
        if template == ReportTemplate.SALES_SUMMARY:
            # Generate sample sales data
            dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
            np.random.seed(42)  # For reproducibility
            
            n_sales = 500
            sales_indices = np.random.choice(len(dates), size=n_sales, replace=True)
            
            return pd.DataFrame({
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
            
            return pd.DataFrame({
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
            
            return pd.DataFrame({
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
            
            return pd.DataFrame({
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
        try:
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
                
                # Add charts
                # 1. Monthly sales trend
                if 'Sale_Date' in df.columns:
                    df['Month'] = df['Sale_Date'].dt.to_period('M').astype(str)
                    monthly_sales = df.groupby('Month').size().reset_index(name='count')
                    
                    fig = go.Figure(data=go.Bar(
                        x=monthly_sales['Month'],
                        y=monthly_sales['count'],
                        marker_color='rgb(55, 83, 109)'
                    ))
                    fig.update_layout(
                        title="Monthly Sales",
                        xaxis_title="Month",
                        yaxis_title="Number of Sales",
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    
                    chart_img = self._fig_to_base64(fig)
                    content["charts"].append({
                        "title": "Monthly Sales",
                        "type": "bar",
                        "image": chart_img
                    })
                
                # 2. Sales by make
                if 'VehicleMake' in df.columns:
                    make_counts = df['VehicleMake'].value_counts().reset_index()
                    make_counts.columns = ['Make', 'Count']
                    
                    fig = go.Figure(data=go.Pie(
                        labels=make_counts['Make'],
                        values=make_counts['Count'],
                        textinfo='label+percent',
                        insidetextorientation='radial'
                    ))
                    fig.update_layout(
                        title="Sales by Make",
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    
                    chart_img = self._fig_to_base64(fig)
                    content["charts"].append({
                        "title": "Sales by Make",
                        "type": "pie",
                        "image": chart_img
                    })
                
                # Add tables
                # 1. Sales summary by month
                if 'Sale_Date' in df.columns and 'Gross_Profit' in df.columns:
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
            
            elif report.template == ReportTemplate.INVENTORY_HEALTH:
                # Add summary
                total_units = len(df)
                avg_days = df['DaysInInventory'].mean()
                aged_90_plus = (df['DaysInInventory'] > 90).sum()
                aged_pct = (aged_90_plus / total_units) * 100
                
                content["summary"] = f"Inventory Health: {total_units} total units with average age of {avg_days:.1f} days. {aged_90_plus} units ({aged_pct:.1f}%) are over 90 days old."
                
                # Add charts
                # 1. Age distribution
                bins = [0, 30, 60, 90, float('inf')]
                labels = ['<30 days', '30-60 days', '61-90 days', '>90 days']
                df['Age Category'] = pd.cut(df['DaysInInventory'], bins=bins, labels=labels)
                age_counts = df['Age Category'].value_counts().reindex(labels)
                
                fig = go.Figure(data=go.Pie(
                    labels=age_counts.index,
                    values=age_counts.values,
                    textinfo='label+percent',
                    insidetextorientation='radial'
                ))
                fig.update_layout(
                    title="Inventory Age Distribution",
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                
                chart_img = self._fig_to_base64(fig)
                content["charts"].append({
                    "title": "Inventory Age Distribution",
                    "type": "pie",
                    "image": chart_img
                })
                
                # 2. Age by make
                if 'VehicleMake' in df.columns:
                    make_age = df.groupby('VehicleMake')['DaysInInventory'].mean().reset_index()
                    make_age.columns = ['Make', 'AvgDays']
                    make_age = make_age.sort_values('AvgDays', ascending=False)
                    
                    fig = go.Figure(data=go.Bar(
                        x=make_age['Make'],
                        y=make_age['AvgDays'],
                        marker_color='rgb(55, 83, 109)'
                    ))
                    fig.update_layout(
                        title="Average Age by Make",
                        xaxis_title="Make",
                        yaxis_title="Average Days in Inventory",
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    
                    chart_img = self._fig_to_base64(fig)
                    content["charts"].append({
                        "title": "Average Age by Make",
                        "type": "bar",
                        "image": chart_img
                    })
                
                # Add tables
                # 1. Age summary
                age_summary = df.groupby('Age Category').agg(
                    Units=('VIN', 'count'),
                    AvgDays=('DaysInInventory', 'mean')
                ).reset_index()
                
                content["tables"].append({
                    "title": "Inventory Age Summary",
                    "columns": ["Age Category", "Units", "Avg Days"],
                    "data": age_summary.to_dict('records')
                })
            
            # Add similar implementations for other templates...
            
            return content
            
        except Exception as e:
            print(f"Error rendering report content: {e}")
            return {
                "title": report.name,
                "generated_at": datetime.now().isoformat(),
                "template": report.template,
                "summary": f"Error generating report: {str(e)}",
                "charts": [],
                "tables": [],
                "metadata": {
                    "report_id": report.report_id,
                    "error": str(e)
                }
            }
    
    def _fig_to_base64(self, fig) -> str:
        """Convert a plotly figure to base64 encoded string."""
        img_bytes = fig.to_image(format="png", engine="kaleido")
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')
        return img_b64
    
    def _format_report(self, content: Dict[str, Any], 
                     format: ReportFormat) -> Union[str, bytes]:
        """
        Format report content based on the specified format.
        
        Args:
            content: Report content dictionary
            format: Desired output format
            
        Returns:
            Formatted report as string or bytes
        """
        if format == ReportFormat.JSON:
            # Simple JSON serialization
            return json.dumps(content, indent=2)
        
        elif format == ReportFormat.CSV:
            # Convert tables to CSV
            if not content.get("tables"):
                return "No tabular data available for CSV export."
            
            # Concatenate all tables into a single CSV
            buffer = io.StringIO()
            
            for i, table in enumerate(content["tables"]):
                table_name = table.get("title", f"Table_{i}")
                data = table.get("data", [])
                
                if data:
                    # Convert to DataFrame and write to CSV
                    df = pd.DataFrame(data)
                    
                    # Add table name as header
                    buffer.write(f"# {table_name}\n")
                    
                    # Write data
                    df.to_csv(buffer, index=False)
                    
                    # Add separator between tables
                    buffer.write("\n\n")
            
            return buffer.getvalue()
        
        elif format == ReportFormat.HTML:
            # Generate HTML report
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
                            if col.lower().endswith(('gross', 'price', 'cost')):
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
        
        elif format == ReportFormat.PDF:
            # For a real implementation, you'd use a library like ReportLab or WeasyPrint
            # For this example, we'll just return HTML content that could be converted
            # For now, we'll return a placeholder
            return "PDF generation not implemented in this example."
        
        else:
            # Default to plain text summary
            text = f"Report: {content.get('title', 'Untitled')}\n"
            text += f"Generated: {content.get('generated_at', 'N/A')}\n\n"
            text += f"Summary: {content.get('summary', 'No summary available.')}\n\n"
            
            for table in content.get("tables", []):
                text += f"Table: {table.get('title', 'Untitled')}\n"
                
                # Add headers
                columns = table.get("columns", [])
                text += " | ".join(columns) + "\n"
                text += "-" * (sum(len(c) + 3 for c in columns) - 1) + "\n"
                
                # Add rows
                for row in table.get("data", []):
                    row_values = []
                    for col in columns:
                        value = row.get(col, "")
                        row_values.append(str(value))
                    text += " | ".join(row_values) + "\n"
                
                text += "\n"
            
            return text
    
    def _deliver_report(self, report_content: Union[str, bytes], 
                      report: ScheduledReport) -> None:
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
        filepath = os.path.join(self.reports_dir, filename)
        
        # Save the file
        mode = "wb" if isinstance(report_content, bytes) else "w"
        with open(filepath, mode) as f:
            f.write(report_content)
        
        print(f"Saved report to {filepath}")
        
        # Handle delivery
        if report.delivery == DeliveryMethod.EMAIL:
            if not report.recipients:
                print(f"Warning: Email delivery selected for report {report.report_id} but no recipients specified.")
                return
            
            # In a real implementation, you'd send an email with the report attached
            print(f"Would send email to {', '.join(report.recipients)} with report {filename} attached.")
        
        # Dashboard delivery is handled separately via the UI
        # Files are already saved to the reports directory for download


def render_report_scheduler() -> None:
    """Render the report scheduler interface."""
    st.title("Report Scheduler")
    
    # Initialize the report scheduler
    scheduler = ReportScheduler()
    
    # Get the current user
    current_user = None
    if 'user' in st.session_state:
        current_user = st.session_state['user'].username
    
    # Create tabs for different functions
    tab1, tab2, tab3 = st.tabs(["Scheduled Reports", "Create Report", "Report History"])
    
    with tab1:
        st.header("Scheduled Reports")
        
        # Get all reports
        all_reports = scheduler.get_all_reports()
        
        if current_user:
            # Filter to show only user's reports and public reports
            user_reports = [r for r in all_reports if r.created_by == current_user]
            
            if user_reports:
                # Create a DataFrame for display
                report_data = []
                for report in user_reports:
                    next_run = "N/A"
                    if report.next_run:
                        next_run = datetime.fromisoformat(report.next_run).strftime("%Y-%m-%d %H:%M")
                    
                    last_run = "Never"
                    if report.last_run:
                        last_run = datetime.fromisoformat(report.last_run).strftime("%Y-%m-%d %H:%M")
                    
                    report_data.append({
                        "Name": report.name,
                        "Template": report.template,
                        "Frequency": report.frequency,
                        "Format": report.format,
                        "Delivery": report.delivery,
                        "Next Run": next_run,
                        "Last Run": last_run,
                        "Status": "Enabled" if report.enabled else "Disabled"
                    })
                
                # Display the reports table
                st.dataframe(report_data, use_container_width=True)
                
                # Report management section
                st.subheader("Manage Reports")
                
                # Select a report to manage
                report_options = [r.name for r in user_reports]
                selected_report_name = st.selectbox("Select Report", report_options)
                
                if selected_report_name:
                    # Find the selected report
                    selected_report = next((r for r in user_reports if r.name == selected_report_name), None)
                    
                    if selected_report:
                        # Display report details
                        with st.expander("Report Details", expanded=True):
                            st.json(selected_report.to_dict())
                        
                        # Management actions
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            # Toggle enabled/disabled
                            current_status = "Disable" if selected_report.enabled else "Enable"
                            if st.button(f"{current_status} Report"):
                                scheduler.update_report(
                                    selected_report.report_id,
                                    enabled=not selected_report.enabled
                                )
                                st.success(f"Report {current_status.lower()}d.")
                                st.rerun()
                        
                        with col2:
                            # Run now button
                            if st.button("Run Now"):
                                st.info("Manually running report...")
                                try:
                                    # In a real implementation, this would actually generate the report
                                    # For this example, we'll just update the last_run time
                                    scheduler.update_report(
                                        selected_report.report_id,
                                        # Add any other parameters to update
                                    )
                                    selected_report.last_run = datetime.now().isoformat()
                                    scheduler.save_reports()
                                    st.success("Report generation started.")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error running report: {e}")
                        
                        with col3:
                            # Delete button
                            if st.button("Delete Report"):
                                st.warning(f"Are you sure you want to delete '{selected_report_name}'?")
                                confirm = st.checkbox("Confirm deletion")
                                
                                if confirm:
                                    scheduler.delete_report(selected_report.report_id)
                                    st.success("Report deleted.")
                                    st.rerun()
            else:
                st.info("You don't have any scheduled reports yet. Create one in the 'Create Report' tab.")
        else:
            st.warning("Please log in to view your scheduled reports.")
    
    with tab2:
        st.header("Create New Report")
        
        if not current_user:
            st.warning("Please log in to create scheduled reports.")
        else:
            # Form for creating a new report
            with st.form("create_report_form"):
                report_name = st.text_input("Report Name", placeholder="Monthly Sales Report")
                
                # Select template
                template = st.selectbox(
                    "Report Template",
                    [t.value for t in ReportTemplate]
                )
                
                # Configure scheduling
                col1, col2 = st.columns(2)
                with col1:
                    frequency = st.selectbox(
                        "Frequency",
                        [f.value for f in ReportFrequency]
                    )
                
                with col2:
                    format = st.selectbox(
                        "Format",
                        [f.value for f in ReportFormat]
                    )
                
                # Configure delivery
                delivery = st.selectbox(
                    "Delivery Method",
                    [d.value for d in DeliveryMethod]
                )
                
                # Email recipients
                recipients = []
                if delivery == DeliveryMethod.EMAIL.value:
                    recipients_text = st.text_area(
                        "Recipients (one email per line)",
                        placeholder="user@example.com\nuser2@example.com"
                    )
                    if recipients_text:
                        recipients = [email.strip() for email in recipients_text.split('\n') if email.strip()]
                
                # Optional parameters based on template
                parameters = {}
                if template == ReportTemplate.SALES_SUMMARY.value:
                    st.subheader("Template Parameters")
                    parameters["include_charts"] = st.checkbox("Include Charts", value=True)
                    parameters["include_tables"] = st.checkbox("Include Tables", value=True)
                
                # Submit button
                submit = st.form_submit_button("Create Report")
                
                if submit:
                    if not report_name:
                        st.error("Report name is required.")
                    else:
                        # Create the report
                        report_id = scheduler.create_report(
                            name=report_name,
                            template=template,
                            frequency=frequency,
                            format=format,
                            delivery=delivery,
                            recipients=recipients,
                            parameters=parameters,
                            created_by=current_user
                        )
                        
                        st.success(f"Report '{report_name}' created successfully.")
    
    with tab3:
        st.header("Report History")
        
        # In a real implementation, this would search the reports directory for past reports
        # For this example, we'll show a placeholder
        
        # List reports directory
        report_files = []
        if os.path.exists(scheduler.reports_dir):
            report_files = [f for f in os.listdir(scheduler.reports_dir) if os.path.isfile(os.path.join(scheduler.reports_dir, f)) and not f.endswith('.json')]
        
        if report_files:
            st.subheader("Generated Reports")
            
            # Display as a dataframe
            file_data = []
            for file in report_files:
                try:
                    # Parse filename for metadata
                    parts = file.split('_')
                    report_id = parts[0] if len(parts) > 0 else "unknown"
                    
                    # Get file attributes
                    file_path = os.path.join(scheduler.reports_dir, file)
                    file_size = os.path.getsize(file_path)
                    modified_time = os.path.getmtime(file_path)
                    
                    # Get report name
                    report_name = "Unknown"
                    report = next((r for r in scheduler.reports.values() if r.report_id == report_id), None)
                    if report:
                        report_name = report.name
                    
                    file_data.append({
                        "Filename": file,
                        "Report": report_name,
                        "Format": file.split('.')[-1].upper(),
                        "Size": f"{file_size / 1024:.1f} KB",
                        "Generated": datetime.fromtimestamp(modified_time).strftime("%Y-%m-%d %H:%M")
                    })
                except Exception as e:
                    print(f"Error parsing file {file}: {e}")
            
            # Sort by generation time (newest first)
            file_data.sort(key=lambda x: x["Generated"], reverse=True)
            
            # Display the files table
            st.dataframe(file_data, use_container_width=True)
            
            # File actions
            st.subheader("Download Reports")
            selected_file = st.selectbox("Select Report File", [f["Filename"] for f in file_data])
            
            if selected_file:
                file_path = os.path.join(scheduler.reports_dir, selected_file)
                with open(file_path, 'rb') as f:
                    file_contents = f.read()
                
                file_ext = selected_file.split('.')[-1]
                mime_type = {
                    'pdf': 'application/pdf',
                    'html': 'text/html',
                    'csv': 'text/csv',
                    'json': 'application/json',
                    'txt': 'text/plain'
                }.get(file_ext, 'application/octet-stream')
                
                st.download_button(
                    label="Download Report",
                    data=file_contents,
                    file_name=selected_file,
                    mime=mime_type
                )
        else:
            st.info("No generated reports found. Schedule a report or use the 'Run Now' button to generate reports.")


if __name__ == "__main__":
    # Example usage
    import streamlit as st
    import numpy as np  # Required for sample data
    
    st.set_page_config(page_title="Report Scheduler Demo", layout="wide")
    
    # Mock user in session state
    if 'user' not in st.session_state:
        from enum import Enum
        class MockUserRole(str, Enum):
            ADMIN = "admin"
        
        class MockUser:
            def __init__(self):
                self.username = "demo_user"
                self.role = MockUserRole.ADMIN
        
        st.session_state['user'] = MockUser()
    
    render_report_scheduler()