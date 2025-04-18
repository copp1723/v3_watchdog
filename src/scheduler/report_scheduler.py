"""
Report Scheduler Module for V3 Watchdog AI.

Provides functionality for scheduling and generating reports at regular intervals.
"""

import os
import uuid
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Callable

from .base_scheduler import BaseScheduler, ScheduledReport, ReportFrequency, ReportFormat, DeliveryMethod
from .notification_service import NotificationService
from .reports.sales_report import SalesReportGenerator
from .reports.inventory_report import InventoryReportGenerator


class ReportTemplate:
    """Available report templates."""
    SALES_SUMMARY = "sales_summary"
    INVENTORY_HEALTH = "inventory_health"
    PROFITABILITY = "profitability"
    LEAD_SOURCE = "lead_source"
    CUSTOM = "custom"


class ReportScheduler(BaseScheduler):
    """Manages scheduled reports and their execution."""
    
    def __init__(self, reports_dir: str = None):
        """
        Initialize the report scheduler.
        
        Args:
            reports_dir: Directory to store reports and configurations
        """
        super().__init__(reports_dir)
        
        # Initialize notification service
        self.notification_service = NotificationService(reports_dir)
        
        # Initialize report generators
        self.report_generators = {
            ReportTemplate.SALES_SUMMARY: SalesReportGenerator(),
            ReportTemplate.INVENTORY_HEALTH: InventoryReportGenerator(),
            # Additional generators can be added here
        }
        
        # Load existing reports
        self.load_reports()
    
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
        
        # Save to file
        self.save_reports()
        
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
    
    def run_due(self) -> None:
        """Run all reports that are currently due."""
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
    
    def _generate_report(self, report: ScheduledReport) -> None:
        """
        Generate a report based on its configuration.
        
        Args:
            report: The report to generate
        """
        # Load data based on template
        df = self._load_data_for_template(report.template, report.parameters)
        
        # Get the appropriate report generator
        generator = self.report_generators.get(report.template)
        if not generator:
            print(f"No generator available for template {report.template}")
            return
        
        # Generate report content
        content = generator.generate(df, report.parameters)
        
        # Add metadata
        content["metadata"] = {
            "report_id": report.report_id,
            "frequency": report.frequency,
            "created_by": report.created_by
        }
        
        # Format report
        formatted_report = self._format_report(content, report.format)
        
        # Deliver report
        self.notification_service.notify(report, formatted_report)
    
    def _load_data_for_template(self, template: str, 
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
        import json
        import io
        
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