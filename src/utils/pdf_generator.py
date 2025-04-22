"""
PDF Generator Module for V3 Watchdog AI.

Provides functionality for generating PDF reports from Streamlit components or data.
"""

import os
import io
import base64
import tempfile
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple, Callable

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from weasyprint import HTML
import pdfkit
import streamlit as st

logger = logging.getLogger(__name__)

# Options for PDF generation
PDF_OPTIONS = {
    'page-size': 'Letter',
    'margin-top': '0.75in',
    'margin-right': '0.75in',
    'margin-bottom': '0.75in',
    'margin-left': '0.75in',
    'encoding': 'UTF-8',
    'custom-header': [
        ('Accept-Encoding', 'gzip')
    ],
    'no-outline': None,
    'enable-local-file-access': None
}

class PDFGenerator:
    """Generator for PDF reports from Streamlit components or raw data."""
    
    def __init__(self, output_dir: str = None):
        """
        Initialize the PDF generator.
        
        Args:
            output_dir: Directory to store generated PDF files
        """
        self.output_dir = output_dir or os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                                    "data", "reports")
        os.makedirs(self.output_dir, exist_ok=True)
        
    def generate_from_html(self, html_content: str, filename: str = None) -> str:
        """
        Generate a PDF file from HTML content.
        
        Args:
            html_content: HTML content to convert to PDF
            filename: Filename for the generated PDF (optional)
            
        Returns:
            Path to the generated PDF file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"report_{timestamp}.pdf"
        
        # Ensure PDF extension
        if not filename.endswith('.pdf'):
            filename += '.pdf'
        
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            # Primary method with pdfkit/wkhtmltopdf
            try:
                pdfkit.from_string(html_content, filepath, options=PDF_OPTIONS)
                logger.info(f"Generated PDF with pdfkit: {filepath}")
                return filepath
            except Exception as e:
                logger.warning(f"pdfkit PDF generation failed: {e}, falling back to WeasyPrint")
                
            # Fallback to WeasyPrint 
            HTML(string=html_content).write_pdf(filepath)
            logger.info(f"Generated PDF with WeasyPrint: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to generate PDF: {e}")
            raise
    
    def generate_from_streamlit(self, report_func: Callable, params: Dict[str, Any] = None, 
                             filename: str = None, width: int = 800, height: int = None) -> str:
        """
        Generate a PDF from a Streamlit component.
        
        This method uses a temporary Streamlit session to render components
        and then captures the output as PDF.
        
        Args:
            report_func: Function that uses Streamlit components to render a report
            params: Parameters to pass to the report function
            filename: Filename for the generated PDF (optional)
            width: Width for the Streamlit render (pixels)
            height: Height for the Streamlit render (pixels, auto if None)
            
        Returns:
            Path to the generated PDF file
        """
        try:
            # Create a temporary HTML file
            with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
                temp_html_path = f.name
            
            # Set up parameters with defaults
            params = params or {}
            
            # Use Streamlit to generate HTML content
            streamlit_html = self._run_streamlit_to_html(report_func, params)
            
            # Save the HTML
            with open(temp_html_path, 'w', encoding='utf-8') as f:
                f.write(streamlit_html)
            
            # Generate the PDF from the HTML
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"streamlit_report_{timestamp}.pdf"
            
            # Ensure PDF extension
            if not filename.endswith('.pdf'):
                filename += '.pdf'
            
            filepath = os.path.join(self.output_dir, filename)
            
            # Convert HTML to PDF
            try:
                pdfkit.from_file(temp_html_path, filepath, options=PDF_OPTIONS)
                logger.info(f"Generated PDF from Streamlit with pdfkit: {filepath}")
            except Exception as e:
                logger.warning(f"pdfkit PDF generation failed: {e}, falling back to WeasyPrint")
                HTML(filename=temp_html_path).write_pdf(filepath)
                logger.info(f"Generated PDF from Streamlit with WeasyPrint: {filepath}")
            
            # Clean up the temporary file
            try:
                os.unlink(temp_html_path)
            except Exception as e:
                logger.warning(f"Failed to remove temporary HTML file: {e}")
            
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to generate PDF from Streamlit: {e}")
            raise
    
    def _run_streamlit_to_html(self, report_func: Callable, params: Dict[str, Any]) -> str:
        """
        Run a Streamlit function and capture the output as HTML.
        
        Args:
            report_func: Function that uses Streamlit components
            params: Parameters to pass to the function
            
        Returns:
            HTML string of rendered Streamlit components
        """
        # For this mock implementation, we'll generate a basic HTML template
        # In a production system, this would use Streamlit's headless mode or
        # a similar approach to capture the rendered HTML
        
        # Placeholder function to simulate the Streamlit rendering
        # In production, you would need to run a headless Streamlit session
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Watchdog AI Report</title>
            <style>
                body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; margin: 0; padding: 0; }
                .content { max-width: 1000px; margin: 0 auto; padding: 20px; }
                .header { background-color: #2c3e50; color: white; padding: 20px; text-align: center; }
                .footer { background-color: #f5f5f5; padding: 10px 20px; font-size: 12px; color: #666; text-align: center; }
                table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .chart-container { margin: 20px 0; text-align: center; }
                h1, h2, h3 { color: #2c3e50; }
                .metrics-container { display: flex; flex-wrap: wrap; justify-content: space-between; margin-bottom: 20px; }
                .metric-box { background: #f5f7fa; border-radius: 5px; padding: 15px; margin-bottom: 15px; width: 45%; }
                .metric-value { font-size: 24px; font-weight: bold; color: #2c3e50; }
                .metric-label { font-size: 14px; color: #7f8c8d; }
                .banner { background-color: #e8f4fc; border-left: 4px solid #3498db; padding: 10px 15px; margin-bottom: 20px; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Watchdog AI Executive Report</h1>
                <p>Generated: {date}</p>
            </div>
            
            <div class="content">
                <div class="banner">
                    <p><strong>Dealership:</strong> {dealership_name} | <strong>Report Type:</strong> {report_type} | <strong>System Status:</strong> {system_status}</p>
                </div>
                
                <!-- Sales Summary Section -->
                <h2>Sales Summary</h2>
                <div class="metrics-container">
                    <div class="metric-box">
                        <div class="metric-value">{total_sales}</div>
                        <div class="metric-label">Total Sales</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value">${total_revenue}</div>
                        <div class="metric-label">Total Revenue</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value">${avg_gross}</div>
                        <div class="metric-label">Average Gross Profit</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value">{avg_days}</div>
                        <div class="metric-label">Average Days to Close</div>
                    </div>
                </div>
                
                <div class="chart-container">
                    <img src="data:image/png;base64,{sales_chart}" alt="Sales Trend" style="max-width: 100%;" />
                </div>
                
                <!-- Lead Source Analysis -->
                <h2>Lead Source Breakdown</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Lead Source</th>
                            <th>Count</th>
                            <th>Conversion Rate</th>
                            <th>Avg. Gross Profit</th>
                        </tr>
                    </thead>
                    <tbody>
                        {lead_source_rows}
                    </tbody>
                </table>
                
                <div class="chart-container">
                    <img src="data:image/png;base64,{lead_chart}" alt="Lead Source Distribution" style="max-width: 100%;" />
                </div>
                
                <!-- Inventory Analysis -->
                <h2>Inventory Heatmap</h2>
                <div class="chart-container">
                    <img src="data:image/png;base64,{inventory_chart}" alt="Inventory Heatmap" style="max-width: 100%;" />
                </div>
                
                <!-- Top Performers -->
                <h2>Top Performing Sales Representatives</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Name</th>
                            <th>Sales</th>
                            <th>Total Gross</th>
                            <th>Avg. Gross</th>
                        </tr>
                    </thead>
                    <tbody>
                        {top_performer_rows}
                    </tbody>
                </table>
                
                <!-- KPI Deltas -->
                <h2>Week-over-Week KPI Changes</h2>
                <table>
                    <thead>
                        <tr>
                            <th>KPI</th>
                            <th>Current Week</th>
                            <th>Previous Week</th>
                            <th>Change</th>
                        </tr>
                    </thead>
                    <tbody>
                        {kpi_delta_rows}
                    </tbody>
                </table>
            </div>
            
            <div class="footer">
                <p>Watchdog AI Executive Report | System Uptime: {uptime} | Data Freshness: {data_freshness}</p>
                <p>&copy; {year} Watchdog AI. All rights reserved.</p>
            </div>
        </body>
        </html>
        """
        
        # Generate dummy data for placeholders
        # In a real implementation, these would come from the Streamlit rendering
        current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        dealership_name = params.get("dealership_name", "Your Dealership")
        report_type = params.get("report_type", "Weekly Executive Summary")
        
        # Sample data
        total_sales = params.get("total_sales", 128)
        total_revenue = params.get("total_revenue", "1,245,500")
        avg_gross = params.get("avg_gross", "3,240")
        avg_days = params.get("avg_days", "45")
        
        # Generate sample charts as base64
        sales_chart = self._generate_sample_sales_chart()
        lead_chart = self._generate_sample_lead_chart()
        inventory_chart = self._generate_sample_inventory_chart()
        
        # Sample rows for tables
        lead_source_rows = self._generate_sample_lead_source_rows()
        top_performer_rows = self._generate_sample_top_performer_rows()
        kpi_delta_rows = self._generate_sample_kpi_delta_rows()
        
        # System stats
        uptime = "99.8%"
        data_freshness = "Last Updated 2 hours ago"
        system_status = "Healthy"
        
        # Fill template
        html_content = html_template.format(
            date=current_date,
            dealership_name=dealership_name,
            report_type=report_type,
            total_sales=total_sales,
            total_revenue=total_revenue,
            avg_gross=avg_gross,
            avg_days=avg_days,
            sales_chart=sales_chart,
            lead_chart=lead_chart,
            inventory_chart=inventory_chart,
            lead_source_rows=lead_source_rows,
            top_performer_rows=top_performer_rows,
            kpi_delta_rows=kpi_delta_rows,
            uptime=uptime,
            data_freshness=data_freshness,
            system_status=system_status,
            year=datetime.now().year
        )
        
        return html_content
    
    def _generate_sample_sales_chart(self) -> str:
        """Generate a sample sales chart and return as base64 image."""
        plt.figure(figsize=(10, 5))
        
        # Sample data
        dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='D')
        sales = np.random.normal(20, 5, size=len(dates)).cumsum()
        
        plt.plot(dates, sales, marker='o', linestyle='-', color='#3498db')
        plt.title('Daily Sales Trend')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Sales')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Convert to base64
        img_b64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return img_b64
    
    def _generate_sample_lead_chart(self) -> str:
        """Generate a sample lead source chart and return as base64 image."""
        plt.figure(figsize=(8, 6))
        
        # Sample data
        lead_sources = ['Website', 'Walk-in', 'Referral', 'Third-party', 'Phone']
        values = [35, 25, 20, 15, 5]
        colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']
        
        plt.pie(values, labels=lead_sources, colors=colors, autopct='%1.1f%%', 
                startangle=90, shadow=False)
        plt.axis('equal')
        plt.title('Lead Source Distribution')
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Convert to base64
        img_b64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return img_b64
    
    def _generate_sample_inventory_chart(self) -> str:
        """Generate a sample inventory heatmap and return as base64 image."""
        plt.figure(figsize=(10, 6))
        
        # Sample data
        makes = ['Toyota', 'Honda', 'Ford', 'Chevrolet', 'BMW']
        types = ['Sedan', 'SUV', 'Truck', 'Coupe']
        
        data = np.random.randint(5, 30, size=(len(makes), len(types)))
        
        plt.imshow(data, cmap='YlGnBu')
        plt.colorbar(label='Units in Stock')
        
        # Set ticks and labels
        plt.xticks(np.arange(len(types)), types)
        plt.yticks(np.arange(len(makes)), makes)
        
        # Rotate x-axis labels
        plt.xticks(rotation=45)
        
        # Loop over data dimensions and add text annotations
        for i in range(len(makes)):
            for j in range(len(types)):
                plt.text(j, i, data[i, j],
                        ha="center", va="center", color="black")
        
        plt.title('Inventory Heatmap by Make and Type')
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Convert to base64
        img_b64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return img_b64
    
    def _generate_sample_lead_source_rows(self) -> str:
        """Generate sample HTML table rows for lead sources."""
        data = [
            {"source": "Website", "count": 45, "conversion": "18.5%", "avg_gross": "$3,450"},
            {"source": "Walk-in", "count": 32, "conversion": "22.1%", "avg_gross": "$3,870"},
            {"source": "Referral", "count": 28, "conversion": "31.2%", "avg_gross": "$4,120"},
            {"source": "Third-party", "count": 22, "conversion": "14.8%", "avg_gross": "$2,980"},
            {"source": "Phone", "count": 8, "conversion": "20.5%", "avg_gross": "$3,210"}
        ]
        
        rows = ""
        for item in data:
            rows += f"""
            <tr>
                <td>{item['source']}</td>
                <td>{item['count']}</td>
                <td>{item['conversion']}</td>
                <td>{item['avg_gross']}</td>
            </tr>
            """
        
        return rows
    
    def _generate_sample_top_performer_rows(self) -> str:
        """Generate sample HTML table rows for top performers."""
        data = [
            {"name": "John Doe", "sales": 18, "total_gross": "$72,450", "avg_gross": "$4,025"},
            {"name": "Jane Smith", "sales": 15, "total_gross": "$61,800", "avg_gross": "$4,120"},
            {"name": "Robert Johnson", "sales": 14, "total_gross": "$53,900", "avg_gross": "$3,850"},
            {"name": "Emily Wilson", "sales": 12, "total_gross": "$43,200", "avg_gross": "$3,600"},
            {"name": "Michael Brown", "sales": 10, "total_gross": "$35,500", "avg_gross": "$3,550"}
        ]
        
        rows = ""
        for item in data:
            rows += f"""
            <tr>
                <td>{item['name']}</td>
                <td>{item['sales']}</td>
                <td>{item['total_gross']}</td>
                <td>{item['avg_gross']}</td>
            </tr>
            """
        
        return rows
    
    def _generate_sample_kpi_delta_rows(self) -> str:
        """Generate sample HTML table rows for KPI deltas."""
        data = [
            {"kpi": "Total Sales", "current": 128, "previous": 115, "change": "+11.3%", "positive": True},
            {"kpi": "Conversion Rate", "current": "18.5%", "previous": "17.2%", "change": "+1.3%", "positive": True},
            {"kpi": "Avg. Days to Close", "current": 45, "previous": 48, "change": "-6.3%", "positive": True},
            {"kpi": "Avg. Gross Profit", "current": "$3,240", "previous": "$3,310", "change": "-2.1%", "positive": False},
            {"kpi": "Inventory Turnover", "current": "34.5%", "previous": "32.8%", "change": "+1.7%", "positive": True}
        ]
        
        rows = ""
        for item in data:
            change_color = 'green' if item['positive'] else 'red'
            rows += f"""
            <tr>
                <td>{item['kpi']}</td>
                <td>{item['current']}</td>
                <td>{item['previous']}</td>
                <td style="color: {change_color};">{item['change']}</td>
            </tr>
            """
        
        return rows


def generate_executive_pdf(
    data: Dict[str, Any] = None,
    output_file: str = None,
    title: str = "Weekly Executive Summary",
    dealership: str = None
) -> str:
    """
    Generate an executive PDF report using Streamlit components.
    
    Args:
        data: Data dictionary containing report data
        output_file: Path for output file
        title: Report title
        dealership: Dealership name
        
    Returns:
        Path to the generated PDF file
    """
    # Create PDF generator
    generator = PDFGenerator()
    
    # Create parameters for report
    params = {
        "report_type": title,
        "dealership_name": dealership or "Your Dealership"
    }
    
    # Update with provided data
    if data:
        params.update(data)
    
    # Generate the report
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"executive_report_{timestamp}.pdf"
    
    # Generate PDF using the Streamlit wrapper
    pdf_path = generator.generate_from_streamlit(
        report_func=_build_executive_report,
        params=params,
        filename=output_file
    )
    
    return pdf_path


def generate_from_streamlit_ui(block: Callable, filename: str = None, **kwargs) -> str:
    """
    Generate a PDF from a Streamlit UI block.
    
    Args:
        block: Streamlit UI function to render
        filename: Output filename (optional)
        **kwargs: Additional parameters to pass to the UI function
        
    Returns:
        Path to the generated PDF file
    """
    generator = PDFGenerator()
    return generator.generate_from_streamlit(block, kwargs, filename)


def _build_executive_report(params: Dict[str, Any]) -> None:
    """
    Build a Streamlit executive report.
    
    Note: This function would normally use Streamlit's st.* functions to build
    a complete UI that would be captured for the PDF. This is a simplified version.
    
    Args:
        params: Parameters for the report
    """
    # In a real implementation, this would have all the st.* commands to build the report
    st.title(params.get("report_type", "Weekly Executive Summary"))
    st.write(f"Dealership: {params.get('dealership_name', 'Your Dealership')}")
    st.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Would continue with charts, tables, and other Streamlit UI components...
    # For this example, we're using the mock HTML template in PDFGenerator._run_streamlit_to_html


# Streamlit component that can be directly embedded in a Streamlit app
def render_pdf_export_button(report_data: Dict[str, Any], title: str = "Export to PDF"):
    """
    Render a button to export the current report to PDF.
    
    Args:
        report_data: Data for the report
        title: Button title
    """
    if st.button(title):
        with st.spinner("Generating PDF report..."):
            try:
                pdf_path = generate_executive_pdf(report_data)
                
                # Create a download link
                with open(pdf_path, "rb") as f:
                    pdf_data = f.read()
                
                b64_pdf = base64.b64encode(pdf_data).decode('utf-8')
                download_filename = os.path.basename(pdf_path)
                
                href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="{download_filename}">Download PDF Report</a>'
                st.markdown(href, unsafe_allow_html=True)
                st.success(f"PDF generated successfully: {download_filename}")
                
            except Exception as e:
                st.error(f"Failed to generate PDF: {e}")