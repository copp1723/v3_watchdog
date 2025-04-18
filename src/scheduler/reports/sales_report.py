"""
Sales Report Generator Module for V3 Watchdog AI.

Provides functionality for generating sales summary reports.
"""

import base64
from datetime import datetime
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, Any, Optional

from . import ReportGenerator

class SalesReportGenerator(ReportGenerator):
    """Generator for sales summary reports."""
    
    def generate(self, data: pd.DataFrame, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a sales report from data.
        
        Args:
            data: DataFrame containing sales data
            parameters: Additional parameters for report generation
            
        Returns:
            Dictionary with report content
        """
        parameters = parameters or {}
        include_charts = parameters.get("include_charts", True)
        include_tables = parameters.get("include_tables", True)
        
        content = {
            "title": parameters.get("title", "Sales Summary Report"),
            "generated_at": datetime.now().isoformat(),
            "summary": "",
            "charts": [],
            "tables": [],
            "metadata": parameters.get("metadata", {})
        }
        
        if data.empty:
            content["summary"] = "No data available for this report."
            return content
        
        # Generate summary
        total_sales = len(data)
        if 'Gross_Profit' in data.columns:
            total_gross = data['Gross_Profit'].sum()
            avg_gross = data['Gross_Profit'].mean()
            content["summary"] = f"Sales Summary: {total_sales} total sales with average gross of ${avg_gross:.2f} and total gross of ${total_gross:.2f}."
        else:
            content["summary"] = f"Sales Summary: {total_sales} total sales."
        
        # Add charts if requested
        if include_charts:
            self._add_charts(data, content)
        
        # Add tables if requested
        if include_tables:
            self._add_tables(data, content)
        
        return content
    
    def _add_charts(self, data: pd.DataFrame, content: Dict[str, Any]) -> None:
        """
        Add charts to the report content.
        
        Args:
            data: DataFrame containing sales data
            content: Report content dictionary to update
        """
        # 1. Monthly sales trend
        if 'Sale_Date' in data.columns:
            data['Month'] = data['Sale_Date'].dt.to_period('M').astype(str)
            monthly_sales = data.groupby('Month').size().reset_index(name='count')
            
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
        if 'VehicleMake' in data.columns:
            make_counts = data['VehicleMake'].value_counts().reset_index()
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
    
    def _add_tables(self, data: pd.DataFrame, content: Dict[str, Any]) -> None:
        """
        Add tables to the report content.
        
        Args:
            data: DataFrame containing sales data
            content: Report content dictionary to update
        """
        # 1. Sales summary by month
        if 'Sale_Date' in data.columns and 'Gross_Profit' in data.columns:
            data['Month'] = data['Sale_Date'].dt.to_period('M').astype(str)
            monthly_summary = data.groupby('Month').agg(
                Sales=('VIN', 'count'),
                AvgGross=('Gross_Profit', 'mean'),
                TotalGross=('Gross_Profit', 'sum')
            ).reset_index()
            
            content["tables"].append({
                "title": "Monthly Sales Summary",
                "columns": ["Month", "Sales", "Avg Gross", "Total Gross"],
                "data": monthly_summary.to_dict('records')
            })
        
        # 2. Sales by lead source
        if 'LeadSource' in data.columns:
            lead_summary = data.groupby('LeadSource').agg(
                Sales=('VIN', 'count')
            ).reset_index()
            
            if 'Gross_Profit' in data.columns:
                lead_profit = data.groupby('LeadSource').agg(
                    AvgGross=('Gross_Profit', 'mean'),
                    TotalGross=('Gross_Profit', 'sum')
                ).reset_index()
                
                lead_summary = lead_summary.merge(lead_profit, on='LeadSource')
                
                content["tables"].append({
                    "title": "Sales by Lead Source",
                    "columns": ["Lead Source", "Sales", "Avg Gross", "Total Gross"],
                    "data": lead_summary.to_dict('records')
                })
            else:
                content["tables"].append({
                    "title": "Sales by Lead Source",
                    "columns": ["Lead Source", "Sales"],
                    "data": lead_summary.to_dict('records')
                })
    
    def _fig_to_base64(self, fig) -> str:
        """Convert a plotly figure to base64 encoded string."""
        img_bytes = fig.to_image(format="png", engine="kaleido")
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')
        return img_b64