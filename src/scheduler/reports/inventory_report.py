"""
Inventory Report Generator Module for V3 Watchdog AI.

Provides functionality for generating inventory health reports.
"""

import base64
from datetime import datetime
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, Any, Optional

from . import ReportGenerator

class InventoryReportGenerator(ReportGenerator):
    """Generator for inventory health reports."""
    
    def generate(self, data: pd.DataFrame, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate an inventory report from data.
        
        Args:
            data: DataFrame containing inventory data
            parameters: Additional parameters for report generation
            
        Returns:
            Dictionary with report content
        """
        parameters = parameters or {}
        include_charts = parameters.get("include_charts", True)
        include_tables = parameters.get("include_tables", True)
        
        content = {
            "title": parameters.get("title", "Inventory Health Report"),
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
        total_units = len(data)
        if 'DaysInInventory' in data.columns:
            avg_days = data['DaysInInventory'].mean()
            aged_90_plus = (data['DaysInInventory'] > 90).sum()
            aged_pct = (aged_90_plus / total_units) * 100
            
            content["summary"] = f"Inventory Health: {total_units} total units with average age of {avg_days:.1f} days. {aged_90_plus} units ({aged_pct:.1f}%) are over 90 days old."
        else:
            content["summary"] = f"Inventory Health: {total_units} total units."
        
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
            data: DataFrame containing inventory data
            content: Report content dictionary to update
        """
        if 'DaysInInventory' in data.columns:
            # 1. Age distribution
            bins = [0, 30, 60, 90, float('inf')]
            labels = ['<30 days', '30-60 days', '61-90 days', '>90 days']
            data['Age Category'] = pd.cut(data['DaysInInventory'], bins=bins, labels=labels)
            age_counts = data['Age Category'].value_counts().reindex(labels)
            
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
            if 'VehicleMake' in data.columns:
                make_age = data.groupby('VehicleMake')['DaysInInventory'].mean().reset_index()
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
    
    def _add_tables(self, data: pd.DataFrame, content: Dict[str, Any]) -> None:
        """
        Add tables to the report content.
        
        Args:
            data: DataFrame containing inventory data
            content: Report content dictionary to update
        """
        if 'DaysInInventory' in data.columns:
            # Add age category if not already present
            if 'Age Category' not in data.columns:
                bins = [0, 30, 60, 90, float('inf')]
                labels = ['<30 days', '30-60 days', '61-90 days', '>90 days']
                data['Age Category'] = pd.cut(data['DaysInInventory'], bins=bins, labels=labels)
            
            # 1. Age summary
            age_summary = data.groupby('Age Category').agg(
                Units=('VIN', 'count'),
                AvgDays=('DaysInInventory', 'mean')
            ).reset_index()
            
            content["tables"].append({
                "title": "Inventory Age Summary",
                "columns": ["Age Category", "Units", "Avg Days"],
                "data": age_summary.to_dict('records')
            })
            
            # 2. Make/Model summary
            if 'VehicleMake' in data.columns and 'VehicleModel' in data.columns:
                make_model_summary = data.groupby(['VehicleMake', 'VehicleModel']).agg(
                    Units=('VIN', 'count'),
                    AvgDays=('DaysInInventory', 'mean')
                ).reset_index()
                
                content["tables"].append({
                    "title": "Inventory by Make/Model",
                    "columns": ["Make", "Model", "Units", "Avg Days"],
                    "data": make_model_summary.to_dict('records')
                })
    
    def _fig_to_base64(self, fig) -> str:
        """Convert a plotly figure to base64 encoded string."""
        img_bytes = fig.to_image(format="png", engine="kaleido")
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')
        return img_b64