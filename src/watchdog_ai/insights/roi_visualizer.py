"""
Lead Source ROI visualization components.

This module provides visualization tools for ROI metrics.
"""

import pandas as pd
import numpy as np
import altair as alt
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ROIVisualizer:
    """Lead source ROI visualization utilities."""
    
    def __init__(self, 
                width: int = 700, 
                height: int = 400,
                color_scheme: str = "category10"):
        """
        Initialize the ROI visualizer.
        
        Args:
            width: Chart width
            height: Chart height
            color_scheme: Color scheme name
        """
        self.width = width
        self.height = height
        self.color_scheme = color_scheme
    
    def create_roi_bar_chart(self, df: pd.DataFrame, limit: int = 10) -> alt.Chart:
        """
        Create a bar chart showing ROI by lead source.
        
        Args:
            df: DataFrame from LeadSourceROI.process_dataframe
            limit: Maximum number of sources to display
            
        Returns:
            Altair chart object
        """
        if df is None or df.empty:
            return self._create_empty_chart("No ROI data available")
        
        # Limit to top sources by ROI
        if len(df) > limit:
            plot_df = df.nlargest(limit, 'ROI')
        else:
            plot_df = df.copy()
            
        # Add color category based on ROI value
        def get_roi_category(roi):
            if not np.isfinite(roi):
                return "Infinite"
            elif roi < 0:
                return "Negative"
            elif roi <= 1:
                return "Low"
            else:
                return "High"
                
        plot_df['ROI_Category'] = plot_df['ROI'].apply(get_roi_category)
        
        # Define color scale
        color_scale = alt.Scale(
            domain=['Negative', 'Low', 'High', 'Infinite'],
            range=['#e74c3c', '#f39c12', '#2ecc71', '#3498db']
        )
        
        # Create chart
        chart = alt.Chart(plot_df).mark_bar().encode(
            x=alt.X('ROI:Q', title='ROI (Return on Investment)'),
            y=alt.Y('LeadSource:N', 
                   sort=alt.EncodingSortField(field="ROI", order="descending"),
                   title='Lead Source'),
            color=alt.Color('ROI_Category:N', scale=color_scale, legend=None),
            tooltip=[
                alt.Tooltip('LeadSource:N', title='Lead Source'),
                alt.Tooltip('ROI:Q', title='ROI', format='.2f'),
                alt.Tooltip('ROIPercentage:N', title='ROI %'),
                alt.Tooltip('LeadCount:Q', title='Lead Count'),
                alt.Tooltip('TotalRevenue:Q', title='Revenue', format='$,.2f'),
                alt.Tooltip('MonthlyCost:Q', title='Monthly Cost', format='$,.2f')
            ]
        ).properties(
            width=self.width,
            height=self.height,
            title='Lead Source ROI Comparison'
        )
        
        # Add a rule to mark breakeven point (ROI = 0)
        breakeven = alt.Chart(pd.DataFrame({'x': [0]})).mark_rule(
            strokeDash=[4, 4],
            color='gray'
        ).encode(x='x:Q')
        
        return chart + breakeven
    
    def create_roi_rank_chart(self, df: pd.DataFrame) -> alt.Chart:
        """
        Create a ranked lead source ROI chart.
        
        Args:
            df: DataFrame from LeadSourceROI.process_dataframe
            
        Returns:
            Altair chart object
        """
        if df is None or df.empty:
            return self._create_empty_chart("No ROI data available")
        
        # Create a ranking column
        ranked_df = df.copy()
        ranked_df['Rank'] = range(1, len(df) + 1)
        
        # Calculate text position
        max_roi = ranked_df['ROI'].max()
        if max_roi == float('inf'):
            # Find the second highest ROI for scaling
            filtered = ranked_df[ranked_df['ROI'] != float('inf')]
            max_roi = filtered['ROI'].max() if not filtered.empty else 3
        
        # Set a reasonable max for infinite values
        ranked_df['ROI_Capped'] = ranked_df['ROI'].apply(
            lambda x: 1.2 * max_roi if x == float('inf') else x
        )
        
        # Create rank chart
        chart = alt.Chart(ranked_df).mark_bar().encode(
            x=alt.X('ROI_Capped:Q', title='ROI'),
            y=alt.Y('Rank:O', title='Rank', sort='ascending'),
            color=alt.Color('LeadSource:N', scale=alt.Scale(scheme=self.color_scheme)),
            tooltip=[
                alt.Tooltip('Rank:O', title='Rank'),
                alt.Tooltip('LeadSource:N', title='Lead Source'),
                alt.Tooltip('ROIPercentage:N', title='ROI %'),
                alt.Tooltip('LeadCount:Q', title='Lead Count'),
                alt.Tooltip('CostPerLead:Q', title='Cost Per Lead', format='$,.2f')
            ]
        ).properties(
            width=self.width,
            height=self.height,
            title='Lead Source ROI Ranking'
        )
        
        # Add source labels
        text = alt.Chart(ranked_df).mark_text(
            align='left',
            baseline='middle',
            dx=5
        ).encode(
            x=alt.X('ROI_Capped:Q'),
            y=alt.Y('Rank:O'),
            text=alt.Text('LeadSource:N')
        )
        
        return chart + text
    
    def create_roi_trend_chart(self, 
                              trend_data: pd.DataFrame,
                              source_col: str = 'LeadSource',
                              date_col: str = 'Date',
                              roi_col: str = 'ROI') -> alt.Chart:
        """
        Create a line chart showing ROI trend over time.
        
        Args:
            trend_data: DataFrame with ROI trend data
            source_col: Column with lead source names
            date_col: Column with dates
            roi_col: Column with ROI values
            
        Returns:
            Altair chart object
        """
        if trend_data is None or trend_data.empty:
            return self._create_empty_chart("No trend data available")
        
        # Ensure date column is datetime
        if pd.api.types.is_datetime64_any_dtype(trend_data[date_col]):
            trend_data = trend_data.copy()
            trend_data[date_col] = pd.to_datetime(trend_data[date_col])
        
        # Create chart
        chart = alt.Chart(trend_data).mark_line(point=True).encode(
            x=alt.X(f'{date_col}:T', title='Date'),
            y=alt.Y(f'{roi_col}:Q', title='ROI'),
            color=alt.Color(f'{source_col}:N', scale=alt.Scale(scheme=self.color_scheme)),
            tooltip=[
                alt.Tooltip(f'{date_col}:T', title='Date'),
                alt.Tooltip(f'{source_col}:N', title='Lead Source'),
                alt.Tooltip(f'{roi_col}:Q', title='ROI', format='.2f')
            ]
        ).properties(
            width=self.width,
            height=self.height,
            title='ROI Trend Over Time'
        )
        
        # Add a rule to mark breakeven point (ROI = 0)
        breakeven = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(
            strokeDash=[4, 4],
            color='gray'
        ).encode(y='y:Q')
        
        return chart + breakeven
    
    def create_metrics_tooltip_chart(self, 
                                   df: pd.DataFrame,
                                   metrics: List[str] = ['LeadCount', 'MonthlyCost', 'CostPerLead']) -> alt.Chart:
        """
        Create a chart with metrics tooltips.
        
        Args:
            df: DataFrame from LeadSourceROI.process_dataframe
            metrics: List of metrics to include
            
        Returns:
            Altair chart object
        """
        if df is None or df.empty:
            return self._create_empty_chart("No data available")
        
        # Filter available metrics
        available_metrics = [m for m in metrics if m in df.columns]
        if not available_metrics:
            return self._create_empty_chart("No valid metrics selected")
        
        # Prepare data
        plot_df = df.copy()
        # Melt the DataFrame to get metrics in long format
        melted = pd.melt(
            plot_df, 
            id_vars=['LeadSource'], 
            value_vars=available_metrics,
            var_name='Metric', 
            value_name='Value'
        )
        
        # Create chart
        chart = alt.Chart(melted).mark_bar().encode(
            x=alt.X('Value:Q', title='Value'),
            y=alt.Y('LeadSource:N', title='Lead Source'),
            row=alt.Row('Metric:N', title=''),
            color=alt.Color('LeadSource:N', scale=alt.Scale(scheme=self.color_scheme)),
            tooltip=[
                alt.Tooltip('LeadSource:N', title='Lead Source'),
                alt.Tooltip('Metric:N', title='Metric'),
                alt.Tooltip('Value:Q', title='Value', format=',.2f')
            ]
        ).properties(
            width=self.width // 2,
            height=self.height // len(available_metrics),
            title='Lead Source Metrics Comparison'
        )
        
        return chart
    
    def create_combined_dashboard(self, df: pd.DataFrame, trend_data: Optional[pd.DataFrame] = None) -> alt.VConcatChart:
        """
        Create a complete ROI visualization dashboard.
        
        Args:
            df: DataFrame from LeadSourceROI.process_dataframe
            trend_data: Optional DataFrame with trend data
            
        Returns:
            Altair chart object
        """
        if df is None or df.empty:
            return self._create_empty_chart("No ROI data available")
        
        # Create ROI bar chart
        roi_chart = self.create_roi_bar_chart(df)
        
        # Create metrics chart
        metric_chart = self.create_metrics_tooltip_chart(df)
        
        # Create trend chart if data available
        if trend_data is not None and not trend_data.empty:
            trend_chart = self.create_roi_trend_chart(trend_data)
            return alt.vconcat(roi_chart, alt.hconcat(metric_chart, trend_chart))
        
        # Create default dashboard without trend data
        return alt.vconcat(roi_chart, metric_chart)
    
    def _create_empty_chart(self, message: str = "No data available") -> alt.Chart:
        """
        Create an empty chart with a message.
        
        Args:
            message: Message to display
            
        Returns:
            Altair chart object
        """
        return alt.Chart(pd.DataFrame({'text': [message]})).mark_text(
            align='center',
            baseline='middle',
            fontSize=18,
            color='gray'
        ).encode(
            text='text:N'
        ).properties(
            width=self.width,
            height=self.height
        )