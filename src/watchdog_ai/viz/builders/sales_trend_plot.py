"""
Sales trend visualization chart builder.

This module provides functions to create sales trend charts with time-series data,
supporting responsive sizing for both desktop and mobile views.
"""

import pandas as pd
import altair as alt
import logging
from typing import Dict, Optional, Any, Union, List

from watchdog_ai.viz.utils import get_responsive_dimensions, configure_responsive_chart
from watchdog_ai.viz.constants import (
    DEFAULT_COLOR_SCHEME,
    DEFAULT_ANIMATION_DURATION
)

logger = logging.getLogger(__name__)

def sales_trend_plot(
    df: pd.DataFrame,
    date_column: Optional[str] = None,
    value_column: Optional[str] = None,
    category_column: Optional[str] = None,
    title: Optional[str] = "Sales Trend",
    x_label: Optional[str] = None,
    y_label: Optional[str] = "Sales",
    color_scheme: str = DEFAULT_COLOR_SCHEME,
    container_width: Optional[int] = None,
    height: Optional[int] = None,
    show_points: bool = True,
    animate: bool = True,
    tooltip: Union[bool, List[str]] = True,
    interactive: bool = True
) -> alt.Chart:
    """
    Create a sales trend line chart with time-series data.
    
    Args:
        df: DataFrame containing sales data
        date_column: Column name with date/time data (auto-detected if None)
        value_column: Column name with sales values (auto-detected if None)
        category_column: Optional column for series grouping (e.g., by product category)
        title: Chart title
        x_label: Label for x-axis (defaults to date_column name if None)
        y_label: Label for y-axis
        color_scheme: Color scheme for multiple series
        container_width: Width of the container for responsive sizing
        height: Explicit chart height
        show_points: Whether to show data points on the line
        animate: Whether to add entrance animation
        tooltip: True for default tooltip, list of column names for custom tooltip
        interactive: Enable pan and zoom
        
    Returns:
        Altair Chart object representing the sales trend
        
    Raises:
        ValueError: If the DataFrame is empty or required columns cannot be determined
    """
    try:
        # Validate input data
        if df.empty:
            raise ValueError("DataFrame is empty")
            
        # Make a copy to avoid modifying the original
        plot_df = df.copy()
        
        # Determine date column if not specified
        date_col = date_column or _find_date_column(plot_df)
        if not date_col:
            raise ValueError("Could not determine date column. Please specify date_column.")
            
        # Determine value column if not specified
        value_col = value_column or _find_numeric_column(plot_df)
        if not value_col:
            raise ValueError("Could not determine value column. Please specify value_column.")
            
        # Ensure date column is datetime type
        if not pd.api.types.is_datetime64_any_dtype(plot_df[date_col]):
            try:
                plot_df[date_col] = pd.to_datetime(plot_df[date_col])
            except Exception as e:
                logger.warning(f"Could not convert {date_col} to datetime: {str(e)}")
                # Continue anyway, Altair will attempt conversion
        
        # Create base chart
        base = alt.Chart(plot_df)
        
        # Define encoding
        encoding = {
            'x': alt.X(f'{date_col}:T', title=x_label or date_col),
            'y': alt.Y(f'{value_col}:Q', title=y_label or value_col),
            'tooltip': _get_tooltip_encoding(plot_df, date_col, value_col, category_column, tooltip)
        }
        
        # Add color encoding if category column specified
        if category_column and category_column in plot_df.columns:
            encoding['color'] = alt.Color(
                f'{category_column}:N', 
                scale=alt.Scale(scheme=color_scheme),
                legend=alt.Legend(title=category_column)
            )
            
        # Create the line chart
        line_chart = base.mark_line(
            point=show_points,
            opacity=0.8
        ).encode(**encoding)
        
        # Add points for better interaction on mobile
        if show_points:
            point_opacity = 0 if animate else 0.8
            point_chart = base.mark_point(
                opacity=point_opacity,
                size=60,
                filled=True
            ).encode(**encoding)
            
            # Combine line and points
            chart = alt.layer(line_chart, point_chart)
        else:
            chart = line_chart
        
        # Set chart properties
        chart = chart.properties(
            title=title
        )
        
        # Add responsive sizing
        responsive_props = get_responsive_dimensions(
            container_width=container_width,
            height=height
        )
        if responsive_props:
            chart = chart.properties(**responsive_props)
            
        # Add interactive features
        if interactive:
            selection = alt.selection_interval(bind='scales', encodings=['x'])
            chart = chart.add_selection(selection)
            
        # Add animation
        if animate:
            chart = chart.configure_mark(
                opacity=alt.value(0)
            ).transform_filter(
                {"not": alt.expr("isNaN(datum.sum)")}
            ).encode(
                opacity=alt.value(1)
            )
            
        return chart
        
    except Exception as e:
        logger.error(f"Error creating sales trend chart: {str(e)}")
        return _create_error_chart(str(e))

def _find_date_column(df: pd.DataFrame) -> Optional[str]:
    """Find appropriate column for date/time data."""
    # Check for date/time columns
    date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
    if date_cols:
        return date_cols[0]
        
    # Look for columns with date/time-related names
    date_indicators = ['date', 'time', 'day', 'month', 'year', 'period']
    date_cols = [col for col in df.columns if any(indicator in col.lower() for indicator in date_indicators)]
    if date_cols:
        return date_cols[0]
        
    return None
    
def _find_numeric_column(df: pd.DataFrame) -> Optional[str]:
    """Find appropriate numeric column for sales values."""
    # Look for sales-related column names first
    sales_indicators = ['sales', 'revenue', 'amount', 'value', 'price', 'total', 'sum']
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Find numeric columns with sales-related names
    sales_cols = [col for col in numeric_cols if any(indicator in col.lower() for indicator in sales_indicators)]
    if sales_cols:
        return sales

