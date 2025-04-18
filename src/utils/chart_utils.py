"""
Chart utilities for Watchdog AI.

This module provides functions for creating, formatting, and displaying charts
using Plotly and other visualization libraries.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
import sentry_sdk
import json
import re

# Configure logger
logger = logging.getLogger(__name__)

def extract_chart_data(llm_response: str) -> Dict[str, Any]:
    """
    Extract chart data from an LLM response.
    
    Args:
        llm_response: Response from LLM that may contain chart data
        
    Returns:
        Dictionary with extracted chart data, or empty dict if none found
    """
    try:
        # Look for JSON chart data in the response
        chart_match = re.search(r'```json\s*({[\s\S]*?})\s*```', llm_response)
        if not chart_match:
            # Try alternative format without language specification
            chart_match = re.search(r'```\s*({[\s\S]*?})\s*```', llm_response)
            
        if chart_match:
            chart_json = chart_match.group(1)
            chart_data = json.loads(chart_json)
            return chart_data
        else:
            logger.info("No chart data found in LLM response")
            return {}
    except Exception as e:
        logger.error(f"Error extracting chart data: {str(e)}")
        sentry_sdk.capture_exception(e)
        return {}

def determine_chart_type(df: pd.DataFrame, columns: Dict[str, str] = None) -> str:
    """
    Determine the most appropriate chart type based on the data.
    
    Args:
        df: DataFrame containing the data
        columns: Optional mapping of column roles (e.g. {"x": "date", "y": "sales"})
        
    Returns:
        Chart type string ('line', 'bar', 'pie', etc.)
    """
    try:
        if df.empty:
            return 'bar'  # Default fallback
            
        if columns is None:
            columns = {}
            
        # Identify key columns by role or infer them
        x_col = columns.get('x')
        y_col = columns.get('y')
        
        # If no x column specified, attempt to identify it
        if x_col is None:
            # Look for time/date columns
            date_cols = [col for col in df.columns if any(term in col.lower() for term in 
                                                         ['date', 'time', 'day', 'month', 'year'])]
            if date_cols:
                x_col = date_cols[0]
                # Check if it's actually a date type
                if pd.api.types.is_datetime64_any_dtype(df[x_col]):
                    return 'line'  # Time series data is best for line charts
            
            # Look for category columns
            category_cols = [col for col in df.columns if any(term in col.lower() for term in 
                                                            ['category', 'type', 'name', 'source', 'rep', 'model'])]
            if category_cols:
                x_col = category_cols[0]
                n_categories = df[x_col].nunique()
                
                # If few categories, use a pie chart
                if n_categories <= 5 and y_col and df[x_col].nunique() <= 7:
                    return 'pie'
                # Otherwise use a bar chart
                return 'bar'
        
        # If an x column is specified or was identified
        if x_col:
            # Check if it's a date/time column
            if pd.api.types.is_datetime64_any_dtype(df[x_col]):
                return 'line'  # Time series data is best for line charts
                
            # Check number of unique values
            if df[x_col].nunique() <= 10:
                # Few categories, good for bar or pie
                if df[x_col].nunique() <= 5 and len(df) <= 10:
                    return 'pie'
                else:
                    return 'bar'
            else:
                # Many categories, better for bar or line
                return 'bar'
        
        # Fallback to bar chart as the most versatile
        return 'bar'
        
    except Exception as e:
        logger.error(f"Error determining chart type: {str(e)}")
        sentry_sdk.capture_exception(e)
        return 'bar'  # Default fallback

def create_chart(
    df: pd.DataFrame, 
    chart_type: str = None, 
    x: str = None, 
    y: str = None,
    color: str = None,
    title: str = None,
    x_title: str = None,
    y_title: str = None,
    legend_title: str = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Create a chart specification based on input data.
    
    Args:
        df: DataFrame containing the data
        chart_type: Type of chart to create ('line', 'bar', 'pie', etc.)
        x: Column to use for the x-axis
        y: Column to use for the y-axis
        color: Column to use for color encoding
        title: Chart title
        x_title: X-axis title
        y_title: Y-axis title
        legend_title: Legend title
        **kwargs: Additional arguments to pass to the chart creation function
        
    Returns:
        Dictionary with chart specification
    """
    try:
        if df.empty:
            logger.warning("Empty DataFrame provided for chart creation")
            return {
                "error": "No data available for chart",
                "success": False
            }
            
        # Determine chart type if not specified
        if chart_type is None:
            chart_type = determine_chart_type(df, {"x": x, "y": y})
            
        # Auto-detect columns if not specified
        if x is None and len(df.columns) >= 1:
            # For pie charts, prefer category columns
            if chart_type == 'pie':
                category_cols = [col for col in df.columns if any(term in col.lower() for term in 
                                                                ['category', 'type', 'name', 'source'])]
                x = category_cols[0] if category_cols else df.columns[0]
            else:
                # For other charts, prefer time/date columns first
                date_cols = [col for col in df.columns if any(term in col.lower() for term in 
                                                             ['date', 'time', 'day', 'month', 'year'])]
                x = date_cols[0] if date_cols else df.columns[0]
        
        if y is None and len(df.columns) >= 2:
            # Prefer numeric columns
            numeric_cols = [col for col in df.columns if np.issubdtype(df[col].dtype, np.number)]
            if numeric_cols and numeric_cols[0] != x:
                y = numeric_cols[0]
            else:
                # Skip the x column
                y_candidates = [col for col in df.columns if col != x]
                y = y_candidates[0] if y_candidates else None
                
        # Basic validation
        missing_cols = []
        if x and x not in df.columns:
            missing_cols.append(x)
        if y and y not in df.columns:
            missing_cols.append(y)
        if color and color not in df.columns:
            missing_cols.append(color)
            
        if missing_cols:
            logger.warning(f"Missing columns in DataFrame: {missing_cols}")
            return {
                "error": f"Columns not found in data: {', '.join(missing_cols)}",
                "success": False
            }
            
        # Create the chart based on the type
        if chart_type == 'bar':
            return _create_bar_chart(df, x, y, color, title, x_title, y_title, legend_title, **kwargs)
        elif chart_type == 'line':
            return _create_line_chart(df, x, y, color, title, x_title, y_title, legend_title, **kwargs)
        elif chart_type == 'pie':
            return _create_pie_chart(df, x, y, title, legend_title, **kwargs)
        else:
            logger.warning(f"Unsupported chart type: {chart_type}")
            return {
                "error": f"Unsupported chart type: {chart_type}",
                "success": False
            }
    
    except Exception as e:
        logger.error(f"Error creating chart: {str(e)}")
        sentry_sdk.capture_exception(e)
        return {
            "error": str(e),
            "success": False
        }

def _create_bar_chart(
    df: pd.DataFrame, 
    x: str, 
    y: str = None, 
    color: str = None,
    title: str = None,
    x_title: str = None,
    y_title: str = None,
    legend_title: str = None,
    **kwargs
) -> Dict[str, Any]:
    """Create a bar chart specification."""
    # Clean up data - handle null values
    df = df.dropna(subset=[x])
    if y:
        df = df.dropna(subset=[y])
    
    # Default aggregation if y is provided
    if y:
        # Convert to string for categorical x-axis
        if not pd.api.types.is_numeric_dtype(df[x]) and not pd.api.types.is_datetime64_any_dtype(df[x]):
            df[x] = df[x].astype(str)
            
        # Create bar chart
        fig = px.bar(
            df, 
            x=x, 
            y=y,
            color=color,
            title=title or f"{y} by {x}",
            labels={
                x: x_title or x.replace('_', ' ').title(),
                y: y_title or y.replace('_', ' ').title()
            }
        )
    else:
        # Count by x categories
        value_counts = df[x].value_counts().reset_index()
        value_counts.columns = [x, 'count']
        
        # Create bar chart of counts
        fig = px.bar(
            value_counts, 
            x=x, 
            y='count',
            title=title or f"Count by {x}",
            labels={
                x: x_title or x.replace('_', ' ').title(),
                'count': y_title or 'Count'
            }
        )
    
    # Additional formatting
    fig.update_layout(
        margin=dict(l=10, r=10, t=50, b=10),
        legend_title=legend_title or (color.replace('_', ' ').title() if color else None),
        xaxis_title=x_title or x.replace('_', ' ').title(),
        yaxis_title=y_title or (y.replace('_', ' ').title() if y else 'Count')
    )
    
    # Return chart spec
    return {
        "chart_type": "bar",
        "figure": fig.to_dict(),
        "success": True
    }

def _create_line_chart(
    df: pd.DataFrame, 
    x: str, 
    y: str = None, 
    color: str = None,
    title: str = None,
    x_title: str = None,
    y_title: str = None,
    legend_title: str = None,
    **kwargs
) -> Dict[str, Any]:
    """Create a line chart specification."""
    # Clean up data - handle null values
    df = df.dropna(subset=[x])
    if y:
        df = df.dropna(subset=[y])
        
    # Sort by x for proper line sequencing
    if pd.api.types.is_datetime64_any_dtype(df[x]) or pd.api.types.is_numeric_dtype(df[x]):
        df = df.sort_values(by=x)
    
    # Default aggregation if y is provided
    if y:
        # Create line chart
        fig = px.line(
            df, 
            x=x, 
            y=y,
            color=color,
            title=title or f"{y} over {x}",
            labels={
                x: x_title or x.replace('_', ' ').title(),
                y: y_title or y.replace('_', ' ').title()
            }
        )
    else:
        # Count by x categories
        value_counts = df[x].value_counts().reset_index()
        value_counts.columns = [x, 'count']
        value_counts = value_counts.sort_values(by=x)
        
        # Create line chart of counts
        fig = px.line(
            value_counts, 
            x=x, 
            y='count',
            title=title or f"Count over {x}",
            labels={
                x: x_title or x.replace('_', ' ').title(),
                'count': y_title or 'Count'
            }
        )
    
    # Additional formatting
    fig.update_layout(
        margin=dict(l=10, r=10, t=50, b=10),
        legend_title=legend_title or (color.replace('_', ' ').title() if color else None),
        xaxis_title=x_title or x.replace('_', ' ').title(),
        yaxis_title=y_title or (y.replace('_', ' ').title() if y else 'Count')
    )
    
    # Return chart spec
    return {
        "chart_type": "line",
        "figure": fig.to_dict(),
        "success": True
    }

def _create_pie_chart(
    df: pd.DataFrame, 
    names: str, 
    values: str = None, 
    title: str = None,
    legend_title: str = None,
    **kwargs
) -> Dict[str, Any]:
    """Create a pie chart specification."""
    # Clean up data - handle null values
    df = df.dropna(subset=[names])
    
    # If values column is provided
    if values and values in df.columns:
        df = df.dropna(subset=[values])
        
        # Create pie chart
        fig = px.pie(
            df, 
            names=names, 
            values=values,
            title=title or f"{values} by {names}",
            labels={
                names: names.replace('_', ' ').title(),
                values: values.replace('_', ' ').title()
            }
        )
    else:
        # Count by categories
        value_counts = df[names].value_counts().reset_index()
        value_counts.columns = [names, 'count']
        
        # Create pie chart of counts
        fig = px.pie(
            value_counts, 
            names=names, 
            values='count',
            title=title or f"Distribution by {names}",
            labels={
                names: names.replace('_', ' ').title(),
                'count': 'Count'
            }
        )
    
    # Additional formatting
    fig.update_layout(
        margin=dict(l=10, r=10, t=50, b=10),
        legend_title=legend_title or names.replace('_', ' ').title()
    )
    
    # Return chart spec
    return {
        "chart_type": "pie",
        "figure": fig.to_dict(),
        "success": True
    }

def get_chart_data_for_insight(
    insight_data: Dict[str, Any], 
    chart_type: str = None
) -> Tuple[Optional[Dict[str, Any]], Optional[pd.DataFrame]]:
    """
    Extract and prepare chart data from an insight result.
    
    Args:
        insight_data: The insight data dictionary
        chart_type: Optional chart type to override automatic detection
        
    Returns:
        Tuple of (chart_spec, dataframe)
    """
    try:
        # If chart data already exists in the insight
        if "chart_data" in insight_data and isinstance(insight_data["chart_data"], pd.DataFrame):
            df = insight_data["chart_data"]
            
            # Use provided chart encoding if available
            if "chart_encoding" in insight_data and isinstance(insight_data["chart_encoding"], dict):
                encoding = insight_data["chart_encoding"]
                chart_spec = create_chart(
                    df,
                    chart_type=encoding.get("chart_type", chart_type),
                    x=encoding.get("x"),
                    y=encoding.get("y"),
                    color=encoding.get("color"),
                    title=encoding.get("title") or insight_data.get("title"),
                    x_title=encoding.get("x_title"),
                    y_title=encoding.get("y_title"),
                    legend_title=encoding.get("legend_title")
                )
                return chart_spec, df
            
            # Otherwise, create a chart automatically
            chart_spec = create_chart(
                df, 
                chart_type=chart_type,
                title=insight_data.get("title")
            )
            return chart_spec, df
            
        # Try to extract chart data from various insight formats
        if "model_metrics" in insight_data and isinstance(insight_data["model_metrics"], list):
            # Convert list of dictionaries to dataframe
            df = pd.DataFrame(insight_data["model_metrics"])
            chart_spec = create_chart(
                df, 
                chart_type=chart_type or "bar",
                x="category",
                y="avg_days",
                title=insight_data.get("title") or "Model Metrics"
            )
            return chart_spec, df
            
        if "rep_metrics" in insight_data and isinstance(insight_data["rep_metrics"], list):
            # Convert list of dictionaries to dataframe
            df = pd.DataFrame(insight_data["rep_metrics"])
            chart_spec = create_chart(
                df, 
                chart_type=chart_type or "bar",
                x="rep_name",
                y="avg_gross_per_deal",
                title=insight_data.get("title") or "Sales Representative Performance"
            )
            return chart_spec, df
            
        if "time_based" in insight_data and "monthly_data" in insight_data["time_based"]:
            # Time-based data
            df = pd.DataFrame(insight_data["time_based"]["monthly_data"])
            chart_spec = create_chart(
                df, 
                chart_type=chart_type or "line",
                x="month",
                y="total_gross",
                title=insight_data.get("title") or "Monthly Performance"
            )
            return chart_spec, df
        
        # No chart data found
        logger.info("No suitable chart data found in insight")
        return None, None
        
    except Exception as e:
        logger.error(f"Error extracting chart data from insight: {str(e)}")
        sentry_sdk.capture_exception(e)
        return None, None