"""
Utility functions for chart visualization and responsive sizing.

This module provides helper functions for creating responsive visualizations
that work well on both desktop and mobile devices when using Streamlit.
"""

import altair as alt
import pandas as pd
import logging
from typing import Dict, Optional, Tuple, Union, Any, List, Callable

from .constants import (
    DEFAULT_CHART_WIDTH,
    DEFAULT_CHART_HEIGHT,
    DEFAULT_MOBILE_CHART_HEIGHT,
    MOBILE_WIDTH_THRESHOLD,
    LANDSCAPE_RATIO,
    PORTRAIT_RATIO,
    SQUARE_RATIO,
    TITLE_FONT_SIZE,
    MOBILE_TITLE_FONT_SIZE,
    AXIS_LABEL_FONT_SIZE,
    MOBILE_AXIS_LABEL_FONT_SIZE,
    TICK_LABEL_FONT_SIZE,
    MOBILE_TICK_LABEL_FONT_SIZE,
    LEGEND_FONT_SIZE,
    MOBILE_LEGEND_FONT_SIZE
)

logger = logging.getLogger(__name__)

def get_responsive_dimensions(
    container_width: Optional[int] = None,
    height: Optional[int] = None,
    aspect_ratio: float = LANDSCAPE_RATIO,
    min_height: int = 200,
    max_height: Optional[int] = None
) -> Dict[str, Any]:
    """
    Calculate responsive dimensions for charts based on container width.
    
    This function determines the appropriate width and height for a chart
    based on the container width (typically from Streamlit) and desired
    aspect ratio. It handles both desktop and mobile sizes.
    
    Args:
        container_width: Width of the containing element (e.g., from st.container)
        height: Explicit height to use, overrides calculated height
        aspect_ratio: Desired aspect ratio (width/height)
        min_height: Minimum height in pixels
        max_height: Maximum height in pixels
        
    Returns:
        Dictionary with width and height properties for chart configuration
    
    Example:
        ```python
        import streamlit as st
        container_width = st.container().width
        chart_props = get_responsive_dimensions(container_width)
        chart = alt.Chart(df).properties(**chart_props)
        ```
    """
    properties = {}
    
    # If no container width provided, use responsive container
    if container_width is None:
        properties["width"] = "container"
    else:
        # For explicit container width, calculate dimensions
        is_mobile = container_width <= MOBILE_WIDTH_THRESHOLD
        
        if is_mobile:
            # For mobile, use narrower width with small margins
            width = min(container_width - 20, DEFAULT_CHART_WIDTH)
            calculated_height = height or int(width / aspect_ratio)
            calculated_height = min(calculated_height, DEFAULT_MOBILE_CHART_HEIGHT)
        else:
            # For desktop, use provided width with some margin
            width = min(container_width - 40, DEFAULT_CHART_WIDTH)
            calculated_height = height or int(width / aspect_ratio)
        
        # Apply min/max height constraints
        if calculated_height < min_height:
            calculated_height = min_height
        if max_height and calculated_height > max_height:
            calculated_height = max_height
            
        properties["width"] = width
        properties["height"] = calculated_height
    
    # If only height is provided (without container_width)
    if container_width is None and height is not None:
        properties["height"] = height
        
    return properties

def make_chart_responsive(
    chart: alt.Chart, 
    container_width: Optional[int] = None, 
    height: Optional[int] = None
) -> alt.Chart:
    """
    Apply responsive sizing to an existing Altair chart.
    
    Args:
        chart: Altair Chart object to make responsive
        container_width: Width of the container for responsive sizing
        height: Explicit chart height (overrides calculated height)
        
    Returns:
        Chart with responsive sizing applied
    """
    responsive_props = get_responsive_dimensions(container_width, height)
    return chart.properties(**responsive_props)

def get_mobile_optimized_chart_config(
    is_mobile: Optional[bool] = None,
    container_width: Optional[int] = None
) -> Dict[str, Any]:
    """
    Get chart configuration optimized for current device size.
    
    Provides configuration settings appropriate for the current device,
    with smaller fonts and tighter spacing for mobile displays.
    
    Args:
        is_mobile: Explicitly set mobile mode, otherwise determined by container_width
        container_width: Container width to determine if mobile mode should be used
        
    Returns:
        Dictionary with device-optimized chart configuration
    """
    # Determine if mobile mode based on container width if not explicitly set
    if is_mobile is None and container_width is not None:
        is_mobile = container_width <= MOBILE_WIDTH_THRESHOLD
    elif is_mobile is None:
        is_mobile = False
    
    # Select appropriate font sizes based on device type
    title_size = MOBILE_TITLE_FONT_SIZE if is_mobile else TITLE_FONT_SIZE
    axis_label_size = MOBILE_AXIS_LABEL_FONT_SIZE if is_mobile else AXIS_LABEL_FONT_SIZE
    tick_label_size = MOBILE_TICK_LABEL_FONT_SIZE if is_mobile else TICK_LABEL_FONT_SIZE
    legend_size = MOBILE_LEGEND_FONT_SIZE if is_mobile else LEGEND_FONT_SIZE
    
    return {
        "view": {"stroke": "transparent"},
        "axis": {
            "labelFontSize": tick_label_size,
            "titleFontSize": axis_label_size,
            "labelLimit": 100,
            "grid": not is_mobile  # Disable grid on mobile for cleaner look
        },
        "header": {
            "labelFontSize": tick_label_size,
            "titleFontSize": axis_label_size
        },
        "legend": {
            "labelFontSize": legend_size,
            "titleFontSize": legend_size,
            "titleLimit": 120,
            "symbolSize": 80 if is_mobile else 100,  # Slightly smaller symbols on mobile
            "orient": "top" if is_mobile else "right"  # Legend position based on device
        },
        "title": {
            "fontSize": title_size,
            "subtitleFontSize": title_size - 2,
            "anchor": "start"
        },
        "range": {
            "category": {"scheme": "tableau10"}  # Default color scheme
        }
    }

def configure_responsive_chart(
    chart: alt.Chart, 
    container_width: Optional[int] = None
) -> alt.Chart:
    """
    Apply device-optimized configuration to a chart.
    
    Args:
        chart: Altair Chart object
        container_width: Container width for responsive configuration
        
    Returns:
        Configured chart with device-appropriate settings
    """
    config = get_mobile_optimized_chart_config(container_width=container_width)
    return chart.configure(**config)

def create_error_chart(
    error_message: str, 
    chart_type: str = "bar"
) -> alt.Chart:
    """
    Create a fallback chart for error cases.
    
    Args:
        error_message: Error message to display in the chart
        chart_type: The type of chart to create as fallback
        
    Returns:
        Simple chart with error message as title
    """
    if chart_type == "pie":
        df = pd.DataFrame({'category': ['Error'], 'value': [1]})
        return (
            alt.Chart(df)
            .mark_arc()
            .encode(theta='value', color='category')
            .properties(title=f"Error creating chart: {error_message}")
        )
    else:
        # Default to bar/line chart
        df = pd.DataFrame({'x': [0, 1], 'y': [0, 0]})
        return (
            alt.Chart(df)
            .mark_line()
            .encode(x='x', y='y')
            .properties(title=f"Error creating chart: {error_message}")
        )

def detect_device_type(container_width: Optional[int] = None) -> str:
    """
    Determine the device type based on container width.
    
    Args:
        container_width: Container width in pixels
        
    Returns:
        Device type: 'mobile', 'tablet', or 'desktop'
    """
    if container_width is None:
        return 'desktop'  # Default to desktop
    
    if container_width <= MOBILE_WIDTH_THRESHOLD:
        return 'mobile'
    elif container_width <= 992:  # Bootstrap lg breakpoint
        return 'tablet'
    else:
        return 'desktop'

def filter_dataframe_for_performance(
    df: pd.DataFrame, 
    container_width: Optional[int] = None,
    max_points_mobile: int = 500,
    max_points_desktop: int = 2000,
    date_column: Optional[str] = None,
    category_column: Optional[str] = None
) -> pd.DataFrame:
    """
    Filter or aggregate dataframe to manageable size based on device.
    
    For improved performance, especially on mobile devices, this function
    will sample or aggregate large datasets to a reasonable size.
    
    Args:
        df: Input DataFrame
        container_width: Container width to determine device type
        max_points_mobile: Maximum points to display on mobile
        max_points_desktop: Maximum points to display on desktop
        date_column: Explicit date column name (if known)
        category_column: Explicit category column name (if known)
        
    Returns:
        Filtered or aggregated DataFrame
    """
    if df.empty:
        return df
        
    device = detect_device_type(container_width)
    
    # Set max points based on device type
    if device == 'mobile':
        max_points = max_points_mobile
    elif device == 'tablet':
        max_points = (max_points_mobile + max_points_desktop) // 2  # Average for tablets
    else:
        max_points = max_points_desktop
    
    # If dataframe is already small enough, return as is
    if len(df) <= max_points:
        return df
    
    logger.info(f"Optimizing dataframe with {len(df)} rows for {device} display (max: {max_points})")
    
    # For large dataframes, sample or aggregate
    # Try to find date columns if not specified
    if date_column and date_column in df.columns:
        date_cols = [date_column]
    else:
        date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
    
    # If we have date columns, use time-based resampling
    if date_cols:
        time_col = date_cols[0]  # Use the first date column found
        
        # Determine appropriate resampling frequency based on data size and device
        if len(df) > max_points * 4:
            freq = 'M'  # Monthly for very large datasets
        elif len(df) > max_points * 2:
            freq = 'W'  # Weekly for large datasets
        else:
            freq = 'D'  # Daily for medium datasets
            
        # For mobile, use coarser aggregation
        if device == 'mobile' and len(df) > max_points * 3:
            freq = 'M'  # Monthly for mobile with large datasets
        
        try:
            # If category column is specified, group by both time and category
            if category_column and category_column in df.columns:
                # Set the date column as index
                temp_df = df.set_index(time_col)
                
                # Group by time frequency and category
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                
                # Create grouper for time frequency
                time_grouper = pd.Grouper(freq=freq)
                
                # Group and aggregate
                result = temp_df.groupby([time_grouper, category_column])[numeric_cols].mean().reset_index()
                logger.info(f"Time-series resampled with frequency {freq} by category: {len(result)} rows")
                return result
            else:
                # Simple time-based resampling for all numeric columns
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                
                # Set the date column as index
                temp_df = df.set_index(time_col)
                
                # Resample and aggregate numeric columns
                result = temp_df[numeric_cols].resample(freq).mean().reset_index()
                
                # Add back any non-numeric columns with forward fill
                for col in df.columns:
                    if col not in result.columns and col != time_col:
                        # For categorical data, take the most common value in each time period
                        if pd.api.types.is_categorical_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
                            most_common = temp_df.groupby(pd.Grouper(freq=freq))[col].agg(
                                lambda x: x.mode().iloc[0] if not x.mode().empty else None
                            )
                            result[col] = most_common.values
                
                logger.info(f"Time-series resampled with frequency {freq}: {len(result)} rows")
                return result
        except Exception as e:
            logger.warning(f"Time-based resampling failed: {str(e)}. Falling back to sampling.")
    
    # For non-time series data, or if time resampling failed:
    # 1. Try to preserve category distribution if categorical data exists
    if category_column and category_column in df.columns:
        # Stratified sampling to maintain category distribution
        categories = df[category_column].unique()
        
        # Calculate sampling fraction based on device
        if device == 'mobile':
            frac = min(0.7, max_points / len(df))
        else:
            frac = min(0.9, max_points / len(df))
            
        # Sample from each category
        samples = []
        for category in categories:
            category_df = df[df[category_column] == category]
            # Ensure we take at least 1 sample from each category
            n_samples = max(1, int(len(category_df) * frac))
            samples.append(category_df.sample(n=min(n_samples, len(category_df))))
            
        result = pd.concat(samples)
        logger.info(f"Stratified sampling by category: {len(result)} rows")
        return result
    
    # If no categorical data or date columns, use simple random sampling
    sample_size = min(max_points, len(df))
    result = df.sample(n=sample_size)
    logger.info(f"Random sampling: {len(result)} rows")
    return result

