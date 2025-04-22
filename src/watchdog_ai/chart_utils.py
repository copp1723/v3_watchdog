

def extract_chart_data(chart_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract chart data from a dictionary.
    
    DEPRECATED: This function is deprecated.
    Please use watchdog_ai.core.visualization.extract_chart_data instead.
    
    Args:
        chart_data: Dictionary containing chart data
        
    Returns:
        Processed chart data dictionary
    """
    warnings.warn(
        "extract_chart_data() is deprecated. "
        "Please use watchdog_ai.core.visualization.extract_chart_data instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Use new implementation if available
    if _extract_chart_data is not None:
        result, _ = _extract_chart_data(chart_data)
        return result or {}
    
    # Simple fallback implementation
    if isinstance(chart_data, dict):
        return chart_data
    return {}

def build_chart(chart_data: Dict[str, Any], chart_type: str = None) -> Any:
    """
    Build a chart object from data.
    
    DEPRECATED: This function is deprecated.
    Please use watchdog_ai.core.visualization.build_chart instead.
    
    Args:
        chart_data: Dictionary with chart data
        chart_type: Optional chart type override
        
    Returns:
        Chart object ready for rendering
    """
    warnings.warn(
        "build_chart() is deprecated. "
        "Please use watchdog_ai.core.visualization.build_chart instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Use new implementation if available
    if _build_chart is not None:
        config = None
        if chart_type:
            config = ChartConfig(chart_type=chart_type)
        return _build_chart(chart_data, chart_type, config)
    
    # No fallback implementation - this requires chart libraries
    raise NotImplementedError(
        "build_chart() requires the new visualization system. "
        "Please import watchdog_ai.core.visualization."
    )

def render_chart(chart_data: Dict[str, Any], chart_type: str = None) -> bool:
    """
    Render a chart in Streamlit.
    
    DEPRECATED: This function is deprecated.
    Please use watchdog_ai.core.visualization.render_chart instead.
    
    Args:
        chart_data: Dictionary with chart data
        chart_type: Optional chart type override
        
    Returns:
        True if successful, False otherwise
    """
    warnings.warn(
        "render_chart() is deprecated. "
        "Please use watchdog_ai.core.visualization.render_chart instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Use new implementation if available
    if _render_chart is not None:
        config = None
        if chart_type:
            config = ChartConfig(chart_type=chart_type)
        return _render_chart(chart_data, chart_type, config)
    
    # No fallback implementation - this requires chart libraries and Streamlit
    raise NotImplementedError(
        "render_chart() requires the new visualization system. "
        "Please import watchdog_ai.core.visualization."
    )

"""
Chart utilities for Watchdog AI.

DEPRECATED: This module is deprecated and will be removed in v4.0.0.
Please use 'watchdog_ai.core.visualization' instead.
"""

import warnings
import pandas as pd
from typing import Dict, Any, Optional

warnings.warn(
    "The 'watchdog_ai.chart_utils' module is deprecated and will be removed in v4.0.0. "
    "Please use 'watchdog_ai.core.visualization' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import from new location for backward compatibility
try:
    from watchdog_ai.core.visualization import (
        build_chart_data as _build_chart_data,
        extract_chart_data as _extract_chart_data,
        build_chart as _build_chart,
        render_chart as _render_chart,
        ChartConfig
    )
except ImportError:
    # If import fails, log a warning but don't fail
    warnings.warn(
        "Could not import from 'watchdog_ai.core.visualization'. "
        "Using deprecated local implementations.",
        ImportWarning
    )
    _build_chart_data = None
    _extract_chart_data = None
    _build_chart = None
    _render_chart = None

def build_chart_data(df: pd.DataFrame, 
                    x_column: str, 
                    y_column: str,
                    chart_type: str = 'bar',
                    title: Optional[str] = None) -> Dict[str, Any]:
    """
    Build chart data dictionary from DataFrame.
    
    DEPRECATED: This function is deprecated.
    Please use watchdog_ai.core.visualization.build_chart_data instead.
    
    Args:
        df: Source DataFrame
        x_column: Column name for x-axis
        y_column: Column name for y-axis
        chart_type: Type of chart ('bar' or 'line')
        title: Optional chart title
        
    Returns:
        Dictionary with chart data
    """
    warnings.warn(
        "build_chart_data() is deprecated. "
        "Please use watchdog_ai.core.visualization.build_chart_data instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Use new implementation if available
    if _build_chart_data is not None:
        return _build_chart_data(df, x_column, y_column, chart_type, title)
    
    # Legacy implementation
    return {
        "type": chart_type,
        "data": {
            "x": df[x_column].tolist(),
            "y": df[y_column].tolist()
        },
        "title": title
    }
