"""
Chart utilities for Watchdog AI.
Provides functions for building and extracting chart data.

DEPRECATED: This module is deprecated and will be removed in v4.0.0.
Please use 'watchdog_ai.core.visualization' instead.
"""

import json
import pandas as pd
from typing import Dict, Any, List, Optional
import warnings

warnings.warn(
    "The 'chart_utils' module is deprecated and will be removed in v4.0.0. "
    "Please use 'watchdog_ai.core.visualization' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import from new location for backward compatibility
try:
    from watchdog_ai.core.visualization import (
        build_chart_data as _build_chart_data,
        extract_chart_data_from_llm_response as _extract_chart_data_from_llm_response,
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
    _extract_chart_data_from_llm_response = None
    _build_chart = None
    _render_chart = None
import warnings

warnings.warn(
    "The 'chart_utils' module is deprecated and will be removed in v4.0.0. "
    "Please use 'watchdog_ai.core.visualization' instead.",
    DeprecationWarning,
    stacklevel=2
)

import pandas as pd
import json
from typing import Dict, Any, Optional, List, Tuple


# Import from new location for backward compatibility
try:
    from watchdog_ai.core.visualization.chart_utils import (
        build_chart_data as _build_chart_data,
        extract_chart_data_from_llm_response as _extract_chart_data_from_llm_response,
        _determine_chart_type,
        _identify_target_field,
        _identify_time_field,
        _generate_pie_chart_data,
        _generate_bar_chart_data,
        _generate_line_chart_data,
        _get_default_chart_data
    )
except ImportError:
    # If import fails, log a warning but don't fail
    warnings.warn(
        "Could not import from 'watchdog_ai.core.visualization'. "
        "Using deprecated local implementations.",
        ImportWarning
    )
    _build_chart_data = None
    _extract_chart_data_from_llm_response = None


def build_chart_data(df: pd.DataFrame, chart_type: str, x_col: str, y_col: str) -> Dict[str, Any]:
    """
    DEPRECATED: This function is deprecated.
    Please use watchdog_ai.core.visualization.build_chart_data instead.
    """
    warnings.warn(
        "build_chart_data() is deprecated. "
        "Please use watchdog_ai.core.visualization.build_chart_data instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Use new implementation if available
    if _build_chart_data is not None:
        try:
            return _build_chart_data(df, x_col, y_col, chart_type)
        except Exception as e:
            warnings.warn(
                f"Error using new implementation: {str(e)}. Falling back to legacy implementation.",
                RuntimeWarning
            )
    """
    Build chart data dictionary for visualization.
    
    Args:
        df: DataFrame containing the data
        chart_type: Type of chart ('line', 'bar', 'pie')
        x_col: Column name for x-axis
        y_col: Column name for y-axis
        
    Returns:
        Dictionary containing chart data
    """
    """
    DEPRECATED: This function is deprecated. 
    Please use watchdog_ai.core.visualization.build_chart_data instead.
    """
    warnings.warn(
        "build_chart_data() is deprecated. "
        "Please use watchdog_ai.core.visualization.build_chart_data instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Use new implementation if available
    if _build_chart_data is not None:
        return _build_chart_data(df, x_col, y_col, chart_type)
    
    # Legacy implementation for backward compatibility
    try:
        if chart_type == 'pie':
            # For pie charts, we need labels and values
            data = {
                'type': 'pie',
                'data': {
                    'labels': df[x_col].tolist(),
                    'values': df[y_col].tolist()
                }
            }
        else:
            # For line and bar charts, we need x and y arrays
            data = {
                'type': chart_type,
                'data': {
                    'x': df[x_col].tolist(),
                    'y': df[y_col].tolist()
                }
            }
        
        return data
    except Exception as e:
        print(f"Error building chart data: {str(e)}")
        return {
            'type': 'error',
            'data': {'error': str(e)}
        }
def _determine_chart_type(data: Dict[str, Any]) -> str:
    """
    Determine the best chart type based on the data.
    
    Args:
        data: Dictionary containing the data
        
    Returns:
        Chart type string ('line', 'bar', 'pie')
    """
    # Check if data suggests a time series
    if 'time' in str(data).lower() or 'date' in str(data).lower():
        return 'line'
    
    # Check if data suggests categories
    if any(cat in str(data).lower() for cat in ['category', 'type', 'group']):
        return 'bar'
    
    # Check if data suggests parts of a whole
    if any(part in str(data).lower() for part in ['share', 'percentage', 'distribution']):
        return 'pie'
    
    # Default to bar chart
    return 'bar'

def _identify_target_field(data: Dict[str, Any], target_terms: List[str]) -> Optional[str]:
    """
    Identify the target field from data based on terms.
    
    Args:
        data: Dictionary containing the data
        target_terms: List of terms to look for
        
    Returns:
        Field name if found, None otherwise
    """
    for field in data.keys():
        if any(term in field.lower() for term in target_terms):
            return field
    return None

def _identify_time_field(data: Dict[str, Any]) -> Optional[str]:
    """
    Identify the time/date field in the data.
    
    Args:
        data: Dictionary containing the data
        
    Returns:
        Field name if found, None otherwise
    """
    time_terms = ['time', 'date', 'month', 'year', 'period']
    return _identify_target_field(data, time_terms)

def _generate_pie_chart_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate data for a pie chart.
    
    Args:
        data: Dictionary containing the data
        
    Returns:
        Chart data dictionary
    """
    # Look for value and label fields
    value_field = _identify_target_field(data, ['value', 'count', 'amount'])
    label_field = _identify_target_field(data, ['label', 'name', 'category'])
    
    if value_field and label_field:
        return {
            'type': 'pie',
            'data': {
                'labels': data[label_field],
                'values': data[value_field]
            }
        }
    return None

def _generate_bar_chart_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate data for a bar chart.
    
    Args:
        data: Dictionary containing the data
        
    Returns:
        Chart data dictionary
    """
    # Look for x and y fields
    x_field = _identify_target_field(data, ['category', 'name', 'type'])
    y_field = _identify_target_field(data, ['value', 'count', 'amount'])
    
    if x_field and y_field:
        return {
            'type': 'bar',
            'data': {
                'x': data[x_field],
                'y': data[y_field]
            }
        }
    return None

def _generate_line_chart_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate data for a line chart.
    
    Args:
        data: Dictionary containing the data
        
    Returns:
        Chart data dictionary
    """
    # Look for time and value fields
    x_field = _identify_time_field(data)
    y_field = _identify_target_field(data, ['value', 'count', 'amount'])
    
    if x_field and y_field:
        return {
            'type': 'line',
            'data': {
                'x': data[x_field],
                'y': data[y_field]
            }
        }
    return None

def _get_default_chart_data() -> Dict[str, Any]:
    """
    Get default chart data when no valid data is available.
    
    Returns:
        Default chart data dictionary
    """
    return {
        'type': 'bar',
        'data': {
            'x': ['No Data'],
            'y': [0]
        }
    }


def extract_chart_data_from_llm_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    DEPRECATED: This function is deprecated.
    Please use watchdog_ai.core.visualization.extract_chart_data_from_llm_response instead.
    """
    warnings.warn(
        "extract_chart_data_from_llm_response() is deprecated. "
        "Please use watchdog_ai.core.visualization.extract_chart_data_from_llm_response instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Use new implementation if available
    if _extract_chart_data_from_llm_response is not None:
        try:
            return _extract_chart_data_from_llm_response(response)
        except Exception as e:
            warnings.warn(
                f"Error using new implementation: {str(e)}. Falling back to legacy implementation.",
                RuntimeWarning
            )
    """
    Extract chart data from LLM response.
    
    Args:
        response: Dictionary containing the LLM response
        
    Returns:
        Chart data dictionary
    """
    DEPRECATED: This function is deprecated. 
    Please use watchdog_ai.core.visualization.extract_chart_data_from_llm_response instead.
    """
    warnings.warn(
        "extract_chart_data_from_llm_response() is deprecated. "
        "Please use watchdog_ai.core.visualization.extract_chart_data_from_llm_response instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Use new implementation if available
    if _extract_chart_data_from_llm_response is not None:
        return _extract_chart_data_from_llm_response(response)
    
    # Legacy implementation for backward compatibility
    try:
        # Check if response contains chart data
        if 'chart_data' not in response:
            return None
        
        data = response['chart_data']
        
        # If data is a string (JSON), parse it
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                print("Error parsing chart data JSON")
                return None
        
        # If chart type is specified, use it
        if 'type' in data:
            chart_type = data['type']
        else:
            chart_type = _determine_chart_type(data)
        
        # Generate chart data based on type
        if chart_type == 'pie':
            chart_data = _generate_pie_chart_data(data)
        elif chart_type == 'bar':
            chart_data = _generate_bar_chart_data(data)
        elif chart_type == 'line':
            chart_data = _generate_line_chart_data(data)
        else:
            chart_data = None
        
        # Return chart data or default
        return chart_data if chart_data else _get_default_chart_data()
        
    except Exception as e:
        print(f"Error extracting chart data: {str(e)}")
        return _get_default_chart_data()
