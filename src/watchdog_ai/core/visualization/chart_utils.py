
"""
Chart utilities for Watchdog AI.

This module consolidates chart utilities from:
- src/chart_utils.py
- src/utils/chart_utils.py
- src/watchdog_ai/chart_utils.py

Provides functions for building and extracting chart data.
"""

import pandas as pd
import json
from typing import Dict, Any, Optional, List, Tuple, Union
import logging

logger = logging.getLogger(__name__)

def build_chart_data(
    df: pd.DataFrame, 
    x_col: str, 
    y_col: str, 
    chart_type: str = 'bar',
    title: Optional[str] = None
) -> Dict[str, Any]:
    """
    Build chart data dictionary for visualization.
    
    Args:
        df: DataFrame containing the data
        x_col: Column name for x-axis
        y_col: Column name for y-axis
        chart_type: Type of chart ('line', 'bar', 'pie')
        title: Optional chart title
        
    Returns:
        Dictionary containing chart data
    """
    try:
        # Ensure expected column types
        if x_col not in df.columns:
            logger.warning(f"Column '{x_col}' not found in DataFrame")
            return {
                'type': 'error',
                'data': {'error': f"Column '{x_col}' not found"}
            }
            
        if y_col not in df.columns:
            logger.warning(f"Column '{y_col}' not found in DataFrame")
            return {
                'type': 'error',
                'data': {'error': f"Column '{y_col}' not found"}
            }
            
        if chart_type == 'pie':
            # For pie charts, we need labels and values
            data = {
                'type': chart_type,
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
        
        # Add title if provided
        if title:
            data['title'] = title
            
        return data
    except Exception as e:
        logger.error(f"Error building chart data: {str(e)}")
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
    # Check if type is explicitly specified
    if isinstance(data, dict) and 'type' in data:
        return data['type']
        
    data_str = str(data).lower()
    
    # Check if data suggests a time series
    if any(time_term in data_str for time_term in ['time', 'date', 'month', 'year', 'period']):
        return 'line'
    
    # Check if data suggests categories
    if any(cat in data_str for cat in ['category', 'type', 'group']):
        return 'bar'
    
    # Check if data suggests parts of a whole
    if any(part in data_str for part in ['share', 'percentage', 'distribution']):
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

def _generate_pie_chart_data(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
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

def _generate_bar_chart_data(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
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

def _generate_line_chart_data(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
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

def extract_chart_data_from_llm_response(response: Union[Dict[str, Any], str]) -> Optional[Dict[str, Any]]:
    """
    Extract chart data from LLM response.
    
    Args:
        response: Dictionary or string containing the LLM response
        
    Returns:
        Chart data dictionary
    """
    try:
        # If response is a string, try to parse as JSON
        if isinstance(response, str):
            try:
                response = json.loads(response)
            except json.JSONDecodeError:
                logger.warning("Error parsing LLM response JSON")
                return None
                
        # Check if response contains chart data
        if not isinstance(response, dict) or 'chart_data' not in response:
            return None
        
        data = response['chart_data']
        
        # If data is a string (JSON), parse it
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                logger.warning("Error parsing chart data JSON")
                return None
        
        # If chart type is specified, use it

