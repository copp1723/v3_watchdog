"""
Chart rendering utilities for Watchdog AI.

Provides high-level functions for rendering charts from various data formats.
"""

import streamlit as st
import pandas as pd
import altair as alt
import logging
from typing import Dict, Any, Optional, Union, List, Tuple

from .chart_base import ChartBase, ChartConfig, ChartType
from .chart_utils import extract_chart_data_from_llm_response, build_chart_data

logger = logging.getLogger(__name__)

def render_chart(
    chart_data: Union[Dict[str, Any], pd.DataFrame, alt.Chart],
    chart_type: Optional[str] = None,
    config: Optional[ChartConfig] = None,
    use_container_width: bool = True
) -> bool:
    """
    Render a chart in Streamlit based on provided data.
    
    Args:
        chart_data: Dictionary, DataFrame or Chart with data to visualize
        chart_type: Optional chart type to override inferred type
        config: Optional chart configuration
        use_container_width: Whether to use container width
        
    Returns:
        True if chart was rendered successfully, False otherwise
    """
    try:
        # If already an Altair chart, render directly
        if isinstance(chart_data, alt.Chart):
            st.altair_chart(chart_data, use_container_width=use_container_width)
            return True
        
        # Process the data and create chart
        chart = build_chart(chart_data, chart_type, config)
        
        if chart is None:
            logger.warning("Failed to build chart, no valid data or configuration")
            return False
        
        # Render in Streamlit based on chart type
        if isinstance(chart, alt.Chart):
            st.altair_chart(chart, use_container_width=use_container_width)
        elif hasattr(chart, 'update_layout'):
            # Plotly chart
            st.plotly_chart(chart, use_container_width=use_container_width)
        elif isinstance(chart, pd.DataFrame):
            # Fallback to basic Streamlit charts
            if chart_type == 'bar':
                st.bar_chart(chart, use_container_width=use_container_width)
            elif chart_type == 'line':
                st.line_chart(chart, use_container_width=use_container_width)
            elif chart_type == 'area':
                st.area_chart(chart, use_container_width=use_container_width)
            else:
                st.line_chart(chart, use_container_width=use_container_width)
        else:
            logger.warning(f"Unknown chart type: {type(chart)}")
            return False
            
        return True
    except Exception as e:
        logger.error(f"Error rendering chart: {str(e)}")
        st.error(f"Could not render chart: {str(e)}")
        return False

def extract_chart_data(
    data: Union[Dict[str, Any], pd.DataFrame, str]
) -> Tuple[Optional[Union[Dict[str, Any], pd.DataFrame]], Optional[str]]:
    """
    Extract chart data from various formats.
    
    Args:
        data: Input data (dict, DataFrame, or string)
        
    Returns:
        Tuple of (chart_data, chart_type)
    """
    try:
        # Handle DataFrame directly
        if isinstance(data, pd.DataFrame):
            return data, None
            
        # Handle LLM responses
        if isinstance(data, (dict, str)) and not isinstance(data, pd.DataFrame):
            extracted = extract_chart_data_from_llm_response(data)
            if extracted:
                chart_type = extracted.get('type', 'bar')
                chart_data = extracted.get('data', {})
                return chart_data, chart_type
        
        # Handle dictionary with 'data' and 'encoding' (Altair/Vega-Lite format)
        if isinstance(data, dict):
            # Standard format with records
            if 'data' in data and isinstance(data['data'], list):
                # Convert to DataFrame
                df = pd.DataFrame(data['data'])
                chart_type = data.get('type', 'bar')
                return df, chart_type
            
            # Vega-Lite specification
            if 'data' in data and 'encoding' in data:
                # Get chart type from mark or markType if available
                chart_type = None
                if 'mark' in data:
                    if isinstance(data['mark'], str):
                        chart_type = data['mark']
                    elif isinstance(data['mark'], dict) and 'type' in data['mark']:
                        chart_type = data['mark']['type']
                
                # Extract data values
                if isinstance(data['data'], dict) and 'values' in data['data']:
                    df = pd.DataFrame(data['data']['values'])
                    return df, chart_type
                else:
                    return data, chart_type
            
            # Standard x/y data format
            if 'data' in data and ('x' in data['data'] and 'y' in data['data']):
                x_values = data['data']['x']
                y_values = data['data']['y']
                df = pd.DataFrame({
                    'x': x_values,
                    'y': y_values
                })
                return df, data.get('type', 'bar')
            
            # Pie chart format
            if 'data' in data and ('labels' in data['data'] and 'values' in data['data']):
                labels = data['data']['labels']
                values = data['data']['values']
                df = pd.DataFrame({
                    'label': labels,
                    'value': values
                })
                return df, 'pie'
                
            # If just a data object with keys that might be columns
            if len(data) > 0 and not any(k in ['type', 'data', 'encoding'] for k in data):
                # Try to interpret as a single row to be converted to columns
                try:
                    df = pd.DataFrame([data])
                    return df, None
                except:
                    pass
        
        # If we got here, we couldn't extract chart data properly
        logger.warning(f"Could not extract chart data from: {type(data)}")
        return None, None
    
    except Exception as e:
        logger.error(f"Error extracting chart data: {str(e)}")
        return None, None

def build_chart(
    chart_data: Union[Dict[str, Any], pd.DataFrame],
    chart_type: Optional[str] = None,
    config: Optional[ChartConfig] = None
) -> Any:
    """
    Build a chart object from data.
    
    Args:
        chart_data: Dictionary or DataFrame with chart data
        chart_type: Optional chart type to override inferred type
        config: Optional chart configuration
        
    Returns:
        Chart object ready for rendering
    """
    try:
        # Extract data and infer chart type if needed
        data, inferred_type = extract_chart_data(chart_data)
        
        if data is None:
            logger.warning("No valid data could be extracted for chart building")
            return None
            
        # Use provided chart type or inferred type
        final_chart_type = chart_type or inferred_type or 'bar'
        
        # Create configuration if not provided
        if config is None:
            config = ChartConfig(chart_type=final_chart_type)
        elif chart_type:
            config.chart_type = ChartType.from_string(final_chart_type)
        
        # Use factory method to get correct chart implementation
        from .chart_implementations import (
            BarChart, LineChart, PieChart, 
            ScatterChart, AreaChart, HeatmapChart
        )
        
        chart_classes = {
            'bar': BarChart,
            'line': LineChart,
            'pie': PieChart,
            'scatter': ScatterChart,
            'area': AreaChart,
            'heatmap': HeatmapChart
        }
        
        chart_class = chart_classes.get(final_chart_type.lower(), BarChart)
        chart = chart_class()
        
        # Set configuration
        chart.config = config
        
        # Generate chart from data
        if isinstance(data, pd.DataFrame):
            return chart.from_dataframe(data)
        elif isinstance(data, dict):
            return chart.from_dict(data)
        else:
            logger.warning(f"Unsupported data type: {type(data)}")
            return None
            
    except Exception as e:
        logger.error(f"Error building chart: {str(e)}")
        return None

def chart_data_to_string(data: Any) -> str:
    """
    Convert chart data to a string representation for debugging.
    
    Args:
        data: Chart data in any format
        
    Returns:
        String representation of the data
    """
    if isinstance(data, pd.DataFrame):
        return f"DataFrame with columns: {list(data.columns)} and shape: {data.shape}"
    elif isinstance(data, dict):
        if 'data' in data:
            inner_data = data['data']
            if isinstance(inner_data, dict):
                return f"Dict with keys: {list(data.keys())}, data keys: {list(inner_data.keys())}"
            elif isinstance(inner_data, list):
                return f"Dict with keys: {list(data.keys())}, data is list of length: {len(inner_data)}"
            else:
                return f"Dict with keys: {list(data.keys())}, data type: {type(inner_data)}"
        else:
            return f"Dict with keys: {list(data.keys())}"
    elif isinstance(data, alt.Chart):
        return "Altair Chart object"
    else:
        return f"Unknown data type: {type(data)}"

"""
Chart rendering utilities for Watchdog AI.

Provides high-level functions for rendering charts from various data formats.
"""

import streamlit as st
import pandas as pd
import altair as alt
import logging
from typing import Dict, Any, Optional, Union, List, Tuple

from .chart_base import ChartBase, ChartConfig, ChartType
from .chart_utils import extract_chart_data_from_llm_response, build_chart_data

logger = logging.getLogger(__name__)

def render_chart(
    chart_data: Union[Dict[str, Any], pd.DataFrame],
    chart_type: Optional[str] = None,
    config: Optional[ChartConfig] = None,
    use_container_width: bool = True
) -> bool:
    """
    Render a chart in Streamlit based on provided data.
    
    Args:
        chart_data: Dictionary or DataFrame with chart data
        chart_type: Optional chart type to override inferred type
        config: Optional chart configuration
        use_container_width: Whether to use container width
        
    Returns:
        True if chart was rendered successfully, False otherwise
    """
    try:
        # Process the data and create chart
        chart = build_chart(chart_data, chart_type, config)
        
        if chart is None:
            return False
        
        # Render in Streamlit
        if isinstance(chart, alt.Chart):
            st.altair_chart(chart, use_container_width=use_container_width)
        elif hasattr(chart, 'update_layout'):
            # Plotly chart
            st.plotly_chart(chart, use_container_width=use_container_width)
        elif isinstance(chart, pd.DataFrame):
            # Fallback to basic Streamlit charts
            if chart_type == 'bar':
                st.bar_chart(chart, use_container_width=use_container_width)
            elif chart_type == 'line':
                st.line_chart(chart, use_container_width=use_container_width)
            else:
                st.line_chart(chart, use_container_width=use_container_width)
        else:
            logger.warning(f"Unknown chart type: {type(chart)}")
            return False
            
        return True
    except Exception as e:
        logger.error(f"Error rendering chart: {str(e)}")
        return False

def extract_chart_data(
    data: Union[Dict[str, Any], pd.DataFrame, str]
) -> Tuple[Optional[Union[Dict[str, Any], pd.DataFrame]], Optional[str]]:
    """
    Extract chart data from various formats.
    
    Args:
        data: Input data (dict, DataFrame, or string)
        
    Returns:
        Tuple of (chart_data, chart_type)
    """
    try:
        # Handle LLM responses
        if isinstance(data, (dict, str)) and not isinstance(data, pd.DataFrame):
            extracted = extract_chart_data_from_llm_response(data)
            if extracted:
                chart_type = extracted.get('type', 'bar')
                chart_data = extracted.get('data', {})
                return chart_data, chart_type
        
        # Handle dictionary with 'data' and 'encoding' (Altair/Vega-Lite format)
        if isinstance(data, dict):
            if 'data' in data and isinstance(data['data'], list):
                # Convert to DataFrame
                df = pd.DataFrame(data['data'])
                chart_type = data.get('type', 'bar')
                return df, chart_type
            
            if 'data' in data and 'encoding' in data:
                # Vega-Lite specification
                if isinstance(data['data'], dict) and 'values' in data['data']:
                    df = pd.DataFrame(data['data']['values'])
                else:
                    # Extract data based on encoding
                    x_field = data['encoding'].get('x', {}).get('field')
                    y_field = data['encoding'].get('y', {}).get('field')
                    
                    if x_field and y_field and 'data' in data:
                        return data['data'], data.get('type', 'bar')
                        
            # Standard format we defined
            if 'data' in data and ('x' in data['data'] and 'y' in data['data']):
                x_values = data['data']['x']
                y_values = data['data']['y']
                df = pd.DataFrame({
                    'x': x_values,
                    'y': y_values
                })
                return df, data.get('type', 'bar')
            
            # Pie chart format
            if 'data' in data and ('labels' in data['data'] and 'values' in data['data']):
                labels = data['data']['labels']
                values = data['data']['values']
                df = pd.DataFrame({
                    'label': labels,
                    'value': values
                })
                return df, 'pie'
        
        # Handle DataFrame directly
        if isinstance(data, pd.DataFrame):
            return data, 'bar'  # Default to bar
            
        return None, None
    
    except Exception as e:
        logger.error(f"Error extracting chart data: {str(e)}")
        return None, None

def build_chart(
    chart_data: Union[Dict[str, Any], pd.DataFrame],
    chart_type: Optional[str] = None,
    config: Optional[ChartConfig] = None
) -> Any:
    """
    Build a chart object from data.
    
    Args:
        chart_data: Dictionary or DataFrame with chart data
        chart_type: Optional chart type to override inferred type
        config: Optional chart configuration
        
    Returns:
        Chart object ready for rendering
    """
    try:
        # Extract data and infer chart type if needed
        data, inferred_type = extract_chart_data(chart_data)
        
        if data is None:
            return None
            
        # Use provided chart type or inferred type
        final_chart_type = chart_type or inferred_type or 'bar'
        
        # Create configuration if not provided
        if config is None:
            config = ChartConfig(chart_type=final_chart_type)
        elif chart_type:
            config.chart_type = ChartType.from_string(final_chart_type)
        
        # Use factory method to get correct chart implementation
        chart = ChartBase.get_chart_for_type(final_chart_type)
        
        # Set configuration
        chart.config = config
        
        # Generate chart from data
        if isinstance(data, pd.DataFrame):
            return chart.from_dataframe(data)
        elif isinstance(data, dict):
            return chart.from_dict(data)
        else:
            logger.warning(f"Unsupported data type: {type(data)}")
            return None
            
    except Exception as e:
        logger.error(f"Error building chart: {str(e)}")
        return None

