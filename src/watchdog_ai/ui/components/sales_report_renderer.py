"""
Sales report renderer components.
Provides modular, functional components for rendering sales insights.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from typing import Dict, Any, List, Optional, Union
from datetime import timedelta
from watchdog_ai.core.data_utils import format_metric_value

# Import new visualization system
from watchdog_ai.core.visualization import (
    render_chart as viz_render_chart,
    ChartConfig, 
    ChartType,
    build_chart
)

def format_timedelta(td: timedelta) -> str:
    """
    Format timedelta into human-readable string.
    
    Args:
        td: Timedelta to format
        
    Returns:
        Formatted string representation
    """
    hours = td.total_seconds() / 3600
    if hours < 1:
        minutes = int(hours * 60)
        return f"{minutes} {'minute' if minutes == 1 else 'minutes'}"
    elif hours < 24:
        return f"{hours:.1f} {'hour' if hours == 1 else 'hours'}"
    else:
        days = hours / 24
        return f"{days:.1f} {'day' if days == 1 else 'days'}"
def render_bar_chart(data: pd.DataFrame, x_col: str, y_col: str, title: str = None):
    """
    Render a bar chart using the new visualization system.
    
    Args:
        data: DataFrame containing the data
        x_col: Column to use for x-axis
        y_col: Column to use for y-axis
        title: Optional chart title
    """
    config = ChartConfig(
        chart_type='bar',
        title=title,
        x_column=x_col,
        y_column=y_col
    )
    
    return viz_render_chart(data, 'bar', config)
def render_report(self, data: pd.DataFrame, legacy_charts: bool = False):
    """Render sales report with optional legacy chart support."""
    pass
    """
    Render a complete insight block with all components.
    
    Args:
        insight_data: Complete insight data dictionary containing metrics,
                     visualizations, findings, and recommendations
        use_legacy_charts: Whether to use legacy chart rendering system
    """
        - data: Chart data
            - title: Optional chart title
            - x_axis: X-axis label
            - y_axis: Y-axis label
            - other type-specific options
        use_legacy: Whether to use legacy rendering system instead of new visualization
            
    Returns:
        True if chart was rendered successfully, False otherwise
    """
    if not chart_data:
        return False
        
    chart_type = chart_data.get('type', '').lower()
    if not chart_type or 'data' not in chart_data:
        return False
    
    # Use legacy rendering for backward compatibility or when explicitly requested
    if use_legacy:
        # Use legacy rendering system
        if chart_type == 'time_series':
            render_time_series_chart(chart_data)
        elif chart_type == 'scatter':
            render_scatter_chart(chart_data)
        elif chart_type == 'bar':
            st.bar_chart(chart_data['data'])
        else:
            st.warning(f"Unsupported chart type: {chart_type}")
            return False
        return True
        
    # Use new visualization system
    try:
        # Map chart types
        type_mapping = {
            'time_series': 'line',
            'scatter': 'scatter',
            'bar': 'bar',
            'pie': 'pie'
        }
        viz_type = type_mapping.get(chart_type, chart_type)
        
        # Create chart configuration
        config = ChartConfig(
            chart_type=viz_type,
            title=chart_data.get('title'),
            x_label=chart_data.get('x_axis'),
            y_label=chart_data.get('y_axis'),
            width=chart_data.get('width'),
            height=chart_data.get('height', 400)
        )
        
        # Special handling for time series
        if chart_type == 'time_series' and isinstance(chart_data['data'], dict):
            # Convert dict to DataFrame
            df = pd.DataFrame({
                'x': list(chart_data['data'].keys()),
                'y': list(chart_data['data'].values())
            })
            
            # Apply additional styling if needed
            if chart_data.get('cumulative', False):
                # For cumulative case, we currently need to use legacy
                render_time_series_chart(chart_data)
                return True
                
            return viz_render_chart(df, 'line', config)
            
        # Special handling for scatter plots with trend lines
        elif chart_type == 'scatter' and chart_data.get('trend_line'):
            # For trend lines, use legacy renderer
            render_scatter_chart(chart_data)
            return True
        
        # Use the new visualization system for standard charts
        return viz_render_chart(chart_data['data'], viz_type, config)
    
    except Exception as e:
        # Fallback to legacy rendering if new system fails
        st.warning(f"Using legacy chart rendering due to: {str(e)}")
        return render_chart(chart_data, use_legacy=True)
    Args:
        metrics: Response time metrics dictionary containing:
            - avg_response_time: Average response time (timedelta)
            - within_1hour: Percentage of responses within 1 hour (float)
            - within_24hours: Percentage of responses within 24 hours (float)
            - response_rate: Overall response rate (float)
            - response_distribution: Optional distribution data
    """
    if not metrics:
        return
        
    st.markdown("### Lead Response Time Analysis")
    
    # Create columns for metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Avg Response Time",
            format_timedelta(metrics['avg_response_time']),
            help="Average time between lead creation and first response"
        )
    
    with col2:
        st.metric(
            "Within 1 Hour",
            f"{metrics['within_1hour']:.1f}%",
            help="Percentage of leads responded to within one hour"
        )
        
    with col3:
        st.metric(
            "Response Rate",
            f"{metrics['response_rate']:.1f}%",
            help="Percentage of leads that received any response"
        )
    
    # Display response time distribution if available
    if response_dist := metrics.get('response_distribution'):
        st.markdown("#### Response Time Distribution")
        fig = px.bar(
            response_dist,
            title="Lead Response Time Distribution",
            labels={"index": "Response Time", "value": "Number of Leads"}
        )
        fig.update_layout(
            height=250,
            showlegend=True,
            margin=dict(l=10, r=10, t=30, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

def render_inventory_age_metrics(metrics: Dict[str, Any]):
    """
    Render inventory age and profit metrics with charts.
    
    Args:
        metrics: Inventory age metrics dictionary containing:
            - avg_days_on_lot: Average days on lot (float)
            - best_performing_age: Best performing age group (str)
            - total_aged_inventory: Count of aged inventory (int)
            - avg_profit_by_age: Dict of age groups to average profits
            - profit_correlation: Correlation coefficient between age and profit
    """
    if not metrics:
        return
        
    st.markdown("### Inventory Age & Profit Analysis")
    
    # Create columns for metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Avg Days on Lot",
            f"{metrics['avg_days_on_lot']:.1f}",
            help="Average time vehicles spend on lot"
        )
    
    with col2:
        st.metric(
            "Best Age Group",
            metrics['best_performing_age'],
            help="Age group with highest average profit"
        )
        
    with col3:
        st.metric(
            "Aged Inventory",
            str(metrics['total_aged_inventory']),
            help="Number of units over 90 days old"
        )
    
    # Render profit by age group chart
    if age_profit := metrics.get('avg_profit_by_age'):
        st.markdown("#### Profit by Age Group")
        profit_data = pd.DataFrame(
            list(age_profit.items()),
            columns=['Age Group', 'Average Profit']
        )
        
        fig = px.bar(
            profit_data,
            x='Age Group',
            y='Average Profit',
            title='Average Profit by Inventory Age',
            labels={'Average Profit': 'Average Gross Profit ($)'}
        )
        fig.update_layout(
            height=300,
            showlegend=False,
            margin=dict(l=10, r=10, t=30, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Show correlation indicator
        if correlation := metrics.get('profit_correlation'):
            correlation_color = 'red' if correlation < -0.3 else 'green' if correlation > 0.3 else 'orange'
            st.markdown(
                f"<p style='color: {correlation_color}'>Profit-Age Correlation: {correlation:.2f}</p>",
                unsafe_allow_html=True,
                help="Correlation between inventory age and profit"
            )

def render_chart(chart_data: Dict[str, Any]):
    """
    Render a chart based on specification.
    
    Args:
        chart_data: Chart specification containing:
            - type: Chart type (time_series, scatter, bar)
            - data: Chart data
            - title: Optional chart title
            - x_axis: X-axis label
            - y_axis: Y-axis label
            - other type-specific options
    """
    if not chart_data:
        return
        
    chart_type = chart_data.get('type', '').lower()
    if not chart_type or 'data' not in chart_data:
        return
        
    if chart_type == 'time_series':
        render_time_series_chart(chart_data)
    elif chart_type == 'scatter':
        render_scatter_chart(chart_data)
    elif chart_type == 'bar':
        st.bar_chart(chart_data['data'])
    else:
        st.warning(f"Unsupported chart type: {chart_type}")

def render_time_series_chart(chart_data: Dict[str, Any]):
    """Render a time series chart using plotly."""
    fig = go.Figure()
    
    x_data = list(chart_data['data'].keys())
    y_data = list(chart_data['data'].values())
    
    # Add main series
    fig.add_trace(go.Scatter(
        x=x_data,
        y=y_data,
        mode='lines+markers',
        name=chart_data.get('y_axis', 'Value')
    ))
    
    # Add cumulative line if requested
    if chart_data.get('cumulative', False):
        cumsum = np.cumsum(y_data)
        fig.add_trace(go.Scatter(
            x=x_data,
            y=cumsum,
            mode='lines',
            name='Cumulative',
            line=dict(dash='dash')
        ))
    
    # Update layout
    fig.update_layout(
        title=chart_data.get('title', ''),
        xaxis_title=chart_data.get('x_axis', ''),
        yaxis_title=chart_data.get('y_axis', ''),
        height=300,
        showlegend=True,
        margin=dict(l=10, r=10, t=30, b=10)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_scatter_chart(chart_data: Dict[str, Any]):
    """Render a scatter plot using plotly."""
    fig = go.Figure()
    
    # Add scatter points
    fig.add_trace(go.Scatter(
        x=chart_data['data']['x'],
        y=chart_data['data']['y'],
        mode='markers',
        name='Data Points',
        marker=dict(
            size=8,
            color=chart_data.get('color', None),
            colorscale=chart_data.get('colorscale', 'Viridis')
        )
    ))
    
    # Add trend line if requested
    if chart_data.get('trend_line'):
        x_data = np.array(chart_data['data']['x'])
        y_data = np.array(chart_data['data']['y'])
        z = np.polyfit(x_data, y_data, 1)
        p = np.poly1d(z)
        x_range = [min(x_data), max(x_data)]
        
        fig.add_trace(go.Scatter(
            x=x_range,
            y=[p(x) for x in x_range],
            mode='lines',
            name='Trend',
            line=dict(color='red', dash='dash')
        ))
    
    # Update layout
    fig.update_layout(
        title=chart_data.get('title', ''),
        xaxis_title=chart_data.get('x_axis', ''),
        yaxis_title=chart_data.get('y_axis', ''),
        height=400,
        showlegend=True,
        margin=dict(l=10, r=10, t=30, b=10)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_insight_block(insight_data: Optional[Dict[str, Any]] = None):
    """
    Render a complete insight block with all components.
    
    Args:
        insight_data: Complete insight data dictionary containing metrics,
                     visualizations, findings, and recommendations
    """
    if not insight_data:
        st.warning("No insight data available")
        return
    
    # Main summary
    st.markdown(f"### {insight_data.get('summary', 'Insight Analysis')}")
    
    # Render metrics if present
    if base_metrics := insight_data.get('metrics'):
        render_metric_group({
            "Total Records": base_metrics.get('total_records', 0),
            "Time Period": base_metrics.get('time_period', 'N/A'),
            "Data Quality": f"{base_metrics.get('data_quality_score', 0)*100:.1f}%"
        }, "Overview")
    
    # Render type-specific metrics
    if response_metrics := insight_data.get('metrics', {}).get('response_time_metrics'):
        render_response_time_metrics(response_metrics)
    
    if inventory_metrics := insight_data.get('metrics', {}).get('inventory_age_metrics'):
        render_inventory_age_metrics(inventory_metrics)
    
    # Render visualization based on type
    if viz := insight_data.get('visualization'):
        render_chart(viz, use_legacy=use_legacy_charts)
    # Create two columns for findings and recommendations
    col1, col2 = st.columns(2)
    
    # Render key findings
    with col1:
        if findings := insight_data.get('key_findings'):
            st.markdown("### Key Findings")
            for finding in findings:
                st.markdown(f"• {finding}")
    
    # Render recommendations
    with col2:
        if recommendations := insight_data.get('recommendations'):
            st.markdown("### Recommendations")
            for rec in recommendations:
                st.markdown(f"• {rec}")
    
    # Show confidence level with appropriate color
    if confidence := insight_data.get('confidence'):
        confidence_color = {
            'high': 'green',
            'medium': 'orange',
            'low': 'red'
        }.get(confidence.lower(), 'grey')
        
        st.markdown(
            f"<p style='color: {confidence_color}'>Confidence Level: {confidence.title()}</p>",
            unsafe_allow_html=True,
            help="Indicates the reliability of the insight based on data quality"
        )

def render_error_state(error_message: str):
    """
    Render an error state with consistent styling.
    
    Args:
        error_message: Error message to display
    """
    st.error(
        f"⚠️ {error_message}",
        icon="🚫"
    )

# For backward compatibility
class SalesReportRenderer:
    """Legacy class wrapper for backward compatibility."""
    
    def render_insight_block(self, insight_data: Dict[str, Any] = None):
        """Delegates to the functional implementation."""
        render_insight_block(insight_data)
