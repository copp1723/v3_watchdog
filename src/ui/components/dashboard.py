"""
Dashboard Component for Watchdog AI.
Provides UI components for displaying data insights in a dashboard format.
"""

import streamlit as st
import pandas as pd
import altair as alt
from typing import Dict, Any, List, Optional

def render_kpi_metrics(metrics: Dict[str, Any]) -> None:
    """
    Render KPI metrics in a dashboard layout.
    
    Args:
        metrics: Dictionary containing KPI metrics
    """
    # Create columns for metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Total Sales",
            metrics.get("total_sales", "$0"),
            metrics.get("sales_delta", "0%"),
            help="Total sales value"
        )
    
    with col2:
        st.metric(
            "Average Gross",
            metrics.get("avg_gross", "$0"),
            metrics.get("gross_delta", "0%"),
            help="Average gross profit per sale"
        )
    
    with col3:
        st.metric(
            "Lead Conversion",
            metrics.get("conversion_rate", "0%"),
            metrics.get("conversion_delta", "0%"),
            help="Lead to sale conversion rate"
        )

def render_sales_dashboard(df: pd.DataFrame) -> None:
    """
    Render a sales performance dashboard.
    
    Args:
        df: DataFrame containing sales data
    """
    st.markdown("### Sales Performance")
    
    # Create sales trend chart
    if 'date' in df.columns and 'sales' in df.columns:
        chart = alt.Chart(df).mark_line().encode(
            x='date:T',
            y='sales:Q',
            tooltip=['date:T', 'sales:Q']
        ).properties(
            width=600,
            height=300
        )
        st.altair_chart(chart, use_container_width=True)

def render_inventory_dashboard(df: pd.DataFrame) -> None:
    """
    Render an inventory health dashboard.
    
    Args:
        df: DataFrame containing inventory data
    """
    st.markdown("### Inventory Health")
    
    # Create inventory metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Total Units",
            len(df),
            help="Total inventory units"
        )
    
    with col2:
        avg_age = df['age'].mean() if 'age' in df.columns else 0
        st.metric(
            "Average Age",
            f"{avg_age:.1f} days",
            help="Average inventory age"
        )
    
    with col3:
        total_value = df['value'].sum() if 'value' in df.columns else 0
        st.metric(
            "Total Value",
            f"${total_value:,.2f}",
            help="Total inventory value"
        )

def render_interactive_chart(data: Dict[str, Any]) -> None:
    """
    Render an interactive chart based on data.
    
    Args:
        data: Dictionary containing chart data
    """
    chart_type = data.get('type', 'bar')
    chart_data = data.get('data', {})
    
    if not chart_data:
        st.warning("No chart data available")
        return
    
    # Convert data to DataFrame
    df = pd.DataFrame(chart_data)
    
    # Create chart based on type
    if chart_type == 'bar':
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X('x:N', title=None),
            y=alt.Y('y:Q', title=None),
            tooltip=['x:N', 'y:Q']
        )
    elif chart_type == 'line':
        chart = alt.Chart(df).mark_line().encode(
            x=alt.X('x:T', title=None),
            y=alt.Y('y:Q', title=None),
            tooltip=['x:T', 'y:Q']
        )
    elif chart_type == 'pie':
        # For pie charts, we need a special encoding
        chart = alt.Chart(df).mark_arc().encode(
            theta='y:Q',
            color='x:N',
            tooltip=['x:N', 'y:Q']
        )
    else:
        st.warning(f"Unsupported chart type: {chart_type}")
        return
    
    # Display the chart
    st.altair_chart(chart, use_container_width=True)

def render_dashboard_from_insight(insight: Dict[str, Any]) -> None:
    """
    Render a dashboard based on an insight response.
    
    Args:
        insight: Dictionary containing insight data
    """
    st.markdown("### Insight Dashboard")
    
    # Display summary
    if 'summary' in insight:
        st.markdown(f"**{insight['summary']}**")
    
    # Display metrics
    if 'metrics' in insight:
        render_kpi_metrics(insight['metrics'])
    
    # Display chart if available
    if 'chart_data' in insight:
        render_interactive_chart(insight['chart_data'])
    
    # Display value insights
    if 'value_insights' in insight:
        st.markdown("#### Key Insights")
        for insight_text in insight['value_insights']:
            st.markdown(f"- {insight_text}")
    
    # Display actionable flags
    if 'actionable_flags' in insight:
        st.markdown("#### Action Items")
        for flag in insight['actionable_flags']:
            st.markdown(f"- {flag}")