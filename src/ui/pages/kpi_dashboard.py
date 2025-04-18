"""
KPI Dashboard for Watchdog AI.
Displays key performance indicators and metrics in a dashboard format.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import sentry_sdk
from typing import Dict, Any, List, Optional, Tuple
import plotly.express as px
from io import BytesIO

from ...utils.chart_utils import create_chart, get_chart_data_for_insight
from ..components.dashboard import (
    render_kpi_metrics,
    render_interactive_chart
)
from ...insights.summarizer import Summarizer
from ...utils.session import record_action

# Configure logger
logger = logging.getLogger(__name__)

def apply_date_filter(df: pd.DataFrame, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Apply date range filter to DataFrame."""
    try:
        date_col = next(
            col for col in df.columns
            if any(term in col.lower() for term in ['date', 'time', 'day'])
        )
        df[date_col] = pd.to_datetime(df[date_col])
        return df[(df[date_col] >= start_date) & (df[date_col] <= end_date)]
    except Exception as e:
        logger.error(f"Error applying date filter: {str(e)}")
        return df

def apply_dimension_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    """Apply dimension filters to DataFrame."""
    try:
        filtered_df = df.copy()
        for col, value in filters.items():
            if value and col in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[col] == value]
        return filtered_df
    except Exception as e:
        logger.error(f"Error applying dimension filters: {str(e)}")
        return df

def get_download_data(df: pd.DataFrame, chart_type: str) -> BytesIO:
    """Prepare data for download."""
    try:
        output = BytesIO()
        df.to_csv(output, index=False)
        return output.getvalue()
    except Exception as e:
        logger.error(f"Error preparing download data: {str(e)}")
        return BytesIO()

def handle_chart_click(chart_id: str, selected_data: Dict[str, Any]) -> None:
    """Handle chart click events."""
    try:
        st.query_params["selected_chart"] = chart_id
        st.query_params["selected_value"] = selected_data.get("points", [{}])[0].get("x", "")
        record_action("chart_click", {
            "chart_id": chart_id,
            "selected_data": selected_data
        })
    except Exception as e:
        logger.error(f"Error handling chart click: {str(e)}")

def calculate_rep_performance(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Calculate sales rep performance metrics.
    
    Args:
        df: DataFrame containing sales data
        
    Returns:
        Tuple of (metrics DataFrame, summary dict)
    """
    try:
        # Find relevant columns
        rep_col = next(
            col for col in df.columns
            if any(term in col.lower() for term in ['rep', 'salesperson', 'employee'])
        )
        gross_col = next(
            col for col in df.columns
            if any(term in col.lower() for term in ['gross', 'profit'])
        )
        
        # Calculate metrics by rep
        rep_metrics = df.groupby(rep_col).agg({
            gross_col: ['count', 'sum', 'mean']
        }).reset_index()
        
        # Flatten column names
        rep_metrics.columns = [rep_col, 'deals', 'total_gross', 'avg_gross']
        
        # Sort by average gross
        rep_metrics = rep_metrics.sort_values('avg_gross', ascending=False)
        
        # Calculate summary metrics
        summary = {
            "top_performer": rep_metrics.iloc[0][rep_col],
            "top_avg_gross": rep_metrics.iloc[0]['avg_gross'],
            "total_deals": rep_metrics['deals'].sum(),
            "total_gross": rep_metrics['total_gross'].sum(),
            "overall_avg_gross": rep_metrics['avg_gross'].mean()
        }
        
        return rep_metrics, summary
        
    except Exception as e:
        logger.error(f"Error calculating rep performance: {str(e)}")
        sentry_sdk.capture_exception(e)
        return pd.DataFrame(), {}

def analyze_inventory_aging(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Analyze inventory aging metrics.
    
    Args:
        df: DataFrame containing inventory data
        
    Returns:
        Tuple of (metrics DataFrame, summary dict)
    """
    try:
        # Find relevant columns
        days_col = next(
            col for col in df.columns
            if any(term in col.lower() for term in ['days', 'age', 'aging'])
        )
        
        # Calculate aging metrics
        aging_metrics = pd.DataFrame({
            'age_range': ['0-30', '31-60', '61-90', '90+'],
            'count': [
                len(df[df[days_col] <= 30]),
                len(df[(df[days_col] > 30) & (df[days_col] <= 60)]),
                len(df[(df[days_col] > 60) & (df[days_col] <= 90)]),
                len(df[df[days_col] > 90])
            ]
        })
        
        # Calculate summary metrics
        summary = {
            "total_units": len(df),
            "avg_age": df[days_col].mean(),
            "aged_inventory": len(df[df[days_col] > 90]),
            "aged_percentage": (len(df[df[days_col] > 90]) / len(df)) * 100
        }
        
        return aging_metrics, summary
        
    except Exception as e:
        logger.error(f"Error analyzing inventory aging: {str(e)}")
        sentry_sdk.capture_exception(e)
        return pd.DataFrame(), {}

def calculate_lead_conversion(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Calculate lead conversion metrics.
    
    Args:
        df: DataFrame containing lead data
        
    Returns:
        Tuple of (metrics DataFrame, summary dict)
    """
    try:
        # Find relevant columns
        source_col = next(
            col for col in df.columns
            if any(term in col.lower() for term in ['source', 'lead', 'channel'])
        )
        status_col = next(
            col for col in df.columns
            if any(term in col.lower() for term in ['status', 'state'])
        )
        
        # Calculate conversion by source
        conversion_metrics = df.groupby(source_col).agg({
            status_col: 'count'
        }).reset_index()
        
        # Calculate conversion rates
        total_leads = len(df)
        converted = len(df[df[status_col].str.lower().isin(['sold', 'closed', 'won'])])
        
        summary = {
            "total_leads": total_leads,
            "converted_leads": converted,
            "conversion_rate": (converted / total_leads) * 100,
            "top_source": conversion_metrics.iloc[0][source_col],
            "top_source_leads": conversion_metrics.iloc[0][status_col]
        }
        
        return conversion_metrics, summary
        
    except Exception as e:
        logger.error(f"Error calculating lead conversion: {str(e)}")
        sentry_sdk.capture_exception(e)
        return pd.DataFrame(), {}

def render_forecast_trends(df: pd.DataFrame) -> None:
    """
    Render forecast trend charts.
    
    Args:
        df: DataFrame containing historical data
    """
    try:
        # Find date and metric columns
        date_col = next(
            col for col in df.columns
            if any(term in col.lower() for term in ['date', 'time', 'day'])
        )
        metric_col = next(
            col for col in df.columns
            if any(term in col.lower() for term in ['gross', 'profit', 'revenue'])
        )
        
        # Create trend chart
        chart_spec = create_chart(
            df,
            chart_type='line',
            x=date_col,
            y=metric_col,
            title='Performance Trends',
            x_title='Date',
            y_title='Gross Profit'
        )
        
        if chart_spec.get("success"):
            render_interactive_chart(chart_spec)
        
    except Exception as e:
        logger.error(f"Error rendering forecast trends: {str(e)}")
        sentry_sdk.capture_exception(e)
        st.error("Unable to render forecast trends")

def kpi_dashboard():
    """Main KPI dashboard page."""
    try:
        # Check authentication
        if not st.session_state.get("is_authenticated"):
            st.warning("Please log in to access the KPI dashboard.")
            return
            
        st.title("KPI Dashboard")
        
        # Check if we have data
        if 'validated_data' not in st.session_state:
            st.warning("Please upload data to view the dashboard.")
            return
        
        df = st.session_state.validated_data
        
        # Track dashboard view
        sentry_sdk.set_tag("page", "kpi_dashboard")
        sentry_sdk.set_tag("data_rows", len(df))
        
        # Date range filter
        st.sidebar.markdown("### Filters")
        date_options = {
            "Last 7 Days": 7,
            "Last 30 Days": 30,
            "Last 90 Days": 90,
            "Custom Range": -1
        }
        date_filter = st.sidebar.selectbox("Date Range", list(date_options.keys()))
        
        if date_filter == "Custom Range":
            start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=30))
            end_date = st.sidebar.date_input("End Date", datetime.now())
        else:
            days = date_options[date_filter]
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
        
        # Dimension filters
        dimension_filters = {}
        if "location" in df.columns:
            dimension_filters["location"] = st.sidebar.selectbox(
                "Dealership Location",
                ["All"] + list(df["location"].unique())
            )
        if "vehicle_type" in df.columns:
            dimension_filters["vehicle_type"] = st.sidebar.selectbox(
                "Vehicle Type",
                ["All"] + list(df["vehicle_type"].unique())
            )
        
        # Apply filters
        with st.spinner("Updating dashboard..."):
            filtered_df = apply_date_filter(df, start_date, end_date)
            filtered_df = apply_dimension_filters(
                filtered_df,
                {k: v for k, v in dimension_filters.items() if v != "All"}
            )
        
        # Generate KPI summary
        if "llm_client" in st.session_state:
            summarizer = Summarizer(st.session_state.llm_client)
            
            # Format metrics for summary
            metrics_table = pd.DataFrame({
                "Metric": ["Total Sales", "Avg Gross", "Aged Inventory"],
                "Value": [
                    len(filtered_df),
                    filtered_df["TotalGross"].mean() if "TotalGross" in filtered_df.columns else 0,
                    len(filtered_df[filtered_df["DaysInStock"] > 90]) if "DaysInStock" in filtered_df.columns else 0
                ]
            })
            
            summary = summarizer.summarize(
                "kpi_summary.tpl",
                entity_name="Dealership",
                date_range=f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                metrics_table=metrics_table.to_markdown()
            )
            
            st.info(summary)
        
        # Create layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Average Gross per Rep")
            rep_metrics, rep_summary = calculate_rep_performance(filtered_df)
            if not rep_metrics.empty:
                # Create rep performance chart
                fig = px.bar(
                    rep_metrics,
                    x='SalesRepName',
                    y='avg_gross',
                    title='Average Gross by Sales Rep'
                )
                
                # Enable clicking
                selected_points = st.plotly_chart(
                    fig,
                    use_container_width=True,
                    on_click=lambda data: handle_chart_click("rep_performance", data)
                )
                
                # Show drill-down if rep selected
                if selected_points:
                    rep_name = selected_points[0].get("x")
                    if rep_name:
                        st.markdown(f"#### Details for {rep_name}")
                        rep_data = filtered_df[filtered_df["SalesRepName"] == rep_name]
                        st.dataframe(rep_data)
                
                # Download button
                st.download_button(
                    "Download Rep Performance Data",
                    get_download_data(rep_metrics, "rep_performance"),
                    "rep_performance.csv",
                    "text/csv"
                )
        
        with col2:
            st.markdown("### Inventory Aging")
            aging_metrics, aging_summary = analyze_inventory_aging(filtered_df)
            if not aging_metrics.empty:
                # Create aging chart
                fig = px.pie(
                    aging_metrics,
                    names='age_range',
                    values='count',
                    title='Inventory Age Distribution'
                )
                
                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    on_click=lambda data: handle_chart_click("inventory_aging", data)
                )
                
                # Download button
                st.download_button(
                    "Download Inventory Aging Data",
                    get_download_data(aging_metrics, "inventory_aging"),
                    "inventory_aging.csv",
                    "text/csv"
                )
        
        # Full width sections
        st.markdown("### Lead Conversion Rates")
        conversion_metrics, conversion_summary = calculate_lead_conversion(filtered_df)
        if not conversion_metrics.empty:
            # Create conversion chart
            fig = px.bar(
                conversion_metrics,
                x='LeadSource',
                y='count',
                title='Leads by Source'
            )
            
            st.plotly_chart(
                fig,
                use_container_width=True,
                on_click=lambda data: handle_chart_click("lead_conversion", data)
            )
            
            # Download button
            st.download_button(
                "Download Lead Conversion Data",
                get_download_data(conversion_metrics, "lead_conversion"),
                "lead_conversion.csv",
                "text/csv"
            )
        
        st.markdown("### Forecast Trends")
        render_forecast_trends(filtered_df)
        
        # Full data table with download
        st.markdown("### Detailed Data")
        st.dataframe(filtered_df)
        st.download_button(
            "Download Full Dataset",
            get_download_data(filtered_df, "full_data"),
            "full_data.csv",
            "text/csv"
        )
        
    except Exception as e:
        logger.error(f"Error rendering KPI dashboard: {str(e)}")
        sentry_sdk.capture_exception(e)
        st.error("An error occurred while rendering the dashboard")