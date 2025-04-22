"""
Lead Source ROI Dashboard for Watchdog AI.

This module provides a comprehensive dashboard for analyzing
lead source ROI with interactive visualizations.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple

# Import project components
from ...insights.roi_visualizer import ROIVisualizer
from ....validators.lead_source_roi import (
    LeadSourceNormalizer, 
    LeadSourceROI,
    create_lead_source_roi_schema
)

logger = logging.getLogger(__name__)

def render_lead_source_roi_dashboard():
    """Render the Lead Source ROI Dashboard."""
    st.title("Lead Source ROI Dashboard")
    
    # Initialize ROI components
    roi_calculator = LeadSourceROI()
    roi_visualizer = ROIVisualizer()
    
    # Sidebar for controls
    with st.sidebar:
        st.header("Dashboard Controls")
        
        # Date Range Selection
        st.subheader("Date Range")
        date_range = st.selectbox(
            "Select Period",
            options=["Last 30 Days", "Last 90 Days", "Year to Date", "Last Year", "Custom"],
            index=0
        )
        
        if date_range == "Custom":
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Start Date",
                    value=(datetime.now() - timedelta(days=30)).date()
                )
            with col2:
                end_date = st.date_input(
                    "End Date",
                    value=datetime.now().date()
                )
        else:
            today = datetime.now().date()
            if date_range == "Last 30 Days":
                start_date = today - timedelta(days=30)
                end_date = today
            elif date_range == "Last 90 Days":
                start_date = today - timedelta(days=90)
                end_date = today
            elif date_range == "Year to Date":
                start_date = today.replace(month=1, day=1)
                end_date = today
            elif date_range == "Last Year":
                start_date = today.replace(year=today.year-1, month=1, day=1)
                end_date = today.replace(year=today.year-1, month=12, day=31)
        
        # Timeframe Filter
        st.subheader("Timeframe")
        timeframe = st.radio(
            "Calculate ROI by",
            options=["Monthly", "Weekly"],
            index=0
        )
        
        # ROI calculation options
        st.subheader("ROI Options")
        
        normalize_sources = st.checkbox(
            "Normalize Lead Source Names",
            value=True,
            help="Standardize variations of lead source names"
        )
        
        include_zero_cost = st.checkbox(
            "Include Sources with No Cost",
            value=True,
            help="Include sources with no defined cost (infinite ROI)"
        )
        
        # Visualization controls
        st.subheader("Visualization Controls")
        
        chart_limit = st.slider(
            "Max Sources to Display",
            min_value=3,
            max_value=15,
            value=8
        )
        
        color_scheme = st.selectbox(
            "Color Scheme",
            options=["category10", "tableau10", "set1", "set2", "set3"],
            index=0
        )
    
    # Main content area - Tabs
    tabs = st.tabs(["ROI Overview", "Source Cost Management", "Trend Analysis", "Export"])
    
    with tabs[0]:  # ROI Overview
        st.header("Lead Source ROI Overview")
        
        # Load sample data if real data not available
        # In a real implementation, this would load actual data based on date range
        data = _load_sample_data(start_date, end_date)
        
        # Process data for ROI analysis
        roi_df = roi_calculator.process_dataframe(
            data,
            normalize=normalize_sources,
            weekly=(timeframe == "Weekly")
        )
        
        # Display summary metrics
        summary = roi_calculator.get_roi_summary(roi_df)
        
        if "error" not in summary:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Ad Spend",
                    f"${summary['total_cost']:,.2f}"
                )
            
            with col2:
                st.metric(
                    "Total Revenue",
                    f"${summary['total_revenue']:,.2f}"
                )
            
            with col3:
                st.metric(
                    "Overall ROI",
                    summary['overall_roi_percentage']
                )
            
            with col4:
                st.metric(
                    "Total Leads",
                    f"{summary['total_leads']:,}"
                )
        
        # Create and display ROI chart
        roi_chart = roi_visualizer.create_roi_bar_chart(roi_df, limit=chart_limit)
        st.altair_chart(roi_chart, use_container_width=True)
        
        # Display top performers
        if "error" not in summary and summary["top_performers"]:
            st.subheader("Top Performing Sources")
            top_df = pd.DataFrame(summary["top_performers"])
            st.dataframe(top_df, use_container_width=True)
        
        # Display full data table
        st.subheader("Detailed ROI Data")
        st.dataframe(roi_df, use_container_width=True)
    
    with tabs[1]:  # Source Cost Management
        st.header("Lead Source Cost Management")
        
        # Display current cost data
        current_costs = roi_calculator.cost_data['sources']
        
        if current_costs:
            cost_data = []
            for source, data in current_costs.items():
                cost_data.append({
                    "Source": source,
                    "Monthly Cost": data.get("monthly_cost", 0),
                    "Last Updated": datetime.fromisoformat(
                        data.get("history", [{"effective_date": datetime.now().isoformat()}])[-1]["effective_date"]
                    ).strftime("%Y-%m-%d")
                })
            
            cost_df = pd.DataFrame(cost_data)
            st.dataframe(cost_df, use_container_width=True)
        else:
            st.info("No cost data available. Add source costs below.")
        
        # Add/update source cost
        st.subheader("Add/Update Source Cost")
        
        col1, col2 = st.columns(2)
        
        with col1:
            source_name = st.text_input("Lead Source Name")
        
        with col2:
            monthly_cost = st.number_input(
                "Monthly Cost ($)",
                min_value=0.0,
                step=100.0
            )
        
        if st.button("Save Source Cost"):
            if not source_name:
                st.error("Lead source name is required")
            else:
                roi_calculator.update_source_cost(source_name, monthly_cost)
                st.success(f"Updated cost for {source_name} to ${monthly_cost:,.2f}/month")
                st.rerun()  # Refresh the page to show updated data
    
    with tabs[2]:  # Trend Analysis
        st.header("ROI Trend Analysis")
        
        # Generate trend data based on real calculations or sample data
        # In a real implementation, this would query historical data
        trend_data = _generate_trend_data(start_date, end_date)
        
        # Create and display trend chart
        trend_chart = roi_visualizer.create_roi_trend_chart(trend_data)
        st.altair_chart(trend_chart, use_container_width=True)
        
        # Display additional metrics
        st.subheader("Source Performance Metrics")
        metrics_chart = roi_visualizer.create_metrics_tooltip_chart(roi_df)
        st.altair_chart(metrics_chart, use_container_width=True)
    
    with tabs[3]:  # Export
        st.header("Export ROI Data")
        
        export_format = st.radio(
            "Export Format",
            options=["CSV", "Excel", "JSON"],
            horizontal=True
        )
        
        if st.button("Generate Export"):
            # Generate export file
            file_name = f"lead_source_roi_{datetime.now().strftime('%Y%m%d')}"
            
            if export_format == "CSV":
                csv = roi_df.to_csv(index=False)
                st.download_button(
                    "Download CSV",
                    csv,
                    file_name=f"{file_name}.csv",
                    mime="text/csv"
                )
            elif export_format == "Excel":
                # For Excel, we'd need a BytesIO object in a real implementation
                # This is a placeholder
                st.info(
                    "Excel export would be implemented here in the full version. "
                    "Use CSV export for now."
                )
            elif export_format == "JSON":
                json_data = roi_df.to_json(orient="records")
                st.download_button(
                    "Download JSON",
                    json_data,
                    file_name=f"{file_name}.json",
                    mime="application/json"
                )
        
        # Export summary report
        st.subheader("Export Summary Report")
        if st.button("Generate Summary Report"):
            # In a real implementation, this would generate a PDF or detailed report
            st.info(
                "Summary report generation would be implemented here in the full version."
            )


def _load_sample_data(start_date, end_date):
    """Load sample data for demonstration purposes."""
    # Generate random data for testing
    lead_sources = [
        "Website", "CarGurus", "AutoTrader", "Cars.com", 
        "Facebook", "Walk-in", "Referral", "Phone"
    ]
    
    # Calculate date range
    date_range = (end_date - start_date).days
    
    # Generate sample data
    np.random.seed(42)  # For reproducibility
    
    # Create sample DataFrame
    data = {
        "LeadSource": [],
        "LeadDate": [],
        "TotalGross": [],
        "Closed": []
    }
    
    # Generate data for each lead source
    for source in lead_sources:
        # Generate random number of leads for this source
        lead_count = np.random.randint(10, 50)
        
        # Generate random dates within range
        dates = [
            start_date + timedelta(days=np.random.randint(0, date_range)) 
            for _ in range(lead_count)
        ]
        
        # Generate random revenue amounts
        if source in ["Website", "CarGurus", "AutoTrader"]:
            # Higher revenue sources
            revenues = np.random.normal(2000, 800, lead_count)
        else:
            # Lower revenue sources
            revenues = np.random.normal(1500, 600, lead_count)
        
        # Add to data
        data["LeadSource"].extend([source] * lead_count)
        data["LeadDate"].extend(dates)
        data["TotalGross"].extend(revenues)
        data["Closed"].extend([True] * lead_count)
    
    return pd.DataFrame(data)


def _generate_trend_data(start_date, end_date):
    """Generate sample trend data for demonstration purposes."""
    # Create date range by week
    weeks = pd.date_range(start=start_date, end=end_date, freq='W')
    
    # Lead sources to track
    lead_sources = ["website", "cargurus", "autotrader", "facebook"]
    
    # Create trend data
    trend_data = {
        "Date": [],
        "LeadSource": [],
        "ROI": []
    }
    
    # Generate data for each source over time
    for source in lead_sources:
        # Base ROI value
        if source == "website":
            base_roi = 2.5
            amplitude = 0.5
        elif source == "cargurus":
            base_roi = 1.8
            amplitude = 0.3
        elif source == "autotrader":
            base_roi = 1.5
            amplitude = 0.2
        else:
            base_roi = 1.2
            amplitude = 0.4
        
        # Generate ROI values with some trending
        for i, week in enumerate(weeks):
            # Add some seasonal and trend patterns
            seasonal = amplitude * np.sin(i / 4)  # Seasonal component
            trend = 0.05 * (i / len(weeks))  # Slight upward trend
            
            # Calculate ROI with noise
            roi = max(0, base_roi + seasonal + trend + np.random.normal(0, 0.1))
            
            # Add to data
            trend_data["Date"].append(week)
            trend_data["LeadSource"].append(source)
            trend_data["ROI"].append(roi)
    
    return pd.DataFrame(trend_data)