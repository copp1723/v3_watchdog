"""
Insights View for Watchdog AI.

This module provides a dedicated page for displaying and exploring 
the key insights from automotive dealership data.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from typing import Optional, Dict, Any, List

from src.insights.core_insights import (
    get_sales_rep_performance,
    get_inventory_aging_alerts
)
from src.insights.insight_generator import insight_generator
from src.utils.chart_utils import create_chart

def display_sales_rep_performance(df: pd.DataFrame) -> None:
    """
    Display the sales rep performance insight in the UI.
    
    Args:
        df: The DataFrame containing sales data
    """
    with st.spinner("Generating sales rep performance analysis..."):
        try:
            # Get the performance metrics
            performance_df = get_sales_rep_performance(df)
            
            if performance_df.empty:
                st.warning("No sales rep performance data could be generated from the provided dataset.")
                return
            
            # Display the summary
            st.subheader("Sales Representative Performance")
            
            # Create metrics for overview
            cols = st.columns(3)
            with cols[0]:
                st.metric("Total Sales Reps", len(performance_df))
            
            with cols[1]:
                top_performer = performance_df.iloc[0]
                st.metric("Top Performer", top_performer['rep_name'])
            
            with cols[2]:
                total_cars = performance_df['total_cars_sold'].sum()
                st.metric("Total Cars Sold", total_cars)
            
            # Create a bar chart for total cars sold
            fig = px.bar(
                performance_df,
                x='rep_name',
                y='total_cars_sold',
                color='average_gross',
                labels={
                    'rep_name': 'Sales Representative',
                    'total_cars_sold': 'Total Cars Sold',
                    'average_gross': 'Average Gross ($)'
                },
                color_continuous_scale='Viridis',
                title='Total Cars Sold by Sales Representative'
            )
            
            fig.update_layout(
                xaxis_title="Sales Representative",
                yaxis_title="Total Cars Sold",
                coloraxis_colorbar_title="Avg Gross ($)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Create a table for detailed metrics
            st.subheader("Detailed Sales Metrics")
            
            # Format the table for better display
            display_df = performance_df.copy()
            display_df['average_gross'] = display_df['average_gross'].map('${:,.2f}'.format)
            display_df['delta_from_dealership_avg'] = display_df['delta_from_dealership_avg'].map('${:,.2f}'.format)
            display_df.columns = ['Sales Rep', 'Total Cars Sold', 'Average Gross', 'Δ from Dealership Avg']
            
            st.dataframe(display_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error generating sales rep performance: {str(e)}")


def display_inventory_aging_alerts(df: pd.DataFrame) -> None:
    """
    Display the inventory aging alerts insight in the UI.
    
    Args:
        df: The DataFrame containing inventory data
    """
    with st.spinner("Analyzing inventory aging patterns..."):
        try:
            # Get inventory aging alerts with default threshold
            alerts_df = get_inventory_aging_alerts(df)
            
            if alerts_df.empty:
                st.info("No inventory aging alerts found with the default threshold. Try adjusting the threshold.")
                
                # Let user adjust threshold
                threshold = st.slider(
                    "Adjust threshold (days over model average)",
                    min_value=10,
                    max_value=120,
                    value=30,
                    step=5
                )
                
                # Try again with user-specified threshold
                alerts_df = get_inventory_aging_alerts(df, threshold_days=threshold)
                
                if alerts_df.empty:
                    st.warning("No inventory aging alerts found even with adjusted threshold.")
                    return
            
            # Display the results
            st.subheader("Inventory Aging Alerts")
            
            # Create summary metrics
            cols = st.columns(3)
            with cols[0]:
                st.metric("Vehicles Exceeding Threshold", len(alerts_df))
            
            with cols[1]:
                avg_excess = alerts_df['excess_days'].mean()
                st.metric("Avg Days Over Model Average", f"{avg_excess:.1f}")
            
            with cols[2]:
                max_days = alerts_df['days_on_lot'].max()
                st.metric("Longest Days on Lot", int(max_days))
            
            # Create a scatter plot of days on lot vs excess days
            fig = px.scatter(
                alerts_df,
                x='days_on_lot',
                y='excess_days',
                color='model',
                size='excess_days',
                hover_data=['vin', 'model_avg_days'],
                labels={
                    'days_on_lot': 'Days on Lot',
                    'excess_days': 'Days Exceeding Model Average',
                    'model': 'Vehicle Model',
                    'vin': 'Vehicle ID',
                    'model_avg_days': 'Model Average Days'
                },
                title='Inventory Aging Anomalies'
            )
            
            fig.update_layout(
                xaxis_title="Days on Lot",
                yaxis_title="Days Exceeding Model Average"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show the detailed data
            st.subheader("Vehicles Requiring Attention")
            
            # Format for better display
            display_df = alerts_df.copy()
            display_df.columns = ['Vehicle ID', 'Model', 'Days on Lot', 'Model Avg Days', 'Excess Days']
            
            # Convert to integer where appropriate
            display_df['Days on Lot'] = display_df['Days on Lot'].astype(int)
            display_df['Model Avg Days'] = display_df['Model Avg Days'].round(1)
            display_df['Excess Days'] = display_df['Excess Days'].astype(int)
            
            st.dataframe(display_df, use_container_width=True)
            
            # Add recommendations
            st.subheader("Recommendations")
            
            # Example recommendations based on the data
            if len(alerts_df) > 0:
                st.markdown("""
                ### Action Items
                
                1. **Focus on oldest inventory first**: Prioritize vehicles with highest excess days
                2. **Review pricing strategy**: Consider price adjustments for vehicles exceeding model average by >60 days
                3. **Targeted marketing**: Create specific campaigns for slow-moving models
                4. **Evaluate acquisition strategy**: Review purchasing decisions for consistently slow-moving models
                """)
            
        except Exception as e:
            st.error(f"Error analyzing inventory aging: {str(e)}")


def display_monthly_gross_margin(df: pd.DataFrame) -> None:
    """
    Display the monthly gross margin insight in the UI.
    
    Args:
        df: The DataFrame containing sales data
    """
    with st.spinner("Analyzing monthly gross margin vs target..."):
        try:
            # Calculate the insight using insight generator
            insight_result = insight_generator.generate_specific_insight("monthly_gross_margin", df)
            
            if "error" in insight_result:
                st.warning(f"Could not generate monthly gross margin insight: {insight_result['error']}")
                return
                
            # Display the summary
            st.subheader("Monthly Gross Margin vs. Target")
            
            # Extract data for display
            monthly_data = pd.DataFrame(insight_result.get("monthly_data", []))
            if monthly_data.empty:
                st.warning("No monthly data available for analysis.")
                return
                
            target_margin = insight_result.get("target_margin", 0.20)
            
            # Create metrics for summary
            cols = st.columns(3)
            
            with cols[0]:
                if "margin" in monthly_data.columns:
                    current_margin = monthly_data.iloc[-1]["margin"]
                    delta = current_margin - target_margin
                    st.metric(
                        "Current Gross Margin", 
                        f"{current_margin:.1%}", 
                        f"{delta:.1%}",
                        delta_color="normal" if delta >= 0 else "inverse"
                    )
                else:
                    # Show total gross if margin not available
                    current_gross = monthly_data.iloc[-1]["total_gross"]
                    st.metric("Current Month Gross", f"${current_gross:,.2f}")
            
            with cols[1]:
                if "trend_data" in insight_result:
                    trend = insight_result["trend_data"]
                    if "margin_change_ppt" in trend:
                        st.metric(
                            "Margin Trend", 
                            trend.get("margin_trend", "stable").title(),
                            f"{trend['margin_change_ppt']:.1f} pts"
                        )
                    else:
                        st.metric("Trend", trend.get("gross_trend", "stable").title())
                else:
                    st.metric("Target Margin", f"{target_margin:.1%}")
            
            with cols[2]:
                if "deal_count" in monthly_data.columns:
                    total_deals = monthly_data["deal_count"].sum()
                    st.metric("Total Deals", f"{total_deals:,}")
                else:
                    avg_gross = monthly_data["avg_gross_per_deal"].mean()
                    st.metric("Avg Gross/Deal", f"${avg_gross:,.2f}")
            
            # Create chart
            if "margin" in monthly_data.columns:
                fig = px.line(
                    monthly_data,
                    x="month_str", 
                    y=["margin", "target_margin"],
                    labels={
                        "month_str": "Month",
                        "margin": "Gross Margin",
                        "target_margin": "Target"
                    },
                    title="Monthly Gross Margin vs. Target"
                )
                
                # Update y-axis to show as percentage
                fig.update_layout(
                    yaxis_tickformat='.1%',
                    hovermode="x unified",
                    legend_title_text=''
                )
                
                # Add color coding
                fig.update_traces(
                    line=dict(width=3),
                    selector=dict(name="margin")
                )
                fig.update_traces(
                    line=dict(width=2, dash='dash'),
                    selector=dict(name="target_margin")
                )
                
            else:
                # Fallback to gross profit chart
                fig = px.line(
                    monthly_data,
                    x="month_str", 
                    y="total_gross",
                    labels={
                        "month_str": "Month",
                        "total_gross": "Total Gross Profit"
                    },
                    title="Monthly Gross Profit"
                )
                
                # Update y-axis format
                fig.update_layout(
                    yaxis_tickformat='$,.0f',
                    hovermode="x unified"
                )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display insights and recommendations
            if "insights" in insight_result and insight_result["insights"]:
                st.subheader("Key Findings")
                for insight in insight_result["insights"]:
                    st.markdown(f"**{insight['title']}**: {insight['description']}")
            
            if "recommendations" in insight_result and insight_result["recommendations"]:
                st.subheader("Recommendations")
                for i, rec in enumerate(insight_result["recommendations"], 1):
                    st.markdown(f"{i}. {rec}")
                    
            # Show data table
            with st.expander("Monthly Data Table"):
                display_df = monthly_data.copy()
                if "margin" in display_df.columns:
                    display_df["margin"] = display_df["margin"].map('{:.1%}'.format)
                    if "target_margin" in display_df.columns:
                        display_df["target_margin"] = display_df["target_margin"].map('{:.1%}'.format)
                
                display_df["total_gross"] = display_df["total_gross"].map('${:,.2f}'.format)
                display_df["avg_gross_per_deal"] = display_df["avg_gross_per_deal"].map('${:,.2f}'.format)
                
                if "month" in display_df.columns:
                    display_df = display_df.drop("month", axis=1)
                
                # Rename columns for display
                display_df = display_df.rename(columns={
                    "month_str": "Month",
                    "total_gross": "Total Gross",
                    "deal_count": "Deal Count",
                    "avg_gross_per_deal": "Avg Gross/Deal",
                    "margin": "Gross Margin",
                    "target_margin": "Target Margin"
                })
                
                st.dataframe(display_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error analyzing monthly gross margin: {str(e)}")


def display_lead_conversion_rate(df: pd.DataFrame) -> None:
    """
    Display the lead conversion rate insight in the UI.
    
    Args:
        df: The DataFrame containing lead/sales data
    """
    with st.spinner("Analyzing lead conversion rates by source..."):
        try:
            # Calculate the insight using insight generator
            insight_result = insight_generator.generate_specific_insight("lead_conversion_rate", df)
            
            if "error" in insight_result:
                st.warning(f"Could not generate lead conversion rate insight: {insight_result['error']}")
                return
                
            # Display the summary
            st.subheader("Lead Conversion Rate by Source")
            
            # Extract data for display
            source_data = pd.DataFrame(insight_result.get("source_data", []))
            if source_data.empty:
                st.warning("No source data available for analysis.")
                return
            
            # Get overall stats
            overall_rate = insight_result.get("overall_conversion_rate", 0)
            total_leads = insight_result.get("total_leads", 0)
            converted_leads = insight_result.get("converted_leads", 0)
            
            # Create metrics for summary
            cols = st.columns(3)
            
            with cols[0]:
                st.metric("Overall Conversion Rate", f"{overall_rate:.1%}")
            
            with cols[1]:
                if "trend_data" in insight_result:
                    trend = insight_result["trend_data"]
                    direction = trend.get("trend_direction", "stable").title()
                    change = trend.get("rate_change_ppt", 0)
                    st.metric(
                        "Trend", 
                        direction,
                        f"{change:.1f} pts",
                        delta_color="normal" if change >= 0 else "inverse"
                    )
                else:
                    st.metric("Total Leads", f"{total_leads:,}")
            
            with cols[2]:
                top_source = source_data.iloc[0]
                source_name = top_source["LeadSource"]
                top_rate = top_source["conversion_rate"]
                st.metric("Best Source", f"{source_name} ({top_rate:.1%})")
            
            # Create source comparison chart
            chart_df = source_data.copy()
            chart_df["conversion_pct"] = chart_df["conversion_rate"] * 100
            
            # Limit to top sources for readability
            if len(chart_df) > 8:
                chart_df = chart_df.head(8)
            
            fig = px.bar(
                chart_df,
                x="LeadSource", 
                y="conversion_pct",
                color="total_count",
                labels={
                    "LeadSource": "Lead Source",
                    "conversion_pct": "Conversion Rate (%)",
                    "total_count": "Total Leads"
                },
                title="Lead Conversion Rate by Source",
                color_continuous_scale="Viridis"
            )
            
            fig.update_layout(
                xaxis_title="Lead Source",
                yaxis_title="Conversion Rate (%)",
                yaxis_ticksuffix="%",
                coloraxis_colorbar_title="Lead Count"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display trend chart if available
            if "monthly_data" in insight_result:
                monthly_data = pd.DataFrame(insight_result["monthly_data"])
                if not monthly_data.empty and "month_str" in monthly_data.columns:
                    # Sort by month for proper display
                    try:
                        monthly_data = monthly_data.sort_values("month", kind="stable")
                    except:
                        # Fallback if sorting fails
                        pass
                    
                    monthly_data["conversion_pct"] = monthly_data["conversion_rate"] * 100
                    
                    trend_fig = px.line(
                        monthly_data,
                        x="month_str", 
                        y="conversion_pct",
                        labels={
                            "month_str": "Month",
                            "conversion_pct": "Conversion Rate (%)"
                        },
                        title="Lead Conversion Rate Trend",
                        markers=True
                    )
                    
                    trend_fig.update_layout(
                        xaxis_title="Month",
                        yaxis_title="Conversion Rate (%)",
                        yaxis_ticksuffix="%"
                    )
                    
                    st.plotly_chart(trend_fig, use_container_width=True)
            
            # Display insights and recommendations
            if "insights" in insight_result and insight_result["insights"]:
                st.subheader("Key Findings")
                for insight in insight_result["insights"]:
                    st.markdown(f"**{insight['title']}**: {insight['description']}")
            
            if "recommendations" in insight_result and insight_result["recommendations"]:
                st.subheader("Recommendations")
                for i, rec in enumerate(insight_result["recommendations"], 1):
                    st.markdown(f"{i}. {rec}")
                    
            # Show source data table
            with st.expander("Source Data Table"):
                display_df = source_data.copy()
                display_df["conversion_rate"] = display_df["conversion_rate"].map('{:.1%}'.format)
                display_df["volume_percentage"] = display_df["volume_percentage"].map('{:.1f}%'.format)
                
                # Rename columns for display
                display_df = display_df.rename(columns={
                    "LeadSource": "Lead Source",
                    "converted_count": "Converted",
                    "total_count": "Total Leads",
                    "conversion_rate": "Conversion Rate",
                    "volume_percentage": "% of Total Volume"
                })
                
                st.dataframe(display_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error analyzing lead conversion rates: {str(e)}")


def insights_view() -> None:
    """
    Main function to render the insights view page.
    """
    st.title("Watchdog AI - Dealership Insights")
    
    st.markdown("""
    This page provides critical insights derived from your dealership data.
    Upload your data file to generate insights about sales performance, inventory aging, and more.
    """)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload your dealership data (CSV or Excel)",
        type=["csv", "xlsx", "xls"],
        help="Upload your sales or inventory data file to generate insights"
    )
    
    if uploaded_file is not None:
        try:
            # Read the file into a DataFrame
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Display basic information about the data
            st.write(f"Successfully loaded {len(df)} records with {len(df.columns)} columns")
            
            # Data preview
            with st.expander("Data Preview"):
                st.dataframe(df.head(), use_container_width=True)
            
            # Create tabs for different insights
            tabs = st.tabs([
                "Monthly Gross Margin", 
                "Lead Conversion Rate", 
                "Sales Rep Performance", 
                "Inventory Aging Alerts"
            ])
            
            with tabs[0]:
                display_monthly_gross_margin(df)
                
            with tabs[1]:
                display_lead_conversion_rate(df)
            
            with tabs[2]:
                display_sales_rep_performance(df)
            
            with tabs[3]:
                display_inventory_aging_alerts(df)
                
        except Exception as e:
            st.error(f"Error reading or processing file: {str(e)}")
    else:
        # Show placeholder when no file is uploaded
        st.info("Upload a file to get started with insights.")
        
        # Example image or placeholder
        st.markdown("""
        ### What insights will you discover?
        
        - **Monthly Gross Margin**: Track your gross margin performance against targets
        - **Lead Conversion Rate**: See which lead sources convert best and trending
        - **Sales Rep Performance**: Identify top performers and opportunities for coaching
        - **Inventory Aging Alerts**: Find vehicles that have been in inventory too long
        
        Upload your dealership data file to begin analysis.
        """)


if __name__ == "__main__":
    insights_view()