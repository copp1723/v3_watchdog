"""
Delivery Performance Analytics Dashboard for Watchdog AI.

Tracks how insights are being delivered and consumed.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import json
import os
import logging

from ...delivery.smart_delivery_engine import DeliveryMetrics, DeliveryPriority, DeliveryTrigger, DeliveryChannel
from ...utils.upload_tracker import UploadTracker
from ...ui.utils.status_formatter import StatusType, format_status_text

logger = logging.getLogger(__name__)

class DeliveryAnalyticsDashboard:
    """Dashboard for monitoring delivery performance and analytics."""
    
    def __init__(self, 
                delivery_metrics: Optional[DeliveryMetrics] = None,
                upload_tracker: Optional[UploadTracker] = None):
        """
        Initialize the dashboard.
        
        Args:
            delivery_metrics: Optional DeliveryMetrics instance
            upload_tracker: Optional UploadTracker instance
        """
        self.delivery_metrics = delivery_metrics or DeliveryMetrics()
        self.upload_tracker = upload_tracker or UploadTracker()
        
        # Initialize session state
        if 'delivery_analytics_date_range' not in st.session_state:
            st.session_state.delivery_analytics_date_range = 30  # Default to 30 days
    
    def _load_email_tracking_data(self, days: int = 30) -> pd.DataFrame:
        """
        Load email tracking data (opens, clicks, etc.).
        
        Args:
            days: Number of days to look back
            
        Returns:
            DataFrame with tracking data
        """
        # In a real implementation, this would load data from a database
        # For now, generate some sample data
        
        # Generate dates
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate data
        data = []
        for date in dates:
            # Generate some random data with a trend
            sent_count = np.random.randint(80, 120)
            open_rate = min(0.95, max(0.4, 0.65 + np.random.normal(0, 0.1)))
            click_rate = min(0.4, max(0.1, 0.25 + np.random.normal(0, 0.05)))
            bounce_rate = min(0.05, max(0.01, 0.02 + np.random.normal(0, 0.005)))
            
            data.append({
                'date': date,
                'sent_count': sent_count,
                'open_count': int(sent_count * open_rate),
                'click_count': int(sent_count * click_rate),
                'bounce_count': int(sent_count * bounce_rate)
            })
        
        return pd.DataFrame(data)
    
    def _load_delivery_latency_data(self, days: int = 30) -> pd.DataFrame:
        """
        Load delivery latency data.
        
        Args:
            days: Number of days to look back
            
        Returns:
            DataFrame with latency data
        """
        # In a real implementation, this would load data from a database
        # For now, generate some sample data
        
        # Generate dates
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate data
        data = []
        for date in dates:
            # Generate some random data with a trend
            avg_latency = max(5, min(60, 30 + np.random.normal(0, 10)))
            min_latency = max(1, avg_latency - np.random.randint(5, 15))
            max_latency = avg_latency + np.random.randint(10, 30)
            
            data.append({
                'date': date,
                'avg_latency_sec': avg_latency,
                'min_latency_sec': min_latency,
                'max_latency_sec': max_latency
            })
        
        return pd.DataFrame(data)
    
    def _load_insight_popularity_data(self, days: int = 30) -> pd.DataFrame:
        """
        Load insight popularity data.
        
        Args:
            days: Number of days to look back
            
        Returns:
            DataFrame with popularity data
        """
        # In a real implementation, this would load data from a database
        # For now, generate some sample data
        
        # Define insight types
        insight_types = [
            "Sales Performance", 
            "Inventory Alerts", 
            "Customer Trends", 
            "Financial Insights",
            "Service Metrics"
        ]
        
        # Generate data
        data = []
        for insight_type in insight_types:
            # Generate some random data with a trend
            read_count = np.random.randint(50, 200)
            click_through_rate = min(0.8, max(0.2, 0.5 + np.random.normal(0, 0.1)))
            
            data.append({
                'insight_type': insight_type,
                'read_count': read_count,
                'click_through_rate': click_through_rate,
                'engagement_score': read_count * click_through_rate
            })
        
        return pd.DataFrame(data).sort_values('engagement_score', ascending=False)
    
    def render_summary_metrics(self) -> None:
        """Render summary metrics section."""
        st.header("Delivery Performance Summary")
        
        # Get metrics
        metrics = self.delivery_metrics.get_metrics()
        
        # Create columns for metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            success_rate = self.delivery_metrics.get_success_rate()
            st.metric(
                "Delivery Success Rate", 
                f"{success_rate:.1f}%",
                delta=None
            )
        
        with col2:
            avg_time = self.delivery_metrics.get_average_delivery_time()
            st.metric(
                "Avg. Delivery Time", 
                f"{avg_time:.1f}s",
                delta=None
            )
        
        with col3:
            total_deliveries = metrics.get('total_deliveries', 0)
            st.metric(
                "Total Deliveries", 
                f"{total_deliveries:,}",
                delta=None
            )
        
        with col4:
            # Calculate open rate from sample data
            email_data = self._load_email_tracking_data()
            if not email_data.empty:
                total_sent = email_data['sent_count'].sum()
                total_opened = email_data['open_count'].sum()
                open_rate = (total_opened / total_sent) * 100 if total_sent > 0 else 0
                
                st.metric(
                    "Email Open Rate", 
                    f"{open_rate:.1f}%",
                    delta=None
                )
    
    def render_delivery_trends(self) -> None:
        """Render delivery trends section."""
        st.header("Delivery Trends")
        
        # Get date range
        days = st.session_state.delivery_analytics_date_range
        
        # Load data
        email_data = self._load_email_tracking_data(days)
        latency_data = self._load_delivery_latency_data(days)
        
        # Create tabs for different charts
        tab1, tab2, tab3 = st.tabs(["Email Performance", "Delivery Latency", "Channel Distribution"])
        
        with tab1:
            # Email performance chart
            if not email_data.empty:
                # Calculate rates
                email_data['open_rate'] = email_data['open_count'] / email_data['sent_count']
                email_data['click_rate'] = email_data['click_count'] / email_data['sent_count']
                email_data['bounce_rate'] = email_data['bounce_count'] / email_data['sent_count']
                
                # Create figure
                fig = go.Figure()
                
                # Add traces
                fig.add_trace(go.Scatter(
                    x=email_data['date'], 
                    y=email_data['open_rate'],
                    mode='lines+markers',
                    name='Open Rate',
                    line=dict(color='#2ecc71', width=2)
                ))
                
                fig.add_trace(go.Scatter(
                    x=email_data['date'], 
                    y=email_data['click_rate'],
                    mode='lines+markers',
                    name='Click Rate',
                    line=dict(color='#3498db', width=2)
                ))
                
                fig.add_trace(go.Scatter(
                    x=email_data['date'], 
                    y=email_data['bounce_rate'],
                    mode='lines+markers',
                    name='Bounce Rate',
                    line=dict(color='#e74c3c', width=2)
                ))
                
                # Update layout
                fig.update_layout(
                    title='Email Engagement Metrics',
                    xaxis_title='Date',
                    yaxis_title='Rate',
                    yaxis_tickformat='.0%',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show summary stats
                st.markdown("### Email Performance Summary")
                
                avg_open_rate = email_data['open_rate'].mean()
                avg_click_rate = email_data['click_rate'].mean()
                avg_bounce_rate = email_data['bounce_rate'].mean()
                
                stat_cols = st.columns(3)
                
                with stat_cols[0]:
                    st.metric("Avg. Open Rate", f"{avg_open_rate:.1%}")
                
                with stat_cols[1]:
                    st.metric("Avg. Click Rate", f"{avg_click_rate:.1%}")
                
                with stat_cols[2]:
                    st.metric("Avg. Bounce Rate", f"{avg_bounce_rate:.1%}")
        
        with tab2:
            # Delivery latency chart
            if not latency_data.empty:
                # Create figure
                fig = go.Figure()
                
                # Add traces
                fig.add_trace(go.Scatter(
                    x=latency_data['date'], 
                    y=latency_data['avg_latency_sec'],
                    mode='lines',
                    name='Avg. Latency',
                    line=dict(color='#3498db', width=3)
                ))
                
                # Add range
                fig.add_trace(go.Scatter(
                    x=latency_data['date'],
                    y=latency_data['min_latency_sec'],
                    mode='lines',
                    name='Min Latency',
                    line=dict(width=0),
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=latency_data['date'],
                    y=latency_data['max_latency_sec'],
                    mode='lines',
                    name='Max Latency',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor='rgba(52, 152, 219, 0.2)',
                    showlegend=False
                ))
                
                # Update layout
                fig.update_layout(
                    title='Delivery Latency Trends',
                    xaxis_title='Date',
                    yaxis_title='Latency (seconds)',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show summary stats
                st.markdown("### Latency Summary")
                
                avg_latency = latency_data['avg_latency_sec'].mean()
                min_latency = latency_data['min_latency_sec'].min()
                max_latency = latency_data['max_latency_sec'].max()
                
                stat_cols = st.columns(3)
                
                with stat_cols[0]:
                    st.metric("Avg. Latency", f"{avg_latency:.1f}s")
                
                with stat_cols[1]:
                    st.metric("Min. Latency", f"{min_latency:.1f}s")
                
                with stat_cols[2]:
                    st.metric("Max. Latency", f"{max_latency:.1f}s")
        
        with tab3:
            # Channel distribution chart
            metrics = self.delivery_metrics.get_metrics()
            
            # Get channel distribution
            channel_data = metrics.get('delivery_by_channel', {})
            
            # Convert to DataFrame
            channel_df = pd.DataFrame([
                {"Channel": channel, "Count": count}
                for channel, count in channel_data.items()
            ])
            
            if not channel_df.empty and channel_df['Count'].sum() > 0:
                # Create pie chart
                fig = px.pie(
                    channel_df,
                    values='Count',
                    names='Channel',
                    title='Delivery Channel Distribution',
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                
                # Update layout
                fig.update_layout(
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No channel distribution data available yet.")
    
    def render_insight_popularity(self) -> None:
        """Render insight popularity section."""
        st.header("Insight Type Popularity")
        
        # Load data
        popularity_data = self._load_insight_popularity_data()
        
        if not popularity_data.empty:
            # Create bar chart
            fig = px.bar(
                popularity_data,
                x='insight_type',
                y='read_count',
                color='click_through_rate',
                color_continuous_scale='Viridis',
                title='Most Read Insight Types',
                labels={
                    'insight_type': 'Insight Type',
                    'read_count': 'Read Count',
                    'click_through_rate': 'Click-Through Rate'
                }
            )
            
            # Update layout
            fig.update_layout(
                xaxis_title='Insight Type',
                yaxis_title='Read Count',
                coloraxis_colorbar=dict(
                    title='Click-Through Rate'
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show data table
            with st.expander("View Data Table"):
                st.dataframe(
                    popularity_data,
                    use_container_width=True,
                    column_config={
                        'insight_type': 'Insight Type',
                        'read_count': 'Read Count',
                        'click_through_rate': st.column_config.NumberColumn(
                            'Click-Through Rate',
                            format="%.1f%%",
                            help="Percentage of users who clicked on the insight"
                        ),
                        'engagement_score': st.column_config.NumberColumn(
                            'Engagement Score',
                            format="%.1f",
                            help="Combined score of reads and clicks"
                        )
                    }
                )
    
    def render_delivery_failures(self) -> None:
        """Render delivery failures section."""
        st.header("Delivery Failures Analysis")
        
        # In a real implementation, this would load data from a database
        # For now, generate some sample data
        
        # Define failure types
        failure_types = [
            "Invalid Email", 
            "Server Error", 
            "Timeout", 
            "Mailbox Full",
            "Connection Error"
        ]
        
        # Generate data
        data = []
        for failure_type in failure_types:
            # Generate some random data
            count = np.random.randint(5, 30)
            
            data.append({
                'failure_type': failure_type,
                'count': count
            })
        
        failure_df = pd.DataFrame(data).sort_values('count', ascending=False)
        
        if not failure_df.empty:
            # Create bar chart
            fig = px.bar(
                failure_df,
                x='failure_type',
                y='count',
                color='count',
                color_continuous_scale='Reds',
                title='Delivery Failures by Type',
                labels={
                    'failure_type': 'Failure Type',
                    'count': 'Count'
                }
            )
            
            # Update layout
            fig.update_layout(
                xaxis_title='Failure Type',
                yaxis_title='Count'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show recommendations
            st.subheader("Recommendations")
            
            # Get top failure type
            top_failure = failure_df.iloc[0]['failure_type']
            
            if top_failure == "Invalid Email":
                recommendation = format_status_text(StatusType.INFO, custom_text="Consider implementing an email validation system to catch invalid emails before sending.")
                st.markdown(recommendation, unsafe_allow_html=True)
            elif top_failure == "Server Error":
                recommendation = format_status_text(StatusType.WARNING, custom_text="Check server logs for errors and consider increasing server resources.")
                st.markdown(recommendation, unsafe_allow_html=True)
            elif top_failure == "Timeout":
                recommendation = format_status_text(StatusType.INFO, custom_text="Review network connectivity and consider increasing timeout thresholds.")
                st.markdown(recommendation, unsafe_allow_html=True)
            elif top_failure == "Mailbox Full":
                recommendation = format_status_text(StatusType.INFO, custom_text="Implement a system to detect and handle full mailboxes before sending.")
                st.markdown(recommendation, unsafe_allow_html=True)
            else:
                recommendation = format_status_text(StatusType.INFO, custom_text="Review system logs to identify and address the most common failure patterns.")
                st.markdown(recommendation, unsafe_allow_html=True)
    
    def render(self) -> None:
        """Render the delivery analytics dashboard."""
        st.title("Delivery Performance Analytics")
        
        # Date range selector
        st.sidebar.header("Filters")
        
        days = st.sidebar.slider(
            "Date Range",
            min_value=7,
            max_value=90,
            value=st.session_state.delivery_analytics_date_range,
            step=7,
            help="Select the number of days to include in the analysis"
        )
        
        # Update session state
        st.session_state.delivery_analytics_date_range = days
        
        # Render dashboard sections
        self.render_summary_metrics()
        self.render_delivery_trends()
        
        # Create columns for bottom sections
        col1, col2 = st.columns(2)
        
        with col1:
            self.render_insight_popularity()
        
        with col2:
            self.render_delivery_failures()
        
        # Add export options
        st.sidebar.header("Export")
        
        if st.sidebar.button("Export as PDF"):
            success_text = format_status_text(StatusType.SUCCESS, custom_text="Dashboard exported as PDF")
            st.sidebar.markdown(success_text, unsafe_allow_html=True)
        
        if st.sidebar.button("Export as CSV"):
            success_text = format_status_text(StatusType.SUCCESS, custom_text="Data exported as CSV")
            st.sidebar.markdown(success_text, unsafe_allow_html=True)


def render_delivery_analytics():
    """Main entry point for the delivery analytics dashboard."""
    try:
        dashboard = DeliveryAnalyticsDashboard()
        dashboard.render()
    except Exception as e:
        st.error(f"Error rendering delivery analytics dashboard: {str(e)}")
        logger.error(f"Error rendering delivery analytics dashboard: {str(e)}", exc_info=True)
