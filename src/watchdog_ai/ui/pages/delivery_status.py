"""
Delivery Status Dashboard for monitoring notification delivery and performance.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List, Optional

from ...insights.metrics_logger import MetricsLogger
from ...scheduler.notification_service import NotificationService
from ...utils.config import get_feature_flags
from ...ui.utils.status_formatter import StatusType, format_status_text

class DeliveryStatusDashboard:
    """Dashboard for monitoring notification delivery status and performance."""
    
    def __init__(self, notification_service: Optional[NotificationService] = None,
                 metrics_logger: Optional[MetricsLogger] = None):
        """
        Initialize the dashboard.
        
        Args:
            notification_service: Optional notification service instance
            metrics_logger: Optional metrics logger instance
        """
        self.notification_service = notification_service or NotificationService()
        self.metrics_logger = metrics_logger or MetricsLogger()
        
        # Initialize state
        if 'delivery_filter_days' not in st.session_state:
            st.session_state.delivery_filter_days = 7
        if 'delivery_filter_status' not in st.session_state:
            st.session_state.delivery_filter_status = 'All'
    
    def render(self) -> None:
        """Render the delivery status dashboard."""
        st.title("Delivery Status Dashboard")
        
        # Feature flag check
        # Feature flag check
        if not get_feature_flags().get('enable_notifications', True):
            warning_text = f"{format_status_text(StatusType.WARNING)} Notifications are currently disabled for maintenance."
            st.markdown(warning_text, unsafe_allow_html=True)
            
            if st.button("Re-enable Notifications"):
                # Update feature flag
                success_text = f"{format_status_text(StatusType.SUCCESS)} Notifications re-enabled!"
                st.markdown(success_text, unsafe_allow_html=True)
            return
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["Recent Deliveries", "Performance Metrics", "System Status"])
        
        with tab1:
            self._render_recent_deliveries()
        
        with tab2:
            self._render_performance_metrics()
        
        with tab3:
            self._render_system_status()
    
    def _render_recent_deliveries(self) -> None:
        """Render the recent deliveries section."""
        st.header("Recent Deliveries")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            days = st.selectbox(
                "Time Range",
                options=[1, 7, 14, 30],
                index=1,
                key='delivery_filter_days'
            )
        
        with col2:
            status = st.selectbox(
                "Status",
                options=['All', 'Delivered', 'Failed', 'Retrying'],
                key='delivery_filter_status'
            )
        
        with col3:
            st.write("")  # Spacing
            refresh = st.button("Refresh")
        
        # Get delivery records
        records = self._get_filtered_records(days, status)
        
        if not records:
            st.info("No delivery records found for the selected filters.")
            return
        
        # Create DataFrame for display
        df = pd.DataFrame(records)
        
        # Format timestamps
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Display records
        st.dataframe(
            df,
            use_container_width=True,
            column_config={
                "timestamp": "Time",
                "recipient": "Recipient",
                "type": "Type",
                "status": "Status",
                "attempt_count": "Attempts",
                "error": "Error"
            }
        )
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total = len(records)
            st.metric("Total Deliveries", total)
        
        with col2:
            success = sum(1 for r in records if r['status'] == 'Delivered')
            success_rate = (success / total * 100) if total > 0 else 0
            st.metric("Success Rate", f"{success_rate:.1f}%")
        
        with col3:
            failures = sum(1 for r in records if r['status'] == 'Failed')
            st.metric("Failures", failures)
        
        with col4:
            retries = sum(1 for r in records if r['status'] == 'Retrying')
            st.metric("In Retry", retries)
    
    def _render_performance_metrics(self) -> None:
        """Render the performance metrics section."""
        st.header("Performance Metrics")
        
        # Time range selector
        days = st.slider("Time Range (days)", min_value=1, max_value=30, value=7)
        
        # Get metrics
        metrics = self.metrics_logger.get_delivery_metrics(days=days)
        
        if not metrics:
            st.info("No performance metrics available for the selected time range.")
            return
        
        # Create metrics visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Delivery times chart
            times_df = pd.DataFrame(metrics.get('delivery_times', []))
            if not times_df.empty:
                fig = px.box(
                    times_df,
                    y='duration',
                    title='Delivery Time Distribution (seconds)'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # PDF sizes chart
            sizes_df = pd.DataFrame(metrics.get('pdf_sizes', []))
            if not sizes_df.empty:
                fig = px.histogram(
                    sizes_df,
                    x='size_kb',
                    title='PDF Size Distribution (KB)'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Detailed metrics table
        if st.checkbox("Show Detailed Metrics"):
            st.dataframe(
                pd.DataFrame([metrics]),
                use_container_width=True
            )
    
    def _render_system_status(self) -> None:
        """Render the system status section."""
        st.header("System Status")
        
        # Service status
        status = self._get_system_status()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Service Status")
            
            st.subheader("Service Status")
            
            for service, info in status['services'].items():
                status_type = StatusType.SUCCESS if info['status'] == 'healthy' else StatusType.ERROR
                service_status = format_status_text(status_type, custom_text=f"{service}: {info['status']}")
                st.markdown(service_status, unsafe_allow_html=True)
                    st.caption(info['message'])
        
        with col2:
            st.subheader("Queue Status")
            
            st.metric("Messages in Queue", status['queue']['pending'])
            st.metric("Active Workers", status['queue']['workers'])
            
            if status['queue']['errors']:
                st.error("Recent queue errors detected")
                with st.expander("View Errors"):
                    for error in status['queue']['errors']:
                        st.code(error)
        
        # Maintenance controls
        st.subheader("Maintenance Controls")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Pause Notifications"):
                warning_text = f"{format_status_text(StatusType.WARNING)} Notifications paused"
                st.markdown(warning_text, unsafe_allow_html=True)
        
        with col2:
            if st.button("Clear Error State"):
                success_text = f"{format_status_text(StatusType.SUCCESS)} Error state cleared"
                st.markdown(success_text, unsafe_allow_html=True)
    
    def _get_filtered_records(self, days: int, status: str) -> List[Dict[str, Any]]:
        """
        Get filtered delivery records.
        
        Args:
            days: Number of days to look back
            status: Status to filter by or 'All'
            
        Returns:
            List of delivery records
        """
        # Get records from notification service
        records = []
        
        # Mock data for example
        start_date = datetime.now() - timedelta(days=days)
        
        for i in range(10):
            record = {
                "timestamp": (start_date + timedelta(hours=i*2)).isoformat(),
                "recipient": f"user{i}@example.com",
                "type": "daily_summary" if i % 2 == 0 else "alert",
                "status": "Delivered" if i % 3 == 0 else ("Failed" if i % 3 == 1 else "Retrying"),
                "attempt_count": i % 3 + 1,
                "error": "Connection timeout" if i % 3 == 1 else None
            }
            records.append(record)
        
        # Apply filters
        if status != 'All':
            records = [r for r in records if r['status'] == status]
        
        return records
    
    def _get_system_status(self) -> Dict[str, Any]:
        """
        Get current system status.
        
        Returns:
            Dictionary with status information
        """
        return {
            "services": {
                "Email Service": {
                    "status": "healthy",
                    "message": None
                },
                "PDF Generator": {
                    "status": "healthy",
                    "message": None
                },
                "Metrics Logger": {
                    "status": "healthy",
                    "message": None
                }
            },
            "queue": {
                "pending": 5,
                "workers": 2,
                "errors": []
            }
        }