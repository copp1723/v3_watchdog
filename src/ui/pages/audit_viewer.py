"""
Audit log viewer interface for V3 Watchdog AI.

This module provides a UI for viewing, filtering, and analyzing audit logs.
It retrieves audit data from Redis (recent logs) and S3 (archived logs).
"""

import streamlit as st
import pandas as pd
import json
import redis
import boto3
import os
from datetime import datetime, timedelta
from io import StringIO
import time

# Import audit log utilities
from src.utils.audit_log import redis_client, AUDIT_LOG_KEY, get_audit_log_ttl


def render_audit_viewer():
    """Render the audit log viewer interface."""
    st.title("Audit Log Viewer")
    
    # Display information about retention policy
    with st.expander("Retention Policy", expanded=False):
        st.info(f"""
        ### Audit Log Retention
        
        - **Redis Logs**: {get_audit_log_ttl() // 86400} days ({get_audit_log_ttl()} seconds)
        - **S3 Archive**: 365 days total (in Glacier after 30 days)
        
        This retention configuration complies with industry standards and regulatory requirements.
        """)
    
    # Time range filter
    col1, col2 = st.columns(2)
    with col1:
        days = st.number_input("Days of History", min_value=1, max_value=365, value=7)
    with col2:
        end_date = st.date_input("End Date", value=datetime.now())
    
    start_date = end_date - timedelta(days=days)
    
    # Create filter options
    st.subheader("Filters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        filter_event = st.selectbox(
            "Event Type", 
            options=["All", "file_upload", "report_generation", "user_login", "data_export", "insight_generation"]
        )
    
    with col2:
        filter_status = st.selectbox(
            "Status", 
            options=["All", "success", "error", "warning"]
        )
    
    with col3:
        filter_user = st.text_input("User ID", placeholder="Filter by user ID")
    
    # Get audit logs
    if st.button("Load Audit Logs"):
        with st.spinner("Loading audit logs..."):
            df = load_audit_logs(start_date, end_date)
            
            if df is not None and not df.empty:
                # Apply filters
                if filter_event != "All":
                    df = df[df["event"] == filter_event]
                
                if filter_status != "All":
                    df = df[df["status"] == filter_status]
                
                if filter_user:
                    df = df[df["user_id"].str.contains(filter_user, case=False, na=False)]
                
                # Display results
                st.subheader(f"Audit Logs ({len(df)} records)")
                st.dataframe(df, use_container_width=True)
                
                # Allow export
                if not df.empty:
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Export as CSV",
                        data=csv,
                        file_name=f"audit_logs_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv",
                        mime='text/csv',
                    )
            else:
                st.warning("No audit logs found for the selected time period and filters.")


def load_audit_logs(start_date, end_date):
    """
    Load audit logs from Redis and S3 based on date range.
    
    Args:
        start_date: Start date for log retrieval
        end_date: End date for log retrieval
        
    Returns:
        pandas.DataFrame containing the audit logs
    """
    # Convert dates to datetime with time
    start_datetime = datetime.combine(start_date, datetime.min.time())
    end_datetime = datetime.combine(end_date, datetime.max.time())
    
    # Placeholder for logs
    logs = []
    
    # Try to get logs from Redis first (recent logs)
    if redis_client:
        try:
            # Get all logs from Redis
            redis_logs = redis_client.lrange(AUDIT_LOG_KEY, 0, -1)
            
            for log_json in redis_logs:
                try:
                    log = json.loads(log_json)
                    log_time = datetime.fromisoformat(log.get("timestamp", ""))
                    
                    # Filter by date range
                    if start_datetime <= log_time <= end_datetime:
                        logs.append(log)
                except Exception as e:
                    st.error(f"Error parsing log entry: {e}")
        except Exception as e:
            st.error(f"Error retrieving logs from Redis: {e}")
    
    # Get logs from S3 for older data (if configured)
    s3_bucket = os.environ.get("WATCHDOG_S3_AUDIT_BUCKET")
    s3_prefix = os.environ.get("WATCHDOG_S3_AUDIT_PREFIX", "audit_logs/")
    
    if s3_bucket:
        try:
            s3_client = boto3.client('s3')
            
            # List objects in the audit logs directory
            # This is simplified - in a real implementation, you would:
            # 1. Optimize this to only request objects in the relevant date range
            # 2. Handle pagination for large result sets
            # 3. Possibly use S3 Select for filtering server-side
            
            # Placeholder for S3 logs
            # In a real implementation, this would fetch from S3
            # For this example, we'll simulate it
            simulate_s3_logs = False
            
            if simulate_s3_logs:
                # Simulate loading logs from S3
                # This is just for demonstration - real implementation would use boto3
                for i in range(5):
                    log_time = start_datetime + timedelta(days=i, hours=i)
                    if start_datetime <= log_time <= end_datetime:
                        logs.append({
                            "event": "file_upload",
                            "user_id": f"user{i}",
                            "session_id": f"session{i}",
                            "timestamp": log_time.isoformat(),
                            "ip_address": "192.168.1.1",
                            "resource_type": "file",
                            "resource_id": f"file{i}",
                            "status": "success",
                            "details": {"source": "s3_archive"}
                        })
        except Exception as e:
            st.error(f"Error retrieving logs from S3: {e}")
    
    # Convert to DataFrame
    if logs:
        # Convert logs to proper structure
        structured_logs = []
        for log in logs:
            # Extract details from nested structure
            details = log.pop("details", {}) if isinstance(log.get("details"), dict) else {}
            
            # Merge log and details
            log_entry = {**log, **details}
            structured_logs.append(log_entry)
        
        df = pd.DataFrame(structured_logs)
        
        # Sort by timestamp (descending)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp", ascending=False)
        
        return df
    else:
        return pd.DataFrame()


if __name__ == "__main__":
    st.set_page_config(
        page_title="Audit Log Viewer | V3 Watchdog AI",
        page_icon="🔍",
        layout="wide"
    )
    render_audit_viewer()