"""
Ingestion Pipeline Health Dashboard

This script provides a Streamlit UI for monitoring the health 
and status of the Nova Act data ingestion pipeline.
"""

import os
import sys
import json
import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime, timedelta, timezone
import time
import asyncio
import nest_asyncio

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.nova_act.monitoring import get_monitor
from src.nova_act.enhanced_scheduler import get_scheduler
from src.nova_act.enhanced_credentials import get_credential_manager

# Set page configuration
st.set_page_config(
    page_title="Watchdog AI - Ingestion Pipeline Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS for styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f5f5f5;
        border-radius: 5px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .metric-label {
        font-size: 14px;
        color: #666;
    }
    .success {
        color: #28a745;
    }
    .warning {
        color: #ffc107;
    }
    .error {
        color: #dc3545;
    }
    .status-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: bold;
    }
    .badge-success {
        background-color: #d4edda;
        color: #155724;
    }
    .badge-warning {
        background-color: #fff3cd;
        color: #856404;
    }
    .badge-error {
        background-color: #f8d7da;
        color: #721c24;
    }
    .header-text {
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .card {
        background-color: white;
        border-radius: 5px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'refresh_interval' not in st.session_state:
    st.session_state.refresh_interval = 5  # minutes
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now()
if 'selected_dealer' not in st.session_state:
    st.session_state.selected_dealer = None
if 'selected_vendor' not in st.session_state:
    st.session_state.selected_vendor = None
if 'selected_report' not in st.session_state:
    st.session_state.selected_report = None

# Sidebar
st.sidebar.title("Control Panel")

# Refresh settings
st.sidebar.header("Refresh Settings")
st.session_state.refresh_interval = st.sidebar.slider(
    "Auto-refresh interval (minutes)",
    min_value=1,
    max_value=60,
    value=st.session_state.refresh_interval
)

if st.sidebar.button("Refresh Now"):
    st.session_state.last_refresh = datetime.now()
    st.experimental_rerun()

# Filter settings
st.sidebar.header("Filter Settings")

# Get monitor instance
monitor = get_monitor()
scheduler = get_scheduler()
cred_manager = get_credential_manager()

# Auto-refresh logic
time_since_refresh = (datetime.now() - st.session_state.last_refresh).total_seconds() / 60
if time_since_refresh >= st.session_state.refresh_interval:
    st.session_state.last_refresh = datetime.now()
    st.experimental_rerun()

# Get available dealers and vendors
try:
    dealers = cred_manager.list_dealers()
    vendors = cred_manager.list_vendors()
except:
    dealers = []
    vendors = []

# Dealer selector
dealer_options = ["All Dealers"] + dealers
selected_dealer_index = 0 if st.session_state.selected_dealer is None else dealer_options.index(st.session_state.selected_dealer) if st.session_state.selected_dealer in dealer_options else 0
st.session_state.selected_dealer = st.sidebar.selectbox(
    "Dealer",
    dealer_options,
    index=selected_dealer_index
)

# Vendor selector
vendor_options = ["All Vendors"] + vendors
selected_vendor_index = 0 if st.session_state.selected_vendor is None else vendor_options.index(st.session_state.selected_vendor) if st.session_state.selected_vendor in vendor_options else 0
st.session_state.selected_vendor = st.sidebar.selectbox(
    "Vendor",
    vendor_options,
    index=selected_vendor_index
)

# Report type selector
report_options = ["All Reports", "sales", "inventory", "leads"]
selected_report_index = 0 if st.session_state.selected_report is None else report_options.index(st.session_state.selected_report) if st.session_state.selected_report in report_options else 0
st.session_state.selected_report = st.sidebar.selectbox(
    "Report Type",
    report_options,
    index=selected_report_index
)

# Time range selector
time_range = st.sidebar.radio(
    "Time Range",
    ["Last 24 Hours", "Last 7 Days", "Last 30 Days"]
)

# Convert time range to days
if time_range == "Last 24 Hours":
    days = 1
elif time_range == "Last 7 Days":
    days = 7
else:
    days = 30

# Get filtered data
dealer_id = None if st.session_state.selected_dealer == "All Dealers" else st.session_state.selected_dealer
vendor_id = None if st.session_state.selected_vendor == "All Vendors" else st.session_state.selected_vendor
report_type = None if st.session_state.selected_report == "All Reports" else st.session_state.selected_report

# Main dashboard
st.title("Ingestion Pipeline Health Dashboard")
st.write(f"Last refreshed: {st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M:%S')}")

# Get status data
status_data = monitor.get_ingestion_status(
    dealer_id=dealer_id,
    vendor_id=vendor_id,
    report_type=report_type,
    days=days
)

# Top metrics row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">{status_data['counts']['total']}</div>
            <div class="metric-label">Total Ingestion Runs</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value success">{status_data['counts']['success']}</div>
            <div class="metric-label">Successful Runs</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value error">{status_data['counts']['failure']}</div>
            <div class="metric-label">Failed Runs</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col4:
    success_rate = 0 if status_data['counts']['total'] == 0 else (status_data['counts']['success'] / status_data['counts']['total'] * 100)
    
    color_class = "success"
    if success_rate < 70:
        color_class = "error"
    elif success_rate < 90:
        color_class = "warning"
    
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value {color_class}">{success_rate:.1f}%</div>
            <div class="metric-label">Success Rate</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Group statistics
if 'group_stats' in status_data and status_data['group_stats']:
    st.header("Success Rates by Group")
    
    # Convert to dataframe
    group_df = pd.DataFrame(status_data['group_stats'])
    
    # Create a bar chart
    chart = alt.Chart(group_df).mark_bar().encode(
        x=alt.X('success_rate:Q', title='Success Rate (%)'),
        y=alt.Y('dealer_id:N', title='Dealer ID', sort='-x'),
        color=alt.Color('success_rate:Q', scale=alt.Scale(scheme='greenblue'), legend=None),
        tooltip=['dealer_id', 'vendor_id', 'report_type', 'count', 'success_count', 'success_rate']
    ).properties(
        width=800,
        height=400,
        title='Success Rate by Group'
    )
    
    st.altair_chart(chart, use_container_width=True)

# Recent failures
if 'recent_failures' in status_data and status_data['recent_failures']:
    st.header("Recent Failures")
    
    # Convert to dataframe
    failures_df = pd.DataFrame(status_data['recent_failures'])
    
    # Add a time column for better display
    if 'processing_time' in failures_df.columns:
        failures_df['time'] = pd.to_datetime(failures_df['processing_time']).dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Show as a table
    st.dataframe(
        failures_df[['dealer_id', 'vendor_id', 'report_type', 'time', 'error']],
        use_container_width=True
    )

# Scheduled tasks
st.header("Scheduled Tasks")

# Get tasks
tasks = scheduler.get_all_tasks()

if tasks:
    # Convert to dataframe
    tasks_df = pd.DataFrame(tasks)
    
    # Format timestamps
    tasks_df['next_run'] = pd.to_datetime(tasks_df['next_run']).dt.strftime('%Y-%m-%d %H:%M:%S')
    tasks_df['last_run'] = pd.to_datetime(tasks_df['last_run']).dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Add status badge
    def status_badge(status):
        if status == "success":
            return '<span class="status-badge badge-success">Success</span>'
        elif status == "error":
            return '<span class="status-badge badge-error">Error</span>'
        elif status == "queued":
            return '<span class="status-badge badge-warning">Queued</span>'
        elif status == "running":
            return '<span class="status-badge badge-warning">Running</span>'
        else:
            return f'<span class="status-badge">{status}</span>'
    
    tasks_df['status_badge'] = tasks_df['last_status'].apply(status_badge)
    
    # Show as a table
    st.write(
        tasks_df[['id', 'dealer_id', 'vendor_id', 'report_type', 'schedule', 'next_run', 'last_run', 'status_badge']].to_html(escape=False),
        unsafe_allow_html=True
    )
else:
    st.info("No scheduled tasks found.")

# Action buttons
if st.session_state.selected_dealer != "All Dealers" and st.session_state.selected_vendor != "All Vendors":
    st.header("Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Run Collection Now"):
            try:
                # Get report types
                if st.session_state.selected_report == "All Reports":
                    report_types = ["sales", "inventory", "leads"]
                else:
                    report_types = [st.session_state.selected_report]
                
                # Queue a task for each report type
                for report_type in report_types:
                    # Create a one-time task
                    scheduler.schedule_task(
                        dealer_id=st.session_state.selected_dealer,
                        vendor_id=st.session_state.selected_vendor,
                        report_type=report_type,
                        schedule="once",
                        schedule_config={"time": datetime.now(timezone.utc).isoformat()}
                    )
                
                st.success(f"Scheduled immediate collection for {st.session_state.selected_dealer}/{st.session_state.selected_vendor}")
            except Exception as e:
                st.error(f"Error scheduling collection: {str(e)}")
    
    with col2:
        if st.button("Schedule Daily Collection"):
            try:
                # Get report types
                if st.session_state.selected_report == "All Reports":
                    report_types = ["sales", "inventory", "leads"]
                else:
                    report_types = [st.session_state.selected_report]
                
                # Queue a task for each report type
                for report_type in report_types:
                    # Create a daily task at 1:00 AM
                    scheduler.schedule_task(
                        dealer_id=st.session_state.selected_dealer,
                        vendor_id=st.session_state.selected_vendor,
                        report_type=report_type,
                        schedule="daily",
                        schedule_config={"hour": 1, "minute": 0}
                    )
                
                st.success(f"Scheduled daily collection for {st.session_state.selected_dealer}/{st.session_state.selected_vendor}")
            except Exception as e:
                st.error(f"Error scheduling collection: {str(e)}")
    
    with col3:
        if st.button("View Dealer Details"):
            # Store selection and navigate to detail view
            st.session_state.detail_dealer = st.session_state.selected_dealer
            st.session_state.detail_vendor = st.session_state.selected_vendor
            st.experimental_rerun()

# Run the scheduler if not already running
if 'scheduler_started' not in st.session_state:
    try:
        # Start scheduler in background
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def start_scheduler_async():
            await scheduler.start()
        
        loop.run_until_complete(start_scheduler_async())
        st.session_state.scheduler_started = True
    except Exception as e:
        st.error(f"Error starting scheduler: {str(e)}")

st.write("---")
st.caption("Watchdog AI Ingestion Pipeline Dashboard")
last_refresh_seconds = (datetime.now() - st.session_state.last_refresh).total_seconds()
st.caption(f"Next refresh in {int(st.session_state.refresh_interval * 60 - last_refresh_seconds)} seconds")