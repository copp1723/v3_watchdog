"""
Demo script for Nova Act integration.
"""

import streamlit as st

# Set page config first, before any other Streamlit commands
st.set_page_config(
    page_title="Nova Act Demo",
    page_icon="🤖",
    layout="wide"
)

import os
import sys
from datetime import datetime, timedelta
import asyncio
import pandas as pd
import nest_asyncio

# Enable nested event loops
nest_asyncio.apply()

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from src.nova_act import NovaAct, TaskPriority
    from src.nova_act.core import NovaActClient
    from src.nova_act.scheduler import NovaScheduler
    from src.nova_act.health_check import health_checker
except ImportError as e:
    st.error(f"Failed to import Nova Act modules: {e}")
    st.stop()

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.client = None
    st.session_state.scheduler = None
    st.session_state.health_status = None
    st.session_state.initialization_error = None

def init_async():
    """Initialize async components with proper error handling."""
    try:
        # Create event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Initialize client
        st.session_state.client = NovaActClient(headless=True)
        loop.run_until_complete(st.session_state.client.start())
        
        # Initialize scheduler
        st.session_state.scheduler = NovaScheduler(
            client=st.session_state.client,
            config={
                "dealersocket": {
                    "base_url": "https://api.dealersocket.com/v1",
                    "report_path": "/reports/daily"
                },
                "vinsolutions": {
                    "base_url": "https://api.vinsolutions.com/v2",
                    "report_path": "/exports/daily"
                }
            }
        )
        
        # Start health monitoring
        loop.run_until_complete(health_checker.start_monitoring())
        
        # Get initial health status
        st.session_state.health_status = loop.run_until_complete(
            health_checker.get_all_health_status()
        )
        
        st.session_state.initialized = True
        st.session_state.initialization_error = None
        
    except Exception as e:
        st.session_state.initialization_error = str(e)
        st.session_state.initialized = False
    finally:
        loop.close()

def render_system_status():
    """Render the system status section."""
    st.header("System Status")
    
    if st.session_state.health_status:
        for vendor, info in st.session_state.health_status["vendors"].items():
            with st.expander(f"{vendor} Status"):
                st.write(f"Status: {info['status']}")
                st.write(f"Last Check: {info['timestamp']}")
                if "metrics" in info:
                    st.write("Performance Metrics:")
                    for metric, value in info["metrics"].items():
                        st.metric(metric, value)
    else:
        st.info("System status not yet available...")

def render_scheduled_tasks():
    """Render the scheduled tasks section."""
    st.header("Scheduled Tasks")
    
    if st.session_state.scheduler:
        tasks = st.session_state.scheduler.get_all_tasks()
        if tasks:
            for task in tasks:
                with st.expander(f"Task {task['id']}"):
                    st.write(f"Status: {task['status']}")
                    st.write(f"Schedule: {task['schedule']}")
                    st.write(f"Next Run: {task['schedule_time']}")
                    if task.get('last_run'):
                        st.write(f"Last Run: {task['last_run']}")
        else:
            st.info("No tasks scheduled")
    else:
        st.info("Task scheduler not yet initialized...")

def render_task_scheduling():
    """Render the task scheduling section."""
    st.header("Schedule New Task")
    
    if not st.session_state.scheduler:
        st.warning("Scheduler not yet initialized...")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        vendor = st.selectbox(
            "Select Vendor",
            ["dealersocket", "vinsolutions"]
        )
        
        schedule_type = st.selectbox(
            "Schedule Type",
            ["once", "daily", "interval"]
        )
    
    with col2:
        if schedule_type == "once":
            schedule_time = st.date_input(
                "Select Date",
                min_value=datetime.now()
            )
        elif schedule_type == "daily":
            schedule_time = st.time_input("Select Time")
        else:
            interval = st.number_input(
                "Interval (minutes)",
                min_value=5,
                value=60
            )
    
    if st.button("Schedule Task"):
        try:
            if schedule_type == "once":
                st.session_state.scheduler.schedule_task(
                    st.session_state.scheduler.run_collection,
                    schedule="once",
                    schedule_time=schedule_time,
                    task_kwargs={"vendor": vendor}
                )
            elif schedule_type == "daily":
                st.session_state.scheduler.schedule_task(
                    st.session_state.scheduler.run_collection,
                    schedule="daily",
                    schedule_time=datetime.combine(datetime.now().date(), schedule_time),
                    task_kwargs={"vendor": vendor}
                )
            else:
                st.session_state.scheduler.schedule_task(
                    st.session_state.scheduler.run_collection,
                    schedule="interval",
                    interval=interval * 60,
                    task_kwargs={"vendor": vendor}
                )
            
            st.success("Task scheduled successfully!")
            st.experimental_rerun()
            
        except Exception as e:
            st.error(f"Failed to schedule task: {e}")

def render_manual_collection():
    """Render the manual collection section."""
    st.header("Manual Collection")
    
    if not st.session_state.client:
        st.warning("Client not yet initialized...")
        return
    
    vendor = st.selectbox(
        "Select Vendor for Manual Collection",
        ["dealersocket", "vinsolutions"],
        key="manual_collection_vendor"
    )
    
    if st.button("Run Collection Now"):
        try:
            with st.spinner("Collecting data..."):
                # Create a placeholder for the result
                result_placeholder = st.empty()
                
                # Run the collection
                if st.session_state.client and st.session_state.scheduler:
                    result = asyncio.run(st.session_state.client.collect_report(
                        vendor,
                        st.session_state.scheduler.config[vendor]
                    ))
                    
                    if result:
                        result_placeholder.success(f"Collection completed: {result}")
                    else:
                        result_placeholder.warning("No data collected")
                else:
                    result_placeholder.error("Client or scheduler not initialized")
                    
        except Exception as e:
            st.error(f"Collection failed: {e}")

def main():
    """Main demo function."""
    st.title("Nova Act Integration Demo")
    
    # Initialize components if needed
    if not st.session_state.initialized:
        with st.spinner("Initializing components..."):
            init_async()
            
        # Check initialization status
        if st.session_state.initialization_error:
            st.error(f"Initialization failed: {st.session_state.initialization_error}")
            if st.button("Retry Initialization"):
                st.experimental_rerun()
            st.stop()
        elif not st.session_state.initialized:
            st.warning("Components not properly initialized. Please refresh the page.")
            if st.button("Retry Initialization"):
                st.experimental_rerun()
            st.stop()
    
    # Render UI sections
    render_system_status()
    render_scheduled_tasks()
    render_task_scheduling()
    render_manual_collection()
    
    # Add refresh button
    if st.button("Refresh Status"):
        if st.session_state.initialized:
            with st.spinner("Refreshing status..."):
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    st.session_state.health_status = loop.run_until_complete(
                        health_checker.get_all_health_status()
                    )
                    loop.close()
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Failed to refresh status: {e}")

if __name__ == "__main__":
    main()