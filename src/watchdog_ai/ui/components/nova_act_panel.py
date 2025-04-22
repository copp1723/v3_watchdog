"""
Nova Act automation panel component for the Watchdog AI UI.

This component provides a user interface for managing Nova Act data sync automation,
including viewing sync status, configuring credentials, and scheduling report collection.
"""

import os
import sys
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

# Add parent directories to path to import Nova Act modules
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import Nova Act modules
from src.nova_act.core import NovaActConnector
from src.nova_act.scheduler_bridge import NovaActSchedulerBridge, get_scheduler_bridge
from src.nova_act.enhanced_credentials import EnhancedCredentialManager
from src.scheduler.report_scheduler import ReportFrequency

def format_datetime(dt_str: str) -> str:
    """Format datetime string to readable format."""
    try:
        dt = datetime.fromisoformat(dt_str)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return dt_str

def nova_act_panel():
    """Render the Nova Act automation panel."""
    st.header("Nova Act Data Automation")
    
    # Initialize session state
    if "nova_act_tab" not in st.session_state:
        st.session_state.nova_act_tab = "status"
    
    # Create tabs
    tabs = st.tabs(["Status", "Schedule", "Credentials", "Settings"])
    
    # Status tab
    with tabs[0]:
        render_status_tab()
    
    # Schedule tab
    with tabs[1]:
        render_schedule_tab()
    
    # Credentials tab
    with tabs[2]:
        render_credentials_tab()
    
    # Settings tab
    with tabs[3]:
        render_settings_tab()

def render_status_tab():
    """Render the status tab showing the status of synced dealers."""
    st.subheader("Data Sync Status")
    
    # Get sync status
    if "nova_act_sync_status" in st.session_state:
        sync_status = st.session_state.nova_act_sync_status
    else:
        # Get from bridge if available
        try:
            bridge = get_scheduler_bridge()
            sync_status = bridge.get_sync_status()
        except:
            sync_status = {}
    
    if not sync_status:
        st.info("No sync data available yet. Schedule a data collection first.")
        return
    
    # Convert to list for display
    status_list = []
    for key, status in sync_status.items():
        # Format date
        last_sync = format_datetime(status.get("last_sync", ""))
        
        # Create status entry
        status_list.append({
            "Vendor": status.get("vendor_id", ""),
            "Dealer": status.get("dealer_id", ""),
            "Report Type": status.get("report_type", ""),
            "Last Sync": last_sync,
            "Status": "✅ Success" if status.get("success", False) else "❌ Failed",
            "Error": status.get("error", "")
        })
    
    # Convert to DataFrame for display
    df = pd.DataFrame(status_list)
    st.dataframe(df, use_container_width=True)
    
    # Add refresh button
    if st.button("Refresh", key="refresh_status"):
        st.rerun()

def render_schedule_tab():
    """Render the schedule tab for configuring sync schedules."""
    st.subheader("Schedule Data Collection")
    
    # Add forms for scheduling
    with st.form("schedule_form"):
        # Vendor selector
        vendor_options = ["dealersocket", "vinsolutions", "eleads"]
        vendor = st.selectbox("Vendor", vendor_options, key="schedule_vendor")
        
        # Dealer ID input
        dealer_id = st.text_input("Dealer ID", key="schedule_dealer_id")
        
        # Report type selector
        report_options = ["sales", "inventory", "leads"]
        report_type = st.selectbox("Report Type", report_options, key="schedule_report_type")
        
        # Frequency selector
        frequency_options = ["daily", "weekly", "monthly"]
        frequency = st.selectbox("Frequency", frequency_options, key="schedule_frequency")
        
        # Daily schedule options
        if frequency == "daily":
            hour = st.number_input("Hour (24-hour format)", min_value=0, max_value=23, value=2)
            minute = st.number_input("Minute", min_value=0, max_value=59, value=0)
        
        # Weekly schedule options
        elif frequency == "weekly":
            day_options = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            day_of_week = st.selectbox("Day of Week", day_options, index=0)
            hour = st.number_input("Hour (24-hour format)", min_value=0, max_value=23, value=2)
            minute = st.number_input("Minute", min_value=0, max_value=59, value=0)
        
        # Monthly schedule options
        elif frequency == "monthly":
            day_of_month = st.number_input("Day of Month", min_value=1, max_value=31, value=1)
            hour = st.number_input("Hour (24-hour format)", min_value=0, max_value=23, value=2)
            minute = st.number_input("Minute", min_value=0, max_value=59, value=0)
        
        # Submit button
        submitted = st.form_submit_button("Schedule")
        
        if submitted:
            if not dealer_id:
                st.error("Dealer ID is required")
            else:
                try:
                    # Get scheduler bridge
                    bridge = get_scheduler_bridge()
                    
                    # Convert frequency
                    freq_map = {
                        "daily": ReportFrequency.DAILY,
                        "weekly": ReportFrequency.WEEKLY,
                        "monthly": ReportFrequency.MONTHLY
                    }
                    report_freq = freq_map.get(frequency, ReportFrequency.DAILY)
                    
                    # Prepare schedule params
                    schedule_params = {}
                    if frequency == "daily":
                        schedule_params = {"hour": hour, "minute": minute}
                    elif frequency == "weekly":
                        day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, 
                                  "Friday": 4, "Saturday": 5, "Sunday": 6}
                        schedule_params = {"day_of_week": day_map[day_of_week], "hour": hour, "minute": minute}
                    elif frequency == "monthly":
                        schedule_params = {"day_of_month": day_of_month, "hour": hour, "minute": minute}
                    
                    # Schedule the report
                    report_id = bridge.schedule_report_collection(
                        vendor_id=vendor,
                        dealer_id=dealer_id,
                        report_type=report_type,
                        frequency=report_freq,
                        **schedule_params
                    )
                    
                    st.success(f"Successfully scheduled {report_type} report collection for {dealer_id}")
                    
                except Exception as e:
                    st.error(f"Error scheduling report: {str(e)}")
    
    # Add section for triggering immediate sync
    st.subheader("Trigger Immediate Sync")
    with st.form("trigger_form"):
        # Vendor selector
        vendor = st.selectbox("Vendor", vendor_options, key="trigger_vendor")
        
        # Dealer ID input
        dealer_id = st.text_input("Dealer ID", key="trigger_dealer_id")
        
        # Report type selector
        report_type = st.selectbox("Report Type", report_options, key="trigger_report_type")
        
        # Submit button
        trigger_submitted = st.form_submit_button("Sync Now")
        
        if trigger_submitted:
            if not dealer_id:
                st.error("Dealer ID is required")
            else:
                try:
                    # Get scheduler bridge
                    bridge = get_scheduler_bridge()
                    
                    # Trigger immediate sync
                    result = bridge.trigger_sync_now(
                        vendor_id=vendor,
                        dealer_id=dealer_id,
                        report_type=report_type
                    )
                    
                    st.success(f"Triggered immediate sync for {dealer_id}. Check the Status tab for updates.")
                    
                except Exception as e:
                    st.error(f"Error triggering sync: {str(e)}")

def render_credentials_tab():
    """Render the credentials tab for managing dealer credentials."""
    st.subheader("Manage Credentials")
    
    # Add form for adding credentials
    with st.form("credentials_form"):
        # Vendor selector
        vendor_options = ["dealersocket", "vinsolutions", "eleads"]
        vendor = st.selectbox("Vendor", vendor_options, key="cred_vendor")
        
        # Dealer ID input
        dealer_id = st.text_input("Dealer ID", key="cred_dealer_id")
        
        # Username and password inputs
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        # URL input
        url = st.text_input("Login URL", value=f"https://login.{vendor}.com")
        
        # 2FA method selector
        tfa_options = ["None", "SMS", "Email", "Authenticator"]
        tfa_method = st.selectbox("2FA Method", tfa_options)
        
        # 2FA configuration
        if tfa_method == "SMS":
            phone = st.text_input("Phone Number")
        elif tfa_method == "Email":
            email = st.text_input("Email Address")
        elif tfa_method == "Authenticator":
            totp_secret = st.text_input("TOTP Secret")
        
        # Submit button
        submitted = st.form_submit_button("Save Credentials")
        
        if submitted:
            if not all([dealer_id, username, password]):
                st.error("Dealer ID, username, and password are required")
            else:
                try:
                    # Create credential manager
                    cred_manager = EnhancedCredentialManager()
                    
                    # Prepare 2FA config
                    tfa_config = {}
                    tfa_code = None
                    if tfa_method == "SMS":
                        tfa_code = "sms"
                        tfa_config = {"phone_number": phone}
                    elif tfa_method == "Email":
                        tfa_code = "email"
                        tfa_config = {"email": email}
                    elif tfa_method == "Authenticator":
                        tfa_code = "authenticator"
                        tfa_config = {"totp_secret": totp_secret}
                    
                    # Create cred_id
                    cred_id = f"{vendor}:{dealer_id}"
                    
                    # Add credentials
                    cred_manager.add_credential(
                        system_id=cred_id,
                        username=username,
                        password=password,
                        metadata={
                            "url": url,
                            "2fa_method": tfa_code,
                            "2fa_config": tfa_config
                        }
                    )
                    
                    st.success(f"Successfully saved credentials for {dealer_id}")
                    
                except Exception as e:
                    st.error(f"Error saving credentials: {str(e)}")

def render_settings_tab():
    """Render the settings tab for configuring Nova Act."""
    st.subheader("Nova Act Settings")
    
    # Headless mode setting
    headless = st.checkbox("Run in headless mode", value=True, 
                          help="When enabled, browsers run without visible windows")
    
    # Max concurrent setting
    max_concurrent = st.slider("Max concurrent sessions", min_value=1, max_value=10, value=3,
                             help="Maximum number of concurrent browser sessions")
    
    # Download directory setting
    download_dir = st.text_input("Download directory", value="/tmp/nova_act_downloads",
                               help="Directory where downloaded files are stored")
    
    # Save button
    if st.button("Save Settings"):
        try:
            # Store settings in session state
            if "nova_act_settings" not in st.session_state:
                st.session_state.nova_act_settings = {}
            
            st.session_state.nova_act_settings.update({
                "headless": headless,
                "max_concurrent": max_concurrent,
                "download_dir": download_dir
            })
            
            st.success("Settings saved successfully")
            
        except Exception as e:
            st.error(f"Error saving settings: {str(e)}")
    
    # System status section
    st.subheader("System Status")
    
    # Check if scheduler bridge is running
    try:
        bridge = get_scheduler_bridge()
        bridge_running = bridge.running
    except:
        bridge_running = False
    
    if bridge_running:
        st.success("Nova Act system is running")
    else:
        st.warning("Nova Act system is not running")
        
        # Add start button
        if st.button("Start Nova Act System"):
            try:
                # Get settings
                settings = st.session_state.get("nova_act_settings", {})
                
                # Create connector with settings
                connector = NovaActConnector(
                    headless=settings.get("headless", True),
                    max_concurrent=settings.get("max_concurrent", 3),
                    download_dir=settings.get("download_dir", "/tmp/nova_act_downloads")
                )
                
                # Create bridge with connector
                bridge = NovaActSchedulerBridge(connector=connector)
                
                # Start bridge
                import asyncio
                asyncio.run(bridge.start())
                
                # Update bridge singleton
                global _bridge
                _bridge = bridge
                
                st.success("Nova Act system started successfully")
                st.rerun()
                
            except Exception as e:
                st.error(f"Error starting Nova Act system: {str(e)}")
    
    # Add stop button if running
    if bridge_running:
        if st.button("Stop Nova Act System"):
            try:
                import asyncio
                asyncio.run(bridge.stop())
                st.success("Nova Act system stopped successfully")
                st.rerun()
            except Exception as e:
                st.error(f"Error stopping Nova Act system: {str(e)}")
    
    # Add information about available resources
    st.subheader("System Resources")
    import psutil
    
    # Get CPU and memory info
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    memory_percent = memory.percent
    
    # Display in columns
    col1, col2 = st.columns(2)
    with col1:
        st.metric("CPU Usage", f"{cpu_percent}%")
    with col2:
        st.metric("Memory Usage", f"{memory_percent}%")