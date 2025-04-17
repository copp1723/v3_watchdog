"""
System Connect Component for Watchdog AI.
Provides UI components for connecting to external systems.
"""

import streamlit as st
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

def render_vendor_selection() -> None:
    """Render the vendor selection interface."""
    st.markdown("### Connect to Your DMS")
    
    # Vendor selection
    vendor = st.selectbox(
        "Select your DMS vendor",
        ["DealerSocket", "VinSolutions", "CDK", "Reynolds & Reynolds", "Other"],
        help="Choose your Dealer Management System vendor"
    )
    
    if vendor == "Other":
        st.text_input("Enter your DMS vendor name")

def render_credential_form() -> None:
    """Render the credential input form."""
    st.markdown("### Enter Your Credentials")
    
    # Credential inputs
    username = st.text_input("Username", help="Your DMS username")
    password = st.text_input("Password", type="password", help="Your DMS password")
    
    # Optional fields
    with st.expander("Additional Settings"):
        st.text_input("API Key", help="Optional: Your DMS API key if required")
        st.text_input("Instance URL", help="Optional: Your DMS instance URL")

def render_report_selection() -> None:
    """Render the report selection interface."""
    st.markdown("### Select Reports to Sync")
    
    # Report selection
    st.multiselect(
        "Choose reports to sync",
        [
            "Sales Reports",
            "Inventory Reports",
            "Lead Activity Reports",
            "Deal Summary Reports",
            "F&I Reports"
        ],
        default=["Sales Reports", "Inventory Reports"],
        help="Select which reports to sync from your DMS"
    )

def render_schedule_selection() -> None:
    """Render the sync schedule selection interface."""
    st.markdown("### Set Sync Schedule")
    
    # Schedule selection
    frequency = st.selectbox(
        "Sync frequency",
        ["Daily", "Weekly", "Monthly", "Real-time"],
        help="How often to sync data from your DMS"
    )
    
    if frequency != "Real-time":
        st.time_input("Sync time", help="When to run the sync")
        if frequency == "Weekly":
            st.multiselect("Sync days", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
        elif frequency == "Monthly":
            st.number_input("Day of month", min_value=1, max_value=31, value=1)

def render_system_connect() -> None:
    """Render the complete system connect interface."""
    st.markdown("## System Integration")
    
    # Progress tracking
    if 'connect_step' not in st.session_state:
        st.session_state.connect_step = 1
    
    # Progress bar
    progress = st.progress(st.session_state.connect_step / 4)
    
    # Step navigation
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(f"**Step {st.session_state.connect_step} of 4**")
    with col2:
        if st.session_state.connect_step > 1:
            if st.button("← Back", key="system_connect_back_button"):
                st.session_state.connect_step -= 1
                st.rerun()
    
    # Render current step
    if st.session_state.connect_step == 1:
        render_vendor_selection()
    elif st.session_state.connect_step == 2:
        render_credential_form()
    elif st.session_state.connect_step == 3:
        render_report_selection()
    else:
        render_schedule_selection()
    
    # Next/Submit button
    if st.session_state.connect_step < 4:
        if st.button("Next →", type="primary", key="system_connect_next_button"):
            st.session_state.connect_step += 1
            st.rerun()
    else:
        if st.button("Submit", type="primary", key="system_connect_submit_button"):
            with st.spinner("Setting up connection..."):
                # Here we would actually set up the connection
                st.success("Connection established successfully!")
                st.balloons()

if __name__ == "__main__":
    st.set_page_config(
        page_title="Connect Systems - Watchdog AI",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    render_system_connect()