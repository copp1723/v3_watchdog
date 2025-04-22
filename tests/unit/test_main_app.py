"""
Test file for the main app, with simplified imports to avoid dependency issues.
"""

import streamlit as st
import pandas as pd
from datetime import datetime

# Set page config - must be first Streamlit command
st.set_page_config(
    page_title="Watchdog AI - System Connect",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define CSS styles (Tailwind-inspired)
STYLES = {
    "card": """
        background-color: white;
        border-radius: 0.5rem;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    """,
    "badge_success": """
        background-color: #D1FAE5;
        color: #065F46;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.75rem;
        font-weight: 500;
    """,
    "badge_error": """
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.75rem;
        font-weight: 500;
    """
}

# Initialize session state
if 'connected' not in st.session_state:
    st.session_state.connected = False

if 'last_sync' not in st.session_state:
    st.session_state.last_sync = None

if 'dealer_name' not in st.session_state:
    st.session_state.dealer_name = "Your Dealership"

if 'credentials' not in st.session_state:
    st.session_state.credentials = {
        "vendor": "",
        "email": "",
        "password": "",
        "dealership_id": "",
        "2fa_method": "",
        "reports": []
    }

if 'sync_result' not in st.session_state:
    st.session_state.sync_result = None

# Render header with dealer name
st.markdown(
    f"<div style='display: flex; align-items: center; margin-bottom: 1rem;'>"
    f"<div>"
    f"<h1 style='margin: 0; font-size: 1.8rem;'>Watchdog AI</h1>"
    f"<p style='margin: 0; color: #6B7280;'>{st.session_state.dealer_name}</p>"
    f"</div>"
    f"</div>",
    unsafe_allow_html=True
)

# Create tabs
tabs = st.tabs([
    "Connect Systems", 
    "System Status"
])

# Tab 1: Connect Systems form
with tabs[0]:
    st.markdown("<h2 style='text-align: center;'>Connect My Systems</h2>", unsafe_allow_html=True)
    
    with st.container():
        st.markdown(f"<div style='{STYLES['card']}'>", unsafe_allow_html=True)
        
        # Vendor selection
        vendor = st.selectbox(
            "Select Vendor", 
            ["", "DealerSocket", "VinSolutions", "CDK", "Reynolds & Reynolds", "DealerTrack"],
            index=0,
            help="Select your DMS or CRM vendor"
        )
        
        if vendor:
            # Create two columns for credentials
            col1, col2 = st.columns(2)
            
            with col1:
                email = st.text_input(
                    "Email", 
                    value=st.session_state.credentials.get("email", ""),
                    help="Your login email for the vendor system"
                )
                
                dealership_id = st.text_input(
                    "Dealership ID",
                    value=st.session_state.credentials.get("dealership_id", ""),
                    help="Your dealership identifier"
                )
                
                dealer_name = st.text_input(
                    "Dealership Name",
                    value=st.session_state.dealer_name,
                    help="Your dealership's display name"
                )
            
            with col2:
                password = st.text_input(
                    "Password", 
                    type="password",
                    help="Your login password for the vendor system"
                )
                
                two_factor_method = st.selectbox(
                    "2FA Method",
                    ["None", "SMS", "Email", "Authenticator"],
                    index=0,
                    help="Two-factor authentication method"
                )
            
            # Report selection
            st.subheader("Select Reports to Sync")
            col1, col2 = st.columns(2)
            
            with col1:
                sales_report = st.checkbox("Sales Reports", value=True)
                inventory_report = st.checkbox("Inventory Reports", value=True)
            
            with col2:
                leads_report = st.checkbox("Lead Reports")
                service_report = st.checkbox("Service Reports")
            
            # Collect selected reports
            selected_reports = []
            if sales_report:
                selected_reports.append("sales")
            if inventory_report:
                selected_reports.append("inventory")
            if leads_report:
                selected_reports.append("leads")
            if service_report:
                selected_reports.append("service")
            
            # Sync frequency
            sync_frequency = st.selectbox(
                "Sync Frequency",
                ["Hourly", "Daily", "Weekly", "Monthly"],
                index=1,
                help="How often to automatically sync data"
            )
            
            # Submit button
            if st.button("Connect System", key="connect_system"):
                with st.spinner("Connecting to vendor system..."):
                    # Store values in session state
                    st.session_state.credentials = {
                        "vendor": vendor,
                        "email": email,
                        "password": password,
                        "dealership_id": dealership_id,
                        "2fa_method": two_factor_method.lower() if two_factor_method != "None" else "",
                        "reports": selected_reports
                    }
                    st.session_state.dealer_name = dealer_name
                    
                    # Simulate connection
                    import time
                    time.sleep(1.5)  # Simulate API call
                    
                    # Update connection state
                    st.session_state.connected = True
                    st.session_state.last_sync = datetime.now().isoformat()
                    
                    st.success(f"✅ Successfully connected to {vendor}!")
                    st.session_state.sync_result = {
                        "status": "success",
                        "message": f"Connected to {vendor} successfully.",
                        "timestamp": datetime.now().isoformat()
                    }
        
        st.markdown("</div>", unsafe_allow_html=True)

# Tab 2: System Status
with tabs[1]:
    st.markdown("<h2 style='text-align: center;'>System Status</h2>", unsafe_allow_html=True)
    
    with st.container():
        st.markdown(f"<div style='{STYLES['card']}'>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if st.session_state.connected:
                st.markdown(f"<div style='{STYLES['badge_success']}'>Connected</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='{STYLES['badge_error']}'>Disconnected</div>", unsafe_allow_html=True)
            
            if st.session_state.last_sync:
                last_sync = datetime.fromisoformat(st.session_state.last_sync)
                st.markdown(f"Last Sync: {last_sync.strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                st.markdown("Last Sync: Never")
        
        with col2:
            if st.session_state.connected:
                st.markdown(f"**Vendor**: {st.session_state.credentials.get('vendor', 'Unknown')}")
                st.markdown(f"**Reports**: {', '.join(st.session_state.credentials.get('reports', ['None']))}")
        
        if st.session_state.connected:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                if st.button("Sync Now", key="sync_now"):
                    with st.spinner("Syncing data from vendor system..."):
                        # Simulate sync operation
                        import time
                        time.sleep(2)  # Simulate API call
                        
                        # Update sync timestamp
                        st.session_state.last_sync = datetime.now().isoformat()
                        
                        st.session_state.sync_result = {
                            "status": "success",
                            "message": "Data synchronized successfully.",
                            "files": [
                                {"name": "sales_report.csv", "records": 128},
                                {"name": "inventory_report.csv", "records": 76}
                            ],
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        st.success("✅ Data synchronized successfully!")
            
            with col2:
                if st.button("Disconnect", key="disconnect"):
                    st.session_state.connected = False
                    st.session_state.last_sync = None
                    st.warning("System disconnected. You can reconnect at any time.")
        
        if st.session_state.sync_result:
            st.subheader("Latest Sync Result")
            
            result = st.session_state.sync_result
            if result["status"] == "success":
                status_style = STYLES["badge_success"]
            else:
                status_style = STYLES["badge_error"]
            
            st.markdown(f"<div style='{status_style}'>{result['status'].upper()}</div>", unsafe_allow_html=True)
            st.markdown(f"**Message**: {result['message']}")
            
            if "files" in result:
                for file in result["files"]:
                    st.markdown(f"• {file['name']} ({file['records']} records)")
            
            if "timestamp" in result:
                timestamp = datetime.fromisoformat(result["timestamp"])
                st.markdown(f"**Time**: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        
        st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    f"<div style='text-align: center;'>"
    f"<p>Watchdog AI v3.0 | {datetime.now().strftime('%Y-%m-%d')}</p>"
    f"</div>", 
    unsafe_allow_html=True
)