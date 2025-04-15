"""
Main Streamlit app for Watchdog AI.

This application integrates the validation profile system with the insight generation pipeline.
It provides a clean UI for uploading files, validating data, and generating insights.
"""

import streamlit as st
import pandas as pd
import os
from typing import Optional, Dict, Any

# Import our validation components
from validators.validator_service import (
    process_uploaded_file,
    render_data_validation_interface,
    ValidatorService
)

# Import the insight generation pipeline (Dev 2's flow)
# This is a placeholder - you would typically import the actual insight generation code
def render_insight_card(df: pd.DataFrame) -> None:
    """Placeholder for Dev 2's insight generation pipeline."""
    st.subheader("🧠 Insights Generation")
    st.write("This is where the insight generation pipeline would be integrated.")
    st.write("For now, here's a preview of the cleaned data:")
    st.dataframe(df.head())


def get_active_data() -> Optional[Dict[str, Any]]:
    """Get the active validated data from the session state."""
    if "active_upload" in st.session_state and "validated_data" in st.session_state:
        upload_key = st.session_state["active_upload"]
        if upload_key in st.session_state["validated_data"]:
            return st.session_state["validated_data"][upload_key]
    return None


def main():
    """Main application entry point."""
    # Set page configuration
    st.set_page_config(
        page_title="Watchdog AI V3",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Sidebar
    with st.sidebar:
        st.title("🔍 Watchdog AI V3")
        st.subheader("Data Quality & Insights")
        
        st.write("---")
        
        # Path to the profiles directory - in a real application, you might want
        # to store this in a more permanent location
        profiles_dir = os.path.join(os.path.dirname(__file__), "profiles")
        os.makedirs(profiles_dir, exist_ok=True)
        
        # Add options for auto-cleaning
        auto_clean = st.checkbox("Auto-clean data when uploading", value=False)
        
        # File upload section
        st.subheader("📂 Upload Data")
        uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])
        
        if uploaded_file:
            # Process the uploaded file
            with st.spinner("Processing file..."):
                df, summary, validator = process_uploaded_file(
                    uploaded_file, 
                    profiles_dir,
                    apply_auto_cleaning=auto_clean
                )
            
            if df is not None:
                # Show success message
                st.success(f"File uploaded: {uploaded_file.name}")
            else:
                st.error(f"Error: {summary.get('error', 'Unknown error')}")
        
        # Navigation section
        st.write("---")
        st.subheader("📊 Navigation")
        
        # Check if we have active data
        active_data = get_active_data()
        
        if active_data:
            # Show options based on data state
            if active_data.get("cleaned", False):
                insights_page = st.button("🧠 Insights", key="nav_insights", use_container_width=True)
                if insights_page:
                    st.session_state["page"] = "insights"
            
            validation_page = st.button("🔍 Validation", key="nav_validation", use_container_width=True)
            if validation_page:
                st.session_state["page"] = "validation"
        
        # About section
        st.write("---")
        st.caption("© 2024 Watchdog AI - V3")
    
    # Main content area
    if "page" not in st.session_state:
        st.session_state["page"] = "upload"
    
    # Get the active data
    active_data = get_active_data()
    
    # Show different pages based on navigation state
    if active_data is None:
        # Show upload message if no data is available
        st.title("🔍 Watchdog AI V3")
        st.subheader("Please upload a file to begin.")
        
        # Create sample data for demo purposes
        if st.button("📊 Load Demo Data"):
            # Create sample data
            data = {
                'VIN': ['1HGCM82633A123456', '1HGCM82633A123456', '5TFBW5F13AX123457', '789', '', 'WBAGH83576D123458'],
                'Make': ['Honda', 'Honda', 'Toyota', 'Chevrolet', 'Ford', 'BMW'],
                'Model': ['Accord', 'Accord', 'Tundra', 'Malibu', 'F-150', '7 Series'],
                'Year': [2019, 2019, 2020, 2018, 2021, 2018],
                'Sale_Date': ['2023-01-15', '2023-02-10', '2023-02-20', '2023-03-01', '2023-03-15', '2023-03-05'],
                'Sale_Price': [28500.00, 27000.00, 45750.00, 22000.00, 35000.00, 62000.00],
                'Cost': [25000.00, 28000.00, 40000.00, 20000.00, 32000.00, 55000.00],
                'Gross_Profit': [3500.00, -1000.00, 5750.00, 2000.00, 3000.00, 7000.00],
                'Lead_Source': ['Website', None, '', 'Google', 'Autotrader', 'Walk-in'],
                'Salesperson': ['John Smith', 'Jane Doe', 'Jane Doe', 'Bob Johnson', 'John Smith', 'Bob Johnson']
            }
            df = pd.DataFrame(data)
            
            # Process the demo data
            profiles_dir = os.path.join(os.path.dirname(__file__), "profiles")
            validator = ValidatorService(profiles_dir)
            validated_df, validation_summary = validator.validate_dataframe(df)
            
            # Store in session state
            if "validated_data" not in st.session_state:
                st.session_state["validated_data"] = {}
            
            import datetime
            upload_key = f"demo_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            st.session_state["validated_data"][upload_key] = {
                "df": validated_df,
                "summary": validation_summary,
                "timestamp": datetime.datetime.now().isoformat(),
                "cleaned": False,
                "name": "demo_data.csv"
            }
            
            st.session_state["active_upload"] = upload_key
            st.session_state["page"] = "validation"
            
            # Refresh the page
            st.rerun()
    
    elif st.session_state["page"] == "validation":
        # Show validation page
        df = active_data["df"]
        profiles_dir = os.path.join(os.path.dirname(__file__), "profiles")
        validator = ValidatorService(profiles_dir)
        
        # Define what happens when continuing to insights
        def continue_to_insights(cleaned_df):
            # Update the session state
            upload_key = st.session_state["active_upload"]
            st.session_state["validated_data"][upload_key]["df"] = cleaned_df
            st.session_state["validated_data"][upload_key]["cleaned"] = True
            
            # Navigate to insights page
            st.session_state["page"] = "insights"
            st.rerun()
        
        # Render the validation interface
        render_data_validation_interface(df, validator, continue_to_insights)
    
    elif st.session_state["page"] == "insights":
        # Show insights page
        df = active_data["df"]
        
        # Call Dev 2's flow
        render_insight_card(df)


if __name__ == "__main__":
    main()