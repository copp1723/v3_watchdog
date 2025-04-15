"""
Data Upload Component for Watchdog AI.

This module provides UI components for uploading and processing data files.
It integrates with the validation system to ensure data quality before processing.
"""

import streamlit as st
import pandas as pd
import os
from typing import Optional, Dict, Any, Tuple, Callable

# Import the validation service
from ...validators.validator_service import process_uploaded_file


def render_file_upload(profiles_dir: str, 
                      on_upload_success: Optional[Callable[[pd.DataFrame, Dict[str, Any]], None]] = None) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Render a file upload component that processes files through validation.
    
    Args:
        profiles_dir: Directory containing validation profiles
        on_upload_success: Optional callback when file is uploaded and processed successfully
        
    Returns:
        Tuple of (DataFrame if successful, status info)
    """
    status_info = {}
    
    # Create a container for the file upload section
    with st.container():
        st.subheader("📂 Upload Your Data")
        
        # File upload widget
        uploaded_file = st.file_uploader(
            "Upload a CSV or Excel file",
            type=["csv", "xlsx", "xls"],
            help="Upload your dealership data for analysis."
        )
        
        # Auto-clean option
        auto_clean = st.checkbox(
            "Automatically clean data issues",
            value=False,
            help="Apply standard cleaning operations to fix common issues."
        )
        
        # Process the uploaded file
        if uploaded_file:
            with st.spinner("Processing file..."):
                df, summary, validator = process_uploaded_file(
                    uploaded_file,
                    profiles_dir,
                    apply_auto_cleaning=auto_clean
                )
                
                if df is not None:
                    status_info = {
                        "success": True,
                        "message": f"File uploaded successfully: {uploaded_file.name}",
                        "summary": summary,
                        "validator": validator
                    }
                    
                    # Display basic file info
                    st.success(f"File uploaded: {uploaded_file.name}")
                    st.write(f"Rows: {len(df)}, Columns: {len(df.columns)}")
                    
                    # Display data preview in an expander
                    with st.expander("Preview Data", expanded=False):
                        st.dataframe(df.head(10), use_container_width=True)
                    
                    # Execute callback if provided
                    if on_upload_success:
                        on_upload_success(df, summary)
                        
                    return df, status_info
                else:
                    status_info = {
                        "success": False,
                        "message": f"Error processing file: {summary.get('error', 'Unknown error')}",
                        "summary": summary
                    }
                    
                    st.error(status_info["message"])
        
        return None, status_info


def render_sample_data_option(profiles_dir: str,
                             on_sample_load: Optional[Callable[[pd.DataFrame, Dict[str, Any]], None]] = None) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Render a component to load sample data for demonstration purposes.
    
    Args:
        profiles_dir: Directory containing validation profiles
        on_sample_load: Optional callback when sample data is loaded
        
    Returns:
        Tuple of (DataFrame if sample is loaded, status info)
    """
    status_info = {}
    
    with st.container():
        st.write("---")
        st.caption("Don't have a file? Try our sample data:")
        
        if st.button("📊 Load Sample Dealership Data", use_container_width=True):
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
            
            # Create a file-like object
            import io
            file_obj = io.BytesIO()
            df.to_csv(file_obj, index=False)
            file_obj.seek(0)
            
            # Process like a regular file upload
            with st.spinner("Loading sample data..."):
                # Add name attribute to make it behave like a real UploadedFile
                file_obj.name = "sample_dealership_data.csv"
                
                df, summary, validator = process_uploaded_file(
                    file_obj,
                    profiles_dir,
                    apply_auto_cleaning=False
                )
                
                if df is not None:
                    status_info = {
                        "success": True,
                        "message": "Sample data loaded successfully",
                        "summary": summary,
                        "validator": validator,
                        "is_sample": True
                    }
                    
                    # Display basic file info
                    st.success("Sample data loaded successfully")
                    st.write(f"Rows: {len(df)}, Columns: {len(df.columns)}")
                    
                    # Display data preview in an expander
                    with st.expander("Preview Data", expanded=True):
                        st.dataframe(df, use_container_width=True)
                    
                    # Execute callback if provided
                    if on_sample_load:
                        on_sample_load(df, summary)
                        
                    return df, status_info
                else:
                    status_info = {
                        "success": False,
                        "message": f"Error loading sample data: {summary.get('error', 'Unknown error')}",
                        "summary": summary
                    }
                    
                    st.error(status_info["message"])
        
        return None, status_info


if __name__ == "__main__":
    # Sample code for testing
    import streamlit as st
    
    st.set_page_config(page_title="Data Upload Demo", layout="wide")
    st.title("Watchdog AI - Data Upload Component Demo")
    
    # Create a test profiles directory
    profiles_dir = "test_profiles"
    os.makedirs(profiles_dir, exist_ok=True)
    
    # Define a callback function
    def on_data_loaded(df, summary):
        st.session_state["loaded_data"] = df
        st.session_state["data_summary"] = summary
        st.success("Data successfully loaded into session state!")
    
    # Render the file upload component
    df, status = render_file_upload(profiles_dir, on_data_loaded)
    
    # Render the sample data option
    if df is None:
        sample_df, sample_status = render_sample_data_option(profiles_dir, on_data_loaded)
    
    # Display session state data if available
    if "loaded_data" in st.session_state:
        st.subheader("Session State Data")
        st.write(f"Rows: {len(st.session_state['loaded_data'])}, Columns: {len(st.session_state['loaded_data'].columns)}")
        st.write("Summary:", st.session_state["data_summary"])