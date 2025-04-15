"""
Data Upload Component for Watchdog AI.

This module provides UI components for uploading and processing data files.
It integrates with the validation system to ensure data quality before processing.
"""

import streamlit as st
import pandas as pd
import os
from typing import Optional, Dict, Any, Tuple, Callable

# Remove the import for process_uploaded_file since we're no longer using it
# from validators.validator_service import process_uploaded_file


def render_validation_summary(summary: Dict[str, Any]):
    """Render a validation summary with metrics and charts."""
    if summary["status"] == "error":
        st.error(summary["message"])
        return
    
    # Display metrics in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Total Records",
            summary["total_rows"],
            help="Total number of records in the uploaded file"
        )
    
    with col2:
        passed_percentage = (summary["passed_rows"] / summary["total_rows"] * 100) if summary["total_rows"] > 0 else 0
        st.metric(
            "Passed Validation",
            f"{passed_percentage:.1f}%",
            help="Percentage of records that passed all validation rules"
        )
    
    with col3:
        st.metric(
            "Issues Found",
            summary["failed_rows"],
            help="Number of records with validation issues"
        )
    
    # Show detailed flag counts if any
    if summary["flag_counts"]:
        with st.expander("Detailed Validation Results", expanded=True):
            st.write("Issues by type:")
            for flag, count in summary["flag_counts"].items():
                if count > 0:
                    st.write(f"- {flag}: {count} records")

def render_data_preview(df: pd.DataFrame):
    """Render a preview of the DataFrame with expandable details."""
    with st.expander("Data Preview", expanded=False):
        # Show basic DataFrame info
        st.write("DataFrame Info:")
        st.write(f"- Rows: {len(df)}")
        st.write(f"- Columns: {', '.join(df.columns)}")
        
        # Show the first few rows
        st.dataframe(df.head(), use_container_width=True)

def render_file_upload() -> Optional[Any]:
    """
    Render a file upload component that accepts CSV or Excel files.
    
    Returns:
        The uploaded file object if a file was uploaded, None otherwise
    """
    # File upload widget
    uploaded_file = st.file_uploader(
        "Upload a CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        help="Upload your dealership data for analysis."
    )
    
    return uploaded_file


def render_sample_data_option() -> Optional[Any]:
    """
    Render a component to load sample data for demonstration purposes.
    
    Returns:
        A file-like object containing the sample data if the sample option was selected, None otherwise
    """
    st.write("---")
    st.caption("Don't have a file? Try our sample data:")
    
    if st.button("Load Sample Dealership Data", use_container_width=True):
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
        
        # Add name attribute to make it behave like a real UploadedFile
        file_obj.name = "sample_dealership_data.csv"
        
        return file_obj
    
    return None


def render_data_upload(simplified: bool = False) -> Optional[Any]:
    """
    Render a complete data upload interface with file upload and sample data options.
    
    Args:
        simplified: If True, uses a simplified UI without labels and emojis
    
    Returns:
        The uploaded file object if a file was uploaded or sample data was selected, None otherwise
    """
    # Create a container for the file upload section
    with st.container():
        # Only show subheader in non-simplified mode
        if not simplified:
            st.subheader("Upload Your Data")
        
        # Render file upload component
        uploaded_file = render_file_upload()
        
        # If no file was uploaded, offer sample data
        if uploaded_file is None:
            sample_file = render_sample_data_option()
            if sample_file is not None:
                return sample_file
        
        return uploaded_file


if __name__ == "__main__":
    # Sample code for testing
    import streamlit as st
    
    st.set_page_config(page_title="Data Upload Demo", layout="wide")
    st.title("Watchdog AI - Data Upload Component Demo")
    
    # Render the file upload component
    uploaded_file = render_data_upload()
    
    # Display uploaded file if available
    if uploaded_file is not None:
        st.success(f"File uploaded: {uploaded_file.name}")
        
        # Read the file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file type")
            df = None
            
        if df is not None:
            st.write(f"Rows: {len(df)}, Columns: {len(df.columns)}")
            st.dataframe(df.head(), use_container_width=True)