"""
Enhanced data upload component for Watchdog AI.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import logging

from src.utils.data_io import load_data, validate_data
from src.utils.errors import ValidationError

logger = logging.getLogger(__name__)

class DataUploadManager:
    """Handles data upload and validation."""
    
    def __init__(self):
        """Initialize the data upload manager."""
        self._initialize_session_state()
    
    def _initialize_session_state(self) -> None:
        """Initialize session state variables."""
        if 'validated_data' not in st.session_state:
            st.session_state.validated_data = None
        if 'validation_summary' not in st.session_state:
            st.session_state.validation_summary = None
        if 'upload_timestamp' not in st.session_state:
            st.session_state.upload_timestamp = None
    
    def render_upload_section(self) -> None:
        """Render the data upload section."""
        st.markdown("### Data Upload")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload your dealership data (CSV or Excel)",
            type=['csv', 'xlsx', 'xls'],
            help="Upload a file containing your dealership data"
        )
        
        if uploaded_file:
            try:
                with st.spinner("Processing data..."):
                    # Load and validate data
                    df = load_data(uploaded_file)
                    validated_df, summary = validate_data(df)
                    
                    # Store in session state
                    st.session_state.validated_data = validated_df
                    st.session_state.validation_summary = summary
                    st.session_state.upload_timestamp = datetime.now().isoformat()
                    
                    # Show validation summary
                    self._render_validation_summary(summary)
                    
            except ValidationError as e:
                st.error(f"Validation Error: {str(e)}")
                logger.error(f"Validation error: {str(e)}")
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                logger.error(f"Error processing file: {str(e)}")
    
    def _render_validation_summary(self, summary: dict) -> None:
        """Render the validation summary."""
        st.markdown("#### Data Validation Summary")
        
        # Create columns for metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Total Records",
                summary['total_records']
            )
        
        with col2:
            st.metric(
                "Data Quality Score",
                f"{summary['quality_score']:.1f}%"
            )
        
        with col3:
            total_issues = (
                sum(summary['missing_values'].values()) +
                sum(summary['invalid_values'].values())
            )
            st.metric(
                "Total Issues",
                total_issues
            )
        
        # Show detailed issues if any exist
        if total_issues > 0:
            with st.expander("View Data Quality Issues"):
                # Missing values
                if summary['missing_values']:
                    st.markdown("##### Missing Values")
                    for col, count in summary['missing_values'].items():
                        st.write(f"- {col}: {count} missing values")
                
                # Invalid values
                if summary['invalid_values']:
                    st.markdown("##### Invalid Values")
                    for check, count in summary['invalid_values'].items():
                        st.write(f"- {check}: {count} invalid values")
