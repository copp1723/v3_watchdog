"""
Data upload component with proper session state management.
"""

import streamlit as st
import pandas as pd
import logging
from datetime import datetime
import os

logger = logging.getLogger(__name__)

def convert_currency_to_numeric(value):
    """Convert currency string to numeric value."""
    if isinstance(value, str):
        return float(value.replace('$', '').replace(',', ''))
    return value

def render_data_upload():
    """Render the data upload component."""
    st.markdown("### Data Upload")
    
    uploaded_file = st.file_uploader("Upload your data file", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        try:
            # Read the file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Basic validation
            if df.empty:
                st.error("The uploaded file contains no data.")
                return
            
            # Convert currency columns to numeric
            currency_columns = ['listing_price', 'sold_price', 'profit', 'expense']
            for col in currency_columns:
                if col in df.columns:
                    df[col] = df[col].apply(convert_currency_to_numeric)
                
            # Store in session state for direct querying
            st.session_state.validated_data = df
            
            # Display success message
            st.success(f"Successfully loaded data: {len(df)} rows, {len(df.columns)} columns")
            
            # Show data preview
            st.markdown("#### Data Preview")
            st.dataframe(df.head())
            
            # Show column info
            st.markdown("#### Column Information")
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.astype(str),
                'Non-Null Count': df.count(),
                'Null Count': df.isna().sum()
            })
            st.dataframe(col_info)
            
            # Show summary statistics for numeric columns
            st.markdown("#### Numeric Column Statistics")
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            if not numeric_cols.empty:
                st.dataframe(df[numeric_cols].describe())
            
        except Exception as e:
            logger.error(f"Error loading file: {str(e)}")
            st.error(f"Error loading file: {str(e)}")
            return
    else:
        if 'validated_data' not in st.session_state:
            st.info("Please upload a CSV or Excel file to begin analysis.")
        else:
            st.success("Using previously uploaded data.") 