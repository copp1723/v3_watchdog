"""
Data insights module for handling data processing and validation.
"""

import pandas as pd
import streamlit as st
import logging
from typing import Tuple, Dict, Any, Optional
from .config import SessionKeys

logger = logging.getLogger(__name__)

def handle_upload(uploaded_file) -> Tuple[bool, str]:
    """
    Handle file upload with proper data type conversion for Streamlit/Arrow serialization.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)
        
        # Convert object columns to string to fix Arrow serialization
        object_cols = df.select_dtypes(include=['object']).columns
        for col in object_cols:
            df[col] = df[col].astype("string")
            
        # Convert date columns if they exist
        date_cols = [col for col in df.columns 
                    if any(term in col.lower() 
                          for term in ['date', 'time', 'month', 'year'])]
        for col in date_cols:
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception as e:
                logger.warning(f"Could not convert {col} to datetime: {str(e)}")
        
        # Store in session state
        st.session_state[SessionKeys.UPLOADED_DATA] = df
        
        return True, f"Successfully loaded {len(df)} rows with {len(df.columns)} columns"
        
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}", exc_info=True)
        return False, f"Error processing file: {str(e)}"

def validate_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate uploaded data and return summary.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Dictionary with validation results
    """
    summary = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": df.columns.tolist(),
        "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": {col: int(df[col].isna().sum()) for col in df.columns},
        "sample_values": {col: df[col].head().tolist() for col in df.columns}
    }
    
    return summary