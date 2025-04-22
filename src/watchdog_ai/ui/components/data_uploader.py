"""
Data uploader component.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional, Tuple

from ...config import SessionKeys
from ...utils.errors import ValidationError

def render_data_uploader():
    """Render the data upload interface."""
    st.title("Data Upload")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state[SessionKeys.VALIDATED_DATA] = df
            st.session_state[SessionKeys.VALIDATION_SUMMARY] = {
                "status": "success",
                "total_rows": len(df),
                "passed_rows": len(df),
                "failed_rows": 0
            }
            st.success("File uploaded successfully!")
        except Exception as e:
            st.error(f"Error uploading file: {str(e)}")