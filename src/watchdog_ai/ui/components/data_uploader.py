"""
Data uploader component for Watchdog AI.
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging
import json
from typing import Tuple, Optional
from io import StringIO
import csv
from watchdog_ai.config import SessionKeys
from watchdog_ai.utils.adaptive_schema import AdaptiveSchema
from watchdog_ai.utils.data_lineage import DataLineage
from watchdog_ai.ui.components.mapping_feedback import MappingFeedbackUI

logger = logging.getLogger(__name__)

def _safe_convert_value(val) -> str:
    """Convert any value to a string safely."""
    try:
        if pd.isna(val):
            return ""
        if isinstance(val, (dict, list)):
            return json.dumps(val)
        return str(val)
    except:
        return "[UNCONVERTIBLE]"

def render_data_uploader() -> None:
    """Render the data uploader component."""
    try:
        # Initialize schema and lineage
        schema = AdaptiveSchema()
        lineage = DataLineage()
        feedback_ui = MappingFeedbackUI(lineage)
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload your CSV file",
            type=["csv"],
            help="Upload a CSV file to analyze"
        )
        
        if uploaded_file is not None:
            try:
                # Read CSV as text first to force string conversion
                content = StringIO(uploaded_file.getvalue().decode('utf-8'))
                reader = csv.reader(content)
                
                # Get headers and data
                headers = next(reader)
                data = []
                for row in reader:
                    # Convert all values to strings
                    data.append([_safe_convert_value(val) for val in row])
                
                # Create DataFrame with string data
                df = pd.DataFrame(data, columns=headers)
                
                # Validate schema and get suggestions
                result, suggestions = schema.validate_with_suggestions(df)
                
                if not result.is_valid:
                    # Show mapping suggestions and collect feedback
                    feedback_ui.render_mapping_suggestions(suggestions)
                
                # Store in session state
                st.session_state[SessionKeys.UPLOADED_DATA] = df
                
                # Show success message
                st.success("✅ Data uploaded successfully!")
                
                # Show data preview using native Streamlit display
                st.subheader("Data Preview")
                try:
                    st.dataframe(
                        df.head(),
                        use_container_width=True,
                        hide_index=False
                    )
                except Exception as e:
                    st.error(f"Error displaying preview: {str(e)}")
                    st.write("Displaying alternative preview:")
                    st.json(df.head().to_dict('records'))
                
                # Show data info
                st.subheader("Data Info")
                st.write(f"- Total Records: {len(df):,}")
                st.write(f"- Total Columns: {len(df.columns):,}")
                
                # Show column info using markdown table
                st.subheader("Column Info")
                col_info = []
                for col in df.columns:
                    col_info.append({
                        'Column': col,
                        'Type': str(df[col].dtype),
                        'Non-Null Count': df[col].count(),
                        'Null Count': df[col].isnull().sum()
                    })
                st.markdown(pd.DataFrame(col_info).to_markdown(index=False))
                
                # Show feedback history
                feedback_ui.render_feedback_history()
                
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                logger.error(f"Error reading file: {str(e)}", exc_info=True)
                
    except Exception as e:
        st.error(f"Error in uploader: {str(e)}")
        logger.error(f"Error in uploader: {str(e)}", exc_info=True)