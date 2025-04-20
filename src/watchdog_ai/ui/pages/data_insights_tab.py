import streamlit as st
from watchdog_ai.config import SessionKeys
from watchdog_ai.insights.insight_conversation import ConversationManager
from watchdog_ai.ui.components.data_uploader import render_data_uploader

def render_data_insights_tab():
    """Render the data insights tab."""
    
    # Initialize conversation manager if not already done
    if SessionKeys.UPLOADED_DATA not in st.session_state:
        render_data_uploader()
        return

    st.header("Data & Insights")
    
    # Use the consolidated uploader component
    render_data_uploader()
    
    # Clear data if no file is uploaded
    if "uploaded_data" not in st.session_state:
        if SessionKeys.UPLOADED_DATA in st.session_state:
            del st.session_state[SessionKeys.UPLOADED_DATA]