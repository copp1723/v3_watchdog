"""
Simple Streamlit UI for Watchdog AI.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional
import altair as alt
from datetime import datetime

from src.utils.schema import load_and_validate_file, SchemaValidationError
from src.insights.simple_insight import query_insight, InsightResult

def initialize_session_state():
    """Initialize session state variables."""
    if 'data_dict' not in st.session_state:
        st.session_state.data_dict = None
    if 'validation_error' not in st.session_state:
        st.session_state.validation_error = None
    if 'current_question' not in st.session_state:
        st.session_state.current_question = ""
    if 'insight_history' not in st.session_state:
        st.session_state.insight_history = []

def render_header():
    """Render the application header."""
    st.title("Watchdog AI")
    st.markdown("""
    Upload your dealership data and ask questions to get insights.
    """)

def handle_file_upload():
    """Handle file upload and validation."""
    uploaded_file = st.file_uploader(
        "Upload your data file (CSV or Excel)",
        type=['csv', 'xlsx', 'xls'],
        help="The file should contain sales data with required columns."
    )
    
    if uploaded_file:
        try:
            # Save file to temp location
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            # Load and validate file
            data_dict = load_and_validate_file(tmp_path)
            
            # Clean up temp file
            os.unlink(tmp_path)
            
            # Store in session state
            st.session_state.data_dict = data_dict
            st.session_state.validation_error = None
            
            # Show success message
            st.success("File uploaded and validated successfully!")
            
            # Show data preview
            if 'sales' in data_dict:
                with st.expander("Preview Sales Data"):
                    st.dataframe(data_dict['sales'].head())
            
        except SchemaValidationError as e:
            st.session_state.validation_error = str(e)
            st.error(f"Validation Error: {str(e)}")
        except Exception as e:
            st.session_state.validation_error = str(e)
            st.error(f"Error: {str(e)}")

def render_insight_card(result: InsightResult):
    """Render an insight result card."""
    st.markdown(f"### {result.title}")
    st.markdown(result.summary)
    
    # Show metrics in columns
    if result.metrics:
        cols = st.columns(min(len(result.metrics), 4))
        for i, metric in enumerate(result.metrics):
            col_idx = i % len(cols)
            with cols[col_idx]:
                # Handle different metric formats
                if isinstance(metric, dict):
                    if "metric" in metric and "value" in metric:
                        st.metric(metric["metric"], metric["value"])
                    elif "source" in metric:
                        st.metric(
                            metric["source"],
                            metric["total_gross"],
                            f"{metric['deals']} deals"
                        )
                    elif "rep" in metric:
                        st.metric(
                            metric["rep"],
                            metric["total_gross"],
                            f"{metric['deals']} deals"
                        )
    
    # Show chart if available
    if result.chart_data is not None:
        # Determine chart type based on data
        if "Category" in result.chart_data.columns and "Count" in result.chart_data.columns:
            # Bar chart for categories
            chart = alt.Chart(result.chart_data).mark_bar().encode(
                x='Category',
                y='Count',
                tooltip=['Category', 'Count']
            ).properties(height=300)
        elif "Metric" in result.chart_data.columns and "Value" in result.chart_data.columns:
            # Bar chart for metrics
            chart = alt.Chart(result.chart_data).mark_bar().encode(
                x='Metric',
                y='Value',
                tooltip=['Metric', 'Value']
            ).properties(height=300)
        else:
            # Line chart for time series
            chart = alt.Chart(result.chart_data).mark_line().encode(
                x='date:T',
                y='sum:Q',
                tooltip=['date:T', 'sum:Q']
            ).properties(height=300)
        
        st.altair_chart(chart, use_container_width=True)
    
    # Show recommendations
    if result.recommendations:
        st.markdown("#### Recommendations")
        for rec in result.recommendations:
            st.markdown(f"- {rec}")

def handle_question():
    """Handle user question and generate insight."""
    question = st.chat_input("Ask a question about your data...")
    
    if question:
        if st.session_state.data_dict is None:
            st.warning("Please upload data first.")
            return
        
        # Add user message
        with st.chat_message("user"):
            st.write(question)
        
        # Generate insight
        with st.chat_message("assistant"):
            with st.spinner("Analyzing data..."):
                result = query_insight(st.session_state.data_dict, question)
                render_insight_card(result)
                
                # Store in history
                st.session_state.insight_history.append({
                    'question': question,
                    'result': result,
                    'timestamp': datetime.now().isoformat()
                })

def render_insight_history():
    """Render the insight history."""
    if st.session_state.insight_history:
        st.markdown("### Previous Insights")
        
        for entry in reversed(st.session_state.insight_history):
            with st.expander(f"Q: {entry['question']}", expanded=False):
                render_insight_card(entry['result'])

def main():
    """Main application entry point."""
    # Initialize session state
    initialize_session_state()
    
    # Render header
    render_header()
    
    # File upload section
    st.markdown("### 1. Upload Data")
    handle_file_upload()
    
    # Question and insight section
    st.markdown("### 2. Ask Questions")
    handle_question()
    
    # Show history
    render_insight_history()

if __name__ == "__main__":
    main()