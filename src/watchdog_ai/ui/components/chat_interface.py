"""
Chat interface component for Watchdog AI.
Handles interaction, calls ConversationManager for intent processing and analysis, and renders results.
"""

import streamlit as st
import pandas as pd
import altair as alt # Import altair for charts
from typing import Dict, Any, Optional
from datetime import datetime
import logging
import numpy as np
import plotly.express as px

from watchdog_ai.insights.insight_conversation import ConversationManager
from watchdog_ai.config import SessionKeys
from watchdog_ai.insights.utils import validate_numeric_columns

# Configure logging
logger = logging.getLogger(__name__)

def initialize_session_state():
    """Initialize session state variables with defaults."""
    defaults = {
        SessionKeys.CHAT_HISTORY: [],
        SessionKeys.UPLOADED_DATA: None,
        SessionKeys.LAST_INTENT: None,
        SessionKeys.LAST_RESULT: None,
        SessionKeys.LAST_QUERY: None,
        SessionKeys.QUERY_TEXT: "",
        SessionKeys.SELECTED_EXAMPLE: None,
        SessionKeys.INTENT_CACHE: {},
        SessionKeys.RESULT_CACHE: {},
        SessionKeys.CONVERSATION_STATE: {},
        SessionKeys.METRICS_HISTORY: [],
        SessionKeys.ANALYSIS_STATE: {}
    }
    
    # Initialize each key if it doesn't exist
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
            logger.debug(f"Initialized session state key: {key}")
    
    # Ensure chat_history is always a list
    if not isinstance(st.session_state[SessionKeys.CHAT_HISTORY], list):
        st.session_state[SessionKeys.CHAT_HISTORY] = []
        logger.debug("Reset chat_history to empty list")

def clear_chat():
    """Completely reset all chat-related session state."""
    logger.info("Clearing all chat state")
    
    logger.debug(f"BEFORE CLEAR STATE: {st.session_state}")
    
    keys_to_clear = [
        SessionKeys.CHAT_HISTORY,
        SessionKeys.LAST_INTENT,
        SessionKeys.LAST_RESULT,
        SessionKeys.LAST_QUERY,
        SessionKeys.QUERY_TEXT,
        SessionKeys.SELECTED_EXAMPLE,
        SessionKeys.INTENT_CACHE,
        SessionKeys.RESULT_CACHE,
        SessionKeys.CONVERSATION_STATE,
        SessionKeys.METRICS_HISTORY,
        SessionKeys.ANALYSIS_STATE
    ]
    
    for key in keys_to_clear:
        if key in st.session_state:
            st.session_state[key] = [] if key == SessionKeys.CHAT_HISTORY else None
            logger.debug(f"Cleared state key: {key}")
    
    logger.debug(f"AFTER CLEAR STATE: {st.session_state}")
    
    # Reinitialize session state to ensure chat_history exists
    initialize_session_state()
    
    st.rerun()

# Helper function to render the new structured response format
def render_structured_insight(response: Dict[str, Any]):
    """Renders the structured insight response with formatting and charts."""
    if not isinstance(response, dict):
        st.error("Invalid response format received.")
        logger.error(f"render_structured_insight received non-dict response: {response}")
        return

    # Display debug info in expander (optional for developers)
    if "_debug" in response and st.session_state.get("show_debug", False):
        with st.expander("Debug Info"):
            st.json(response["_debug"])

    summary = response.get("summary", "No summary provided.")
    metrics = response.get("metrics", {})
    breakdown = response.get("breakdown", [])
    recommendations = response.get("recommendations", [])
    confidence = response.get("confidence", "unknown")
    error_type = response.get("error_type")

    # Display summary (potentially with warning/error styling)
    if error_type:
        st.warning(f"⚠️ {summary}")
    else:
        st.markdown(f"**Insight:** {summary}")

    # Display Metrics
    if metrics:
        st.write("**Key Metrics:**")
        cols = st.columns(len(metrics))
        i = 0
        for key, value in metrics.items():
            with cols[i]:
                st.metric(label=key.replace("_", " ").title(), value=str(value))
            i = (i + 1) % len(metrics)

    # Display Breakdown with Chart
    if breakdown and isinstance(breakdown, list) and all(isinstance(item, dict) for item in breakdown):
        st.write("**Breakdown:**")
        try:
            breakdown_df = pd.DataFrame(breakdown)
            
            if "category" in breakdown_df.columns and "value" in breakdown_df.columns:
                breakdown_df["value"] = pd.to_numeric(breakdown_df["value"], errors='coerce')
                breakdown_df = breakdown_df.dropna(subset=["value"])
                breakdown_df = breakdown_df[np.isfinite(breakdown_df["value"])]
                
                if breakdown_df.empty:
                    st.warning("No valid data available for visualization.")
                else:
                    chart = alt.Chart(breakdown_df).mark_bar().encode(
                        x=alt.X('value:Q', title='Value'),
                        y=alt.Y('category:N', title='Category', sort='-x'),
                        tooltip=['category:N', 'value:Q', 
                                alt.Tooltip('percentage:N', title='Percentage')] if 'percentage' in breakdown_df.columns 
                                else ['category:N', 'value:Q']
                    ).properties(
                        title="Breakdown by Category"
                    )
                    st.altair_chart(chart, use_container_width=True)
            else:
                st.write("Breakdown data missing required 'category' or 'value' columns.")

        except Exception as e:
            logger.error(f"Error creating breakdown chart: {e}", exc_info=True)
            st.error("Could not display breakdown chart.")
            st.json(breakdown)
    elif breakdown:
        st.write("**Breakdown Data (raw):**")
        st.json(breakdown)

    # Display Recommendations
    if recommendations and isinstance(recommendations, list) and all(isinstance(item, dict) for item in recommendations):
        st.write("**Recommendations:**")
        for rec in recommendations:
            action = rec.get("action", "N/A")
            priority = rec.get("priority", "Unknown")
            impact = rec.get("impact_estimate", "N/A")
            st.markdown(f"- **{action}** (Priority: {priority}, Est. Impact: {impact})")
    elif recommendations:
        st.write("**Recommendations (raw):**")
        st.json(recommendations)

    # Display Confidence
    st.caption(f"Confidence: {confidence.title()}")
    if error_type:
        st.caption(f"Error Type: {error_type}")

class ChatInterface:
    """Handles chat-based interaction and insight generation."""

    def __init__(self):
        """Initialize the chat interface."""
        initialize_session_state()
        self.conversation_manager = ConversationManager(use_mock=False)
        logger.info("ChatInterface using ConversationManager instance.")

    def render_chat_interface(self) -> None:
        """Render the chat interface."""
        st.title("Watchdog AI Chat")
        
        # File upload section
        st.header("Upload Your Data")
        uploaded_file = st.file_uploader("Upload a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                # Validate and convert numeric columns
                df = validate_numeric_columns(df)
                st.session_state[SessionKeys.UPLOADED_DATA] = df
                st.success("Data uploaded successfully!")
                st.write("Preview of your data:")
                st.dataframe(df.head())
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                return
        
        st.header("Chat with Your Data")
        
        if st.button("Clear Chat"):
            clear_chat()
        
        with st.expander("Debug Inspector", expanded=False):
            st.write("Session State:", st.session_state)
            if st.session_state.get(SessionKeys.LAST_INTENT):
                st.write("Last Intent:", st.session_state[SessionKeys.LAST_INTENT])
            if st.session_state.get(SessionKeys.LAST_RESULT):
                st.write("Last Result:", st.session_state[SessionKeys.LAST_RESULT])
            if st.session_state.get(SessionKeys.LAST_QUERY):
                st.write("Last Query:", st.session_state[SessionKeys.LAST_QUERY])
        
        query = st.text_input("Ask about your sales data:", key="query_input")
        
        if query:
            if st.session_state.get(SessionKeys.LAST_QUERY) == query:
                logger.info(f"Skipping duplicate query: '{query}'")
            else:
                logger.info(f"Processing new query: '{query}'")
                df = st.session_state.get(SessionKeys.UPLOADED_DATA)
                if df is None:
                    logger.error("No data in session state")
                    st.error("No data uploaded. Please upload a CSV file.")
                    return
                
                validation_context = {'columns': df.columns.tolist()}
                
                logger.info(f"Using validation context with {len(df.columns)} columns")
                logger.debug(f"DataFrame columns: {list(df.columns)}")
                
                # Process the query using ConversationManager
                result = self.conversation_manager.process_query(query, validation_context)
                
                logger.info(f"Generated result: {result}")
                
                # Update state with new result
                st.session_state[SessionKeys.LAST_RESULT] = result
                st.session_state[SessionKeys.LAST_QUERY] = query
                st.session_state[SessionKeys.CHAT_HISTORY].append({
                    "role": "user",
                    "content": query,
                    "timestamp": datetime.now().isoformat()
                })
                st.session_state[SessionKeys.CHAT_HISTORY].append({
                    "role": "assistant",
                    "content": result,
                    "timestamp": datetime.now().isoformat()
                })
        
        # Display chat history
        for message in st.session_state[SessionKeys.CHAT_HISTORY]:
            if message["role"] == "user":
                st.markdown(f"**You:** {message['content']}")
            else:
                render_structured_insight(message["content"])
            st.markdown("---")

    def render_structured_insight(self, insight: Dict[str, Any]) -> None:
        """Render a structured insight response."""
        logger.info("Received structured insight response: %s", insight)
        
        # Add debug inspector
        with st.expander("Debug Inspector", expanded=False):
            st.subheader("Session State")
            st.json(st.session_state)
            
            st.subheader("Last Query")
            st.text(st.session_state.get("last_query", "None"))
            
            st.subheader("Last Intent")
            st.json(st.session_state.get("last_intent", "None"))
            
            st.subheader("Current Insight")
            st.json(insight)
            
            if "_debug" in insight:
                st.subheader("Debug Info")
                st.json(insight["_debug"])
        
        # Extract summary and metrics
        summary = insight.get("summary", "No summary available.")
        metrics = insight.get("metrics", {})
        breakdown = insight.get("breakdown", [])
        recommendations = insight.get("recommendations", [])
        confidence = insight.get("confidence", "low")
        error_type = insight.get("error_type", None)
        
        # Display summary with appropriate styling
        if error_type:
            st.error(summary)
        elif confidence == "low":
            st.warning(summary)
        else:
            st.success(summary)
        
        # Display metrics if available
        if metrics:
            st.subheader("Key Metrics")
            for key, value in metrics.items():
                # Format the key for display
                display_key = key.replace("_", " ").title()
                
                # Handle different value types
                if isinstance(value, (int, float)):
                    if "price" in key.lower() or "cost" in key.lower() or "profit" in key.lower() or "revenue" in key.lower():
                        st.metric(display_key, f"${value:,.2f}")
                    elif "percent" in key.lower() or "percentage" in key.lower() or "rate" in key.lower():
                        st.metric(display_key, f"{value:.1f}%")
                    else:
                        st.metric(display_key, f"{value:,}")
                else:
                    st.metric(display_key, value)
        
        # Display breakdown if available
        if breakdown:
            st.subheader("Breakdown")
            
            # Convert breakdown to DataFrame for easier handling
            breakdown_df = pd.DataFrame(breakdown)
            
            # Clean data for charts - remove NaN and Infinity values
            if "value" in breakdown_df.columns:
                breakdown_df = breakdown_df.dropna(subset=["value"])
                breakdown_df = breakdown_df[np.isfinite(breakdown_df["value"])]
            
            # Determine the best visualization based on data
            if len(breakdown_df) > 0:
                # Check if we have a category column and a value column
                category_col = None
                value_col = None
                
                # Try to find category column
                for col in ["category", "lead_source", "sales_rep_name", "vehicle_make"]:
                    if col in breakdown_df.columns:
                        category_col = col
                        break
                
                # Try to find value column
                for col in ["value", "profit", "sold_price", "days_to_close"]:
                    if col in breakdown_df.columns:
                        value_col = col
                        break
                
                if category_col and value_col:
                    # Create a bar chart
                    fig = px.bar(
                        breakdown_df, 
                        x=category_col, 
                        y=value_col,
                        title=f"{value_col.replace('_', ' ').title()} by {category_col.replace('_', ' ').title()}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Display the data table
                st.dataframe(breakdown_df, use_container_width=True)
            else:
                st.info("No breakdown data available.")
        
        # Display recommendations if available
        if recommendations:
            st.subheader("Recommendations")
            for rec in recommendations:
                st.info(rec)