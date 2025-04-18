"""
Feedback view component for Watchdog AI.
Allows executives to review and submit feedback on insights.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional
import sentry_sdk

from ...insights.feedback import feedback_manager
from ...insights.models import FeedbackEntry, FeedbackStats

def _initialize_session_state():
    """Initialize session state variables for feedback."""
    if 'feedback_submitted' not in st.session_state:
        st.session_state.feedback_submitted = False
    if 'selected_insight' not in st.session_state:
        st.session_state.selected_insight = None

def _handle_feedback_submission(
    insight_id: str,
    feedback_type: str,
    comment: str,
    session_id: str
):
    """Handle feedback submission and show appropriate messages."""
    try:
        # Create feedback entry
        feedback = FeedbackEntry(
            insight_id=insight_id,
            feedback_type=feedback_type,
            user_id=st.session_state.get('user_id', 'anonymous'),
            session_id=session_id,
            timestamp=datetime.now(),
            comment=comment
        )
        
        # Add Sentry breadcrumb
        sentry_sdk.add_breadcrumb(
            category="feedback",
            message="Submitting feedback",
            data=feedback.to_dict(),
            level="info"
        )
        
        # Record feedback
        success = feedback_manager.record_feedback(**feedback.to_dict())
        
        if success:
            st.success("Thank you for your feedback!")
            st.session_state.feedback_submitted = True
            
            # Capture success in Sentry
            sentry_sdk.capture_message(
                "Feedback submitted successfully",
                level="info",
                tags={
                    "insight_id": insight_id,
                    "session_id": session_id
                }
            )
        else:
            st.error("Failed to submit feedback. Please try again.")
            
    except Exception as e:
        st.error(f"Error submitting feedback: {str(e)}")
        sentry_sdk.capture_exception(e)

def render_feedback_form():
    """Render the feedback submission form."""
    with st.form("feedback_form"):
        # Insight selection
        available_insights = []
        if 'insights' in st.session_state:
            available_insights = [
                (insight.get('insight_type', 'unknown'), insight.get('summary', 'No summary available')[:100] + '...')
                for insight in st.session_state.insights
            ]
        
        selected_insight = st.selectbox(
            "Select Insight",
            options=[i[0] for i in available_insights],
            format_func=lambda x: next((i[1] for i in available_insights if i[0] == x), x),
            help="Choose the insight you want to provide feedback on"
        )
        
        # Feedback type selection
        feedback_type = st.radio(
            "Was this insight helpful?",
            options=["helpful", "somewhat_helpful", "not_helpful"],
            format_func=lambda x: x.replace('_', ' ').title(),
            horizontal=True
        )
        
        # Comment field
        comment = st.text_area(
            "Additional Comments",
            help="Optional: Provide more detailed feedback about this insight"
        )
        
        # Submit button
        submitted = st.form_submit_button("Submit Feedback")
        
        if submitted:
            if selected_insight:
                _handle_feedback_submission(
                    insight_id=selected_insight,
                    feedback_type=feedback_type,
                    comment=comment,
                    session_id=st.session_state.get('session_id', 'unknown')
                )
            else:
                st.warning("Please select an insight to provide feedback.")

def render_feedback_history():
    """Render the feedback history table."""
    try:
        # Get feedback history
        entries = [
            FeedbackEntry.from_dict(entry)
            for entry in feedback_manager.get_feedback()
        ]
        
        if not entries:
            st.info("No feedback entries found.")
            return
        
        # Calculate stats
        stats = FeedbackStats.from_entries(entries)
        
        # Show stats in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Total Feedback",
                stats.total_feedback
            )
        
        with col2:
            st.metric(
                "Helpful Feedback",
                f"{stats.feedback_percentages.get('helpful', 0):.1f}%"
            )
        
        with col3:
            st.metric(
                "Average Rating",
                f"{stats.average_rating:.2f}"
            )
        
        # Convert entries to DataFrame for display
        df = pd.DataFrame([e.to_dict() for e in entries])
        
        # Format timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Reorder and rename columns for display
        display_df = df[[
            'timestamp',
            'insight_id',
            'feedback_type',
            'comment'
        ]].rename(columns={
            'timestamp': 'Timestamp',
            'insight_id': 'Insight Type',
            'feedback_type': 'Feedback',
            'comment': 'Comments'
        })
        
        # Display feedback history
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
        
    except Exception as e:
        st.error(f"Error loading feedback history: {str(e)}")
        sentry_sdk.capture_exception(e)

def feedback_view():
    """Main feedback view component."""
    _initialize_session_state()
    
    st.title("Insight Feedback")
    
    # Create tabs for submission and history
    tab1, tab2 = st.tabs(["Submit Feedback", "Feedback History"])
    
    with tab1:
        st.markdown("""
            ### Submit Your Feedback
            Help us improve our insights by providing feedback on their usefulness and accuracy.
        """)
        render_feedback_form()
    
    with tab2:
        st.markdown("""
            ### Feedback History
            Review previously submitted feedback and overall statistics.
        """)
        render_feedback_history()