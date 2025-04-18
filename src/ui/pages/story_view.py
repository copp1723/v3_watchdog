"""
Story View for Watchdog AI.
Provides a narrative view combining multiple insights into a cohesive story.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import logging
import sentry_sdk
from typing import Dict, Any, List, Optional

from ...insights.engine import InsightEngine
from ...insights.summarizer import Summarizer
from ...insights.prompt_tuner import PromptTuner
from ...utils.session import record_action

# Configure logger
logger = logging.getLogger(__name__)

def generate_story(
    insights: List[str],
    df: pd.DataFrame,
    feedback_stats: Optional[Dict[str, Any]] = None,
    date_range: Optional[str] = None
) -> str:
    """
    Generate a narrative story from selected insights.
    
    Args:
        insights: List of insight types to include
        df: DataFrame containing the data
        feedback_stats: Optional feedback statistics
        date_range: Optional date range string
        
    Returns:
        Markdown formatted story text
    """
    try:
        # Track story generation
        sentry_sdk.set_tag("story_generation", "active")
        sentry_sdk.set_tag("insight_count", len(insights))
        
        # Get LLM client from session state
        if "llm_client" not in st.session_state:
            raise ValueError("LLM client not initialized")
        
        # Create summarizer and tuner
        summarizer = Summarizer(st.session_state.llm_client)
        tuner = PromptTuner()
        
        # Generate insights with adaptive thresholds
        engine = InsightEngine(st.session_state.llm_client)
        insight_results = []
        
        for insight_type in insights:
            try:
                # Get feedback for threshold learning
                feedback = st.session_state.get("feedback_manager", {}).get_feedback(
                    insight_type=insight_type
                )
                
                # Generate insight with learned threshold
                result = engine.generate_specific_insight(
                    insight_type,
                    df,
                    feedback=feedback
                )
                
                if not result.get("is_error"):
                    insight_results.append(result)
            except Exception as e:
                logger.error(f"Error generating {insight_type} insight: {str(e)}")
                sentry_sdk.capture_exception(e)
        
        if not insight_results:
            return "No insights available. Please select at least one insight type."
        
        # Get date range if not provided
        if not date_range:
            date_range = engine._get_date_range(df)
        
        # Get story template
        template = summarizer.load_template("story_prompt.tpl")
        
        # Get feedback for tuning
        feedback = []
        if "feedback_manager" in st.session_state:
            feedback = st.session_state.feedback_manager.get_feedback(
                template_name="story_prompt.tpl"
            )
        
        # Tune the template based on feedback
        tuned_template = tuner.tune_prompt(template, feedback)
        
        # Generate story using tuned template
        story = summarizer.summarize(
            tuned_template,
            entity_name="Dealership",
            date_range=date_range,
            selected_insights=insights,
            insights=insight_results,
            feedback_stats=feedback_stats
        )
        
        # Record successful generation
        record_action("story_generated", {
            "insight_types": insights,
            "date_range": date_range,
            "template_tuned": template != tuned_template
        })
        
        return story
        
    except Exception as e:
        logger.error(f"Error generating story: {str(e)}")
        sentry_sdk.capture_exception(e)
        return f"Error generating story: {str(e)}"

def story_view():
    """Main story view page."""
    try:
        st.title("Story View")
        
        # Check authentication
        if not st.session_state.get("is_authenticated"):
            st.warning("Please log in to access the story view.")
            return
        
        # Check if we have data
        if 'validated_data' not in st.session_state:
            st.warning("Please upload data to view insights.")
            return
        
        df = st.session_state.validated_data
        
        # Track page view
        sentry_sdk.set_tag("page", "story_view")
        sentry_sdk.set_tag("data_rows", len(df))
        
        # Date range filter
        st.sidebar.markdown("### Filters")
        date_options = {
            "Last 7 Days": 7,
            "Last 30 Days": 30,
            "Last 90 Days": 90,
            "Custom Range": -1
        }
        date_filter = st.sidebar.selectbox("Date Range", list(date_options.keys()))
        
        if date_filter == "Custom Range":
            start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=30))
            end_date = st.sidebar.date_input("End Date", datetime.now())
            date_range = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        else:
            days = date_options[date_filter]
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            date_range = f"Last {days} days"
        
        # Insight selection
        st.sidebar.markdown("### Insights to Include")
        available_insights = [
            "monthly_gross_margin",
            "lead_conversion_rate",
            "sales_performance",
            "inventory_anomalies"
        ]
        
        selected_insights = []
        for insight in available_insights:
            if st.sidebar.checkbox(insight.replace("_", " ").title(), value=True):
                selected_insights.append(insight)
        
        if not selected_insights:
            st.warning("Please select at least one insight type.")
            return
        
        # Generate story
        with st.spinner("Generating story..."):
            story = generate_story(
                selected_insights,
                df,
                st.session_state.get("feedback_stats"),
                date_range
            )
        
        # Display story
        st.markdown(story)
        
        # Feedback buttons
        st.markdown("### Was this story helpful?")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("👍 Yes"):
                record_action("story_feedback", {
                    "helpful": True,
                    "insight_types": selected_insights
                })
                st.success("Thanks for your feedback!")
        
        with col2:
            if st.button("👎 No"):
                record_action("story_feedback", {
                    "helpful": False,
                    "insight_types": selected_insights
                })
                st.info("Thanks for your feedback! We'll work on improving the stories.")
        
    except Exception as e:
        logger.error(f"Error rendering story view: {str(e)}")
        sentry_sdk.capture_exception(e)
        st.error("An error occurred while rendering the story view")