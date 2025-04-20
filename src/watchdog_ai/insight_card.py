"""
Insight card component for rendering analysis results.
"""

import streamlit as st
import pandas as pd
import altair as alt
from typing import Dict, Any, Union
from watchdog_ai.models import InsightResponse

def render_insight_card(insight: Union[Dict[str, Any], InsightResponse]) -> None:
    """
    Render an insight card with analysis results.
    
    Args:
        insight: Dictionary or InsightResponse object containing insight data
    """
    try:
        # Convert to dict if needed
        if isinstance(insight, InsightResponse):
            insight_dict = insight.dict()
        else:
            insight_dict = insight
        
        # Handle error insights
        if insight_dict.get("is_error", False):
            st.error(insight_dict.get("summary", "An error occurred"))
            if "error" in insight_dict:
                st.error(f"Error details: {insight_dict['error']}")
            return
        
        # If it's a mock fallback, flag to the user
        if insight_dict.get("is_mock", False):
            st.warning("⚠️ This insight was generated from fallback due to a formatting error from the LLM.")
            
        # Display summary
        st.markdown(f"**{insight_dict.get('summary', 'No summary available')}**")
        
        # Display metrics if available
        metrics = insight_dict.get("metrics", {})
        if metrics:
            cols = st.columns(len(metrics))
            for i, (key, value) in enumerate(metrics.items()):
                with cols[i]:
                    st.metric(
                        label=key.replace("_", " ").title(),
                        value=value
                    )
        
        # Display chart if available
        chart_data = insight_dict.get("chart_data")
        if chart_data:
            try:
                # Create chart based on data structure
                if isinstance(chart_data, dict):
                    # Altair chart specification
                    chart = alt.Chart.from_dict(chart_data)
                    st.altair_chart(chart, use_container_width=True)
                elif isinstance(chart_data, pd.DataFrame):
                    # Simple line chart for time series
                    st.line_chart(chart_data)
                else:
                    st.warning("Unsupported chart data format")
            except Exception as e:
                st.warning(f"Could not render chart: {str(e)}")
        
        # Display recommendations
        recommendations = insight_dict.get("recommendations", [])
        if recommendations:
            st.markdown("### Recommendations")
            for rec in recommendations:
                st.markdown(f"- {rec}")
        
        # Display confidence level
        confidence = insight_dict.get("confidence", "low")
        confidence_color = {
            "high": "green",
            "medium": "orange",
            "low": "red"
        }.get(confidence, "grey")
        
        st.markdown(
            f"<div style='color: {confidence_color}; font-size: 0.8em;'>"
            f"Confidence: {confidence.title()}</div>",
            unsafe_allow_html=True
        )
            
    except Exception as e:
        st.error(f"Error rendering insight card: {str(e)}")