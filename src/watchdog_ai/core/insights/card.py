"""
Insight card component for rendering analysis results.
"""

import streamlit as st
from typing import Dict, Any, Union, Optional
import logging

from .base import InsightBase

logger = logging.getLogger(__name__)

class InsightCard:
    """Class for rendering insight cards in the UI."""
    
    @staticmethod
    def render(insight: Dict[str, Any], index: int = 0, show_buttons: bool = True) -> None:
        """
        Render a structured insight as a card in the UI.
        
        Args:
            insight: Dictionary containing insight data
            index: Unique index for the card
            show_buttons: Whether to show interaction buttons
        """
        try:
            # Ensure insight is properly formatted
            if not isinstance(insight, dict) or "summary" not in insight:
                st.error("Invalid insight data format")
                return
            
            # Extract key fields with defaults
            summary = insight.get("summary", "No summary available")
            value_insights = insight.get("value_insights", [])
            actionable_flags = insight.get("actionable_flags", [])
            recommendations = insight.get("recommendations", [])
            confidence = insight.get("confidence", "medium").lower()
            is_mock = insight.get("is_mock", False)
            is_error = insight.get("is_error", False)
            
            # Render the insight card
            with st.container():
                # Handle error insights
                if is_error:
                    st.error(summary)
                    if "error" in insight:
                        st.error(f"Error details: {insight['error']}")
                    return
                
                # If it's a mock fallback, flag to the user
                if is_mock:
                    st.warning("⚠️ This insight was generated from fallback due to a formatting error from the LLM.")
                
                # Header with confidence indicator
                if confidence == "high":
                    st.markdown(f"#### 🟢 {summary}")
                elif confidence == "medium":
                    st.markdown(f"#### 🟡 {summary}")
                else:
                    st.markdown(f"#### 🔴 {summary}")
                
                # Display metrics if available
                metrics = insight.get("metrics", {})
                if metrics:
                    cols = st.columns(len(metrics))
                    for i, (key, value) in enumerate(metrics.items()):
                        with cols[i]:
                            st.metric(
                                label=key.replace("_", " ").title(),
                                value=value
                            )
                
                # Value insights
                if value_insights:
                    st.markdown("**Key insights:**")
                    for insight_text in value_insights:
                        st.markdown(f"- {insight_text}")
                
                # Actionable flags or recommendations
                items_to_show = actionable_flags or recommendations
                if items_to_show:
                    st.markdown("**Action items:**")
                    for item in items_to_show:
                        if isinstance(item, dict) and "action" in item:
                            # Handle structured recommendations
                            st.markdown(f"- {item['action']} (Priority: {item.get('priority', 'Medium')})")
                        else:
                            # Handle simple string items
                            st.markdown(f"- {item}")
                
                # Display chart if available
                chart_data = insight.get("chart_data")
                if chart_data:
                    try:
                        from ..visualization.chart_renderer import render_chart
                        render_chart(chart_data)
                    except Exception as e:
                        logger.warning(f"Could not render chart: {str(e)}")
                
                # Interaction buttons
                if show_buttons:
                    col1, col2, col3 = st.columns([1, 1, 3])
                    with col1:
                        if st.button(f"🔄 Regenerate", key=f"regenerate_{index}"):
                            st.session_state.regenerate_insight = True
                            st.session_state.regenerate_index = index
                            st.rerun()
                    with col2:
                        if st.button(f"🔍 Follow-up", key=f"followup_{index}"):
                            st.session_state.followup_insight = True
                            st.session_state.followup_index = index
                            st.rerun()
                            
        except Exception as e:
            logger.error(f"Error rendering insight card: {e}")
            st.error("An error occurred while rendering this insight.")

