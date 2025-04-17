"""
Insight Generator Component for Watchdog AI.
Provides UI components for generating and displaying insights.
"""

import streamlit as st
import pandas as pd
import altair as alt
from typing import Dict, Any, Optional
from datetime import datetime

from src.insights.insight_generator import insight_generator
from src.insight_card import render_insight_card

class InsightGenerator:
    """Handles insight generation and display."""
    
    def __init__(self):
        """Initialize the insight generator."""
        self._initialize_session_state()
    
    def _initialize_session_state(self) -> None:
        """Initialize session state variables."""
        if 'insights' not in st.session_state:
            st.session_state.insights = []
        if 'current_insight' not in st.session_state:
            st.session_state.current_insight = None
        if 'selected_example' not in st.session_state:
            st.session_state.selected_example = None
    
    def render_insight_generation(self) -> None:
        """Render the insight generation interface."""
        st.markdown("### Generate Insights")
        
        # Example queries
        examples = [
            "What was our highest grossing lead source?",
            "Show me the sales rep with the most deals",
            "How many deals had negative gross?",
            "Which vehicle make had the highest average profit?"
        ]
        
        # Display examples as buttons
        st.markdown("##### Try asking...")
        cols = st.columns(2)
        for i, example in enumerate(examples):
            with cols[i % 2]:
                if st.button(f"🔍 {example}", key=f"example_{i}", use_container_width=True):
                    st.session_state.selected_example = example
                    st.rerun()
        
        # Input area
        if st.session_state.selected_example:
            prompt = st.text_input(
                "Ask a question about your data",
                value=st.session_state.selected_example,
                key="insight_prompt"
            )
            st.session_state.selected_example = None
        else:
            prompt = st.text_input(
                "Ask a question about your data",
                key="insight_prompt"
            )
        
        # Generate button
        if st.button("Generate Insight", type="primary", key="generate_insight"):
            if prompt:
                with st.spinner("Analyzing data..."):
                    try:
                        # Generate insight
                        if 'validated_data' in st.session_state and isinstance(st.session_state.validated_data, pd.DataFrame):
                            response = insight_generator.generate_insight(
                                prompt=prompt,
                                df=st.session_state.validated_data
                            )
                            
                            # Store insight
                            st.session_state.insights.append({
                                'prompt': prompt,
                                'response': response,
                                'timestamp': datetime.now().isoformat()
                            })
                            st.session_state.current_insight = response
                            
                            # Clear prompt
                            st.session_state.insight_prompt = ""
                            st.rerun()
                            
                    except Exception as e:
                        st.error(f"Error generating insight: {str(e)}")
            else:
                st.warning("Please enter a question first.")
        
        # Display insights
        if st.session_state.insights:
            st.markdown("#### Generated Insights")
            
            for i, entry in enumerate(reversed(st.session_state.insights)):
                with st.container():
                    st.markdown(f"**Q: {entry['prompt']}**")
                    render_insight_card(entry['response'])
                    st.markdown("---")
        
        # Clear button
        if st.session_state.insights:
            if st.button("Clear All", key="clear_insights"):
                st.session_state.insights = []
                st.session_state.current_insight = None
                st.rerun()