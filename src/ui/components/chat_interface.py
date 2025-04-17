"""
Chat Interface Component for Watchdog AI.
Provides UI components for chat-based interaction.
"""

import streamlit as st
import pandas as pd
import altair as alt
from typing import Dict, Any, Optional
from datetime import datetime

from src.insight_conversation import ConversationManager
from src.insight_card import render_insight_card

class ChatInterface:
    """Handles chat-based interaction and insight generation."""
    
    def __init__(self):
        """Initialize the chat interface."""
        self._initialize_session_state()
        self.conversation_manager = ConversationManager()
    
    def _initialize_session_state(self) -> None:
        """Initialize session state variables."""
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'current_insight' not in st.session_state:
            st.session_state.current_insight = None
        if 'chat_input' not in st.session_state:
            st.session_state.chat_input = ""
        if 'selected_example' not in st.session_state:
            st.session_state.selected_example = None
    
    def _clear_state(self) -> None:
        """Clear chat-related session state."""
        st.session_state.chat_history = []
        st.session_state.current_insight = None
        st.session_state.chat_input = ""
        st.session_state.selected_example = None
        st.rerun()
    
    def render_chat_interface(self) -> None:
        """Render the chat interface."""
        st.markdown("### Chat Analysis")
        
        # Chat history
        for entry in st.session_state.chat_history:
            # User message
            with st.chat_message("user"):
                st.write(entry['prompt'])
            
            # Assistant response
            with st.chat_message("assistant"):
                render_insight_card(entry['response'])
        
        # Input area
        prompt = st.chat_input("Ask a question about your data...", key="chat_input")
        
        # Process input
        if prompt:
            # Add user message to chat
            with st.chat_message("user"):
                st.write(prompt)
            
            # Generate response
            with st.spinner("Analyzing data..."):
                try:
                    # Get validation context
                    validation_context = None
                    if 'validated_data' in st.session_state:
                        validation_context = {
                            'df': st.session_state.validated_data,
                            'columns': st.session_state.validated_data.columns.tolist(),
                            'data_types': {col: str(dtype) for col, dtype in st.session_state.validated_data.dtypes.items()}
                        }
                    
                    # Generate insight
                    response = self.conversation_manager.generate_insight(
                        prompt=prompt,
                        validation_context=validation_context
                    )
                    
                    # Add assistant response to chat
                    with st.chat_message("assistant"):
                        render_insight_card(response)
                    
                    # Store in history
                    st.session_state.chat_history.append({
                        'prompt': prompt,
                        'response': response,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                except Exception as e:
                    st.error(f"Error generating insight: {str(e)}")
        
        # Clear chat button
        if st.session_state.chat_history:
            if st.button("Clear Chat", key="clear_chat"):
                self._clear_state()
    
    def _create_chart(self, data: pd.DataFrame, encoding: Dict[str, str]) -> alt.Chart:
        """Create an Altair chart based on data and encoding."""
        if len(data) <= 2:
            # For 2 or fewer points, use a table view
            return None
        
        # Check if we need log scale (for high variance in values)
        y_field = encoding.get('y', 'y')
        y_values = data[y_field].values
        y_max = y_values.max()
        y_min = y_values[y_values > 0].min() if any(y_values > 0) else y_max
        use_log = y_max / y_min > 100  # Use log scale if range is more than 2 orders of magnitude
        
        # Create chart
        chart = alt.Chart(data).mark_bar().encode(
            x=encoding.get('x', 'x'),
            y=alt.Y(
                encoding.get('y', 'y'),
                scale=alt.Scale(type='log') if use_log else alt.Scale(type='linear')
            ),
            tooltip=encoding.get('tooltip', [encoding.get('x', 'x'), encoding.get('y', 'y')])
        ).properties(
            width='container',
            height=300
        )
        
        return chart