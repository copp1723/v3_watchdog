"""
Consolidated chat interface component for Watchdog AI.

Provides UI components for chat-based interaction with improved styling and features.
"""

import streamlit as st
import pandas as pd
import altair as alt
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from src.insight_conversation_consolidated import ConversationManager
from src.insight_card_consolidated import render_insight_card

# Configure logging
logger = logging.getLogger(__name__)

class ChatInterface:
    """Handles chat-based interaction and insight generation."""
    
    def __init__(self):
        """Initialize the chat interface."""
        self._initialize_session_state()
        self.conversation_manager = ConversationManager()
        
        # Add custom CSS for chat interface
        self._add_custom_css()
    
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
    
    def _add_custom_css(self):
        """Add custom CSS for chat interface styling."""
        st.markdown("""
        <style>
        .message {
            margin: 1rem 0;
            display: flex;
            flex-direction: column;
        }
        
        .user-message {
            align-items: flex-end;
        }
        
        .ai-message {
            align-items: flex-start;
        }
        
        .message-content {
            max-width: 80%;
            padding: 1rem;
            border-radius: 1rem;
            position: relative;
        }
        
        .user-message .message-content {
            background-color: #264653;
            color: #FFFFFF;
        }
        
        .ai-message .message-content {
            background-color: #2A9D8F;
            color: #FFFFFF;
        }
        
        .message-timestamp {
            font-size: 0.8em;
            color: #888888;
            margin-top: 0.5rem;
        }
        
        /* Styling for insight cards */
        .insight-card {
            background-color: #2D2D2D;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
            border-left: 4px solid #2A9D8F;
        }
        
        .insight-header {
            font-weight: bold;
            margin-bottom: 10px;
            color: #E9C46A;
        }
        
        .insight-body {
            color: #F4F4F4;
            margin-bottom: 10px;
        }
        
        .insight-actions {
            border-top: 1px solid #444;
            padding-top: 10px;
            color: #E76F51;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def format_message(self, message: Dict[str, Any]) -> str:
        """
        Format a message for display in the chat interface.
        
        Args:
            message: Dictionary containing message data
            
        Returns:
            str: Formatted message HTML
        """
        role = message.get('role', 'user')
        content = message.get('content', '')
        timestamp = message.get('timestamp', datetime.now()).strftime('%H:%M')
        
        if role == 'user':
            return f"""
                <div class="message user-message">
                    <div class="message-content">
                        <div class="message-text">{content}</div>
                        <div class="message-timestamp">{timestamp}</div>
                    </div>
                </div>
            """
        else:
            return f"""
                <div class="message ai-message">
                    <div class="message-content">
                        <div class="message-text">{content}</div>
                        <div class="message-timestamp">{timestamp}</div>
                    </div>
                </div>
            """
    
    def render_chat_interface(self, use_streamlit_chat: bool = True) -> None:
        """
        Render the chat interface.
        
        Args:
            use_streamlit_chat: Whether to use Streamlit's native chat components or custom HTML
        """
        st.markdown("<h3>Chat Analysis</h3>", unsafe_allow_html=True)
        
        # Process regeneration request if present
        if st.session_state.get('regenerate_insight', False):
            index = st.session_state.get('regenerate_index', -1)
            with st.spinner("Regenerating insight..."):
                response = self.conversation_manager.regenerate_insight(index)
            st.session_state.regenerate_insight = False
            st.session_state.regenerate_index = None
            st.rerun()
        
        # Chat history display - choose style based on parameter
        if use_streamlit_chat:
            self._render_streamlit_chat()
        else:
            self._render_custom_chat()
        
        # Clear chat button
        if st.session_state.chat_history:
            if st.button("Clear Chat", key="clear_chat"):
                self._clear_state()
    
    def _render_streamlit_chat(self):
        """Render chat history using Streamlit's native chat components."""
        # Display chat history
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
            self._process_user_input(prompt)
    
    def _render_custom_chat(self):
        """Render chat history using custom HTML/CSS styling."""
        # Chat messages container
        chat_container = st.container()
        
        # Display chat history
        with chat_container:
            for entry in st.session_state.chat_history:
                # Format and display user message
                user_msg = {
                    'role': 'user',
                    'content': entry['prompt'],
                    'timestamp': entry.get('timestamp', datetime.now().isoformat())
                }
                st.markdown(self.format_message(user_msg), unsafe_allow_html=True)
                
                # Display AI response as an insight card
                with st.container():
                    render_insight_card(entry['response'])
        
        # Input section
        with st.container():
            col1, col2 = st.columns([6, 1])
            
            with col1:
                user_input = st.text_input(
                    "Type your question here...",
                    key="user_input",
                    help="Ask questions about your data, e.g., 'How many cars were sold from CarGurus?'"
                )
            
            with col2:
                send_button = st.button("Send", key="send_button")
        
        # Process input
        if send_button and user_input:
            self._process_user_input(user_input)
    
    def _process_user_input(self, prompt: str) -> None:
        """
        Process user input and generate a response.
        
        Args:
            prompt: User's input prompt
        """
        # Add user message to chat UI for immediate feedback
        if st.session_state.get('_display_method', '') == 'streamlit_chat':
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
                
                # Display assistant response for immediate feedback in streamlit chat mode
                if st.session_state.get('_display_method', '') == 'streamlit_chat':
                    with st.chat_message("assistant"):
                        render_insight_card(response)
                
                # Store in history
                st.session_state.chat_history.append({
                    'prompt': prompt,
                    'response': response,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Clear input field in custom mode
                if st.session_state.get('_display_method', '') != 'streamlit_chat':
                    st.session_state.user_input = ""
                    st.rerun()
                
            except Exception as e:
                logger.error(f"Error generating insight: {str(e)}")
                st.error(f"Error generating insight: {str(e)}")
    
    def _create_chart(self, data: pd.DataFrame, encoding: Dict[str, str]) -> alt.Chart:
        """
        Create an Altair chart based on data and encoding.
        
        Args:
            data: DataFrame to visualize
            encoding: Mapping of visual channels to data fields
            
        Returns:
            Altair chart object or None
        """
        if len(data) <= 2:
            # For 2 or fewer points, a table is better
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