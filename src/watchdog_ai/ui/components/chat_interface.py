"""
Chat interface component.
"""

import streamlit as st
from ...insights.insight_conversation import ConversationManager

class ChatInterface:
    """Chat interface component."""
    
    def __init__(self):
        """Initialize the chat interface."""
        self.conversation_manager = ConversationManager()
    
    def render(self):
        """Render the chat interface."""
        st.title("Chat Interface")