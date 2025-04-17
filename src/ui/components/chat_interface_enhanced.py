"""
Enhanced chat interface component with improved UI and message handling.
"""

import streamlit as st
import logging
from datetime import datetime
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

def format_message(message: Dict[str, Any]) -> str:
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

def render_chat_interface():
    """Render the enhanced chat interface with improved UI and message handling."""
    st.markdown("<h2 style='color: #00FF88;'>Ask About Your Data</h2>", unsafe_allow_html=True)
    
    # Initialize chat history if not present
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Chat messages container
    chat_container = st.container()
    
    # Display chat history
    with chat_container:
        for message in st.session_state.chat_history:
            st.markdown(format_message(message), unsafe_allow_html=True)
    
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
    
    # Handle user input
    if send_button and user_input:
        # Add user message to history
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_input,
            'timestamp': datetime.now()
        })
        
        # Clear input
        st.session_state.user_input = ""
        
        # Generate AI response (placeholder for now)
        ai_response = {
            'role': 'assistant',
            'content': "I'm analyzing your data to answer your question. This is a placeholder response.",
            'timestamp': datetime.now()
        }
        
        # Add AI response to history
        st.session_state.chat_history.append(ai_response)
        
        # Log interaction
        logger.info(f"User asked: {user_input}")
        
        # Rerun to update UI
        st.rerun()
    
    # Add custom CSS for chat interface
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
            background-color: #00FF88;
            color: #1E1E1E;
        }
        
        .ai-message .message-content {
            background-color: #2D2D2D;
            color: #D4D4D4;
        }
        
        .message-timestamp {
            font-size: 0.8em;
            color: #888888;
            margin-top: 0.5rem;
        }
        
        .user-message .message-timestamp {
            text-align: right;
        }
        
        .ai-message .message-timestamp {
            text-align: left;
        }
        </style>
    """, unsafe_allow_html=True)