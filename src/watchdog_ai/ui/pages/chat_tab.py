"""
Chat tab page.
"""

import streamlit as st
from ..components.chat_interface import ChatInterface

def render():
    """Render the chat tab."""
    interface = ChatInterface()
    interface.render()