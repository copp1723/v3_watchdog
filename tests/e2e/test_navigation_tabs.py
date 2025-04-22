"""
End-to-end tests for navigation tabs.
"""

import pytest
import streamlit as st
from src.watchdog_ai.ui.components.chat_interface import ChatInterface

def test_chat_interface():
    """Test chat interface initialization."""
    interface = ChatInterface()
    assert interface is not None