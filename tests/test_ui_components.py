"""
Tests for UI components integration.
"""

import unittest
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.ui.components.data_upload_enhanced import DataUploadManager
from src.ui.components.chat_interface import ChatInterface

class TestUIComponents(unittest.TestCase):
    """Test cases for UI components integration."""
    
    def setUp(self):
        """Set up test environment."""
        # Initialize session state
        if 'validated_data' not in st.session_state:
            st.session_state.validated_data = None
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
        if 'last_insight' not in st.session_state:
            st.session_state.last_insight = None
        
        # Create sample data
        self.sample_df = pd.DataFrame({
            'Gross_Profit': [1000, -500, -750, 2000, 1500],
            'Lead_Source': ['Web', 'Phone', 'Walk-in', 'Web', 'Phone'],
            'Sale_Price': [25000, 22000, 20000, 30000, 28000],
            'Sale_Date': pd.date_range(start='2024-01-01', periods=5)
        })
    
    def test_data_upload_manager(self):
        """Test DataUploadManager component."""
        upload_manager = DataUploadManager()
        validation_result = upload_manager._validate_dataframe(self.sample_df)
        
        self.assertTrue(validation_result['is_valid'])
        self.assertEqual(len(validation_result['errors']), 0)
    
    def test_chat_interface(self):
        """Test ChatInterface component."""
        # Set validated data in session state
        st.session_state.validated_data = self.sample_df
        
        # Initialize chat interface
        chat_interface = ChatInterface()
        
        # Test clear functionality
        st.session_state.chat_history = [{"prompt": "test", "response": {"summary": "test"}}]
        st.session_state.current_insight = {"summary": "test"}
        st.session_state.debug_info = {"test": "data"}
        st.session_state.insight_prompt = "test question"
        
        # Clear state
        chat_interface._clear_state()
        
        # Verify state is cleared
        self.assertEqual(len(st.session_state.chat_history), 0)
        self.assertIsNone(st.session_state.current_insight)
        self.assertIsNone(st.session_state.debug_info)
        self.assertIsNone(st.session_state.current_prompt)
        self.assertEqual(st.session_state.insight_prompt, "")
    
    def test_chart_creation(self):
        """Test chart creation with different data scenarios."""
        chat_interface = ChatInterface()
        
        # Test with 2 data points (should use table view)
        small_data = pd.DataFrame({
            'Category': ['A', 'B'],
            'Value': [100, 200]
        })
        chart = chat_interface._create_chart(
            small_data,
            {'x': 'Category', 'y': 'Value'}
        )
        self.assertIsNotNone(chart)
        
        # Test with large value differences (should use log scale)
        varied_data = pd.DataFrame({
            'Category': ['A', 'B', 'C'],
            'Value': [1, 100, 10000]
        })
        chart = chat_interface._create_chart(
            varied_data,
            {'x': 'Category', 'y': 'Value'}
        )
        self.assertIsNotNone(chart)
    
    def test_negative_profit_insight(self):
        """Test generating insight about negative profits."""
        # Set up chat interface with sample data
        st.session_state.dataframe = self.sample_df
        chat_interface = ChatInterface()
        
        # Simulate question about negative profits
        response = chat_interface.conversation_manager.generate_insight(
            prompt="how many negative profit sales were there?",
            validation_context={"df": self.sample_df}
        )
        
        # Verify response
        self.assertIn("title", response)
        self.assertEqual(response["title"], "Negative Profit Analysis")
        self.assertIn("2", response["summary"])  # Should find 2 negative profits
        self.assertTrue(any("$1,250" in insight for insight in response["value_insights"]))  # Total loss
        
        # Verify chart data
        self.assertIn("chart_data", response)
        self.assertIn("chart_encoding", response)
        chart_data = response["chart_data"]
        self.assertEqual(len(chart_data), 2)  # Should have 2 categories (negative vs other)

if __name__ == '__main__':
    unittest.main()