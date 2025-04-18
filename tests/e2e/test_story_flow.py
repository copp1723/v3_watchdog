"""
End-to-end tests for story view.
"""

import pytest
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import requests
from src.ui.pages.story_view import story_view

@pytest.fixture
def mock_session_state():
    """Setup mock session state."""
    # Store original session state
    original_state = st.session_state
    
    # Create test data
    sales_data = pd.DataFrame({
        'SalesRepName': ['Alice', 'Bob', 'Charlie'] * 3,
        'TotalGross': [1000, 2000, 1500, 2000, 1000, 1500, 1000, 2000, 1500],
        'SaleDate': pd.date_range(start='2023-01-01', periods=9, freq='D'),
        'LeadSource': ['Web', 'Phone', 'Walk-in'] * 3,
        'LeadStatus': ['Sold', 'Open', 'Sold', 'Lost', 'Sold', 'Open', 'Sold', 'Lost', 'Sold'],
        'DaysInStock': [15, 25, 45, 55, 75, 85, 95, 105, 115]
    })
    
    # Create mock state
    mock_state = {
        'validated_data': sales_data,
        'validation_summary': {
            'total_records': len(sales_data),
            'quality_score': 95
        },
        'is_authenticated': True,
        'llm_client': MockLLMClient()
    }
    
    # Replace session state
    st.session_state = mock_state
    
    yield mock_state
    
    # Restore original state
    st.session_state = original_state

class MockLLMClient:
    """Mock LLM client for testing."""
    def generate(self, prompt):
        return """Q3 2023 showed strong performance in gross margins, driven by Alice's exceptional sales team performance (+15% vs. target). However, inventory aging metrics reveal an opportunity: 35% of inventory is over 90 days old.

Next Steps:
1. Consider targeted promotions for aged inventory
2. Evaluate pricing strategy for aging vehicles"""

@pytest.mark.usefixtures("streamlit_server")
def test_story_view_rendering(mock_session_state):
    """Test that the story view renders without errors."""
    try:
        # Verify server is running
        response = requests.get("http://localhost:8501/_stcore/health")
        assert response.status_code == 200
        
        # Render the story view
        story_view()
        
        # Check that key elements are present
        assert 'validated_data' in st.session_state
        df = st.session_state.validated_data
        assert len(df) == 9
        
        # Check for story text
        assert any(
            element.type == "markdown" and "performance" in element.body
            for element in st.elements
        )
        
    except Exception as e:
        pytest.fail(f"Story view rendering failed: {str(e)}")

def test_story_view_filters(mock_session_state):
    """Test story view filters."""
    try:
        # Set filter values
        st.session_state["date_filter"] = "Last 7 Days"
        
        # Select insights
        st.session_state["selected_insights"] = [
            "sales_performance",
            "inventory_anomalies"
        ]
        
        # Render story view
        story_view()
        
        # Check for filtered story content
        assert any(
            element.type == "markdown" and "Last 7 days" in element.body.lower()
            for element in st.elements
        )
        
    except Exception as e:
        pytest.fail(f"Filter testing failed: {str(e)}")

def test_story_view_feedback(mock_session_state):
    """Test feedback functionality."""
    try:
        # Render story view
        story_view()
        
        # Check for feedback buttons
        assert any(
            element.type == "button" and "Yes" in element.label
            for element in st.elements
        )
        assert any(
            element.type == "button" and "No" in element.label
            for element in st.elements
        )
        
        # Simulate feedback click
        for element in st.elements:
            if element.type == "button" and "Yes" in element.label:
                element.click()
                break
        
        # Check for success message
        assert any(
            element.type == "success" and "Thanks" in element.body
            for element in st.elements
        )
        
    except Exception as e:
        pytest.fail(f"Feedback testing failed: {str(e)}")

def test_story_view_authentication(mock_session_state):
    """Test authentication requirements."""
    try:
        # Remove authentication
        st.session_state['is_authenticated'] = False
        
        # Render story view
        story_view()
        
        # Should show warning
        assert any(
            element.type == "warning" and "Please log in" in element.body
            for element in st.elements
        )
        
    except Exception as e:
        pytest.fail(f"Authentication testing failed: {str(e)}")

def test_story_view_no_data(mock_session_state):
    """Test story view with no data."""
    try:
        # Remove data
        st.session_state.pop('validated_data', None)
        
        # Render story view
        story_view()
        
        # Should show warning
        assert any(
            element.type == "warning" and "Please upload data" in element.body
            for element in st.elements
        )
        
    except Exception as e:
        pytest.fail(f"No data testing failed: {str(e)}")

def test_story_view_insight_selection(mock_session_state):
    """Test insight selection behavior."""
    try:
        # Render with no insights selected
        st.session_state["selected_insights"] = []
        story_view()
        
        # Should show warning
        assert any(
            element.type == "warning" and "select at least one insight" in element.body.lower()
            for element in st.elements
        )
        
        # Select insights and re-render
        st.session_state["selected_insights"] = ["sales_performance"]
        story_view()
        
        # Should show story
        assert any(
            element.type == "markdown" and "performance" in element.body.lower()
            for element in st.elements
        )
        
    except Exception as e:
        pytest.fail(f"Insight selection testing failed: {str(e)}")