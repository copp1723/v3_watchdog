"""
Unit tests for story generation.
"""

import pytest
import pandas as pd
from datetime import datetime
from src.ui.pages.story_view import generate_story

@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    class MockLLMClient:
        def generate(self, prompt):
            return """Q3 2023 showed strong performance in gross margins, driven by Alice's exceptional sales team performance (+15% vs. target). However, inventory aging metrics reveal an opportunity: 35% of inventory is over 90 days old.

Next Steps:
1. Consider targeted promotions for aged inventory
2. Evaluate pricing strategy for aging vehicles"""
    return MockLLMClient()

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame({
        'SaleDate': pd.date_range(start='2023-01-01', periods=100, freq='D'),
        'SalesRepName': ['Alice', 'Bob'] * 50,
        'TotalGross': range(1000, 101000, 1000),
        'LeadSource': ['Web', 'Phone'] * 50,
        'DaysInStock': [15, 25, 45, 55, 75, 85, 95, 105, 115] * 11 + [15]
    })

def test_story_generation(mock_llm_client, sample_data):
    """Test basic story generation."""
    # Setup session state
    import streamlit as st
    st.session_state.llm_client = mock_llm_client
    
    story = generate_story(
        ["sales_performance", "inventory_anomalies"],
        sample_data,
        date_range="2023 Q3"
    )
    
    assert isinstance(story, str)
    assert len(story) > 0
    assert "performance" in story.lower()
    assert "Next Steps" in story

def test_story_with_feedback_stats(mock_llm_client, sample_data):
    """Test story generation with feedback stats."""
    import streamlit as st
    st.session_state.llm_client = mock_llm_client
    
    feedback_stats = {
        "total_feedback": 10,
        "helpful_percentage": 80
    }
    
    story = generate_story(
        ["sales_performance"],
        sample_data,
        feedback_stats=feedback_stats,
        date_range="2023 Q3"
    )
    
    assert isinstance(story, str)
    assert len(story) > 0

def test_story_with_no_insights():
    """Test story generation with no insights selected."""
    import streamlit as st
    st.session_state.llm_client = mock_llm_client
    
    story = generate_story(
        [],
        pd.DataFrame(),
        date_range="2023 Q3"
    )
    
    assert "No insights available" in story

def test_story_error_handling(mock_llm_client, sample_data):
    """Test error handling in story generation."""
    import streamlit as st
    st.session_state.llm_client = mock_llm_client
    
    # Create invalid data
    invalid_data = pd.DataFrame({'invalid': ['data']})
    
    story = generate_story(
        ["sales_performance"],
        invalid_data,
        date_range="2023 Q3"
    )
    
    assert "Error" in story

def test_multiple_insights_integration(mock_llm_client, sample_data):
    """Test story generation with multiple insights."""
    import streamlit as st
    st.session_state.llm_client = mock_llm_client
    
    story = generate_story(
        [
            "sales_performance",
            "inventory_anomalies",
            "lead_conversion_rate"
        ],
        sample_data,
        date_range="2023 Q3"
    )
    
    assert isinstance(story, str)
    assert len(story) > 0
    assert "Next Steps" in story