"""
End-to-end tests for feedback-driven story tuning.
"""

import pytest
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
from src.ui.pages.story_view import story_view, generate_story
from src.insights.models import FeedbackEntry

@pytest.fixture
def mock_feedback():
    """Create mock feedback entries."""
    return [
        FeedbackEntry(
            id="1",
            insight_id="sales_performance",
            feedback_type="helpful",
            timestamp=datetime.now() - timedelta(days=1),
            metadata={
                "metrics_used": ["gross_profit", "deal_count"],
                "format_used": "bullet_points",
                "confidence": 0.85
            }
        ),
        FeedbackEntry(
            id="2",
            insight_id="inventory_anomalies",
            feedback_type="not_helpful",
            timestamp=datetime.now() - timedelta(days=2),
            metadata={
                "metrics_used": ["days_on_lot"],
                "format_used": "paragraphs",
                "confidence": 0.65,
                "too_many_alerts": True
            }
        )
    ]

@pytest.fixture
def mock_session_state(mock_feedback):
    """Setup mock session state with feedback."""
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
    
    class MockFeedbackManager:
        def get_feedback(self, **kwargs):
            return mock_feedback
    
    # Create mock state
    mock_state = {
        'validated_data': sales_data,
        'validation_summary': {
            'total_records': len(sales_data),
            'quality_score': 95
        },
        'is_authenticated': True,
        'llm_client': MockLLMClient(),
        'feedback_manager': MockFeedbackManager()
    }
    
    # Replace session state
    st.session_state = mock_state
    
    yield mock_state
    
    # Restore original state
    st.session_state = original_state

class MockLLMClient:
    """Mock LLM client for testing."""
    def generate(self, prompt):
        # Check if prompt has been tuned
        emphasis_count = prompt.count("**gross_profit**")
        bullet_point_preference = "Use bullet points" in prompt
        
        if emphasis_count > 0 and bullet_point_preference:
            # Tuned prompt detected
            return """Strong Q3 2023 performance driven by:
• Gross profit up 15% month-over-month
• Deal count increased by 10%
• Inventory aging improved with 20% reduction in aged units

Next Steps:
1. Leverage top-performing sales channels
2. Optimize pricing for aging inventory"""
        else:
            # Default response
            return """Q3 2023 showed mixed performance with some improvements in sales metrics. Inventory aging metrics reveal opportunities for optimization.

Next Steps:
1. Review sales processes
2. Monitor inventory aging"""

def test_feedback_driven_tuning(mock_session_state, mock_feedback):
    """Test that feedback influences prompt tuning."""
    try:
        # Generate story with feedback
        story = generate_story(
            ["sales_performance", "inventory_anomalies"],
            mock_session_state['validated_data'],
            date_range="2023 Q3"
        )
        
        # Verify tuning effects
        assert "bullet points" in story.lower()  # Preferred format applied
        assert "gross profit" in story.lower()  # Emphasized metric
        assert story.count("•") >= 3  # Bullet points used
        
    except Exception as e:
        pytest.fail(f"Feedback-driven tuning test failed: {str(e)}")

def test_threshold_adaptation(mock_session_state, mock_feedback):
    """Test that thresholds adapt based on feedback."""
    try:
        # Generate story
        story = generate_story(
            ["inventory_anomalies"],
            mock_session_state['validated_data'],
            date_range="2023 Q3"
        )
        
        # Verify threshold adaptation
        assert "aged units" in story.lower()  # Adapted threshold reflected
        assert "reduction" in story.lower()  # Improvement noted
        
    except Exception as e:
        pytest.fail(f"Threshold adaptation test failed: {str(e)}")

def test_format_preference_learning(mock_session_state, mock_feedback):
    """Test that format preferences are learned."""
    try:
        # Generate story
        story = generate_story(
            ["sales_performance"],
            mock_session_state['validated_data'],
            date_range="2023 Q3"
        )
        
        # Count bullet points
        bullet_points = len([line for line in story.split('\n') if line.strip().startswith('•')])
        
        # Verify format preference applied
        assert bullet_points >= 3  # Multiple bullet points used
        assert "Next Steps:" in story  # Section headers maintained
        
    except Exception as e:
        pytest.fail(f"Format preference learning test failed: {str(e)}")

def test_metric_emphasis_learning(mock_session_state, mock_feedback):
    """Test that metric emphasis is learned."""
    try:
        # Generate story
        story = generate_story(
            ["sales_performance", "inventory_anomalies"],
            mock_session_state['validated_data'],
            date_range="2023 Q3"
        )
        
        # Verify metric emphasis
        assert "gross profit" in story.lower()  # Emphasized metric present
        assert "up 15%" in story  # Specific metric detail included
        
    except Exception as e:
        pytest.fail(f"Metric emphasis learning test failed: {str(e)}")