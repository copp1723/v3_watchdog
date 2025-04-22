"""
Tests for the Insight Feedback Engine.

This module contains tests for the InsightFeedbackEngine class and its components.
"""

import os
import json
import pytest
from datetime import datetime
from src.insights.feedback_engine import (
    InsightFeedbackEngine,
    FeedbackType,
    UserPersona,
    InsightFeedback,
    UserProfile
)

@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary directory for test data."""
    return str(tmp_path)

@pytest.fixture
def feedback_engine(temp_data_dir):
    """Create a FeedbackEngine instance with temporary data directory."""
    return InsightFeedbackEngine(data_dir=temp_data_dir)

def test_add_feedback(feedback_engine):
    """Test adding feedback."""
    # Add rating feedback
    feedback_id = feedback_engine.add_feedback(
        insight_id="test_insight_1",
        user_id="test_user_1",
        feedback_type=FeedbackType.RATING,
        rating=5
    )
    
    # Verify feedback was added
    feedback = feedback_engine.get_feedback(feedback_id)
    assert feedback is not None
    assert feedback.insight_id == "test_insight_1"
    assert feedback.user_id == "test_user_1"
    assert feedback.feedback_type == FeedbackType.RATING
    assert feedback.rating == 5
    
    # Add thumbs feedback
    feedback_id = feedback_engine.add_feedback(
        insight_id="test_insight_1",
        user_id="test_user_1",
        feedback_type=FeedbackType.THUMBS,
        thumbs_up=True
    )
    
    # Verify feedback was added
    feedback = feedback_engine.get_feedback(feedback_id)
    assert feedback is not None
    assert feedback.feedback_type == FeedbackType.THUMBS
    assert feedback.thumbs_up is True
    
    # Add text feedback
    feedback_id = feedback_engine.add_feedback(
        insight_id="test_insight_1",
        user_id="test_user_1",
        feedback_type=FeedbackType.TEXT,
        text_feedback="This insight was very helpful!"
    )
    
    # Verify feedback was added
    feedback = feedback_engine.get_feedback(feedback_id)
    assert feedback is not None
    assert feedback.feedback_type == FeedbackType.TEXT
    assert feedback.text_feedback == "This insight was very helpful!"
    
    # Add structured feedback
    feedback_id = feedback_engine.add_feedback(
        insight_id="test_insight_1",
        user_id="test_user_1",
        feedback_type=FeedbackType.STRUCTURED,
        structured_feedback={
            "accuracy": "high",
            "relevance": "medium",
            "actionability": "high"
        }
    )
    
    # Verify feedback was added
    feedback = feedback_engine.get_feedback(feedback_id)
    assert feedback is not None
    assert feedback.feedback_type == FeedbackType.STRUCTURED
    assert feedback.structured_feedback == {
        "accuracy": "high",
        "relevance": "medium",
        "actionability": "high"
    }

def test_get_insight_feedback(feedback_engine):
    """Test getting feedback for an insight."""
    # Add feedback for multiple insights
    feedback_engine.add_feedback(
        insight_id="test_insight_1",
        user_id="test_user_1",
        feedback_type=FeedbackType.RATING,
        rating=5
    )
    
    feedback_engine.add_feedback(
        insight_id="test_insight_1",
        user_id="test_user_2",
        feedback_type=FeedbackType.RATING,
        rating=4
    )
    
    feedback_engine.add_feedback(
        insight_id="test_insight_2",
        user_id="test_user_1",
        feedback_type=FeedbackType.RATING,
        rating=3
    )
    
    # Get feedback for test_insight_1
    feedback = feedback_engine.get_insight_feedback("test_insight_1")
    assert len(feedback) == 2
    assert all(f.insight_id == "test_insight_1" for f in feedback)
    
    # Get feedback for test_insight_2
    feedback = feedback_engine.get_insight_feedback("test_insight_2")
    assert len(feedback) == 1
    assert feedback[0].insight_id == "test_insight_2"

def test_get_user_feedback(feedback_engine):
    """Test getting feedback from a user."""
    # Add feedback from multiple users
    feedback_engine.add_feedback(
        insight_id="test_insight_1",
        user_id="test_user_1",
        feedback_type=FeedbackType.RATING,
        rating=5
    )
    
    feedback_engine.add_feedback(
        insight_id="test_insight_2",
        user_id="test_user_1",
        feedback_type=FeedbackType.RATING,
        rating=4
    )
    
    feedback_engine.add_feedback(
        insight_id="test_insight_1",
        user_id="test_user_2",
        feedback_type=FeedbackType.RATING,
        rating=3
    )
    
    # Get feedback from test_user_1
    feedback = feedback_engine.get_user_feedback("test_user_1")
    assert len(feedback) == 2
    assert all(f.user_id == "test_user_1" for f in feedback)
    
    # Get feedback from test_user_2
    feedback = feedback_engine.get_user_feedback("test_user_2")
    assert len(feedback) == 1
    assert feedback[0].user_id == "test_user_2"

def test_add_user_profile(feedback_engine):
    """Test adding and updating user profiles."""
    # Add new user profile
    feedback_engine.add_user_profile(
        user_id="test_user_1",
        name="Test User",
        persona=UserPersona.EXECUTIVE,
        preferences={"theme": "dark"}
    )
    
    # Verify profile was added
    profile = feedback_engine.get_user_profile("test_user_1")
    assert profile is not None
    assert profile.user_id == "test_user_1"
    assert profile.name == "Test User"
    assert profile.persona == UserPersona.EXECUTIVE
    assert profile.preferences == {"theme": "dark"}
    
    # Update existing profile
    feedback_engine.add_user_profile(
        user_id="test_user_1",
        name="Updated Test User",
        persona=UserPersona.MANAGER,
        preferences={"theme": "light"}
    )
    
    # Verify profile was updated
    profile = feedback_engine.get_user_profile("test_user_1")
    assert profile is not None
    assert profile.name == "Updated Test User"
    assert profile.persona == UserPersona.MANAGER
    assert profile.preferences == {"theme": "light"}

def test_record_ab_test_result(feedback_engine):
    """Test recording A/B test results."""
    # Record results for variant A
    feedback_engine.record_ab_test_result(
        test_id="test_1",
        variant="A",
        user_id="test_user_1",
        insight_id="test_insight_1",
        metric="open_rate",
        value=0.75
    )
    
    # Record results for variant B
    feedback_engine.record_ab_test_result(
        test_id="test_1",
        variant="B",
        user_id="test_user_2",
        insight_id="test_insight_1",
        metric="open_rate",
        value=0.85
    )
    
    # Get test results
    results = feedback_engine.get_ab_test_results("test_1")
    assert "variants" in results
    assert "metrics" in results
    assert "A" in results["variants"]
    assert "B" in results["variants"]
    assert "open_rate" in results["metrics"]
    assert "A" in results["metrics"]["open_rate"]
    assert "B" in results["metrics"]["open_rate"]
    
    # Verify variant data
    assert len(results["variants"]["A"]) == 1
    assert results["variants"]["A"][0]["user_id"] == "test_user_1"
    assert len(results["variants"]["B"]) == 1
    assert results["variants"]["B"][0]["user_id"] == "test_user_2"
    
    # Verify metric data
    assert len(results["metrics"]["open_rate"]["A"]) == 1
    assert results["metrics"]["open_rate"]["A"][0]["value"] == 0.75
    assert len(results["metrics"]["open_rate"]["B"]) == 1
    assert results["metrics"]["open_rate"]["B"][0]["value"] == 0.85

def test_get_insight_stats(feedback_engine):
    """Test getting insight statistics."""
    # Add various types of feedback
    feedback_engine.add_feedback(
        insight_id="test_insight_1",
        user_id="test_user_1",
        feedback_type=FeedbackType.RATING,
        rating=5
    )
    
    feedback_engine.add_feedback(
        insight_id="test_insight_1",
        user_id="test_user_2",
        feedback_type=FeedbackType.RATING,
        rating=4
    )
    
    feedback_engine.add_feedback(
        insight_id="test_insight_1",
        user_id="test_user_3",
        feedback_type=FeedbackType.THUMBS,
        thumbs_up=True
    )
    
    feedback_engine.add_feedback(
        insight_id="test_insight_1",
        user_id="test_user_4",
        feedback_type=FeedbackType.THUMBS,
        thumbs_up=False
    )
    
    feedback_engine.add_feedback(
        insight_id="test_insight_1",
        user_id="test_user_5",
        feedback_type=FeedbackType.TEXT,
        text_feedback="Great insight!"
    )
    
    feedback_engine.add_feedback(
        insight_id="test_insight_1",
        user_id="test_user_6",
        feedback_type=FeedbackType.STRUCTURED,
        structured_feedback={"accuracy": "high", "relevance": "medium"}
    )
    
    # Get insight statistics
    stats = feedback_engine.get_insight_stats("test_insight_1")
    
    # Verify total feedback count
    assert stats["total_feedback"] == 6
    
    # Verify rating statistics
    assert "rating_stats" in stats
    assert stats["rating_stats"]["average"] == 4.5
    assert stats["rating_stats"]["count"] == 2
    assert stats["rating_stats"]["distribution"] == {4: 1, 5: 1}
    
    # Verify thumbs statistics
    assert "thumbs_stats" in stats
    assert stats["thumbs_stats"]["up_count"] == 1
    assert stats["thumbs_stats"]["down_count"] == 1
    assert stats["thumbs_stats"]["up_percentage"] == 50.0
    
    # Verify feedback type counts
    assert "feedback_types" in stats
    assert stats["feedback_types"] == {
        "rating": 2,
        "thumbs": 2,
        "text": 1,
        "structured": 1
    }
    
    # Verify text feedback
    assert "text_feedback" in stats
    assert stats["text_feedback"] == ["Great insight!"]
    
    # Verify structured feedback
    assert "structured_feedback" in stats
    assert "accuracy" in stats["structured_feedback"]
    assert "relevance" in stats["structured_feedback"]
    assert stats["structured_feedback"]["accuracy"] == {"high": 1}
    assert stats["structured_feedback"]["relevance"] == {"medium": 1}

def test_get_user_stats(feedback_engine):
    """Test getting user statistics."""
    # Add feedback from a user
    feedback_engine.add_feedback(
        insight_id="test_insight_1",
        user_id="test_user_1",
        feedback_type=FeedbackType.RATING,
        rating=5
    )
    
    feedback_engine.add_feedback(
        insight_id="test_insight_2",
        user_id="test_user_1",
        feedback_type=FeedbackType.THUMBS,
        thumbs_up=True
    )
    
    feedback_engine.add_feedback(
        insight_id="test_insight_3",
        user_id="test_user_1",
        feedback_type=FeedbackType.TEXT,
        text_feedback="Good insights!"
    )
    
    # Get user statistics
    stats = feedback_engine.get_user_stats("test_user_1")
    
    # Verify total feedback count
    assert stats["total_feedback"] == 3
    
    # Verify feedback type counts
    assert "feedback_by_type" in stats
    assert stats["feedback_by_type"] == {
        "rating": 1,
        "thumbs": 1,
        "text": 1
    }
    
    # Verify recent feedback
    assert "recent_feedback" in stats
    assert len(stats["recent_feedback"]) == 3
    
    # Verify insight IDs
    assert "insight_ids" in stats
    assert set(stats["insight_ids"]) == {"test_insight_1", "test_insight_2", "test_insight_3"}

def test_get_persona_based_formatting(feedback_engine):
    """Test getting persona-based formatting preferences."""
    # Add user profiles with different personas
    feedback_engine.add_user_profile(
        user_id="executive_user",
        name="Executive User",
        persona=UserPersona.EXECUTIVE
    )
    
    feedback_engine.add_user_profile(
        user_id="manager_user",
        name="Manager User",
        persona=UserPersona.MANAGER
    )
    
    feedback_engine.add_user_profile(
        user_id="analyst_user",
        name="Analyst User",
        persona=UserPersona.ANALYST
    )
    
    feedback_engine.add_user_profile(
        user_id="operator_user",
        name="Operator User",
        persona=UserPersona.OPERATOR
    )
    
    # Get formatting for each persona
    executive_formatting = feedback_engine.get_persona_based_formatting("executive_user")
    manager_formatting = feedback_engine.get_persona_based_formatting("manager_user")
    analyst_formatting = feedback_engine.get_persona_based_formatting("analyst_user")
    operator_formatting = feedback_engine.get_persona_based_formatting("operator_user")
    
    # Verify executive formatting
    assert executive_formatting["tone"] == "concise"
    assert executive_formatting["verbosity"] == "low"
    assert executive_formatting["detail_level"] == "high-level"
    assert executive_formatting["focus"] == "strategic"
    
    # Verify manager formatting
    assert manager_formatting["tone"] == "balanced"
    assert manager_formatting["verbosity"] == "medium"
    assert manager_formatting["detail_level"] == "balanced"
    assert manager_formatting["focus"] == "operational"
    
    # Verify analyst formatting
    assert analyst_formatting["tone"] == "detailed"
    assert analyst_formatting["verbosity"] == "high"
    assert analyst_formatting["detail_level"] == "detailed"
    assert analyst_formatting["focus"] == "analytical"
    
    # Verify operator formatting
    assert operator_formatting["tone"] == "action-oriented"
    assert operator_formatting["verbosity"] == "medium"
    assert operator_formatting["detail_level"] == "task-focused"
    assert operator_formatting["focus"] == "actionable"
    
    # Test default formatting for unknown user
    default_formatting = feedback_engine.get_persona_based_formatting("unknown_user")
    assert default_formatting["tone"] == "concise"  # Should default to executive

def test_format_insight_for_persona(feedback_engine):
    """Test formatting insights for different personas."""
    # Add user profiles
    feedback_engine.add_user_profile(
        user_id="executive_user",
        name="Executive User",
        persona=UserPersona.EXECUTIVE
    )
    
    feedback_engine.add_user_profile(
        user_id="analyst_user",
        name="Analyst User",
        persona=UserPersona.ANALYST
    )
    
    # Create a test insight
    test_insight = {
        "summary": "This is a detailed summary with multiple paragraphs.\n\n"
                  "This is the second paragraph with more details.\n\n"
                  "This is the third paragraph with even more details.",
        "bullets": [
            "First important point",
            "Second important point",
            "Third important point",
            "Fourth important point",
            "Fifth important point"
        ],
        "charts": [
            {"type": "summary", "data": "..."},
            {"type": "trend", "data": "..."},
            {"type": "comparison", "data": "..."},
            {"type": "distribution", "data": "..."}
        ]
    }
    
    # Format for executive persona
    executive_insight = feedback_engine.format_insight_for_persona(
        test_insight,
        "executive_user"
    )
    
    # Verify executive formatting
    assert len(executive_insight["summary"].split("\n\n")) == 1  # Should be truncated
    assert len(executive_insight["bullets"]) == 3  # Should be limited
    assert len(executive_insight["charts"]) == 1  # Should be limited
    assert executive_insight["formatting_metadata"]["persona"] == "strategic"
    
    # Format for analyst persona
    analyst_insight = feedback_engine.format_insight_for_persona(
        test_insight,
        "analyst_user"
    )
    
    # Verify analyst formatting
    assert len(analyst_insight["summary"].split("\n\n")) == 3  # Should include all paragraphs
    assert len(analyst_insight["bullets"]) == 5  # Should include all bullets
    assert len(analyst_insight["charts"]) == 3  # Should include more charts
    assert analyst_insight["formatting_metadata"]["persona"] == "analytical"

def test_generate_feedback_prompt(feedback_engine):
    """Test generating feedback prompts."""
    # Generate feedback prompt
    prompt = feedback_engine.generate_feedback_prompt("test_insight_1")
    
    # Verify prompt components
    assert "rating_prompt" in prompt
    assert "thumbs_prompt" in prompt
    assert "text_prompt" in prompt
    assert "structured_prompt" in prompt
    assert "structured_options" in prompt
    
    # Verify structured options
    assert "Accuracy" in prompt["structured_options"]
    assert "Relevance" in prompt["structured_options"]
    assert "Actionability" in prompt["structured_options"]
    assert "Clarity" in prompt["structured_options"]
    assert "Completeness" in prompt["structured_options"] 