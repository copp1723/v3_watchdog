"""
Unit tests for feedback-driven prompt tuning.
"""

import pytest
from datetime import datetime, timedelta
from src.insights.prompt_tuner import PromptTuner, TuningConfig
from src.insights.models import FeedbackEntry

@pytest.fixture
def sample_template():
    """Create a sample prompt template."""
    return """
    Analyze the dealership data focusing on gross_profit, deal_count, and lead_source.
    Respond in JSON format with clear insights and recommendations.
    """

@pytest.fixture
def sample_feedback():
    """Create sample feedback entries."""
    now = datetime.now()
    return [
        FeedbackEntry(
            insight_id="sales_performance",
            feedback_type="helpful",
            user_id="user1",
            session_id="session1",
            timestamp=now - timedelta(days=1),
            metadata={
                "metrics_used": ["gross_profit", "deal_count"],
                "format_used": "bullet_points",
                "confidence": 0.9
            }
        ),
        FeedbackEntry(
            insight_id="sales_performance",
            feedback_type="not_helpful",
            user_id="user2",
            session_id="session2",
            timestamp=now - timedelta(days=2),
            metadata={
                "metrics_used": ["days_on_lot"],
                "format_used": "paragraphs",
                "confidence": 0.6
            }
        ),
        # Add more feedback entries to meet minimum threshold
        *[
            FeedbackEntry(
                insight_id="sales_performance",
                feedback_type="helpful",
                user_id=f"user{i}",
                session_id=f"session{i}",
                timestamp=now - timedelta(days=i),
                metadata={
                    "metrics_used": ["gross_profit"],
                    "format_used": "bullet_points",
                    "confidence": 0.85
                }
            )
            for i in range(3, 12)
        ]
    ]

def test_prompt_tuner_initialization():
    """Test initialization of prompt tuner."""
    config = TuningConfig(min_feedback_count=5)
    tuner = PromptTuner(config)
    
    assert tuner.config.min_feedback_count == 5
    assert tuner.last_tune is None
    assert len(tuner.feedback_history) == 0

def test_feedback_filtering(sample_feedback):
    """Test filtering of old feedback."""
    tuner = PromptTuner(
        TuningConfig(feedback_window_days=1)
    )
    
    recent = tuner._filter_recent_feedback(sample_feedback)
    assert len(recent) < len(sample_feedback)
    assert all(
        (datetime.now() - f.timestamp).days <= 1
        for f in recent
    )

def test_weight_updates(sample_feedback):
    """Test metric and format weight updates."""
    tuner = PromptTuner()
    
    # Initial weights should be 1.0
    assert tuner.metric_weights["gross_profit"] == 1.0
    assert tuner.format_weights["bullet_points"] == 1.0
    
    # Update weights
    tuner._update_weights(sample_feedback)
    
    # Gross profit should be weighted higher (mentioned in helpful feedback)
    assert tuner.metric_weights["gross_profit"] > 1.0
    # Days on lot should be weighted lower (mentioned in unhelpful feedback)
    assert tuner.metric_weights["days_on_lot"] < 1.0
    # Bullet points should be weighted higher (preferred format)
    assert tuner.format_weights["bullet_points"] > 1.0

def test_emphasis_adjustment(sample_template, sample_feedback):
    """Test emphasis adjustment in templates."""
    tuner = PromptTuner()
    tuner._update_weights(sample_feedback)
    
    result = tuner._adjust_emphasis(sample_template)
    
    # Important metrics should be emphasized
    assert "**gross_profit**" in result
    # Less important metrics should not be emphasized
    assert "**days_on_lot**" not in result

def test_context_enhancement(sample_template, sample_feedback):
    """Test context enhancement in templates."""
    tuner = PromptTuner()
    tuner._update_weights(sample_feedback)
    
    result = tuner._enhance_context(sample_template)
    
    # Should mention important metrics
    assert "key metrics" in result
    assert "gross_profit" in result

def test_format_optimization(sample_template, sample_feedback):
    """Test format optimization in templates."""
    tuner = PromptTuner()
    tuner._update_weights(sample_feedback)
    
    result = tuner._optimize_format(sample_template)
    
    # Should include preferred format
    assert "bullet points" in result.lower()

def test_confidence_calibration(sample_template, sample_feedback):
    """Test confidence calibration in templates."""
    tuner = PromptTuner()
    tuner.feedback_history = sample_feedback
    
    result = tuner._calibrate_confidence(sample_template)
    
    # Should include confidence threshold
    assert "confidence level" in result.lower()

def test_full_tuning_pipeline(sample_template, sample_feedback):
    """Test the complete tuning pipeline."""
    tuner = PromptTuner()
    
    result = tuner.tune_prompt(sample_template, sample_feedback)
    
    # Check all tuning effects
    assert "**gross_profit**" in result  # Emphasis
    assert "key metrics" in result  # Context
    assert "bullet points" in result.lower()  # Format
    assert "confidence level" in result.lower()  # Confidence

def test_insufficient_feedback():
    """Test handling of insufficient feedback."""
    tuner = PromptTuner(TuningConfig(min_feedback_count=10))
    template = "Original template"
    feedback = [
        FeedbackEntry(
            insight_id="test",
            feedback_type="helpful",
            user_id="user1",
            session_id="session1",
            timestamp=datetime.now(),
            metadata={}
        )
    ]
    
    # Should return original template unchanged
    result = tuner.tune_prompt(template, feedback)
    assert result == template

def test_error_handling():
    """Test error handling in tuning process."""
    tuner = PromptTuner()
    
    # Create invalid feedback entry
    invalid_feedback = [
        FeedbackEntry(
            insight_id="test",
            feedback_type="invalid_type",  # Invalid feedback type
            user_id="user1",
            session_id="session1",
            timestamp=datetime.now(),
            metadata=None  # Missing metadata
        )
    ]
    
    # Should handle error gracefully and return original template
    template = "Original template"
    result = tuner.tune_prompt(template, invalid_feedback)
    assert result == template

def test_weight_normalization():
    """Test weight normalization logic."""
    tuner = PromptTuner(
        TuningConfig(max_emphasis_boost=2.0)
    )
    
    # Create weights that exceed max boost
    weights = {
        "metric1": 3.0,
        "metric2": 1.5,
        "metric3": 1.0
    }
    
    tuner._normalize_weights(weights)
    
    # Check that weights were scaled properly
    assert max(weights.values()) == 2.0
    assert weights["metric1"] == 2.0  # Was 3.0
    assert 1.0 <= weights["metric2"] <= 1.5  # Should be scaled proportionally
    assert weights["metric3"] < 1.0  # Should be scaled proportionally

def test_sentry_integration(sample_template, sample_feedback):
    """Test Sentry tag integration."""
    import sentry_sdk
    
    tuner = PromptTuner()
    tuner.tune_prompt(sample_template, sample_feedback)
    
    # Verify Sentry tags were set
    assert sentry_sdk.get_tag("prompt_tuner") == "active"
    assert sentry_sdk.get_tag("prompt_tuned") == "true"
    assert int(sentry_sdk.get_tag("feedback_count")) == len(sample_feedback)