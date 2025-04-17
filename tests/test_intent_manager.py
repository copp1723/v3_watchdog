"""
Tests for the intent-based insight generation system.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.insights.intent_manager import IntentManager
from src.insights.intents import (
    TopMetricIntent,
    BottomMetricIntent,
    CountMetricIntent,
    HighestCountIntent
)

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame({
        'SalesRepName': ['Alice', 'Bob', 'Alice', 'Charlie', 'Bob'],
        'Total_Gross': [1000, -500, 1500, -750, 2000],
        'DealCount': [1, 1, 1, 1, 1],
        'LeadSource': ['Web', 'Direct', 'Web', 'Social', 'Web']
    })

@pytest.fixture
def intent_manager():
    """Create an IntentManager instance for testing."""
    return IntentManager()

def test_intent_registration(intent_manager):
    """Test that core intents are registered correctly."""
    assert len(intent_manager.intents) == 4
    assert any(isinstance(i, TopMetricIntent) for i in intent_manager.intents)
    assert any(isinstance(i, BottomMetricIntent) for i in intent_manager.intents)
    assert any(isinstance(i, CountMetricIntent) for i in intent_manager.intents)
    assert any(isinstance(i, HighestCountIntent) for i in intent_manager.intents)

def test_find_matching_intent_count(intent_manager):
    """Test matching count-based queries."""
    prompt = "how many deals had negative profit?"
    intent = intent_manager.find_matching_intent(prompt)
    assert isinstance(intent, CountMetricIntent)

def test_find_matching_intent_highest_count(intent_manager):
    """Test matching highest count queries."""
    prompt = "which lead source sold the most vehicles?"
    intent = intent_manager.find_matching_intent(prompt)
    assert isinstance(intent, HighestCountIntent)

def test_find_matching_intent_top_metric(intent_manager):
    """Test matching top metric queries."""
    prompt = "which rep has the highest gross profit?"
    intent = intent_manager.find_matching_intent(prompt)
    assert isinstance(intent, TopMetricIntent)

def test_find_matching_intent_no_match(intent_manager):
    """Test handling queries with no matching intent."""
    prompt = "tell me about the weather"
    intent = intent_manager.find_matching_intent(prompt)
    assert intent is None

def test_generate_insight_count_negative_profit(intent_manager, sample_data):
    """Test generating insight for counting negative profits."""
    prompt = "how many deals had negative profit?"
    result = intent_manager.generate_insight(prompt, sample_data)
    
    assert result["title"] == "Negative Profit Analysis"
    assert "2 deals" in result["summary"]
    assert "-$1,250" in result["summary"]
    assert result["confidence"] == "high"
    assert result["is_direct_calculation"] is True
    assert result["chart_data"] is not None

def test_generate_insight_highest_count_by_source(intent_manager, sample_data):
    """Test generating insight for highest count by lead source."""
    prompt = "which lead source sold the most vehicles?"
    result = intent_manager.generate_insight(prompt, sample_data)
    
    assert "Web" in result["summary"]
    assert "3 deals" in result["summary"]
    assert result["confidence"] == "high"
    assert result["is_direct_calculation"] is True
    assert result["chart_data"] is not None

def test_generate_insight_top_metric_by_rep(intent_manager, sample_data):
    """Test generating insight for top metric by sales rep."""
    prompt = "which rep has the highest gross profit?"
    result = intent_manager.generate_insight(prompt, sample_data)
    
    assert "Bob" in result["summary"]
    assert "$1,500" in result["summary"]
    assert result["confidence"] == "high"
    assert result["is_direct_calculation"] is True
    assert result["chart_data"] is not None

def test_generate_insight_fallback(intent_manager, sample_data):
    """Test fallback response for unsupported queries."""
    prompt = "tell me about the weather"
    result = intent_manager.generate_insight(prompt, sample_data)
    
    assert result["title"] == "I'm Not Sure About That"
    assert "examples" in result["summary"].lower()
    assert len(result["value_insights"]) > 0  # Should have example queries
    assert result["confidence"] == "low"
    assert result["is_direct_calculation"] is True

def test_generate_insight_with_empty_data(intent_manager):
    """Test handling empty DataFrame."""
    empty_df = pd.DataFrame()
    prompt = "how many deals had negative profit?"
    result = intent_manager.generate_insight(prompt, empty_df)
    
    assert result["is_error"] is True
    assert "error" in result

def test_generate_insight_with_missing_columns(intent_manager, sample_data):
    """Test handling missing required columns."""
    df_no_gross = sample_data.drop(columns=['Total_Gross'])
    prompt = "how many deals had negative profit?"
    result = intent_manager.generate_insight(prompt, df_no_gross)
    
    assert result["is_error"] is True
    assert "missing" in result["error"].lower()

def test_performance_with_large_dataset(intent_manager):
    """Test performance with 100k rows."""
    import time
    
    # Create large dataset
    large_df = pd.DataFrame({
        'SalesRepName': np.random.choice(['Alice', 'Bob', 'Charlie'], 100000),
        'Total_Gross': np.random.normal(1500, 500, 100000),
        'DealCount': np.ones(100000),
        'LeadSource': np.random.choice(['Web', 'Direct', 'Social'], 100000)
    })
    
    # Test insight generation performance
    start_time = time.time()
    result = intent_manager.generate_insight(
        "which rep has the highest gross profit?",
        large_df
    )
    end_time = time.time()
    
    # Should complete in under 200ms
    assert (end_time - start_time) < 0.2
    assert not result.get("is_error", False)
    assert result["confidence"] == "high"