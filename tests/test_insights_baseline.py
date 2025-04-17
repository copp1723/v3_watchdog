"""
Baseline tests for core insight scenarios.
Ensures critical functionality works before adding new features.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from src.insights.intents import (
    TopMetricIntent,
    BottomMetricIntent,
    CountMetricIntent,
    HighestCountIntent
)
from src.insights.models import InsightResult

@pytest.fixture
def sample_data():
    """Create minimal test dataset with core metrics."""
    return pd.DataFrame({
        'SalesRepName': ['Alice', 'Bob', 'Alice', 'Charlie', 'Bob'],
        'Total_Gross': [1000, -500, 1500, -750, 2000],
        'DealCount': [1, 1, 1, 1, 1],
        'LeadSource': ['Web', 'Direct', 'Web', 'Social', 'Web']
    })

def test_highest_gross_by_rep(sample_data):
    """T-01: Test highest gross profit by sales rep."""
    intent = TopMetricIntent()
    result = intent.analyze(sample_data, "Who has the highest gross by rep?")
    
    assert result.title == "Top Rep by Gross"
    assert "Bob" in result.summary  # Bob has highest total gross (1500)
    assert "$1,500" in result.summary
    assert result.confidence == "high"
    assert result.error is None
    assert result.chart_data is not None
    assert len(result.recommendations) > 0

def test_lowest_total_gross(sample_data):
    """T-02: Test lowest total gross."""
    intent = BottomMetricIntent()
    result = intent.analyze(sample_data, "What is the lowest total gross?")
    
    assert result.title == "Lowest Gross Analysis"
    assert "Charlie" in result.summary  # Charlie has lowest gross (-750)
    assert "-$750" in result.summary
    assert result.confidence == "high"
    assert result.error is None
    assert result.chart_data is not None

def test_sales_rep_most_deals(sample_data):
    """T-03: Test sales rep with most deals."""
    intent = HighestCountIntent()
    result = intent.analyze(sample_data, "Which sales rep sold the most deals?")
    
    assert result.title == "Top Rep by Deal Count"
    assert "Alice" in result.summary  # Alice has most deals (2)
    assert "2 deals" in result.summary
    assert result.confidence == "high"
    assert result.error is None
    assert result.chart_data is not None
    assert len(result.recommendations) > 0

def test_count_negative_profits(sample_data):
    """T-04: Test count of negative profit sales."""
    intent = CountMetricIntent()
    result = intent.analyze(sample_data, "How many deals had negative profit?")
    
    assert result.title == "Negative Profit Analysis"
    assert "2 deals" in result.summary  # 2 deals with negative profit
    assert "-$1,250" in result.summary  # Total negative profit
    assert result.confidence == "high"
    assert result.error is None
    assert result.chart_data is not None
    assert len(result.recommendations) > 0

def test_intent_matching():
    """Test that intents match appropriate queries."""
    highest_count = HighestCountIntent()
    assert highest_count.matches("which rep sold the most deals")
    assert highest_count.matches("who has the highest number of sales")
    assert not highest_count.matches("what was the highest gross profit")

def test_performance_large_dataset():
    """Test performance with 100k rows."""
    import time
    
    # Create large dataset
    large_df = pd.DataFrame({
        'SalesRepName': np.random.choice(['Alice', 'Bob', 'Charlie'], 100000),
        'Total_Gross': np.random.normal(1500, 500, 100000),
        'DealCount': np.ones(100000),
        'LeadSource': np.random.choice(['Web', 'Direct', 'Social'], 100000)
    })
    
    # Test highest count intent performance
    intent = HighestCountIntent()
    start_time = time.time()
    result = intent.analyze(large_df, "Which rep sold the most deals?")
    end_time = time.time()
    
    # Should complete in under 200ms
    assert (end_time - start_time) < 0.2
    assert result.error is None
    assert result.title == "Top Rep by Deal Count"