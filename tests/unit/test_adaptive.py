"""
Unit tests for adaptive threshold learning.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.insights.adaptive import (
    ThresholdConfig,
    InventoryAgingLearner,
    GrossMarginLearner,
    LeadConversionLearner
)
from src.insights.models import FeedbackEntry

@pytest.fixture
def sample_inventory_data():
    """Create sample inventory data."""
    return pd.DataFrame({
        'days_on_lot': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'model': ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D', 'E', 'E'],
        'price': [20000] * 10
    })

@pytest.fixture
def sample_sales_data():
    """Create sample sales data."""
    return pd.DataFrame({
        'gross_profit': [1000, -500, 2000, 1500, 3000, 2500, 1000, 1500, 2000, 2500],
        'sale_price': [20000] * 10,
        'model': ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D', 'E', 'E']
    })

@pytest.fixture
def sample_leads_data():
    """Create sample leads data."""
    return pd.DataFrame({
        'status': ['Sold', 'Lost', 'Sold', 'Pending', 'Lost', 'Sold', 'Sold', 'Lost', 'Sold', 'Lost'],
        'source': ['Web', 'Web', 'Phone', 'Phone', 'Email', 'Email', 'Walk-in', 'Walk-in', 'Referral', 'Referral']
    })

@pytest.fixture
def sample_feedback():
    """Create sample feedback entries."""
    now = datetime.now()
    return [
        FeedbackEntry(
            insight_id="inventory_aging",
            feedback_type="helpful",
            user_id="user1",
            session_id="session1",
            timestamp=now - timedelta(days=1),
            metadata={"threshold": 45}
        ),
        FeedbackEntry(
            insight_id="inventory_aging",
            feedback_type="not_helpful",
            user_id="user2",
            session_id="session2",
            timestamp=now - timedelta(days=2),
            metadata={"too_many_alerts": True, "threshold": 30}
        )
    ]

def test_inventory_aging_learner_init():
    """Test initialization of inventory aging learner."""
    learner = InventoryAgingLearner()
    assert learner.current_threshold is None
    assert learner.config.min_threshold == 15.0
    assert learner.config.max_threshold == 120.0

def test_inventory_aging_default_threshold(sample_inventory_data):
    """Test default threshold calculation for inventory aging."""
    learner = InventoryAgingLearner()
    threshold = learner._calculate_default_threshold(sample_inventory_data)
    assert 30 <= threshold <= 120
    assert isinstance(threshold, float)

def test_gross_margin_learner_default_threshold(sample_sales_data):
    """Test default threshold calculation for gross margins."""
    learner = GrossMarginLearner()
    threshold = learner._calculate_default_threshold(sample_sales_data)
    assert 0.05 <= threshold <= 0.40
    assert isinstance(threshold, float)

def test_lead_conversion_learner_default_threshold(sample_leads_data):
    """Test default threshold calculation for lead conversion."""
    learner = LeadConversionLearner()
    threshold = learner._calculate_default_threshold(sample_leads_data)
    assert 0.08 <= threshold <= 0.35
    assert isinstance(threshold, float)

def test_threshold_adaptation_with_feedback(sample_inventory_data, sample_feedback):
    """Test threshold adaptation based on feedback."""
    learner = InventoryAgingLearner()
    
    # Initial fit
    learner.fit(sample_inventory_data, [])
    initial_threshold = learner.current_threshold
    
    # Fit with feedback
    learner.fit(sample_inventory_data, sample_feedback)
    adapted_threshold = learner.current_threshold
    
    assert adapted_threshold != initial_threshold
    assert learner.config.min_threshold <= adapted_threshold <= learner.config.max_threshold

def test_threshold_prediction_with_current_data(sample_inventory_data):
    """Test threshold prediction for current data."""
    learner = InventoryAgingLearner()
    learner.fit(sample_inventory_data, [])
    
    # Create modified current data
    current_data = sample_inventory_data.copy()
    current_data['days_on_lot'] = current_data['days_on_lot'] * 1.2
    
    predicted = learner.predict_threshold(current_data)
    assert isinstance(predicted, float)
    assert learner.config.min_threshold <= predicted <= learner.config.max_threshold

def test_feedback_filtering(sample_feedback):
    """Test filtering of old feedback."""
    learner = InventoryAgingLearner(
        ThresholdConfig(feedback_window_days=1)
    )
    
    recent = learner._filter_recent_feedback(sample_feedback)
    assert len(recent) < len(sample_feedback)
    assert all(
        (datetime.now() - f.timestamp).days <= 1
        for f in recent
    )

def test_error_handling_missing_columns():
    """Test handling of missing columns."""
    learner = InventoryAgingLearner()
    empty_df = pd.DataFrame({'wrong_column': [1, 2, 3]})
    
    # Should fall back to default
    threshold = learner._calculate_default_threshold(empty_df)
    assert threshold == 30.0

def test_threshold_bounds():
    """Test enforcement of threshold bounds."""
    config = ThresholdConfig(
        min_threshold=10.0,
        max_threshold=50.0
    )
    learner = InventoryAgingLearner(config)
    
    # Create data that would naturally produce out-of-bounds threshold
    df = pd.DataFrame({
        'days_on_lot': [1000] * 10,  # Would produce very high threshold
        'model': ['A'] * 10
    })
    
    learner.fit(df, [])
    assert learner.current_threshold <= config.max_threshold

def test_learning_rate_application(sample_inventory_data, sample_feedback):
    """Test correct application of learning rate."""
    slow_learner = InventoryAgingLearner(
        ThresholdConfig(learning_rate=0.1)
    )
    fast_learner = InventoryAgingLearner(
        ThresholdConfig(learning_rate=0.5)
    )
    
    # Fit both learners
    slow_learner.fit(sample_inventory_data, sample_feedback)
    fast_learner.fit(sample_inventory_data, sample_feedback)
    
    # Fast learner should deviate more from initial threshold
    initial = slow_learner._calculate_default_threshold(sample_inventory_data)
    assert abs(fast_learner.current_threshold - initial) > \
           abs(slow_learner.current_threshold - initial)

def test_feedback_weighting(sample_inventory_data):
    """Test that recent feedback is weighted more heavily."""
    now = datetime.now()
    feedback = [
        # Old feedback suggesting lower threshold
        FeedbackEntry(
            insight_id="inventory_aging",
            feedback_type="not_helpful",
            user_id="user1",
            session_id="session1",
            timestamp=now - timedelta(days=30),
            metadata={"missed_alerts": True}
        ),
        # Recent feedback suggesting higher threshold
        FeedbackEntry(
            insight_id="inventory_aging",
            feedback_type="not_helpful",
            user_id="user2",
            session_id="session2",
            timestamp=now - timedelta(days=1),
            metadata={"too_many_alerts": True}
        )
    ]
    
    learner = InventoryAgingLearner()
    
    # Initial threshold
    learner.fit(sample_inventory_data, [])
    initial = learner.current_threshold
    
    # Adapted threshold
    learner.fit(sample_inventory_data, feedback)
    adapted = learner.current_threshold
    
    # Should move up due to recent feedback having more weight
    assert adapted > initial