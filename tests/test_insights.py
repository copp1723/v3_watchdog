"""
Tests for the insight generation system.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st
from src.utils.columns import (
    find_column,
    find_metric_column,
    find_category_column,
    normalize_column_name,
    _compute_hash,
    _get_cache_key
)
from src.insights.intents import TopMetricIntent, BottomMetricIntent, AverageMetricIntent

# --- Column Finder Tests ---

def test_normalize_column_name():
    """Test column name normalization."""
    assert normalize_column_name("Total Gross") == "totalgross"
    assert normalize_column_name("lead_source") == "leadsource"
    assert normalize_column_name("SalesRepName") == "salesrepname"
    # Test EU number format
    assert normalize_column_name("Price (€)") == "price"
    assert normalize_column_name("Gross Profit €") == "grossprofit"

def test_compute_hash():
    """Test hash computation stability."""
    items1 = ["a", "b", "c"]
    items2 = ["c", "a", "b"]
    # Same items in different order should have same hash
    assert _compute_hash(items1) == _compute_hash(items2)

def test_cache_key_generation():
    """Test cache key generation."""
    prompt = "find gross profit"
    columns = ["GrossProfit", "Revenue"]
    key1 = _get_cache_key(prompt, columns)
    key2 = _get_cache_key(prompt, ["Revenue", "GrossProfit"])
    # Same inputs in different order should generate same key
    assert key1 == key2

def test_find_column_with_cache():
    """Test column finding with cache."""
    columns = ["Total_Gross", "LeadSource", "SalesRepName"]
    
    # First call should cache the result
    result1 = find_column(columns, ["total gross"])
    assert result1 == "Total_Gross"
    assert "column_finder_cache" in st.session_state
    
    # Second call should use cached result
    result2 = find_column(columns, ["total gross"])
    assert result2 == result1

def test_find_column_eu_formats():
    """Test column finding with European formats."""
    columns = ["Gross Profit €", "Price (€)", "Revenue €"]
    
    # Test exact matches
    assert find_column(columns, ["Gross Profit"]) == "Gross Profit €"
    assert find_column(columns, ["Price"]) == "Price (€)"
    
    # Test fuzzy matches
    assert find_column(columns, ["gross"]) == "Gross Profit €"
    assert find_column(columns, ["price in euros"]) == "Price (€)"

def test_find_column_fuzzy_matching():
    """Test fuzzy matching with RapidFuzz."""
    columns = ["TotalGrossProfit", "LeadSource", "SalesRepName"]
    
    # Test matches above 80% threshold
    assert find_column(columns, ["total gross"]) == "TotalGrossProfit"
    assert find_column(columns, ["sales rep"]) == "SalesRepName"
    
    # Test matches below threshold
    assert find_column(columns, ["something completely different"]) is None

def test_find_metric_column():
    """Test metric column finding."""
    columns = ["Total_Gross", "SalePrice", "Revenue"]
    
    assert find_metric_column(columns, "gross") == "Total_Gross"
    assert find_metric_column(columns, "price") == "SalePrice"
    assert find_metric_column(columns, "revenue") == "Revenue"

def test_find_category_column():
    """Test category column finding."""
    columns = ["SalesRepName", "LeadSource", "VehicleMake"]
    
    assert find_category_column(columns, "rep") == "SalesRepName"
    assert find_category_column(columns, "source") == "LeadSource"
    assert find_category_column(columns, "make") == "VehicleMake"

# --- Intent Tests ---

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame({
        'SalesRepName': ['Alice', 'Bob', 'Alice', 'Charlie'],
        'Total_Gross': [1000, 2000, 1500, 500],
        'LeadSource': ['Web', 'Direct', 'Web', 'Social'],
        'DealCount': [1, 1, 1, 1]
    })

def test_top_metric_intent_matches():
    """Test TopMetricIntent matching."""
    intent = TopMetricIntent()
    
    assert intent.matches("Who has the highest gross?")
    assert intent.matches("Show me the top sales rep")
    assert not intent.matches("What is the average gross?")

def test_top_metric_intent_analyze(sample_data):
    """Test TopMetricIntent analysis."""
    intent = TopMetricIntent()
    result = intent.analyze(sample_data, "Who has the highest gross by rep?")
    
    assert result.title == "Top Rep by Gross"
    assert "Bob" in result.summary
    assert "$2,000" in result.summary
    assert len(result.recommendations) > 0
    assert result.chart_data is not None
    assert result.confidence == "high"

def test_bottom_metric_intent_matches():
    """Test BottomMetricIntent matching."""
    intent = BottomMetricIntent()
    
    assert intent.matches("What is the lowest gross?")
    assert intent.matches("Show me the bottom performer")
    assert not intent.matches("Who has the highest sales?")

def test_average_metric_intent_matches():
    """Test AverageMetricIntent matching."""
    intent = AverageMetricIntent()
    
    assert intent.matches("What is the average gross?")
    assert intent.matches("Show me mean performance")
    assert not intent.matches("Who has the highest sales?")

# --- Integration Tests ---

def test_end_to_end_insight_generation(sample_data):
    """Test the complete insight generation flow."""
    from src.insight_conversation import ConversationManager
    
    manager = ConversationManager()
    result = manager.generate_insight(
        "Who has the highest gross by rep?",
        validation_context={"df": sample_data}
    )
    
    assert not result.get("is_error", False)
    assert "Bob" in result["summary"]
    assert result.get("chart_data") is not None
    assert result.get("confidence") == "high"

def test_performance_with_large_dataset():
    """Test performance with 100k rows."""
    import time
    
    # Create large dataset
    large_df = pd.DataFrame({
        'SalesRepName': np.random.choice(['Alice', 'Bob', 'Charlie'], 100000),
        'Total_Gross': np.random.normal(1500, 500, 100000),
        'LeadSource': np.random.choice(['Web', 'Direct', 'Social'], 100000)
    })
    
    # Test column finding performance
    start_time = time.time()
    result = find_metric_column(large_df.columns, "gross")
    end_time = time.time()
    
    # Should complete in under 200ms
    assert (end_time - start_time) < 0.2
    assert result == "Total_Gross"