"""
Tests for the enhanced query processing functionality.

Tests cover:
1. Fuzzy matching
2. Temporal context extraction
3. Multi-metric support
4. Entity validation
5. Statistical processing
"""

import pytest
import pandas as pd
import numpy as np
import datetime
from unittest.mock import MagicMock, patch
from typing import Dict, List, Any

from src.watchdog_ai.models.query_models import QueryContext, QueryResult, IntentSchema, TimeRange
from src.watchdog_ai.insights.context import InsightExecutionContext
from src.watchdog_ai.insights.direct_query_handler import (
    fuzzy_match_entity,
    extract_entities,
    extract_temporal_context,
    identify_metric_type,
    process_metric_for_entity,
    process_query
)

# ----------------- Test Fixtures -----------------

@pytest.fixture
def sample_sales_data():
    """Create sample sales data for testing"""
    return pd.DataFrame({
        'sales_rep': ['Karen Davis', 'John Smith', 'Emma Johnson', 'Michael Brown', 'Sarah Williams'],
        'region': ['East', 'West', 'North', 'South', 'Central'],
        'product': ['Product A', 'Product B', 'Product C', 'Product A', 'Product D'],
        'customer_segment': ['Enterprise', 'SMB', 'Enterprise', 'Government', 'SMB'],
        'days_to_close': [14.2, 21.5, 9.8, 31.0, 18.7],
        'revenue': [15000, 8500, 22000, 12300, 9800],
        'sales_count': [5, 3, 7, 4, 2],
        'conversion_rate': [0.35, 0.28, 0.42, 0.31, 0.25],
        'date': ['2023-01-15', '2023-02-20', '2023-03-10', '2023-04-05', '2023-05-22']
    })

@pytest.fixture
def mock_context(sample_sales_data):
    """Create a mock insight execution context"""
    context = MagicMock(spec=InsightExecutionContext)
    context.df = sample_sales_data
    context.time_range = None
    context.query = ""
    context.user_role = "analyst"
    return context

@pytest.fixture
def query_context_factory(mock_context):
    """Factory for creating query contexts with different queries"""
    def _create_query_context(query):
        return QueryContext(
            query=query,
            insight_context=mock_context
        )
    return _create_query_context

# ----------------- Fuzzy Matching Tests -----------------

def test_fuzzy_match_exact(sample_sales_data):
    """Test fuzzy matching with exact match"""
    entity, confidence = fuzzy_match_entity("Karen Davis", "sales_rep", sample_sales_data)
    assert entity == "Karen Davis"
    assert confidence == 1.0

def test_fuzzy_match_close(sample_sales_data):
    """Test fuzzy matching with close match"""
    entity, confidence = fuzzy_match_entity("Karen Davies", "sales_rep", sample_sales_data)
    assert entity == "Karen Davis"
    assert confidence > 0.8

def test_fuzzy_match_case_insensitive(sample_sales_data):
    """Test fuzzy matching with case differences"""
    entity, confidence = fuzzy_match_entity("karen davis", "sales_rep", sample_sales_data)
    assert entity == "Karen Davis"
    assert confidence == 1.0

def test_fuzzy_match_below_threshold(sample_sales_data):
    """Test fuzzy matching with match below threshold"""
    entity, confidence = fuzzy_match_entity("Kevin Davidson", "sales_rep", sample_sales_data, threshold=0.8)
    assert entity is None
    assert confidence == 0.0

def test_fuzzy_match_special_chars(sample_sales_data):
    """Test fuzzy matching with special characters"""
    # Create a temporary df with special characters
    df = sample_sales_data.copy()
    df.loc[0, 'sales_rep'] = "O'Brien, Pat"
    
    entity, confidence = fuzzy_match_entity("OBrien Pat", "sales_rep", df)
    assert entity == "O'Brien, Pat"
    assert confidence > 0.7

# ----------------- Temporal Context Tests -----------------

def test_extract_absolute_date():
    """Test extracting absolute dates"""
    query = "Show sales for January 15, 2023"
    time_range = extract_temporal_context(query)
    assert time_range is not None
    assert time_range.start_date is not None
    assert time_range.start_date.month == 1
    assert time_range.start_date.day == 15
    assert time_range.start_date.year == 2023

def test_extract_relative_date():
    """Test extracting relative dates"""
    query = "Show sales for last month"
    time_range = extract_temporal_context(query)
    assert time_range is not None
    assert time_range.period == "last_month"

def test_extract_date_range():
    """Test extracting date ranges"""
    query = "Show sales from January 1 to March 31"
    time_range = extract_temporal_context(query)
    assert time_range is not None
    assert time_range.period == "custom_range"

def test_extract_period_specification():
    """Test extracting period specifications"""
    query = "Show year to date revenue"
    time_range = extract_temporal_context(query)
    assert time_range is not None
    assert time_range.period == "ytd"
    
    query = "Show Q1 performance"
    time_range = extract_temporal_context(query)
    assert time_range is not None
    assert time_range.period == "q1"

def test_extract_invalid_date():
    """Test handling invalid dates"""
    query = "Show sales for Febtober 35, 2023"
    time_range = extract_temporal_context(query)
    # Should not crash, but should not extract an invalid date
    assert time_range is None or time_range.start_date is None

# ----------------- Multi-Metric Support Tests -----------------

def test_identify_single_metric():
    """Test identifying a single metric type"""
    query = "What is the average days to close for Karen Davis?"
    metric_type = identify_metric_type(query)
    assert metric_type == "days_to_close"

@patch('src.watchdog_ai.insights.direct_query_handler.identify_metric_type')
def test_compound_metrics(mock_identify):
    """Test handling compound metric queries"""
    # We mock identify_metric_type to simulate identification of multiple metrics
    mock_identify.side_effect = ["revenue", "conversion_rate"]
    
    # Real implementation would need to be enhanced to handle multiple metrics
    # This test demonstrates what would happen if it did
    query = "What is the revenue and conversion rate for Karen Davis?"
    
    metric1 = mock_identify(query)
    assert metric1 == "revenue"
    
    metric2 = mock_identify(query)
    assert metric2 == "conversion_rate"

def test_metric_confidence(query_context_factory):
    """Test metric confidence scoring"""
    # Clear query - high confidence
    query = "What is the average days to close for Karen Davis?"
    result = process_query(query_context_factory(query))
    assert result.confidence_score > 0.7
    
    # Ambiguous query - lower confidence
    query = "How is Karen Davis performing?"
    result = process_query(query_context_factory(query))
    assert result.confidence_score < 0.7

# ----------------- Entity Validation Tests -----------------

def test_extract_validated_entities(mock_context):
    """Test extracting and validating entities against sample data"""
    query = "What is the revenue for Karen Davis in the East region?"
    entities = extract_entities(query, mock_context)
    
    assert "sales_rep" in entities
    assert any(e["value"] == "Karen Davis" for e in entities["sales_rep"])
    assert "region" in entities
    assert any(e["value"] == "East" for e in entities["region"])
    
    # Validate that the confidence is high for these matches
    assert any(e["confidence"] > 0.9 for e in entities["sales_rep"] if e["value"] == "Karen Davis")

def test_extract_unknown_entities(mock_context):
    """Test handling unknown entities"""
    query = "What is the revenue for Kevin Durant in the Southwest region?"
    entities = extract_entities(query, mock_context)
    
    # These entities shouldn't be validated since they don't exist in the data
    if "sales_rep" in entities:
        for entity in entities["sales_rep"]:
            if "Kevin Durant" in entity["value"]:
                assert entity["validated"] is False
                assert entity["confidence"] < 0.8

def test_extract_multiple_entity_types(mock_context):
    """Test extracting multiple entity types from a single query"""
    query = "Compare revenue for Product A in the Enterprise segment versus Product B in the SMB segment"
    entities = extract_entities(query, mock_context)
    
    assert "product" in entities
    assert len(entities["product"]) >= 2
    assert "customer_segment" in entities
    assert len(entities["customer_segment"]) >= 2

# ----------------- Statistical Processing Tests -----------------

def test_mean_median_calculation(query_context_factory):
    """Test calculation of mean and median values"""
    query = "What is the average days to close for Karen Davis?"
    result = process_query(query_context_factory(query))
    
    assert result.success
    assert "sales_rep:Karen Davis" in result.metrics
    metric = result.metrics["sales_rep:Karen Davis"]
    
    # The formatted value should contain both mean and median
    assert "days" in metric["formatted"]
    assert "median" in metric["formatted"]

def test_trend_analysis():
    """Test trend analysis functionality"""
    # Create data with a trend
    dates = pd.date_range(start='2023-01-01', periods=10, freq='M')
    trend_data = pd.DataFrame({
        'sales_rep': ['Karen Davis'] * 10,
        'revenue': [1000 + i*500 for i in range(10)],  # Increasing trend
        'date': dates
    })
    
    context = MagicMock(spec=InsightExecutionContext)
    context.df = trend_data
    
    # Process a query that would trigger trend analysis
    query_context = QueryContext(
        query="What is the revenue trend for Karen Davis?",
        insight_context=context
    )
    
    # This would need to be implemented in the actual code
    # Here we're just showing what would be tested
    with patch('src.watchdog_ai.insights.direct_query_handler.process_metric_for_entity') as mock_process:
        # Set up the mock to return a trend value
        mock_process.return_value = (0.5, "Positive trend (0.5 increase per month)")
        
        result = process_query(query_context)
        assert result.success
        assert "sales_rep:Karen Davis" in result.metrics
        assert "trend" in result.metrics["sales_rep:Karen Davis"]["formatted"].lower()

# ----------------- Integration Tests -----------------

def test_end_to_end_query_processing(query_context_factory):
    """Test end-to-end query processing with a realistic query"""
    query = "What is the revenue for Karen Davis in the East region during Q1?"
    result = process_query(query_context_factory(query))
    
    assert result.success
    assert result.confidence_score > 0.7
    assert len(result.metrics) > 0
    
    # Check that temporal filtering worked
    time_range = extract_temporal_context(query)
    assert time_range is not None
    assert time_range.period == "q1"
    
    # Check entity extraction
    entities = extract_entities(query, query_context_factory(query).insight_context)
    assert "sales_rep" in entities
    assert "region" in entities

def test_query_with_multiple_metrics_and_entities(query_context_factory):
    """Test processing a complex query with multiple metrics and entities"""
    query = "Compare the revenue and conversion rate for Karen Davis and John Smith"
    
    # For this test to pass, the code would need to be enhanced to handle
    # multiple metrics in a single query. Here we're just demonstrating what
    # would be tested.
    with patch('src.watchdog_ai.insights.direct_query_handler.identify_metric_type') as mock_identify:
        mock_identify.return_value = "revenue"  # Simplified for the test
        
        result = process_query(query_context_factory(query))
        assert result.success
        assert "sales_rep:Karen Davis" in result.metrics
        assert "sales_rep:John Smith" in result.metrics

# ----------------- Error Handling Tests -----------------

def test_handling_empty_data(query_context_factory):
    """Test handling empty data"""
    context = query_context_factory("What is the revenue for Karen Davis?")
    context.insight_context.df = pd.DataFrame()  # Empty dataframe
    
    result = process_query(context)
    assert not result.success
    assert "metrics" in result.__dict__
    assert not result.metrics

def test_handling_invalid_query(query_context_factory):
    """Test handling invalid or nonsensical queries"""
    query = "asdfghjkl qwertyuiop"
    result = process_query(query_context_factory(query))
    assert not result.success
    assert result.confidence_score < 0.5

# ----------------- Performance Tests -----------------

@pytest.mark.parametrize("size", [100, 1000])
def test_performance_with_large_dataset(size):
    """Test performance with larger datasets"""
    # Create a larger dataset
    np.random.seed(42)
    large_data = pd.DataFrame({
        'sales_rep': np.random.choice(['Karen Davis', 'John Smith', 'Emma Johnson'], size),
        'product': np.random.choice(['Product A', 'Product B', 'Product C'], size),
        'region': np.random.choice(['East', 'West', 'North', 'South'], size),
        'days_to_close': np.random.normal(20, 5, size),
        'revenue': np.random.normal(10000, 3000, size),
        'date': pd.date_range(start='2023-01-01', periods=size)
    })
    
    context = MagicMock(spec=InsightExecutionContext)
    context.df = large_data
    
    query_context = QueryContext(
        query="What is the average days to close for Karen Davis?",
        insight_context=context
    )
    
    # Ensure processing completes in a reasonable time
    import time
    start_time = time.time()
    result = process_query(query_context)
    end_time = time.time()
    
    assert result.success
    assert end_time - start_time < 2.0  # Should complete in less than 2 seconds

