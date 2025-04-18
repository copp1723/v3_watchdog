"""
Unit tests for the statistical baseline calculations.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.insights.baselines import (
    calculate_inventory_aging_stats,
    calculate_sales_performance_stats,
    detect_inventory_anomalies
)

@pytest.fixture
def sample_inventory_data():
    """Create sample inventory data for testing."""
    return pd.DataFrame({
        'days_in_stock': [30, 45, 60, 90, 120, 150, 30, 45],
        'model': ['Civic'] * 4 + ['Accord'] * 4,
        'trim': ['LX', 'EX'] * 4,
        'price': [20000, 22000, 25000, 27000, 28000, 30000, 21000, 23000]
    })

@pytest.fixture
def sample_sales_data():
    """Create sample sales data for testing."""
    return pd.DataFrame({
        'gross': [1000, -500, 2000, 1500, 3000, 2500, 1000, 1500],
        'sales_rep': ['John', 'John', 'Jane', 'Jane', 'Bob', 'Bob', 'Alice', 'Alice'],
        'date': [datetime.now() - timedelta(days=x) for x in range(8)]
    })

def test_calculate_inventory_aging_stats(sample_inventory_data):
    """Test inventory aging statistics calculation."""
    result = calculate_inventory_aging_stats(sample_inventory_data)
    
    assert 'overall_stats' in result
    assert 'model_stats' in result
    assert 'model_trim_stats' in result
    assert 'outlier_stats' in result
    
    # Check overall stats
    overall = result['overall_stats']
    assert isinstance(overall['mean_days'], float)
    assert isinstance(overall['median_days'], float)
    assert isinstance(overall['std_days'], float)
    assert overall['total_vehicles'] == 8
    
    # Check model stats
    assert 'Civic' in result['model_stats']
    assert 'Accord' in result['model_stats']
    assert result['model_stats']['Civic']['total_vehicles'] == 4
    
    # Check model/trim stats
    assert 'Civic' in result['model_trim_stats']
    assert 'LX' in result['model_trim_stats']['Civic']

def test_calculate_sales_performance_stats(sample_sales_data):
    """Test sales performance statistics calculation."""
    result = calculate_sales_performance_stats(sample_sales_data)
    
    assert 'overall_stats' in result
    assert 'rep_stats' in result
    assert 'trend_stats' in result
    assert 'benchmarks' in result
    
    # Check overall stats
    overall = result['overall_stats']
    assert isinstance(overall['total_gross'], float)
    assert isinstance(overall['mean_gross'], float)
    assert overall['total_deals'] == 8
    assert overall['negative_gross_count'] == 1
    
    # Check rep stats
    assert 'John' in result['rep_stats']
    assert 'Jane' in result['rep_stats']
    assert result['rep_stats']['John']['total_deals'] == 2
    
    # Check benchmarks
    benchmarks = result['benchmarks']
    assert isinstance(benchmarks['top_quartile_gross'], float)
    assert isinstance(benchmarks['deals_per_rep_mean'], float)

def test_detect_inventory_anomalies(sample_inventory_data):
    """Test inventory anomaly detection."""
    anomalies = detect_inventory_anomalies(sample_inventory_data)
    
    assert isinstance(anomalies, list)
    assert len(anomalies) > 0
    
    # Check anomaly structure
    for anomaly in anomalies:
        assert 'type' in anomaly
        assert 'days_in_stock' in anomaly
        assert 'z_score' in anomaly
        assert 'severity' in anomaly
        assert anomaly['severity'] in ['high', 'medium']

def test_error_handling():
    """Test error handling with invalid data."""
    # Test with empty DataFrame
    empty_df = pd.DataFrame()
    result = calculate_inventory_aging_stats(empty_df)
    assert 'error' in result
    
    # Test with invalid column names
    invalid_df = pd.DataFrame({'wrong_column': [1, 2, 3]})
    result = calculate_sales_performance_stats(invalid_df)
    assert 'error' in result
    
    # Test with non-numeric data
    bad_data_df = pd.DataFrame({
        'days_in_stock': ['invalid', 'data', 'here'],
        'model': ['A', 'B', 'C']
    })
    anomalies = detect_inventory_anomalies(bad_data_df)
    assert len(anomalies) == 1
    assert 'error' in anomalies[0]