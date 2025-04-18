"""
Unit tests for KPI dashboard components.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.ui.pages.kpi_dashboard import (
    calculate_rep_performance,
    analyze_inventory_aging,
    calculate_lead_conversion
)

@pytest.fixture
def sample_sales_data():
    """Create sample sales data."""
    return pd.DataFrame({
        'SalesRepName': ['Alice', 'Bob', 'Charlie', 'Alice', 'Bob'],
        'TotalGross': [1000, 2000, 1500, 2000, 1000],
        'SaleDate': pd.date_range(start='2023-01-01', periods=5, freq='D')
    })

@pytest.fixture
def sample_inventory_data():
    """Create sample inventory data."""
    return pd.DataFrame({
        'VIN': [f'VIN{i}' for i in range(10)],
        'DaysInStock': [15, 25, 45, 55, 75, 85, 95, 105, 115, 125],
        'ListPrice': [20000 + i * 1000 for i in range(10)]
    })

@pytest.fixture
def sample_leads_data():
    """Create sample leads data."""
    return pd.DataFrame({
        'LeadSource': ['Web', 'Phone', 'Walk-in'] * 5,
        'LeadStatus': ['Sold', 'Open', 'Sold', 'Lost', 'Sold'] * 3,
        'LeadDate': pd.date_range(start='2023-01-01', periods=15, freq='D')
    })

def test_calculate_rep_performance(sample_sales_data):
    """Test sales rep performance calculation."""
    metrics, summary = calculate_rep_performance(sample_sales_data)
    
    assert not metrics.empty
    assert len(metrics) == 3  # Three unique reps
    assert 'deals' in metrics.columns
    assert 'total_gross' in metrics.columns
    assert 'avg_gross' in metrics.columns
    
    # Verify summary metrics
    assert summary['top_performer'] in ['Alice', 'Bob', 'Charlie']
    assert summary['total_deals'] == 5
    assert summary['total_gross'] == 7500
    assert 1000 <= summary['overall_avg_gross'] <= 2000

def test_analyze_inventory_aging(sample_inventory_data):
    """Test inventory aging analysis."""
    metrics, summary = analyze_inventory_aging(sample_inventory_data)
    
    assert not metrics.empty
    assert len(metrics) == 4  # Four age ranges
    assert all(col in metrics.columns for col in ['age_range', 'count'])
    
    # Verify summary metrics
    assert summary['total_units'] == 10
    assert 60 <= summary['avg_age'] <= 70  # Based on sample data
    assert summary['aged_inventory'] == 4  # Units over 90 days
    assert 35 <= summary['aged_percentage'] <= 45  # Should be 40%

def test_calculate_lead_conversion(sample_leads_data):
    """Test lead conversion calculation."""
    metrics, summary = calculate_lead_conversion(sample_leads_data)
    
    assert not metrics.empty
    assert len(metrics) == 3  # Three lead sources
    assert all(source in metrics['LeadSource'].values for source in ['Web', 'Phone', 'Walk-in'])
    
    # Verify summary metrics
    assert summary['total_leads'] == 15
    assert summary['converted_leads'] == 6  # 'Sold' status count
    assert 35 <= summary['conversion_rate'] <= 45  # Should be 40%

def test_empty_dataframe_handling():
    """Test handling of empty DataFrames."""
    empty_df = pd.DataFrame()
    
    # Test each function with empty data
    metrics1, summary1 = calculate_rep_performance(empty_df)
    assert metrics1.empty
    assert not summary1
    
    metrics2, summary2 = analyze_inventory_aging(empty_df)
    assert metrics2.empty
    assert not summary2
    
    metrics3, summary3 = calculate_lead_conversion(empty_df)
    assert metrics3.empty
    assert not summary3

def test_missing_columns_handling(sample_sales_data):
    """Test handling of missing columns."""
    # Remove required columns
    df_no_rep = sample_sales_data.drop('SalesRepName', axis=1)
    df_no_gross = sample_sales_data.drop('TotalGross', axis=1)
    
    # Each function should handle missing columns gracefully
    metrics1, summary1 = calculate_rep_performance(df_no_rep)
    assert metrics1.empty
    assert not summary1
    
    metrics2, summary2 = calculate_rep_performance(df_no_gross)
    assert metrics2.empty
    assert not summary2

def test_data_type_handling():
    """Test handling of different data types."""
    # Create DataFrame with mixed types
    df = pd.DataFrame({
        'SalesRepName': ['Alice', 'Bob', None, 'Alice'],
        'TotalGross': ['1000', '2000', 'invalid', '2000'],
        'SaleDate': ['2023-01-01', '2023-01-02', 'invalid', '2023-01-04']
    })
    
    # Functions should handle type conversion gracefully
    metrics, summary = calculate_rep_performance(df)
    assert not metrics.empty
    assert summary
    assert len(metrics) == 2  # Two valid reps after cleaning