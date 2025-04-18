"""
Unit tests for KPI dashboard filters.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from src.ui.pages.kpi_dashboard import (
    apply_date_filter,
    apply_dimension_filters
)

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame({
        'SaleDate': pd.date_range(start='2023-01-01', periods=100, freq='D'),
        'location': ['North', 'South'] * 50,
        'vehicle_type': ['New', 'Used'] * 50,
        'TotalGross': range(100)
    })

def test_date_filter():
    """Test date range filtering."""
    df = pd.DataFrame({
        'SaleDate': pd.date_range(start='2023-01-01', periods=10, freq='D'),
        'value': range(10)
    })
    
    start_date = datetime(2023, 1, 3)
    end_date = datetime(2023, 1, 7)
    
    filtered = apply_date_filter(df, start_date, end_date)
    
    assert len(filtered) == 5
    assert filtered['SaleDate'].min() >= start_date
    assert filtered['SaleDate'].max() <= end_date

def test_dimension_filters(sample_data):
    """Test dimension filtering."""
    filters = {
        'location': 'North',
        'vehicle_type': 'New'
    }
    
    filtered = apply_dimension_filters(sample_data, filters)
    
    assert len(filtered) == 25
    assert all(filtered['location'] == 'North')
    assert all(filtered['vehicle_type'] == 'New')

def test_empty_filters(sample_data):
    """Test filtering with empty filters."""
    filtered = apply_dimension_filters(sample_data, {})
    assert len(filtered) == len(sample_data)

def test_invalid_column_filter(sample_data):
    """Test filtering with invalid column."""
    filters = {'invalid_column': 'value'}
    filtered = apply_dimension_filters(sample_data, filters)
    assert len(filtered) == len(sample_data)

def test_date_filter_invalid_date():
    """Test date filter with invalid dates."""
    df = pd.DataFrame({
        'SaleDate': ['invalid_date'] * 5,
        'value': range(5)
    })
    
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 1, 31)
    
    filtered = apply_date_filter(df, start_date, end_date)
    assert len(filtered) == len(df)

def test_multiple_date_columns():
    """Test date filter with multiple date-like columns."""
    df = pd.DataFrame({
        'sale_date': pd.date_range(start='2023-01-01', periods=5),
        'transaction_time': pd.date_range(start='2023-01-01', periods=5),
        'value': range(5)
    })
    
    start_date = datetime(2023, 1, 2)
    end_date = datetime(2023, 1, 4)
    
    filtered = apply_date_filter(df, start_date, end_date)
    assert len(filtered) == 3

def test_combined_filters(sample_data):
    """Test combining date and dimension filters."""
    start_date = datetime(2023, 1, 10)
    end_date = datetime(2023, 1, 20)
    
    # Apply date filter first
    date_filtered = apply_date_filter(sample_data, start_date, end_date)
    
    # Then apply dimension filter
    dimension_filtered = apply_dimension_filters(
        date_filtered,
        {'location': 'North'}
    )
    
    assert len(dimension_filtered) == 5
    assert all(dimension_filtered['location'] == 'North')
    assert all(
        (dimension_filtered['SaleDate'] >= start_date) &
        (dimension_filtered['SaleDate'] <= end_date)
    )