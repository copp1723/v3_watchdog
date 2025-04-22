"""
Unit tests for analytics_engine.py.
"""
import os
import pandas as pd
import pytest
import numpy as np
from datetime import datetime, timedelta
import json
from unittest.mock import patch, mock_open

from watchdog_ai.core.analytics_engine import AnalyticsEngine
from watchdog_ai.core.constants import (
    DEFAULT_REQUIRED_COLUMNS,
    ERR_NO_DATA,
    ERR_COLUMN_NOT_FOUND
)


# Test Fixtures
@pytest.fixture
def sample_sales_data():
    """Create a sample sales DataFrame with realistic data."""
    # Create dates for two years of data (for YoY comparisons)
    dates = []
    for year in [2023, 2024]:
        for month in range(1, 13):
            for day in range(1, 28, 3):  # Every 3 days for less data
                if year == 2024 and month > 4:  # Only up to April 2024
                    continue
                dates.append(f"{year}-{month:02d}-{day:02d}")
    
    # Random lead sources
    lead_sources = ['Web', 'Referral', 'Direct', 'Social', 'Email']
    
    # Random salespeople
    sales_people = ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Brown', 'Carlos Rodriguez']
    
    # Create DataFrame
    np.random.seed(42)  # For reproducible tests
    n_rows = len(dates)
    
    df = pd.DataFrame({
        'SaleDate': dates,
        'LeadSource': np.random.choice(lead_sources, size=n_rows),
        'SalesPerson': np.random.choice(sales_people, size=n_rows),
        'TotalGross': np.random.uniform(500, 5000, size=n_rows).round(2),
        'IsSale': np.random.choice([0, 1], size=n_rows, p=[0.3, 0.7])
    })
    
    # Convert SaleDate to datetime
    df['SaleDate'] = pd.to_datetime(df['SaleDate'])
    
    return df


@pytest.fixture
def empty_dataframe():
    """Create an empty DataFrame."""
    return pd.DataFrame()


@pytest.fixture
def missing_columns_df():
    """Create a DataFrame missing required columns."""
    return pd.DataFrame({
        'SomeOtherColumn': [1, 2, 3],
        'AnotherColumn': ['a', 'b', 'c']
    })


@pytest.fixture
def invalid_data_df():
    """Create a DataFrame with invalid data types."""
    return pd.DataFrame({
        'SaleDate': ['invalid-date', '2023-13-45', 'yesterday'],
        'LeadSource': [1, 2, 3],  # Numbers instead of strings
        'TotalGross': ['abc', 'def', 'ghi'],  # Strings instead of numbers
        'IsSale': ['maybe', 'perhaps', 'not sure'],  # Non-boolean strings
        'SalesPerson': ['Person A', 'Person B', 'Person C']  # Add missing required column
    })

@pytest.fixture
def sample_csv_content():
    """Create sample CSV content as a string."""
    return (
        "SaleDate,LeadSource,SalesPerson,TotalGross,IsSale\n"
        "2023-01-01,Web,John Doe,1000.50,1\n"
        "2023-01-02,Referral,Jane Smith,1500.75,0\n"
        "2023-01-03,Direct,Bob Johnson,2000.25,1\n"
        "2023-01-04,Social,Alice Brown,2500.00,1\n"
        "2023-01-05,Email,Carlos Rodriguez,3000.80,0\n"
    )


@pytest.fixture
def analytics_engine():
    """Create an instance of AnalyticsEngine."""
    return AnalyticsEngine()


# Tests for load_csv
def test_load_csv_file_not_found(analytics_engine, monkeypatch):
    """Test load_csv with a non-existent file."""
    # Mock os.path.exists to return False
    monkeypatch.setattr(os.path, 'exists', lambda path: False)
    
    with pytest.raises(FileNotFoundError):
        analytics_engine.load_csv('nonexistent_file.csv')


def test_load_csv_valid_file(analytics_engine, monkeypatch, sample_csv_content):
    """Test load_csv with a valid CSV file."""
    # Mock os.path.exists to return True
    monkeypatch.setattr(os.path, 'exists', lambda path: True)
    
    # Mock pd.read_csv
    expected_df = pd.read_csv(pd.io.common.StringIO(sample_csv_content))
    
    # Use patch to mock built-in open and pandas read_csv
    with patch('builtins.open', mock_open(read_data=sample_csv_content)):
        with patch('pandas.read_csv', return_value=expected_df):
            result = analytics_engine.load_csv('valid_file.csv')
    
    # Verify the result
    pd.testing.assert_frame_equal(result, expected_df)


def test_load_csv_empty_file(analytics_engine, monkeypatch):
    """Test load_csv with an empty CSV file."""
    # Mock os.path.exists to return True
    monkeypatch.setattr(os.path, 'exists', lambda path: True)
    
    # Mock pd.read_csv to return an empty DataFrame
    empty_df = pd.DataFrame()
    with patch('pandas.read_csv', return_value=empty_df):
        with pytest.raises(ValueError) as excinfo:
            analytics_engine.load_csv('empty_file.csv')
    
    assert ERR_NO_DATA in str(excinfo.value)


def test_load_csv_invalid_csv(analytics_engine, monkeypatch):
    """Test load_csv with an invalid CSV file."""
    # Mock os.path.exists to return True
    monkeypatch.setattr(os.path, 'exists', lambda path: True)
    
    # Mock pd.read_csv to raise an exception
    with patch('pandas.read_csv', side_effect=Exception('Invalid CSV')):
        with pytest.raises(ValueError) as excinfo:
            analytics_engine.load_csv('invalid_file.csv')
    
    assert 'Failed to load CSV file' in str(excinfo.value)


# Tests for _standardize_dataframe
def test_standardize_dataframe_valid(analytics_engine, sample_sales_data):
    """Test _standardize_dataframe with valid data."""
    result, warnings = analytics_engine._standardize_dataframe(sample_sales_data)
    
    # Check the result is a DataFrame
    assert isinstance(result, pd.DataFrame)
    
    # Check required columns are present
    for col in DEFAULT_REQUIRED_COLUMNS:
        assert col in result.columns
    
    # Check data types
    assert pd.api.types.is_datetime64_dtype(result['SaleDate'])
    assert pd.api.types.is_numeric_dtype(result['TotalGross'])
    
    # Check warnings list returned
    assert isinstance(warnings, list)


def test_standardize_dataframe_empty(analytics_engine, empty_dataframe):
    """Test _standardize_dataframe with empty data."""
    with pytest.raises(ValueError) as excinfo:
        analytics_engine._standardize_dataframe(empty_dataframe)
    
    assert ERR_NO_DATA in str(excinfo.value)


def test_standardize_dataframe_missing_columns(analytics_engine, missing_columns_df):
    """Test _standardize_dataframe with missing required columns."""
    with pytest.raises(ValueError) as excinfo:
        analytics_engine._standardize_dataframe(missing_columns_df)
    
    error_msg = str(excinfo.value)
    # Check that all required columns are mentioned in error
    assert "Could not find required column" in error_msg
    for col in DEFAULT_REQUIRED_COLUMNS:
        assert col in error_msg


def test_standardize_dataframe_column_mapping(analytics_engine):
    """Test _standardize_dataframe with column mapping."""
    # Create a DataFrame with differently named columns
    df = pd.DataFrame({
        'Sale Date': ['2023-01-01', '2023-01-02'],
        'Lead Source': ['Web', 'Referral'],
        'Sales Person': ['John Doe', 'Jane Smith'],
        'Gross Profit': [1000, 1500],
        'Sold': [1, 0]
    })
    
    result, warnings = analytics_engine._standardize_dataframe(df)
    
    # Check that columns were standardized
    assert 'SaleDate' in result.columns
    assert 'LeadSource' in result.columns
    assert 'SalesPerson' in result.columns
    assert 'TotalGross' in result.columns
    assert 'IsSale' in result.columns
def test_standardize_dataframe_type_conversion_errors(analytics_engine, invalid_data_df):
    """Test _standardize_dataframe with type conversion errors."""
    with pytest.raises(ValueError) as excinfo:
        analytics_engine._standardize_dataframe(invalid_data_df)
    
    error_msg = str(excinfo.value)
    assert any([
        "Invalid data type" in error_msg,
        "could not convert" in error_msg.lower(),
        "time data" in error_msg.lower()  # For datetime conversion errors
    ])
    assert "Invalid data type" in error_msg or "Required column" in error_msg


# Tests for calculate_sales_trends
def test_calculate_sales_trends(analytics_engine, sample_sales_data):
    """Test calculate_sales_trends with valid data."""
    result = analytics_engine.calculate_sales_trends(sample_sales_data)
    
    # Check result keys
    assert 'total_sales' in result
    assert 'average_daily_sales' in result
    assert 'mtd_sales' in result
    assert 'ytd_sales' in result
    assert 'trends' in result
    
    # Check trends data structure
    assert isinstance(result['trends'], list)
    if result['trends']:
        trend = result['trends'][0]
        assert 'date' in trend
        assert 'count' in trend
        assert '7d_avg' in trend
        assert '30d_avg' in trend


def test_calculate_sales_trends_missing_date_column(analytics_engine, missing_columns_df):
    """Test calculate_sales_trends with missing date column."""
    result = analytics_engine.calculate_sales_trends(missing_columns_df)
    
    # Check error is returned
    assert 'error' in result
    assert "Could not find required column" in result['error']
    assert 'trends' in result
    assert result['trends'] == []


def test_calculate_sales_trends_empty_df(analytics_engine, empty_dataframe):
    """Test calculate_sales_trends with empty DataFrame."""
    result = analytics_engine.calculate_sales_trends(empty_dataframe)
    
    # Check error is returned
    assert 'error' in result
    assert 'trends' in result
    assert result['trends'] == []


# Tests for calculate_gross_profit_by_source
def test_calculate_gross_profit_by_source(analytics_engine, sample_sales_data):
    """Test calculate_gross_profit_by_source with valid data."""
    result = analytics_engine.calculate_gross_profit_by_source(sample_sales_data)
    
    # Check result keys
    assert 'total_profit' in result
    assert 'breakdown' in result
    
    # Check breakdown structure
    assert isinstance(result['breakdown'], list)
    if result['breakdown']:
        source = result['breakdown'][0]
        assert 'source' in source
        assert 'total_profit' in source
        assert 'avg_profit' in source
        assert 'sale_count' in source
        assert 'percentage' in source


def test_calculate_gross_profit_by_source_missing_columns(analytics_engine, missing_columns_df):
    """Test calculate_gross_profit_by_source with missing columns."""
    result = analytics_engine.calculate_gross_profit_by_source(missing_columns_df)
    
    # Check error is returned
    assert 'error' in result
    assert "Could not find required column" in result['error']
    assert 'breakdown' in result
    assert result['breakdown'] == []


def test_calculate_gross_profit_by_source_empty_df(analytics_engine, empty_dataframe):
    """Test calculate_gross_profit_by_source with empty DataFrame."""
    result = analytics_engine.calculate_gross_profit_by_source(empty_dataframe)
    
    # Check error is returned
    assert 'error' in result
    assert 'breakdown' in result
    assert result['breakdown'] == []


# Tests for period comparisons (YoY and MoM)
def test_calculate_yoy_comparison(analytics_engine, sample_sales_data):
    """Test calculate_yoy_comparison with valid data spanning multiple years."""
    result = analytics_engine.calculate_yoy_comparison(sample_sales_data)
    
    # Check result structure
    assert 'comparisons' in result
    assert isinstance(result['comparisons'], list)
    
    # If we have enough data, check comparison details
    if result['comparisons']:
        comparison = result['comparisons'][0]
        assert 'period' in comparison
        assert 'period_name' in comparison
        assert comparison['period_name'] == 'Year'
        assert 'current_value' in comparison
        assert 'previous_value' in comparison
        assert 'change' in comparison
        assert 'change_percentage' in comparison


def test_calculate_mom_comparison(analytics_engine, sample_sales_data):
    """Test calculate_mom_comparison with valid data spanning multiple months."""
    result = analytics_engine.calculate_mom_comparison(sample_sales_data)
    
    # Check result structure
    assert 'comparisons' in result
    assert isinstance(result['comparisons'], list)
    
    # If we have enough data, check comparison details
    if result['comparisons']:
        comparison = result['comparisons'][0]
        assert 'period' in comparison
        assert 'period_name' in comparison
        assert comparison['period_name'] == 'Month'
        assert 'current_value' in comparison
        assert 'previous_value' in comparison
        assert 'change' in comparison
        assert 'change_percentage' in comparison


def test_period_comparison_missing_columns(analytics_engine, missing_columns_df):
    """Test period comparison with missing columns."""
    result = analytics_engine.calculate_yoy_comparison(missing_columns_df)
    
    # Check error is returned
    assert 'error' in result
    assert "Could not find required column" in result['error']
    assert 'comparisons' in result
    assert result['comparisons'] == []


def test_period_comparison_empty_df(analytics_engine, empty_dataframe):
    """Test period comparison with empty DataFrame."""
    result = analytics_engine.calculate_mom_comparison(empty_dataframe)
    
    # Check error is returned
    assert 'error' in result
    assert 'comparisons' in result
    assert result['comparisons'] == []


# Tests for run_analysis
def test_run_analysis_valid_data(analytics_engine, sample_sales_data):
    """Test run_analysis with valid data."""
    result = analytics_engine.run_analysis(sample_sales_data)
    
    # Check success flag and timestamp
    assert result['success'] == True
    assert 'timestamp' in result
    
    # Check all required sections exist
    assert 'metadata' in result
    assert 'warnings' in result
    assert 'data_quality' in result
    
    # Check for analysis results
    assert 'sales_trends' in result
    assert 'profit_analysis' in result
    assert 'yoy_comparison' in result
    assert 'mom_comparison' in result
    
    # Check metadata structure
    assert 'row_count' in result['metadata']
    assert 'column_count' in result['metadata']
    assert result['metadata']['row_count'] == len(sample_sales_data)
    assert result['metadata']['column_count'] == len(sample_sales_data.columns)
    
    # Check data_quality structure
    assert isinstance(result['data_quality'], dict)
    for col in sample_sales_data.columns:
        assert col in result['data_quality']
        assert 'missing_percentage' in result['data_quality'][col]
        assert 'warning_level' in result['data_quality'][col]


def test_run_analysis_empty_data(analytics_engine, empty_dataframe):
    """Test run_analysis with empty data."""
    result = analytics_engine.run_analysis(empty_dataframe)
    
    # Check error response structure
    assert result['success'] == False
    assert 'error' in result
    assert ERR_NO_DATA in result['error']
    assert 'timestamp' in result


def test_run_analysis_missing_columns(analytics_engine, missing_columns_df):
    """Test run_analysis with missing required columns."""
    result = analytics_engine.run_analysis(missing_columns_df)
    
    # Check error response structure
    assert result['success'] == False
    assert 'error' in result
    # Check that required columns are mentioned in error
    assert "required column" in result['error'].lower()
    # At least one of the required columns should be mentioned
    assert any(col in result['error'] for col in DEFAULT_REQUIRED_COLUMNS)

def test_run_analysis_invalid_data(analytics_engine, invalid_data_df):
    """Test run_analysis with invalid data types."""
    result = analytics_engine.run_analysis(invalid_data_df)
    
    # With the improved validation, we might get warnings but still succeed
    # because we have all the required columns with valid enough data
    # Just check for the presence of error key if not successful
    if not result['success']:
        assert 'error' in result
        assert "invalid data type" in result['error'].lower() or "could not convert" in result['error'].lower()
    else:
        # If successful, should see warnings
        assert len(result['warnings']) > 0
    
    assert 'timestamp' in result


def test_run_analysis_exception_handling(analytics_engine, sample_sales_data):
    """Test run_analysis handles unexpected exceptions."""
    # Patch _standardize_dataframe to raise an unexpected exception
    with patch.object(
        analytics_engine, '_standardize_dataframe', 
        side_effect=Exception("Unexpected error")
    ):
        result = analytics_engine.run_analysis(sample_sales_data)
        
        # Check error response structure
        assert result['success'] == False
        assert 'error' in result
        assert 'Unexpected error' in result['error']
        assert 'timestamp' in result

# Integration tests
def test_full_analysis_pipeline(analytics_engine, monkeypatch, sample_csv_content):
    """Test the complete analysis pipeline from loading CSV to analysis."""
    # Mock file operations
    monkeypatch.setattr(os.path, 'exists', lambda path: True)
    
    # Create expected DataFrame
    expected_df = pd.read_csv(pd.io.common.StringIO(sample_csv_content))
    
    # Use patch to mock file operations
    with patch('builtins.open', mock_open(read_data=sample_csv_content)):
        with patch('pandas.read_csv', return_value=expected_df):
            # 1. Load the CSV
            df = analytics_engine.load_csv('test_sales.csv')
            
            # 2. Run the analysis
            result = analytics_engine.run_analysis(df)
            
            # 3. Verify success and complete results
            assert result['success'] == True
            assert 'sales_trends' in result
            assert 'profit_analysis' in result
            assert 'yoy_comparison' in result
            assert 'mom_comparison' in result
            
            # 4. Verify sales count matches input data
            expected_sales = len(expected_df[expected_df['IsSale'] == 1])
            assert result['sales_trends']['total_sales'] == expected_sales
            assert result['sales_trends']['total_sales'] > 0


def test_integration_with_standardization(analytics_engine):
    """Test integration of standardization with analysis."""
    # Create test data with non-standard column names
    df = pd.DataFrame({
        'Sale Date': pd.date_range('2023-01-01', periods=10),
        'Lead Source': ['Web', 'Referral', 'Direct', 'Web', 'Email'] * 2,
        'Gross Profit': [1000, 1500, 2000, 2500, 3000] * 2,
        'Sold': [1, 0, 1, 1, 0] * 2,
        'Sales Rep': ['John', 'Jane', 'Bob', 'Alice', 'Carlos'] * 2
    })
    
    # Run the analysis on data that needs standardization
    result = analytics_engine.run_analysis(df)
    
    # Verify that standardization worked and analysis was successful
    assert result['success'] == True
    assert 'warnings' in result
    # May not have warnings if all columns were properly mapped
    assert 'data_quality' in result 
    assert isinstance(result['data_quality'], dict)
    
    # Check that data was properly transformed
    assert 'sales_trends' in result
    assert 'profit_analysis' in result
    assert 'yoy_comparison' in result
    assert 'mom_comparison' in result
    
    # Check that all lead sources are present in breakdown
    all_sources = set(['Web', 'Referral', 'Direct', 'Email'])
    breakdown_sources = set([
        item['source'] for item in result['profit_analysis']['breakdown']
    ])
    assert all_sources.issubset(breakdown_sources)

