"""
Unit tests for data_utils.py module.
"""
import pandas as pd
import pytest
import numpy as np
from datetime import datetime

from watchdog_ai.core.data_utils import (
    find_matching_column,
    normalize_boolean_column,
    clean_numeric_data,
    analyze_data_quality,
    format_metric_value,
    get_error_response
)
from watchdog_ai.core.constants import (
    COLUMN_MAPPINGS,
    NAN_WARNING_THRESHOLD,
    NAN_SEVERE_THRESHOLD
)


# Test Fixtures
@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame with various data types."""
    return pd.DataFrame({
        'SaleDate': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', None],
        'LeadSource': ['Web', 'Referral', 'Direct', 'Web', None],
        'TotalGross': [1000, 1500.50, None, '$2,000.75', 0],
        'IsSale': [1, 'Yes', True, 'Sold', 0],
        'SalesPerson': ['John Doe', 'Jane Smith', None, 'Bob Johnson', 'Alice Brown']
    })


@pytest.fixture
def empty_dataframe():
    """Create an empty DataFrame."""
    return pd.DataFrame()


@pytest.fixture
def quality_test_df():
    """Create a DataFrame with specific percentages of missing values."""
    # 100 rows with varying levels of missing data
    df = pd.DataFrame({
        'no_missing': np.arange(100),
        'low_missing': np.arange(100),
        'medium_missing': np.arange(100),
        'high_missing': np.arange(100),
        'all_missing': np.arange(100)
    })
    
    # Insert NaN values to create specific missing percentages
    # Low: 5% (below warning threshold)
    missing_indices = np.random.choice(100, size=5, replace=False)
    df.loc[missing_indices, 'low_missing'] = np.nan
    
    # Medium: 15% (above warning but below severe threshold)
    missing_indices = np.random.choice(100, size=15, replace=False)
    df.loc[missing_indices, 'medium_missing'] = np.nan
    
    # High: 50% (above severe threshold)
    missing_indices = np.random.choice(100, size=50, replace=False)
    df.loc[missing_indices, 'high_missing'] = np.nan
    
    # All missing: 100%
    df['all_missing'] = np.nan
    
    return df


# Tests for find_matching_column
def test_find_matching_column_exact_match(sample_dataframe):
    """Test finding an exact column match."""
    # Direct column match should work now with our updated function
    assert find_matching_column(sample_dataframe, ['LeadSource']) == 'LeadSource'


def test_find_matching_column_alternate_format(sample_dataframe):
    """Test finding a column with alternate format."""
    # Assuming COLUMN_MAPPINGS has 'Sale Date' as an alternate for 'SaleDate'
    # Mock the behavior if necessary
    COLUMN_MAPPINGS['SaleDate'] = ['SaleDate', 'Sale Date', 'Date of Sale']
    
    # Rename column to test matching
    df = sample_dataframe.rename(columns={'SaleDate': 'Sale Date'})
    assert find_matching_column(df, 'SaleDate') == 'Sale Date'


def test_find_matching_column_no_match(sample_dataframe):
    """Test when no matching column is found."""
    assert find_matching_column(sample_dataframe, 'NonExistentColumn') is None


def test_find_matching_column_empty_df(empty_dataframe):
    """Test with an empty DataFrame."""
    assert find_matching_column(empty_dataframe, 'LeadSource') is None


# Tests for normalize_boolean_column
@pytest.mark.parametrize("input_data,expected", [
    ([1, 0, 1, 0], [1, 0, 1, 0]),
    ([True, False, True, False], [1, 0, 1, 0]),
    (['Yes', 'No', 'True', 'False'], [1, 0, 1, 0]),
    (['yes', 'no', 'true', 'false'], [1, 0, 1, 0]),
    (['Y', 'N', '1', '0'], [0, 0, 1, 0]),  # 'Y' is not in the recognized list
    (['Sold', 'Not Sold', 'Pending', None], [1, 0, 0, 0]),
    ([1.5, 0.0, None, np.nan], [1, 0, 0, 0]),
])
def test_normalize_boolean_column(input_data, expected):
    """Test normalizing various boolean representations."""
    series = pd.Series(input_data)
    result = normalize_boolean_column(series)
    pd.testing.assert_series_equal(result, pd.Series(expected))


# Tests for clean_numeric_data
def test_clean_numeric_data_currency(sample_dataframe):
    """Test cleaning currency values with $ and commas."""
    df = clean_numeric_data(sample_dataframe, 'TotalGross')
    
    # Original: [1000, 1500.50, None, '$2,000.75', 0]
    # Expected: [1000, 1500.50, NaN, 2000.75, 0]
    assert df['TotalGross'].iloc[0] == 1000
    assert df['TotalGross'].iloc[1] == 1500.50
    assert pd.isna(df['TotalGross'].iloc[2])
    assert df['TotalGross'].iloc[3] == 2000.75
    assert df['TotalGross'].iloc[4] == 0


def test_clean_numeric_data_empty_df(empty_dataframe):
    """Test with an empty DataFrame."""
    with pytest.raises(KeyError):
        clean_numeric_data(empty_dataframe, 'TotalGross')


def test_clean_numeric_data_non_numeric():
    """Test with values that can't be converted to numeric."""
    df = pd.DataFrame({'Values': ['abc', '1.23', 'xyz', '$45']})
    result = clean_numeric_data(df, 'Values')
    
    # Expected: [NaN, 1.23, NaN, 45]
    assert pd.isna(result['Values'].iloc[0])
    assert result['Values'].iloc[1] == 1.23
    assert pd.isna(result['Values'].iloc[2])
    assert result['Values'].iloc[3] == 45


# Tests for analyze_data_quality
def test_analyze_data_quality_no_missing(quality_test_df):
    """Test quality analysis for column with no missing values."""
    result = analyze_data_quality(quality_test_df, 'no_missing')
    assert result['total_rows'] == 100
    assert result['nan_count'] == 0
    assert result['nan_percentage'] == 0
    assert result['warning_level'] == 'normal'


def test_analyze_data_quality_low_missing(quality_test_df):
    """Test quality analysis for column with few missing values."""
    result = analyze_data_quality(quality_test_df, 'low_missing')
    assert result['total_rows'] == 100
    assert result['nan_count'] == 5
    assert result['nan_percentage'] == 5.0
    assert result['warning_level'] == 'normal'


def test_analyze_data_quality_medium_missing(quality_test_df):
    """Test quality analysis for column with medium missing values."""
    result = analyze_data_quality(quality_test_df, 'medium_missing')
    assert result['total_rows'] == 100
    assert result['nan_count'] == 15
    assert result['nan_percentage'] == 15.0
    assert result['warning_level'] == 'warning'


def test_analyze_data_quality_high_missing(quality_test_df):
    """Test quality analysis for column with high missing values."""
    result = analyze_data_quality(quality_test_df, 'high_missing')
    assert result['total_rows'] == 100
    assert result['nan_count'] == 50
    assert result['nan_percentage'] == 50.0
    assert result['warning_level'] == 'severe'


def test_analyze_data_quality_all_missing(quality_test_df):
    """Test quality analysis for column with all missing values."""
    result = analyze_data_quality(quality_test_df, 'all_missing')
    assert result['total_rows'] == 100
    assert result['nan_count'] == 100
    assert result['nan_percentage'] == 100.0
    assert result['warning_level'] == 'severe'


def test_analyze_data_quality_empty_df(empty_dataframe):
    """Test quality analysis with an empty DataFrame."""
    with pytest.raises(KeyError):
        analyze_data_quality(empty_dataframe, 'any_column')


# Tests for format_metric_value
@pytest.mark.parametrize("value,metric_name,expected", [
    (1234.56, 'price', '$1,234.56'),
    (1234.56, 'cost', '$1,234.56'),
    (1234.56, 'profit', '$1,234.56'),
    (1234.56, 'revenue', '$1,234.56'),
    (1234.56, 'total_cost', '$1,234.56'),
    (45.67, 'percentage', '45.7%'),
    (45.67, 'rate', '45.7%'),
    (45.67, 'conversion_rate', '45.7%'),
    (45.67, 'success_percent', '45.7%'),
    (1234, 'count', '1,234'),
    (1234, 'sales', '1,234'),
    (1234, 'leads', '1,234'),
])
def test_format_metric_value(value, metric_name, expected):
    """Test formatting different metric types."""
    result = format_metric_value(value, metric_name)
    assert result == expected


# Tests for get_error_response
def test_get_error_response_with_details():
    """Test error response with details."""
    result = get_error_response('DATA_ERROR', 'Missing required columns')
    assert result['error_type'] == 'DATA_ERROR'
    assert result['summary'] == '⚠️ Missing required columns'
    assert result['metrics'] == {}
    assert result['breakdown'] == []
    assert result['confidence'] == 'low'


def test_get_error_response_without_details():
    """Test error response without details."""
    result = get_error_response('PROCESSING_ERROR')
    assert result['error_type'] == 'PROCESSING_ERROR'
    assert result['summary'] == '⚠️ An error occurred.'
    assert result['metrics'] == {}
    assert result['breakdown'] == []
    assert result['confidence'] == 'low'

