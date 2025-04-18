"""
Unit tests for KPI dashboard data export functionality.
"""

import pytest
import pandas as pd
import io
from src.ui.pages.kpi_dashboard import get_download_data

@pytest.fixture
def sample_export_data():
    """Create sample data for export testing."""
    return pd.DataFrame({
        'SaleDate': pd.date_range(start='2023-01-01', periods=10),
        'SalesRepName': ['Alice', 'Bob'] * 5,
        'TotalGross': range(1000, 11000, 1000),
        'LeadSource': ['Web', 'Phone'] * 5
    })

def test_basic_export():
    """Test basic CSV export functionality."""
    df = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
    
    csv_data = get_download_data(df, "test")
    
    # Read back the CSV data
    result_df = pd.read_csv(io.StringIO(csv_data.decode('utf-8')))
    
    assert len(result_df) == len(df)
    assert all(col in result_df.columns for col in df.columns)
    assert result_df.values.tolist() == df.values.tolist()

def test_export_with_dates(sample_export_data):
    """Test exporting data with date columns."""
    csv_data = get_download_data(sample_export_data, "sales")
    result_df = pd.read_csv(io.StringIO(csv_data.decode('utf-8')))
    
    assert len(result_df) == len(sample_export_data)
    assert 'SaleDate' in result_df.columns
    assert pd.to_datetime(result_df['SaleDate']).dtype == 'datetime64[ns]'

def test_export_with_numeric_formatting(sample_export_data):
    """Test numeric formatting in exports."""
    csv_data = get_download_data(sample_export_data, "sales")
    result_df = pd.read_csv(io.StringIO(csv_data.decode('utf-8')))
    
    assert result_df['TotalGross'].dtype in ['int64', 'float64']
    assert result_df['TotalGross'].sum() == sample_export_data['TotalGross'].sum()

def test_export_empty_dataframe():
    """Test exporting an empty DataFrame."""
    df = pd.DataFrame(columns=['col1', 'col2'])
    csv_data = get_download_data(df, "empty")
    
    result_df = pd.read_csv(io.StringIO(csv_data.decode('utf-8')))
    assert len(result_df) == 0
    assert list(result_df.columns) == ['col1', 'col2']

def test_export_with_missing_values():
    """Test exporting data with missing values."""
    df = pd.DataFrame({
        'col1': [1, None, 3],
        'col2': ['a', 'b', None]
    })
    
    csv_data = get_download_data(df, "missing")
    result_df = pd.read_csv(io.StringIO(csv_data.decode('utf-8')))
    
    assert len(result_df) == len(df)
    assert result_df['col1'].isna().sum() == 1
    assert result_df['col2'].isna().sum() == 1

def test_export_with_special_characters():
    """Test exporting data with special characters."""
    df = pd.DataFrame({
        'col1': ['test,with,commas', 'test"with"quotes'],
        'col2': ['test\nwith\nnewlines', 'test with spaces']
    })
    
    csv_data = get_download_data(df, "special")
    result_df = pd.read_csv(io.StringIO(csv_data.decode('utf-8')))
    
    assert len(result_df) == len(df)
    assert result_df.values.tolist() == df.values.tolist()

def test_large_dataset_export(sample_export_data):
    """Test exporting a large dataset."""
    # Create a larger dataset by repeating the sample data
    large_df = pd.concat([sample_export_data] * 1000, ignore_index=True)
    
    csv_data = get_download_data(large_df, "large")
    result_df = pd.read_csv(io.StringIO(csv_data.decode('utf-8')))
    
    assert len(result_df) == len(large_df)
    assert all(col in result_df.columns for col in large_df.columns)