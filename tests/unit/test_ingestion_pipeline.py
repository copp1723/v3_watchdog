"""
Tests for the Nova Act ingestion pipeline.
"""

import os
import json
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

from src.nova_act.ingestion_pipeline import (
    map_columns,
    clean_data,
    validate_schema,
    normalize_and_validate,
    COLUMN_MAPPINGS,
    SCHEMAS
)

# Sample test data
@pytest.fixture
def sample_sales_df():
    """Create a sample sales DataFrame for testing."""
    return pd.DataFrame({
        'sale_date': ['2023-01-01', '2023-01-02', '2023-01-03'],
        'customer': ['John Doe', 'Jane Smith', 'Bob Johnson'],
        'gross_profit': [1000, 1500, -200],
        'vehicle_desc': ['2020 Honda Civic', '2019 Toyota Corolla', '2021 Ford F-150'],
        'vin_number': ['ABC123', 'DEF456', 'GHI789'],
        'salesperson': ['Alice', 'Bob', 'Charlie'],
        'lead_source': ['Website', 'Referral', 'Walk-in'],
        'selling_price': ['$25,000', '$20,000', '$35,000']
    })

@pytest.fixture
def sample_inventory_df():
    """Create a sample inventory DataFrame for testing."""
    return pd.DataFrame({
        'stock_#': ['ST001', 'ST002', 'ST003'],
        'vehicle_id': ['ABC123', 'DEF456', 'GHI789'],
        'age': [30, 45, 15],
        'year': [2020, 2019, 2021],
        'make': ['Honda', 'Toyota', 'Ford'],
        'model': ['Civic', 'Corolla', 'F-150'],
        'odometer': [5000, 25000, 1000],
        'price': ['$25,000', '$20,000', '$35,000'],
        'acquisition_cost': [20000, 15000, 30000],
        'certified_flag': [True, False, True],
        'exterior_color': ['Red', 'Blue', 'Black'],
        'interior_color': ['Black', 'Grey', 'Tan'],
        'in_stock_date': ['2022-12-01', '2022-11-15', '2022-12-15']
    })

def test_map_columns_sales(sample_sales_df):
    """Test column mapping for sales data."""
    # Map columns
    mapped_df, mapping_result = map_columns(sample_sales_df, 'dealersocket', 'sales')
    
    # Check standard mapped columns
    assert 'date' in mapped_df.columns
    assert 'gross' in mapped_df.columns
    assert 'vin' in mapped_df.columns
    assert 'lead_source' in mapped_df.columns
    assert 'sale_price' in mapped_df.columns
    
    # Check mapping results
    assert 'gross' in mapping_result['mapped']
    assert mapping_result['mapped']['gross'] == 'gross_profit'
    assert mapping_result['mapped']['vin'] == 'vin_number'
    
    # Check unmapped columns are preserved
    for col in sample_sales_df.columns:
        if col not in mapping_result['mapped'].values():
            assert col in mapping_result['unmapped']

def test_map_columns_inventory(sample_inventory_df):
    """Test column mapping for inventory data."""
    # Map columns
    mapped_df, mapping_result = map_columns(sample_inventory_df, 'dealersocket', 'inventory')
    
    # Check standard mapped columns
    assert 'stock_num' in mapped_df.columns
    assert 'vin' in mapped_df.columns
    assert 'days_in_stock' in mapped_df.columns
    assert 'list_price' in mapped_df.columns
    
    # Check mapping results
    assert 'stock_num' in mapping_result['mapped']
    assert mapping_result['mapped']['stock_num'] == 'stock_#'
    assert mapping_result['mapped']['vin'] == 'vehicle_id'
    
    # Check unmapped columns are preserved
    for col in sample_inventory_df.columns:
        if col not in mapping_result['mapped'].values():
            assert col in mapping_result['unmapped']

def test_clean_data_sales(sample_sales_df):
    """Test data cleaning for sales data."""
    # Clean data
    cleaned_df, cleaning_result = clean_data(sample_sales_df, 'sales')
    
    # Check cleaning actions
    actions = [action['action'] for action in cleaning_result['actions']]
    assert 'convert_type' in actions
    assert 'fill_missing' not in actions  # No missing values in sample data
    
    # Check data types
    if 'date' in cleaned_df.columns:
        assert pd.api.types.is_datetime64_dtype(cleaned_df['date'])
    
    if 'gross' in cleaned_df.columns:
        assert pd.api.types.is_numeric_dtype(cleaned_df['gross'])
    
    if 'sale_price' in cleaned_df.columns:
        assert pd.api.types.is_numeric_dtype(cleaned_df['sale_price'])
        # Check currency parsing
        assert cleaned_df['sale_price'].iloc[0] == 25000  # Parsed from "$25,000"

def test_clean_data_with_missing_values():
    """Test data cleaning with missing values."""
    # Create DataFrame with missing values
    df = pd.DataFrame({
        'date': ['2023-01-01', None, '2023-01-03'],
        'gross': [1000, None, -200],
        'vin': ['ABC123', None, 'GHI789']
    })
    
    # Clean data
    cleaned_df, cleaning_result = clean_data(df, 'sales')
    
    # Check cleaning actions
    fill_missing_actions = [action for action in cleaning_result['actions'] if action['action'] == 'fill_missing']
    assert len(fill_missing_actions) > 0
    
    # Check that missing values were filled
    assert cleaned_df['date'].isna().sum() == 0
    assert cleaned_df['gross'].isna().sum() == 0
    assert cleaned_df['vin'].isna().sum() == 0

def test_validate_schema_valid():
    """Test schema validation with valid data."""
    # Create a DataFrame that matches the sales schema
    df = pd.DataFrame({
        'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
        'gross': [1000.0, 1500.0, -200.0],
        'vin': ['ABC123', 'DEF456', 'GHI789'],
        'lead_source': ['Website', 'Referral', 'Walk-in'],
        'salesperson': ['Alice', 'Bob', 'Charlie'],
        'sale_price': [25000.0, 20000.0, 35000.0]
    })
    
    # Validate against schema
    result = validate_schema(df, 'sales')
    
    # Check validation result
    assert result['valid']
    assert not result['missing_required']
    assert not result['type_mismatches']

def test_validate_schema_invalid():
    """Test schema validation with invalid data."""
    # Create a DataFrame with missing required columns
    df = pd.DataFrame({
        'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
        'vin': ['ABC123', 'DEF456', 'GHI789'],
        # Missing 'gross', 'lead_source', 'salesperson'
        'sale_price': [25000.0, 20000.0, 35000.0]
    })
    
    # Validate against schema
    result = validate_schema(df, 'sales')
    
    # Check validation result
    assert not result['valid']
    assert 'gross' in result['missing_required']
    assert 'lead_source' in result['missing_required']
    assert 'salesperson' in result['missing_required']

def test_validate_schema_type_mismatch():
    """Test schema validation with type mismatches."""
    # Create a DataFrame with type mismatches
    df = pd.DataFrame({
        'date': ['2023-01-01', '2023-01-02', '2023-01-03'],  # Should be datetime
        'gross': ['1000', '1500', '-200'],  # Should be float
        'vin': ['ABC123', 'DEF456', 'GHI789'],
        'lead_source': ['Website', 'Referral', 'Walk-in'],
        'salesperson': ['Alice', 'Bob', 'Charlie'],
        'sale_price': [25000.0, 20000.0, 35000.0]
    })
    
    # Validate against schema
    result = validate_schema(df, 'sales')
    
    # Check validation result
    assert not result['valid']
    assert len(result['type_mismatches']) > 0
    
    # Check specific type mismatches
    date_mismatch = next((m for m in result['type_mismatches'] if m['column'] == 'date'), None)
    assert date_mismatch
    assert date_mismatch['expected'] == 'datetime64[ns]'
    
    gross_mismatch = next((m for m in result['type_mismatches'] if m['column'] == 'gross'), None)
    assert gross_mismatch
    assert gross_mismatch['expected'] == 'float64'

@patch('src.nova_act.ingestion_pipeline.pd.read_csv')
@patch('src.nova_act.ingestion_pipeline.pd.read_excel')
@patch('src.nova_act.ingestion_pipeline.map_columns')
@patch('src.nova_act.ingestion_pipeline.clean_data')
@patch('src.nova_act.ingestion_pipeline.validate_schema')
@patch('src.nova_act.ingestion_pipeline.save_processed_data')
@patch('src.nova_act.ingestion_pipeline.create_metadata_file')
async def test_normalize_and_validate_success(
    mock_create_metadata, mock_save_data, mock_validate, 
    mock_clean, mock_map, mock_read_excel, mock_read_csv
):
    """Test the full normalize_and_validate pipeline with success case."""
    # Mock CSV reading
    mock_df = pd.DataFrame({
        'sale_date': ['2023-01-01', '2023-01-02', '2023-01-03'],
        'gross_profit': [1000, 1500, -200],
        'vin_number': ['ABC123', 'DEF456', 'GHI789'],
        'lead_source': ['Website', 'Referral', 'Walk-in']
    })
    mock_read_csv.return_value = mock_df
    
    # Mock column mapping
    mapped_df = mock_df.copy()
    mock_map.return_value = (mapped_df, {
        'mapped': {'date': 'sale_date', 'gross': 'gross_profit', 'vin': 'vin_number'},
        'unmapped': []
    })
    
    # Mock data cleaning
    cleaned_df = mock_df.copy()
    mock_clean.return_value = (cleaned_df, {'actions': []})
    
    # Mock schema validation
    mock_validate.return_value = {'valid': True, 'message': 'Validation passed'}
    
    # Mock save data
    mock_save_data.return_value = '/path/to/output.csv'
    
    # Mock metadata creation
    mock_create_metadata.return_value = '/path/to/metadata.json'
    
    # Run the pipeline
    result = await normalize_and_validate(
        'test.csv',
        'dealersocket',
        'sales',
        'test_dealer'
    )
    
    # Check pipeline result
    assert result['success']
    assert 'output_path' in result
    assert 'metadata_path' in result
    
    # Verify mock calls
    mock_read_csv.assert_called_once()
    mock_map.assert_called_once()
    mock_clean.assert_called_once()
    mock_validate.assert_called_once()
    mock_save_data.assert_called_once()
    mock_create_metadata.assert_called_once()

@patch('src.nova_act.ingestion_pipeline.pd.read_csv')
async def test_normalize_and_validate_load_error(mock_read_csv):
    """Test the pipeline with a file loading error."""
    # Mock CSV reading to raise an exception
    mock_read_csv.side_effect = Exception("File not found")
    
    # Run the pipeline
    result = await normalize_and_validate(
        'test.csv',
        'dealersocket',
        'sales',
        'test_dealer'
    )
    
    # Check pipeline result
    assert not result['success']
    assert 'error' in result
    assert 'steps' in result
    assert len(result['steps']) == 1
    assert result['steps'][0]['step'] == 'load_data'
    assert not result['steps'][0]['success']

@patch('src.nova_act.ingestion_pipeline.pd.read_csv')
@patch('src.nova_act.ingestion_pipeline.map_columns')
async def test_normalize_and_validate_mapping_error(mock_map, mock_read_csv):
    """Test the pipeline with a column mapping error."""
    # Mock CSV reading
    mock_df = pd.DataFrame({
        'sale_date': ['2023-01-01', '2023-01-02', '2023-01-03'],
        'gross_profit': [1000, 1500, -200]
    })
    mock_read_csv.return_value = mock_df
    
    # Mock column mapping to raise an exception
    mock_map.side_effect = ValueError("Unknown report type")
    
    # Run the pipeline
    result = await normalize_and_validate(
        'test.csv',
        'dealersocket',
        'sales',
        'test_dealer'
    )
    
    # Check pipeline result
    assert not result['success']
    assert 'error' in result
    assert 'steps' in result
    assert len(result['steps']) == 2
    assert result['steps'][0]['step'] == 'load_data'
    assert result['steps'][0]['success']
    assert result['steps'][1]['step'] == 'map_columns'
    assert not result['steps'][1]['success']