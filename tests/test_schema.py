"""
Tests for schema validation module.
"""

import pytest
import pandas as pd
from src.utils.schema import (
    validate_sheet_schema,
    validate_workbook_schema,
    find_matching_column,
    SchemaValidationError
)

@pytest.fixture
def sample_sales_df():
    """Create a sample sales DataFrame."""
    return pd.DataFrame({
        'sale_date': ['2024-01-01'],
        'total_gross': [1000],
        'lead_source': ['CarGurus'],
        'sales_rep': ['John Doe'],
        'vin_number': ['12345']
    })

@pytest.fixture
def sample_inventory_df():
    """Create a sample inventory DataFrame."""
    return pd.DataFrame({
        'vin': ['12345'],
        'days_in_stock': [30],
        'list_price': [25000]
    })

@pytest.fixture
def sample_leads_df():
    """Create a sample leads DataFrame."""
    return pd.DataFrame({
        'lead_date': ['2024-01-01'],
        'source': ['Website'],
        'status': ['New']
    })

def test_find_matching_column():
    """Test column matching with aliases."""
    df = pd.DataFrame({
        'SaleDate': [],
        'GrossProfit': [],
        'lead_source': []
    })
    
    assert find_matching_column(df, ['date', 'sale_date']) == 'SaleDate'
    assert find_matching_column(df, ['gross', 'gross_profit']) == 'GrossProfit'
    assert find_matching_column(df, ['lead_source', 'source']) == 'lead_source'
    assert find_matching_column(df, ['nonexistent']) is None

def test_validate_sheet_schema_success(sample_sales_df):
    """Test successful sheet schema validation."""
    success, column_map, missing = validate_sheet_schema(sample_sales_df, 'sales')
    assert success
    assert len(missing) == 0
    assert column_map['date'] == 'sale_date'
    assert column_map['gross'] == 'total_gross'
    assert column_map['lead_source'] == 'lead_source'
    assert column_map['sales_rep'] == 'sales_rep'
    assert column_map['vin'] == 'vin_number'

def test_validate_sheet_schema_missing_columns():
    """Test sheet schema validation with missing columns."""
    df = pd.DataFrame({
        'sale_date': [],
        'total_gross': []
    })
    success, column_map, missing = validate_sheet_schema(df, 'sales')
    assert not success
    assert 'lead_source' in missing
    assert 'sales_rep' in missing
    assert 'vin' in missing

def test_validate_workbook_schema_success(sample_sales_df, sample_inventory_df, sample_leads_df):
    """Test successful workbook schema validation."""
    sheets = {
        'sales': sample_sales_df,
        'inventory': sample_inventory_df,
        'leads': sample_leads_df
    }
    success, column_maps, missing = validate_workbook_schema(sheets)
    assert success
    assert len(missing) == 0
    assert 'sales' in column_maps
    assert 'inventory' in column_maps
    assert 'leads' in column_maps

def test_validate_workbook_schema_missing_required():
    """Test workbook schema validation with missing required sheet."""
    sheets = {
        'inventory': pd.DataFrame(),
        'leads': pd.DataFrame()
    }
    with pytest.raises(SchemaValidationError) as exc:
        validate_workbook_schema(sheets)
    assert 'sales' in exc.value.missing_sheets

def test_validate_workbook_schema_missing_columns():
    """Test workbook schema validation with missing columns."""
    sheets = {
        'sales': pd.DataFrame({
            'sale_date': [],
            'total_gross': []
        })
    }
    with pytest.raises(SchemaValidationError) as exc:
        validate_workbook_schema(sheets)
    assert 'sales' in exc.value.missing_columns
    assert 'lead_source' in exc.value.missing_columns['sales']

def test_schema_validation_error_str():
    """Test SchemaValidationError string representation."""
    error = SchemaValidationError(
        "Validation failed",
        missing_sheets=['sales'],
        missing_columns={'inventory': ['vin', 'price']}
    )
    error_str = str(error)
    assert "Validation failed" in error_str
    assert "Missing required sheets: sales" in error_str
    assert "Sheet 'inventory' missing columns: vin, price" in error_str