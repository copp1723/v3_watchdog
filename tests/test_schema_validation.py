"""
Tests for the enhanced schema validation system.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from src.utils.schema import DatasetSchema, ValidationResult, ValidationError

@pytest.fixture
def sample_valid_data():
    """Fixture providing valid sample data."""
    return pd.DataFrame({
        'total_gross': [1000.0, 2000.0, 3000.0],
        'lead_source': ['Website', 'Walk-in', 'CarGurus'],
        'sale_date': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'vin': ['1HGCM82633A123456', '5TFBW5F13AX123457', 'WBAGH83576D123458'],
        'sales_rep': ['John Doe', 'Jane Smith', 'Bob Wilson']
    })

@pytest.fixture
def sample_invalid_data():
    """Fixture providing invalid sample data."""
    return pd.DataFrame({
        'total_gross': ['invalid', 2000.0, 3000.0],
        'source': ['Website', '', 'CarGurus'],  # Missing values and wrong column name
        'transaction_date': ['2024-01-01', 'invalid', '2024-01-03'],
        'vin': ['invalid', '5TFBW5F13AX123457', 'short'],
        'sales_rep': ['John Doe', 'Jane Smith', 'Bob Wilson']
    })

def test_schema_initialization():
    """Test schema initialization and validation."""
    schema = DatasetSchema()
    assert 'required_columns' in schema.schema
    assert 'optional_columns' in schema.schema
    
    # Test invalid schema
    with pytest.raises(ValueError):
        DatasetSchema({'invalid_key': {}})

def test_validate_valid_data(sample_valid_data):
    """Test validation of valid data."""
    schema = DatasetSchema()
    result = schema.validate(sample_valid_data)
    
    assert result.is_valid
    assert not result.missing_required
    assert not result.type_errors
    assert not result.validation_errors
    assert result.column_mapping

def test_validate_invalid_data(sample_invalid_data):
    """Test validation of invalid data."""
    schema = DatasetSchema()
    result = schema.validate(sample_invalid_data)
    
    assert not result.is_valid
    assert result.type_errors
    assert 'gross' in result.type_errors
    assert result.validation_errors

def test_column_aliases():
    """Test column alias recognition."""
    schema = DatasetSchema()
    df = pd.DataFrame({
        'GrossProft': [1000.0],  # Misspelled
        'LeadSource': ['Website'],
        'SaleDate': ['2024-01-01']
    })
    
    result = schema.validate(df)
    assert not result.is_valid  # Should fail due to misspelled column
    
    # Fix the spelling
    df = pd.DataFrame({
        'GrossProfit': [1000.0],
        'LeadSource': ['Website'],
        'SaleDate': ['2024-01-01']
    })
    
    result = schema.validate(df)
    assert result.is_valid

def test_standardize_dataframe(sample_valid_data):
    """Test DataFrame standardization."""
    schema = DatasetSchema()
    std_df, mapping = schema.standardize_dataframe(sample_valid_data)
    
    # Check column renaming
    assert 'gross' in std_df.columns
    assert 'lead_source' in std_df.columns
    assert 'sale_date' in std_df.columns
    
    # Check type conversion
    assert pd.api.types.is_float_dtype(std_df['gross'])
    assert pd.api.types.is_datetime64_dtype(std_df['sale_date'])

def test_validation_with_missing_required():
    """Test validation with missing required columns."""
    schema = DatasetSchema()
    df = pd.DataFrame({
        'lead_source': ['Website'],
        'sale_date': ['2024-01-01']
        # Missing 'gross' column
    })
    
    result = schema.validate(df)
    assert not result.is_valid
    assert 'gross' in result.missing_required

def test_validation_with_invalid_types():
    """Test validation with invalid data types."""
    schema = DatasetSchema()
    df = pd.DataFrame({
        'gross': ['not a number'],
        'lead_source': ['Website'],
        'sale_date': ['2024-01-01']
    })
    
    result = schema.validate(df)
    assert not result.is_valid
    assert 'gross' in result.type_errors

def test_get_column_info():
    """Test retrieval of column information."""
    schema = DatasetSchema()
    info = schema.get_column_info()
    
    assert 'gross' in info
    assert 'type' in info['gross']
    assert 'aliases' in info['gross']
    assert 'description' in info['gross']
    assert 'required' in info['gross']

def test_validation_with_custom_schema():
    """Test validation with a custom schema definition."""
    custom_schema = {
        'required_columns': {
            'custom_field': {
                'type': str,
                'aliases': ['custom', 'field'],
                'description': 'Custom field',
                'validation': lambda x: len(str(x)) > 0
            }
        },
        'optional_columns': {}
    }
    
    schema = DatasetSchema(custom_schema)
    df = pd.DataFrame({
        'custom': ['value1', 'value2']
    })
    
    result = schema.validate(df)
    assert result.is_valid

def test_standardize_invalid_dataframe(sample_invalid_data):
    """Test standardization of invalid DataFrame."""
    schema = DatasetSchema()
    with pytest.raises(ValidationError):
        schema.standardize_dataframe(sample_invalid_data)