"""
Tests for the data_cleaner module.

This module contains tests for the data cleaning functions that standardize
column names, normalize data formats, and handle missing values.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
from typing import Dict, List

# Import the module to test
try:
    from v2_watchdog_ai.data_cleaner import (
        clean_column_names,
        normalize_date_format,
        normalize_currency_format,
        normalize_vin,
        handle_missing_values,
        clean_dataframe
    )
except ImportError:
    # This structure allows for testing before the module is fully implemented
    # Just define empty functions with the same signatures
    def clean_column_names(df): return df
    def normalize_date_format(df, date_columns): return df
    def normalize_currency_format(df, currency_columns): return df
    def normalize_vin(df, vin_column): return df
    def handle_missing_values(df, strategy): return df
    def clean_dataframe(df, date_columns=None, currency_columns=None, missing_value_strategy=None, vin_column='vin'): return df


class TestColumnNameCleaning:
    """Tests for the column name cleaning functionality."""
    
    def test_clean_simple_column_names(self):
        """Test cleaning of simple column names."""
        # Arrange
        df = pd.DataFrame({
            'First Name': ['John', 'Jane'],
            'Last Name': ['Doe', 'Smith'],
            'Age': [30, 25]
        })
        
        # Act
        result = clean_column_names(df)
        
        # Assert
        expected_columns = ['first_name', 'last_name', 'age']
        assert list(result.columns) == expected_columns
        assert df.equals(result[df.columns]) # Data should be preserved
    
    def test_clean_complex_column_names(self):
        """Test cleaning of complex column names with special characters."""
        # Arrange
        df = pd.DataFrame({
            'User\'s First-Name!': ['John', 'Jane'],
            '$$$ Amount (USD)': [100, 200],
            '   Spaces   ': ['Many', 'Spaces']
        })
        
        # Act
        result = clean_column_names(df)
        
        # Assert
        expected_columns = ['user_s_first_name', 'amount_usd', 'spaces']
        assert list(result.columns) == expected_columns
    
    def test_handle_duplicate_column_names(self):
        """Test handling of duplicate column names after cleaning."""
        # Arrange
        df = pd.DataFrame({
            'First Name': ['John', 'Jane'],
            'First-Name': ['Doe', 'Smith'], # Will clean to same as above
            'first_name': [30, 25] # Already in snake_case
        })
        
        # Act
        result = clean_column_names(df)
        
        # Assert
        # Should have unique names like first_name, first_name_1, first_name_2
        assert len(set(result.columns)) == 3
        assert all('first_name' in col for col in result.columns)
        
    def test_preserve_original_data(self):
        """Test that original data is preserved during column name cleaning."""
        # Arrange
        df = pd.DataFrame({
            'First Name': ['John', 'Jane'],
            'Last Name': ['Doe', 'Smith'],
            'Age': [30, 25]
        })
        
        # Act
        result = clean_column_names(df)
        
        # Assert - check if data is unchanged
        assert result['first_name'].tolist() == ['John', 'Jane']
        assert result['last_name'].tolist() == ['Doe', 'Smith']
        assert result['age'].tolist() == [30, 25]


class TestDateNormalization:
    """Tests for date format normalization."""
    
    def test_normalize_standard_date_formats(self):
        """Test normalization of standard date formats."""
        # Arrange
        df = pd.DataFrame({
            'date_1': ['2023-01-15', '2023-02-20'],
            'date_2': ['01/15/2023', '02/20/2023'],
            'date_3': ['15-Jan-2023', '20-Feb-2023']
        })
        
        # Act
        result = normalize_date_format(df, ['date_1', 'date_2', 'date_3'])
        
        # Assert
        for col in ['date_1', 'date_2', 'date_3']:
            assert pd.api.types.is_datetime64_dtype(result[col])
        
        # Check if dates were normalized correctly
        assert result['date_1'][0] == result['date_2'][0] == result['date_3'][0]
        assert result['date_1'][1] == result['date_2'][1] == result['date_3'][1]
    
    def test_handle_invalid_dates(self):
        """Test handling of invalid date values."""
        # Arrange
        df = pd.DataFrame({
            'date_col': ['2023-01-15', 'invalid', '2023-02-20', None]
        })
        
        # Act
        result = normalize_date_format(df, ['date_col'])
        
        # Assert
        assert pd.api.types.is_datetime64_dtype(result['date_col'])
        assert pd.isna(result['date_col'][1])  # Invalid date should be NaT
        assert pd.isna(result['date_col'][3])  # None should be NaT
        
    def test_nonexistent_columns(self):
        """Test behavior with nonexistent columns."""
        # Arrange
        df = pd.DataFrame({
            'existing_date': ['2023-01-15', '2023-02-20']
        })
        
        # Act
        result = normalize_date_format(df, ['existing_date', 'nonexistent_date'])
        
        # Assert
        assert pd.api.types.is_datetime64_dtype(result['existing_date'])
        assert 'nonexistent_date' not in result.columns


class TestCurrencyNormalization:
    """Tests for currency format normalization."""
    
    def test_normalize_standard_currency_formats(self):
        """Test normalization of standard currency formats."""
        # Arrange
        df = pd.DataFrame({
            'price_1': ['$1,234.56', '$2,345.67'],
            'price_2': ['1,234.56', '2,345.67'],
            'price_3': ['$1234.56', '$2345.67']
        })
        
        # Act
        result = normalize_currency_format(df, ['price_1', 'price_2', 'price_3'])
        
        # Assert
        for col in ['price_1', 'price_2', 'price_3']:
            assert pd.api.types.is_numeric_dtype(result[col])
        
        # Check if values were normalized correctly
        assert result['price_1'][0] == result['price_2'][0] == result['price_3'][0] == 1234.56
        assert result['price_1'][1] == result['price_2'][1] == result['price_3'][1] == 2345.67
    
    def test_handle_invalid_currency(self):
        """Test handling of invalid currency values."""
        # Arrange
        df = pd.DataFrame({
            'price_col': ['$1,234.56', 'invalid', '$2,345.67', None]
        })
        
        # Act
        result = normalize_currency_format(df, ['price_col'])
        
        # Assert
        assert pd.api.types.is_numeric_dtype(result['price_col'])
        assert pd.isna(result['price_col'][1])  # Invalid currency should be NaN
        assert pd.isna(result['price_col'][3])  # None should be NaN
        
    def test_negative_currency(self):
        """Test handling of negative currency values."""
        # Arrange
        df = pd.DataFrame({
            'price_col': ['$1,234.56', '-$500.00', '($750.25)']
        })
        
        # Act
        result = normalize_currency_format(df, ['price_col'])
        
        # Assert
        assert result['price_col'][0] == 1234.56
        assert result['price_col'][1] == -500.00
        # Parentheses notation for negative values should be handled
        assert result['price_col'][2] == -750.25


class TestVINNormalization:
    """Tests for VIN format normalization."""
    
    def test_normalize_standard_vins(self):
        """Test normalization of standard VINs."""
        # Arrange
        df = pd.DataFrame({
            'vin': ['1hgcm82633a123456', '5TFBW5F13AX123457', ' WBAGH83576D123458 ']
        })
        
        # Act
        result = normalize_vin(df)
        
        # Assert
        expected_vins = ['1HGCM82633A123456', '5TFBW5F13AX123457', 'WBAGH83576D123458']
        assert result['vin'].tolist() == expected_vins
    
    def test_flag_invalid_vins(self, caplog):
        """Test flagging of invalid VINs (not 17 characters)."""
        # Arrange
        df = pd.DataFrame({
            'vin': ['1HGCM82633A123456', 'INVALID', '5TFBW5F13AX123457']
        })
        
        # Act
        result = normalize_vin(df)
        
        # Assert
        assert result['vin'][0] == '1HGCM82633A123456'
        assert result['vin'][1] == 'INVALID'
        assert "invalid VINs" in caplog.text.lower()
    
    def test_custom_vin_column_name(self):
        """Test using a custom column name for VIN."""
        # Arrange
        df = pd.DataFrame({
            'vehicle_id_number': ['1hgcm82633a123456', '5TFBW5F13AX123457']
        })
        
        # Act
        result = normalize_vin(df, vin_column='vehicle_id_number')
        
        # Assert
        expected_vins = ['1HGCM82633A123456', '5TFBW5F13AX123457']
        assert result['vehicle_id_number'].tolist() == expected_vins


class TestMissingValueHandling:
    """Tests for missing value handling."""
    
    def test_default_strategy(self):
        """Test default strategy (leave missing values as is)."""
        # Arrange
        df = pd.DataFrame({
            'col1': [1, None, 3],
            'col2': ['a', 'b', None]
        })
        
        # Act
        result = handle_missing_values(df)
        
        # Assert
        assert pd.isna(result['col1'][1])
        assert pd.isna(result['col2'][2])
    
    def test_mean_strategy(self):
        """Test mean imputation strategy."""
        # Arrange
        df = pd.DataFrame({
            'col1': [1, None, 3],
            'col2': [10, 20, None]
        })
        
        strategy = {
            'col1': 'mean',
            'col2': 'mean'
        }
        
        # Act
        result = handle_missing_values(df, strategy)
        
        # Assert
        assert result['col1'][1] == 2  # Mean of 1 and 3
        assert result['col2'][2] == 15  # Mean of 10 and 20
    
    def test_custom_value_strategy(self):
        """Test custom value imputation strategy."""
        # Arrange
        df = pd.DataFrame({
            'col1': [1, None, 3],
            'col2': ['a', 'b', None]
        })
        
        strategy = {
            'col1': 0,
            'col2': 'MISSING'
        }
        
        # Act
        result = handle_missing_values(df, strategy)
        
        # Assert
        assert result['col1'][1] == 0
        assert result['col2'][2] == 'MISSING'
    
    def test_mixed_strategies(self):
        """Test using different strategies for different columns."""
        # Arrange
        df = pd.DataFrame({
            'col1': [1, None, 3],
            'col2': [10, 20, None],
            'col3': ['a', 'b', None]
        })
        
        strategy = {
            'col1': 'mean',
            'col2': 0,
            'col3': 'none'  # Explicit none strategy
        }
        
        # Act
        result = handle_missing_values(df, strategy)
        
        # Assert
        assert result['col1'][1] == 2  # Mean of 1 and 3
        assert result['col2'][2] == 0
        assert pd.isna(result['col3'][2])  # Should be left as is


class TestCleanDataFrame:
    """Tests for the main clean_dataframe function."""
    
    def test_clean_dataframe_integration(self, messy_csv_data):
        """Test the full cleaning process with a messy DataFrame."""
        # Arrange
        # Use the messy_csv_data fixture
        
        # Act
        result = clean_dataframe(
            messy_csv_data,
            date_columns=['Date of Sale'],
            currency_columns=['Price $', 'Dealer Cost', 'Gross'],
            missing_value_strategy={'Dealer Cost': 0},
            vin_column='VIN #'
        )
        
        # Assert
        # Column names should be cleaned
        assert 'vin' in result.columns
        assert 'vehicle_make' in result.columns
        assert 'price' in result.columns
        
        # Dates should be datetime
        assert pd.api.types.is_datetime64_dtype(result['date_of_sale'])
        
        # Currency columns should be numeric
        assert pd.api.types.is_numeric_dtype(result['price'])
        assert pd.api.types.is_numeric_dtype(result['dealer_cost'])
        assert pd.api.types.is_numeric_dtype(result['gross'])
        
        # Missing values should be handled according to strategy
        assert result['dealer_cost'].isna().sum() == 0  # Should be filled with 0
        
        # VINs should be uppercase
        assert result['vin'][0] == '1HGCM82633A123456'
    
    def test_clean_dataframe_empty(self):
        """Test cleaning with an empty DataFrame."""
        # Arrange
        df = pd.DataFrame()
        
        # Act
        result = clean_dataframe(df)
        
        # Assert
        assert result.empty
    
    def test_clean_dataframe_all_defaults(self, messy_csv_data):
        """Test cleaning with all default parameters."""
        # Arrange
        # Use the messy_csv_data fixture
        
        # Act
        # Call with only the required dataframe parameter
        result = clean_dataframe(messy_csv_data)
        
        # Assert
        # Column names should be cleaned
        assert all(col == col.lower().replace(' ', '_') for col in result.columns)
        
        # Data should be preserved
        assert len(result) == len(messy_csv_data)
