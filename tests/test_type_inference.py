"""
Tests for the type_inference module.

This module tests the type inference functionality that identifies semantic types
for DataFrame columns based on content and column names.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
from typing import Dict, List

# Import the module to test
try:
    from v2_watchdog_ai.type_inference import (
        infer_column_types,
        infer_types_pandas,
        infer_types_visions,
        get_column_schema,
        AUTO_TYPES
    )
    HAS_VISIONS = True
except ImportError:
    # Define empty functions with the same signatures for testing
    def infer_column_types(df): return {}
    def infer_types_pandas(df): return {}
    def infer_types_visions(df): return {}
    def get_column_schema(df): return {}
    AUTO_TYPES = {}
    HAS_VISIONS = False


class TestTypeInferencePandas:
    """Tests for pandas-based type inference (fallback method)."""
    
    def test_basic_type_detection(self):
        """Test detection of basic pandas dtypes."""
        # Arrange
        df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, 3.3],
            'str_col': ['a', 'b', 'c'],
            'bool_col': [True, False, True],
            'date_col': pd.to_datetime(['2023-01-01', '2023-02-01', '2023-03-01'])
        })
        
        # Act
        result = infer_types_pandas(df)
        
        # Assert
        assert result['int_col'] == 'integer'
        assert result['float_col'] == 'float'
        assert result['str_col'] == 'string'
        assert result['bool_col'] == 'boolean'
        assert result['date_col'] == 'date'
    
    def test_categorical_detection(self):
        """Test detection of categorical data."""
        # Arrange
        df = pd.DataFrame({
            'low_cardinality': ['red', 'blue', 'green', 'red', 'blue', 'green', 'red', 'blue', 'green', 'red', 'blue'],
            'high_cardinality': [f"value_{i}" for i in range(100)]
        })
        
        # Act
        result = infer_types_pandas(df)
        
        # Assert
        assert result['low_cardinality'] == 'categorical'
        assert result['high_cardinality'] == 'string'
    
    def test_automotive_specific_types(self):
        """Test detection of automotive-specific column types."""
        # Arrange
        df = pd.DataFrame({
            'vin': ['1HGCM82633A123456', '5TFBW5F13AX123457', 'WBAGH83576D123458'],
            'price': [28500, 45750, 62000],
            'sale_date': ['2023-01-15', '2023-02-20', '2023-03-05'],
            'lead_source': ['Website', 'Walk-in', 'CarGurus'],
            'make': ['Honda', 'Toyota', 'BMW']
        })
        
        # Act
        result = infer_types_pandas(df)
        
        # Assert
        assert result['vin'] == 'vin'
        assert result['price'] == 'price'
        assert result['sale_date'] == 'date'
        assert result['lead_source'] == 'lead_source'
        assert 'vehicle' in result['make'] or result['make'] == 'string'
    
    def test_mixed_types(self):
        """Test handling of columns with mixed data types."""
        # Arrange
        df = pd.DataFrame({
            'mixed_col': [1, 'string', 3.5]
        })
        
        # Act
        result = infer_types_pandas(df)
        
        # Assert
        assert result['mixed_col'] == 'string'  # pandas converts to object/string


@pytest.mark.skipif(not HAS_VISIONS, reason="visions package not installed")
class TestTypeInferenceVisions:
    """Tests for visions-based type inference."""
    
    def test_basic_visions_types(self):
        """Test detection of basic types using visions."""
        # Arrange
        df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, 3.3],
            'str_col': ['a', 'b', 'c'],
            'bool_col': [True, False, True],
            'date_col': pd.to_datetime(['2023-01-01', '2023-02-01', '2023-03-01'])
        })
        
        # Act
        result = infer_types_visions(df)
        
        # Assert
        assert result['int_col'] in ['integer', 'count']
        assert result['float_col'] == 'float'
        assert result['str_col'] == 'string'
        assert result['bool_col'] == 'boolean'
        assert result['date_col'] == 'date'
    
    def test_advanced_visions_types(self):
        """Test detection of advanced types using visions."""
        # This test should only run if visions is available
        # Arrange
        df = pd.DataFrame({
            'categorical': pd.Categorical(['red', 'blue', 'green', 'red']),
            'url': ['https://example.com', 'https://test.org', 'https://site.net']
        })
        
        # Act
        result = infer_types_visions(df)
        
        # Assert
        assert result['categorical'] == 'categorical'
        assert result['url'] in ['url', 'string']  # might be url if pattern is recognized


class TestMainTypeInference:
    """Tests for the main type inference function."""
    
    def test_infer_column_types(self):
        """Test the main entry point for type inference."""
        # Arrange
        df = pd.DataFrame({
            'vin': ['1HGCM82633A123456', '5TFBW5F13AX123457', 'WBAGH83576D123458'],
            'price': [28500, 45750, 62000],
            'sale_date': ['2023-01-15', '2023-02-20', '2023-03-05']
        })
        
        # Act
        result = infer_column_types(df)
        
        # Assert
        assert isinstance(result, dict)
        assert len(result) == 3
        assert all(col in result for col in df.columns)
    
    def test_empty_dataframe(self):
        """Test type inference with an empty DataFrame."""
        # Arrange
        df = pd.DataFrame()
        
        # Act
        result = infer_column_types(df)
        
        # Assert
        assert isinstance(result, dict)
        assert len(result) == 0


class TestColumnSchema:
    """Tests for the column schema generation function."""
    
    def test_schema_structure(self):
        """Test the structure of the generated schema."""
        # Arrange
        df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, 3.3],
            'str_col': ['a', 'b', 'c']
        })
        
        # Act
        schema = get_column_schema(df)
        
        # Assert
        assert isinstance(schema, dict)
        for col in df.columns:
            assert col in schema
            assert 'type' in schema[col]
            assert 'dtype' in schema[col]
            assert 'unique_count' in schema[col]
            assert 'missing_count' in schema[col]
            assert 'sample_values' in schema[col]
    
    def test_numeric_metadata(self):
        """Test that numeric columns have min/max metadata."""
        # Arrange
        df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, 3.3],
            'str_col': ['a', 'b', 'c']
        })
        
        # Act
        schema = get_column_schema(df)
        
        # Assert
        assert 'min' in schema['int_col']
        assert 'max' in schema['int_col']
        assert 'min' in schema['float_col']
        assert 'max' in schema['float_col']
        assert 'min' not in schema['str_col']
        assert 'max' not in schema['str_col']
    
    def test_sample_values(self):
        """Test that sample values are correctly captured."""
        # Arrange
        df = pd.DataFrame({
            'col': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        })
        
        # Act
        schema = get_column_schema(df)
        
        # Assert
        assert len(schema['col']['sample_values']) <= 5  # Should have at most 5 samples
        assert all(v in df['col'].values for v in schema['col']['sample_values'])
    
    def test_missing_values(self):
        """Test handling of missing values in schema."""
        # Arrange
        df = pd.DataFrame({
            'col': [1, None, 3, None, 5]
        })
        
        # Act
        schema = get_column_schema(df)
        
        # Assert
        assert schema['col']['missing_count'] == 2
        assert len(schema['col']['sample_values']) <= 3  # Only 3 non-null values
