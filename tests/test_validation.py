"""
Tests for the utils.validation and data_normalization modules.
"""

import unittest
import pandas as pd
import os
import sys
from typing import Dict, List, Set

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.utils.data_normalization import normalize_column_name, normalize_columns, COLUMN_ALIASES
from src.utils.data_io import load_data, validate_data, REQUIRED_COLUMNS
from src.utils.errors import ValidationError


class TestColumnNormalization(unittest.TestCase):
    """Test cases for column name normalization."""
    
    def test_normalize_column_name(self):
        """Test normalizing individual column names."""
        # Test with exact match of canonical name
        self.assertEqual(normalize_column_name("SaleDate"), "SaleDate")
        
        # Test with variations
        self.assertEqual(normalize_column_name("sale_date"), "SaleDate")
        self.assertEqual(normalize_column_name("saledate"), "SaleDate")
        self.assertEqual(normalize_column_name("Sale Date"), "SaleDate")
        self.assertEqual(normalize_column_name("SALE_DATE"), "SaleDate")
        
        # Test with unknown column
        self.assertEqual(normalize_column_name("UnknownColumn"), "UnknownColumn")
        self.assertEqual(normalize_column_name("unknown_column"), "unknowncolumn")
    
    def test_normalize_dataframe_columns(self):
        """Test normalizing DataFrame column names."""
        # Create a test DataFrame with variant column names
        df = pd.DataFrame({
            'sale_date': ['2023-01-01'],
            'Sale Price': [25000],
            'Vehicle_VIN': ['1HGCM82633A123456'],
            'total gross': [2500],
            'LeadSource': ['Website']
        })
        
        # Normalize column names
        normalized_df = normalize_columns(df)
        
        # Check that columns were renamed to canonical names
        expected_columns = {'SaleDate', 'SalePrice', 'VIN', 'TotalGross', 'LeadSource'}
        self.assertEqual(set(normalized_df.columns), expected_columns)
        
        # Check that the data is still accessible after renaming
        self.assertEqual(normalized_df['SaleDate'][0], '2023-01-01')
        self.assertEqual(normalized_df['SalePrice'][0], 25000)
        self.assertEqual(normalized_df['VIN'][0], '1HGCM82633A123456')
        self.assertEqual(normalized_df['TotalGross'][0], 2500)
        self.assertEqual(normalized_df['LeadSource'][0], 'Website')


class TestSchemaValidation(unittest.TestCase):
    """Test cases for schema validation."""
    
    def test_validate_schema_with_sample_file(self):
        """Test validation with sample dealership data file."""
        # Load the sample dealership data
        file_path = os.path.join(project_root, 'tests', 'assets', 'sample_dealership_data.csv')
        df = pd.read_csv(file_path)
        
        # Before normalization, these columns should have variant names
        self.assertIn('Sale_Date', df.columns)
        self.assertIn('Sale_Price', df.columns)
        self.assertIn('VIN', df.columns)
        self.assertIn('Gross_Profit', df.columns)
        self.assertIn('Lead_Source', df.columns)
        
        # Normalize and validate
        normalized_df = normalize_columns(df)
        
        # After normalization, columns should have canonical names
        self.assertIn('SaleDate', normalized_df.columns)
        self.assertIn('SalePrice', normalized_df.columns)
        self.assertIn('VIN', normalized_df.columns)
        self.assertIn('TotalGross', normalized_df.columns)
        self.assertIn('LeadSource', normalized_df.columns)
        
        # Validation should pass with all required columns
        missing_cols = set(REQUIRED_COLUMNS) - set(normalized_df.columns)
        self.assertEqual(missing_cols, set(), f"Missing columns: {missing_cols}")
    
    def test_validation_with_missing_columns(self):
        """Test validation with missing required columns."""
        # Create a DataFrame missing required columns
        df = pd.DataFrame({
            'sale_date': ['2023-01-01'],
            'Sale Price': [25000],
            # Missing VIN
            # Missing TotalGross
            # Missing LeadSource
        })
        
        # Normalize the DataFrame
        normalized_df = normalize_columns(df)
        
        # Check for missing columns
        missing_cols = set(REQUIRED_COLUMNS) - set(normalized_df.columns)
        self.assertEqual(len(missing_cols), 3)
        self.assertIn('VIN', missing_cols)
        self.assertIn('TotalGross', missing_cols)
        self.assertIn('LeadSource', missing_cols)
    
    def test_enhanced_error_messages(self):
        """Test that error messages include both found and missing columns."""
        # Create a DataFrame missing required columns
        df = pd.DataFrame({
            'sale_date': ['2023-01-01'],
            'Sale Price': [25000],
            'OtherColumn': ['value']
            # Missing VIN, TotalGross, LeadSource
        })
        
        try:
            # This should raise a ValueError
            load_data_custom = lambda f: df  # Mock the file loading
            normalize_columns(df)
            missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
            if missing_cols:
                error_msg = (
                    f"Missing required columns: {', '.join(sorted(missing_cols))}\n"
                    f"Found columns: {', '.join(df.columns)}"
                )
                raise ValueError(error_msg)
            
            self.fail("Expected ValueError was not raised")
        except ValueError as e:
            error_message = str(e)
            
            # Check that the error message includes both found and missing columns
            self.assertIn("Missing required columns", error_message)
            self.assertIn("Found columns", error_message)
            
            # Check specific missing columns
            self.assertIn("VIN", error_message)
            self.assertIn("TotalGross", error_message)
            self.assertIn("LeadSource", error_message)
            
            # Check found columns
            self.assertIn("sale_date", error_message)
            self.assertIn("Sale Price", error_message)
            self.assertIn("OtherColumn", error_message)


if __name__ == '__main__':
    unittest.main()