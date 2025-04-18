"""
Tests for the term_normalizer module.
"""

import unittest
import pandas as pd
import os
import sys
from typing import Dict, List

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.utils.term_normalizer import TermNormalizer, normalize_terms


class TestTermNormalizer(unittest.TestCase):
    """Test cases for the term normalizer."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create a test normalizer with a set of test rules
        self.normalizer = TermNormalizer()
        
        # Test data
        self.test_data = pd.DataFrame({
            'LeadSource': ['autotrader', 'website', 'walk in', 'CARGURUS', 'facebook mp'],
            'SalesRep': ['John Smith', 'sales_rep', 'Jane Doe', 'sales representative', 'salesperson'],
            'VehicleType': ['Used', 'new car', 'certified pre-owned', 'pre-owned', 'used vehicle']
        })
    
    def test_normalize_term(self):
        """Test normalizing individual terms."""
        # Test some key term normalizations
        self.assertEqual(self.normalizer.normalize_term('autotrader'), 'Auto Trader')
        self.assertEqual(self.normalizer.normalize_term('facebook mp'), 'Facebook')
        self.assertEqual(self.normalizer.normalize_term('walk in'), 'Walk-in')
        self.assertEqual(self.normalizer.normalize_term('sales_rep'), 'Sales Rep')
        self.assertEqual(self.normalizer.normalize_term('new car'), 'New')
        self.assertEqual(self.normalizer.normalize_term('pre-owned'), 'Used')
        
        # Test case insensitivity
        self.assertEqual(self.normalizer.normalize_term('AUTOTRADER'), 'Auto Trader')
        self.assertEqual(self.normalizer.normalize_term('FACEBOOK MP'), 'Facebook')
        
        # Test unknown terms (should return as is)
        self.assertEqual(self.normalizer.normalize_term('Unknown Term'), 'Unknown Term')
        self.assertEqual(self.normalizer.normalize_term('Test'), 'Test')
        
        # Test edge cases
        self.assertEqual(self.normalizer.normalize_term(''), '')
        self.assertIsNone(self.normalizer.normalize_term(None))
    
    def test_normalize_column(self):
        """Test normalizing a column in a DataFrame."""
        # Normalize the LeadSource column
        normalized_df = self.normalizer.normalize_column(self.test_data, 'LeadSource')
        
        # Check that the values were normalized
        expected_lead_sources = ['Auto Trader', 'Website', 'Walk-in', 'CarGurus', 'Facebook']
        self.assertListEqual(normalized_df['LeadSource'].tolist(), expected_lead_sources)
        
        # Check that the original DataFrame was not modified
        self.assertNotEqual(self.test_data['LeadSource'].tolist(), expected_lead_sources)
        
        # Test inplace=True
        df_copy = self.test_data.copy()
        self.normalizer.normalize_column(df_copy, 'LeadSource', inplace=True)
        self.assertListEqual(df_copy['LeadSource'].tolist(), expected_lead_sources)
        
        # Test with a non-existent column
        result = self.normalizer.normalize_column(self.test_data, 'NonExistentColumn')
        self.assertTrue('NonExistentColumn' not in result.columns)
    
    def test_normalize_dataframe(self):
        """Test normalizing multiple columns in a DataFrame."""
        # Normalize LeadSource and VehicleType columns
        normalized_df = self.normalizer.normalize_dataframe(
            self.test_data, columns=['LeadSource', 'VehicleType']
        )
        
        # Check that the values were normalized in both columns
        expected_lead_sources = ['Auto Trader', 'Website', 'Walk-in', 'CarGurus', 'Facebook']
        expected_vehicle_types = ['Used', 'New', 'Certified', 'Used', 'Used']
        
        self.assertListEqual(normalized_df['LeadSource'].tolist(), expected_lead_sources)
        self.assertListEqual(normalized_df['VehicleType'].tolist(), expected_vehicle_types)
        
        # Test with default columns
        default_normalized = normalize_terms(self.test_data)
        self.assertListEqual(default_normalized['LeadSource'].tolist(), expected_lead_sources)
        
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        self.assertTrue(normalize_terms(empty_df).empty)


if __name__ == '__main__':
    unittest.main()