"""
Unit tests for Analytics Engine.

Tests core functionality including data loading, standardization,
and various analysis functions.
"""

import os
import unittest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from watchdog_ai.core.analytics_engine import AnalyticsEngine
from watchdog_ai.core.constants import DEFAULT_REQUIRED_COLUMNS

class TestAnalyticsEngine(unittest.TestCase):
    """Test suite for AnalyticsEngine functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.engine = AnalyticsEngine()
        
        # Create synthetic test data
        dates = pd.date_range(
            start='2023-01-01', 
            end='2023-12-31', 
            freq='D'
        )
        
        np.random.seed(42)  # For reproducible tests
        
        # Generate synthetic sales data
        self.test_data = pd.DataFrame({
            'SaleDate': dates,
            'LeadSource': np.random.choice(
                ['Web', 'Phone', 'Referral', 'Walk-in'],
                size=len(dates)
            ),
            'TotalGross': np.random.normal(10000, 2000, size=len(dates)),
            'SalesPerson': np.random.choice(
                ['John', 'Alice', 'Bob', 'Carol'],
                size=len(dates)
            )
        })
        
        # Ensure positive gross values
        self.test_data['TotalGross'] = self.test_data['TotalGross'].abs()
        
        # Create a test CSV
        self.test_csv_path = 'test_sales_data.csv'
        self.test_data.to_csv(self.test_csv_path, index=False)

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.test_csv_path):
            os.remove(self.test_csv_path)

    def test_load_csv(self):
        """Test CSV loading functionality."""
        # Test valid CSV loading
        df = self.engine.load_csv(self.test_csv_path)
        self.assertFalse(df.empty)
        self.assertEqual(len(df), len(self.test_data))
        
        # Test invalid file path
        with self.assertRaises(FileNotFoundError):
            self.engine.load_csv('nonexistent.csv')
        
        # Test empty CSV
        pd.DataFrame().to_csv('empty.csv')
        with self.assertRaises(ValueError):
            self.engine.load_csv('empty.csv')
        os.remove('empty.csv')

    def test_standardize_dataframe(self):
        """Test DataFrame standardization."""
        # Test with valid data
        std_df, warnings = self.engine._standardize_dataframe(self.test_data)
        self.assertFalse(std_df.empty)
        self.assertEqual(len(warnings), 0)
        
        # Verify required columns
        for col in DEFAULT_REQUIRED_COLUMNS:
            self.assertIn(col, std_df.columns)
        
        # Test with missing required column
        invalid_df = self.test_data.drop('LeadSource', axis=1)
        with self.assertRaises(ValueError):
            self.engine._standardize_dataframe(invalid_df)
        
        # Test with empty DataFrame
        with self.assertRaises(ValueError):
            self.engine._standardize_dataframe(pd.DataFrame())

    def test_calculate_sales_trends(self):
        """Test sales trends calculation."""
        result = self.engine.calculate_sales_trends(self.test_data)
        
        # Verify result structure
        self.assertIn('total_sales', result)
        self.assertIn('average_daily_sales', result)
        self.assertIn('trends', result)
        self.assertIn('mtd_sales', result)
        self.assertIn('ytd_sales', result)
        
        # Verify calculations
        self.assertEqual(result['total_sales'], len(self.test_data))
        self.assertTrue(isinstance(result['trends'], list))
        self.assertTrue(len(result['trends']) > 0)
        
        # Test with invalid date column
        invalid_df = self.test_data.rename(columns={'SaleDate': 'Date'})
        result = self.engine.calculate_sales_trends(invalid_df)
        self.assertIn('error', result)

    def test_calculate_gross_profit_by_source(self):
        """Test gross profit analysis by source."""
        result = self.engine.calculate_gross_profit_by_source(self.test_data)
        
        # Verify result structure
        self.assertIn('total_profit', result)
        self.assertIn('breakdown', result)
        
        # Verify breakdown content
        breakdown = result['breakdown']
        self.assertTrue(isinstance(breakdown, list))
        self.assertTrue(len(breakdown) > 0)
        
        # Verify each source's data
        for source_data in breakdown:
            self.assertIn('source', source_data)
            self.assertIn('total_profit', source_data)
            self.assertIn('avg_profit', source_data)
            self.assertIn('sale_count', source_data)
            self.assertIn('percentage', source_data)

    def test_period_comparisons(self):
        """Test YoY and MoM comparisons."""
        # Add previous year data
        prev_year = self.test_data.copy()
        prev_year['SaleDate'] = prev_year['SaleDate'] - pd.DateOffset(years=1)
        combined_data = pd.concat([self.test_data, prev_year])
        
        # Test YoY comparison
        yoy = self.engine.calculate_yoy_comparison(combined_data)
        self.assertIn('comparisons', yoy)
        self.assertTrue(len(yoy['comparisons']) > 0)
        
        # Test MoM comparison
        mom = self.engine.calculate_mom_comparison(combined_data)
        self.assertIn('comparisons', mom)
        self.assertTrue(len(mom['comparisons']) > 0)
        
        # Verify comparison structure
        for comp in yoy['comparisons']:
            self.assertIn('period', comp)
            self.assertIn('current_value', comp)
            self.assertIn('previous_value', comp)
            self.assertIn('change', comp)
            self.assertIn('change_percentage', comp)

    def test_run_analysis(self):
        """Test complete analysis workflow."""
        result = self.engine.run_analysis(self.test_data)
        
        # Verify success
        self.assertTrue(result['success'])
        
        # Verify result structure
        self.assertIn('timestamp', result)
        self.assertIn('warnings', result)
        self.assertIn('metadata', result)
        self.assertIn('sales_trends', result)
        self.assertIn('profit_by_source', result)
        self.assertIn('year_over_year', result)
        self.assertIn('month_over_month', result)
        self.assertIn('data_quality', result)
        
        # Test with invalid data
        invalid_df = pd.DataFrame({'A': [1, 2, 3]})
        result = self.engine.run_analysis(invalid_df)
        self.assertFalse(result['success'])
        self.assertIn('error', result)

if __name__ == '__main__':
    unittest.main()

