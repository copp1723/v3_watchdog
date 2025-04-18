"""
Tests for the trend_analysis module.
"""

import unittest
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
from unittest.mock import patch

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.trend_analysis import (
    analyze_time_series, 
    calculate_change_metrics, 
    analyze_gross_profit,
    analyze_sales_trend
)


class TestTrendAnalysis(unittest.TestCase):
    """Test cases for trend analysis functions."""
    
    def setUp(self):
        """Set up test data."""
        # Sample data with dates and numeric values
        self.sample_data = pd.DataFrame({
            'date_col': ['2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01'],
            'value_col': [100, 110, 105, 115],
            'gross_col': [1500, 1600, 1550, 1650]
        })
        
        # Sample data with invalid date formats
        self.invalid_date_data = pd.DataFrame({
            'date_col': ['2023-01-01', '01/01/2023', 'invalid-date', '2023-04-01'],
            'value_col': [100, 110, 105, 115]
        })
        
        # Sample data with invalid numeric values
        self.invalid_numeric_data = pd.DataFrame({
            'date_col': ['2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01'],
            'value_col': [100, 'invalid', 105, 115],
            'gross_col': [1500, 'N/A', 1550, 1650]
        })
    
    def test_analyze_time_series_valid_data(self):
        """Test analyze_time_series with valid data."""
        # Run the analysis
        result = analyze_time_series(
            self.sample_data, 'date_col', 'value_col', aggregation='mean', freq='M'
        )
        
        # Check that the result is a DataFrame with the expected shape
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 4)  # Should have 4 monthly data points
        self.assertIn('date_col', result.columns)
        self.assertIn('value_col', result.columns)
        
        # Check that the dates were correctly parsed
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(result['date_col']))
    
    def test_analyze_time_series_invalid_date(self):
        """Test analyze_time_series with invalid date formats."""
        # Should raise ValueError due to invalid dates
        with self.assertRaises(ValueError) as context:
            analyze_time_series(
                self.invalid_date_data, 'date_col', 'value_col', aggregation='mean', freq='M'
            )
        
        # Check error message
        self.assertIn("Invalid date format", str(context.exception))
        
        # The error should mention the problematic data
        self.assertIn("date_col", str(context.exception))
    
    def test_calculate_change_metrics(self):
        """Test calculate_change_metrics function."""
        # Use a simple Series
        series = pd.Series([100, 110, 120, 130])
        
        # Calculate metrics
        metrics = calculate_change_metrics(series)
        
        # Check the results
        self.assertEqual(metrics['absolute_change'], 30)  # 130 - 100
        self.assertEqual(metrics['percentage_change'], 30.0)  # 30%
        self.assertEqual(metrics['average'], 115.0)  # mean
        self.assertEqual(metrics['trend_direction'], 'increasing')  # > 5%
    
    @patch('logging.Logger.error')
    def test_analyze_gross_profit_invalid_numeric(self, mock_logger_error):
        """Test analyze_gross_profit with invalid numeric values."""
        # Run the analysis which should handle the invalid data gracefully
        result = analyze_gross_profit(self.invalid_numeric_data, gross_col='gross_col')
        
        # Check that the function continued despite errors
        self.assertIsInstance(result, dict)
        self.assertTrue('total_gross' in result)
        
        # Check that error was logged
        mock_logger_error.assert_called()
        call_args = mock_logger_error.call_args[0][0]
        self.assertIn("Failed to convert", call_args)
    
    @patch('logging.Logger.error')
    def test_analyze_sales_trend_with_invalid_dates(self, mock_logger_error):
        """Test analyze_sales_trend with invalid date values."""
        with self.assertRaises(ValueError) as context:
            analyze_sales_trend(self.invalid_date_data, date_col='date_col')
        
        # Check that error was logged
        mock_logger_error.assert_called()
        
        # Check error message
        self.assertIn("Unable to process dates", str(context.exception))


if __name__ == '__main__':
    unittest.main()