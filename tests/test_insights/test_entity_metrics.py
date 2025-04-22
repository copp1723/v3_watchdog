"""
Tests for entity-specific metric lookups.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.watchdog_ai.insights.insight_functions import InsightFunctions, DataValidationError

class TestEntityMetrics(unittest.TestCase):
    """Test entity-specific metric lookups."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample data
        dates = pd.date_range(start='2024-01-01', periods=30)
        sales_reps = ['Karen Davis', 'John Smith', 'Alice Brown']
        
        data = []
        for date in dates:
            for rep in sales_reps:
                # Generate some realistic-looking metrics
                gross_profit = np.random.normal(2000, 500)
                days_to_close = np.random.randint(5, 15)
                
                data.append({
                    'sales_rep': rep,
                    'date': date,
                    'gross_profit': gross_profit,
                    'days_to_close': days_to_close,
                    'lead_source': np.random.choice(['Web', 'Phone', 'Walk-in'])
                })
        
        self.test_data = pd.DataFrame(data)
    
    def test_get_entity_metrics_exact_match(self):
        """Test getting metrics for an exact entity match."""
        result = InsightFunctions.get_entity_metrics(
            df=self.test_data,
            entity_name='Karen Davis',
            entity_col='sales_rep',
            metric_col='gross_profit'
        )
        
        self.assertEqual(result['entity_name'], 'Karen Davis')
        self.assertIn('metrics', result)
        self.assertIn('total', result['metrics'])
        self.assertIn('average', result['metrics'])
        self.assertIn('count', result['metrics'])
        self.assertIn('rank', result['metrics'])
        self.assertIn('percentile', result['metrics'])
    
    def test_get_entity_metrics_case_insensitive(self):
        """Test case-insensitive entity matching."""
        result = InsightFunctions.get_entity_metrics(
            df=self.test_data,
            entity_name='karen davis',  # Lowercase
            entity_col='sales_rep',
            metric_col='gross_profit'
        )
        
        self.assertEqual(result['entity_name'], 'karen davis')
        self.assertGreater(result['metrics']['count'], 0)
    
    def test_get_entity_metrics_missing_entity(self):
        """Test handling of missing entity."""
        with self.assertRaises(DataValidationError) as context:
            InsightFunctions.get_entity_metrics(
                df=self.test_data,
                entity_name='Unknown Rep',
                entity_col='sales_rep',
                metric_col='gross_profit'
            )
        
        self.assertIn('No data found', str(context.exception))
        self.assertIn('available_entities', context.exception.details)
    
    def test_get_entity_metrics_with_dates(self):
        """Test metrics calculation with date-based trends."""
        result = InsightFunctions.get_entity_metrics(
            df=self.test_data,
            entity_name='Karen Davis',
            entity_col='sales_rep',
            metric_col='gross_profit'
        )
        
        self.assertIn('trend', result['metrics'])
        self.assertIn('recent_average', result['metrics'])
        self.assertIn('recent_total', result['metrics'])
        self.assertIn('recent_count', result['metrics'])
    
    def test_get_entity_metrics_different_columns(self):
        """Test metrics for different metric columns."""
        result = InsightFunctions.get_entity_metrics(
            df=self.test_data,
            entity_name='Karen Davis',
            entity_col='sales_rep',
            metric_col='days_to_close'
        )
        
        self.assertIn('total', result['metrics'])
        self.assertIn('average', result['metrics'])
        self.assertTrue(isinstance(result['metrics']['average'], float))
    
    def test_get_entity_metrics_invalid_column(self):
        """Test handling of invalid column names."""
        with self.assertRaises(DataValidationError) as context:
            InsightFunctions.get_entity_metrics(
                df=self.test_data,
                entity_name='Karen Davis',
                entity_col='nonexistent_column',
                metric_col='gross_profit'
            )
        
        self.assertIn('Could not find entity column', str(context.exception))
    
    def test_get_entity_metrics_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        with self.assertRaises(DataValidationError) as context:
            InsightFunctions.get_entity_metrics(
                df=pd.DataFrame(),
                entity_name='Karen Davis',
                entity_col='sales_rep',
                metric_col='gross_profit'
            )
        
        self.assertIn('DataFrame is empty', str(context.exception))

if __name__ == '__main__':
    unittest.main()