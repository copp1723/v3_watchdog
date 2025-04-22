"""
Unit tests for the Lead Source ROI module.
"""

import unittest
import pandas as pd
import numpy as np
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

from src.validators.lead_source_roi import (
    LeadSourceNormalizer, 
    LeadSourceROI,
    STANDARD_LEAD_SOURCES,
    create_lead_source_roi_schema
)

class TestLeadSourceNormalizer(unittest.TestCase):
    """Test the lead source name normalization functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.normalizer = LeadSourceNormalizer()
        
    def test_normalize_exact_match(self):
        """Test normalizing names with exact matches."""
        test_cases = [
            ("website", "website"),
            ("Web Lead", "website"),
            ("cargurus", "cargurus"),
            ("CarGurus", "cargurus"),
            ("car  gurus", "cargurus"),
            ("auto trader", "autotrader"),
        ]
        
        for input_name, expected in test_cases:
            with self.subTest(input_name=input_name):
                self.assertEqual(self.normalizer.normalize(input_name), expected)
    
    def test_normalize_fuzzy_match(self):
        """Test normalizing names with fuzzy matches."""
        test_cases = [
            ("dealerwebsite", "website"),
            ("site leads", "website"),
            ("CG leads", "cargurus"),
            ("Facebook Lead", "facebook"),
            ("Google Market", "google"),
            ("Walked In", "walk-in"),
        ]
        
        for input_name, expected in test_cases:
            with self.subTest(input_name=input_name):
                self.assertEqual(self.normalizer.normalize(input_name), expected)
    
    def test_normalize_unknown(self):
        """Test normalizing unrecognized names."""
        test_cases = [
            "",
            None,
            "xyz123",
            "unknown source",
            "miscellaneous"
        ]
        
        for input_name in test_cases:
            with self.subTest(input_name=input_name):
                result = self.normalizer.normalize(input_name)
                self.assertTrue(result in ["other", "unknown"])
    
    def test_normalize_dataframe(self):
        """Test normalizing lead sources in a DataFrame."""
        # Create test DataFrame
        df = pd.DataFrame({
            "LeadSource": ["Website", "Car Gurus", "Auto Trader", "Unknown"],
            "Count": [10, 5, 3, 2]
        })
        
        # Normalize
        result = self.normalizer.normalize_df(df, "LeadSource")
        
        # Check results
        expected = ["website", "cargurus", "autotrader", "other"]
        self.assertListEqual(result["LeadSource"].tolist(), expected)
        
        # Test with different target column
        result = self.normalizer.normalize_df(df, "LeadSource", "NormalizedSource")
        self.assertListEqual(result["NormalizedSource"].tolist(), expected)
        self.assertListEqual(result["LeadSource"].tolist(), df["LeadSource"].tolist())
    
    def test_custom_mappings(self):
        """Test normalizer with custom mappings."""
        custom_mappings = {
            "custom_source": ["custom", "special", "cs"],
            "website": ["dealer site", "dealersite.com"]  # Add to existing
        }
        
        normalizer = LeadSourceNormalizer(custom_mappings)
        
        test_cases = [
            ("custom", "custom_source"),
            ("Special", "custom_source"),
            ("CS leads", "custom_source"),
            ("dealer site", "website"),
            ("dealersite.com", "website"),
        ]
        
        for input_name, expected in test_cases:
            with self.subTest(input_name=input_name):
                self.assertEqual(normalizer.normalize(input_name), expected)


class TestLeadSourceROI(unittest.TestCase):
    """Test the lead source ROI calculation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary file for cost data
        self.temp_dir = tempfile.TemporaryDirectory()
        self.cost_file = os.path.join(self.temp_dir.name, "test_costs.json")
        
        # Create sample cost data
        initial_cost_data = {
            "sources": {
                "website": {
                    "monthly_cost": 1000.0,
                    "history": [
                        {
                            "cost": 1000.0,
                            "effective_date": "2023-01-01T00:00:00",
                            "recorded_by": "test"
                        }
                    ]
                },
                "cargurus": {
                    "monthly_cost": 2000.0,
                    "history": [
                        {
                            "cost": 2000.0,
                            "effective_date": "2023-01-01T00:00:00",
                            "recorded_by": "test"
                        }
                    ]
                },
                "autotrader": {
                    "monthly_cost": 1500.0,
                    "history": [
                        {
                            "cost": 1500.0,
                            "effective_date": "2023-01-01T00:00:00",
                            "recorded_by": "test"
                        }
                    ]
                }
            },
            "updated_at": "2023-01-01T00:00:00"
        }
        
        with open(self.cost_file, 'w') as f:
            json.dump(initial_cost_data, f)
        
        # Initialize ROI calculator with test file
        self.roi_calculator = LeadSourceROI(cost_data_path=self.cost_file)
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            "LeadSource": [
                "Website", "Website", "Website", "Website", "Website",
                "CarGurus", "CarGurus", "CarGurus",
                "Auto Trader", "Auto Trader",
                "Unknown", "Unknown"
            ],
            "SaleDate": pd.date_range(start="2023-01-01", periods=12, freq='D'),
            "TotalGross": [
                1000, 1200, 800, 1500, 1300,
                2000, 1800, 2200,
                1700, 1900,
                500, 700
            ],
            "Closed": [
                True, True, True, True, True,
                True, True, True,
                True, True,
                True, True
            ]
        })
    
    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()
    
    def test_calculate_roi(self):
        """Test ROI calculation formula."""
        test_cases = [
            # revenue, cost, expected ROI
            (1000, 500, 1.0),         # 100% ROI
            (1500, 500, 2.0),         # 200% ROI
            (500, 500, 0.0),          # 0% ROI (break-even)
            (0, 500, -1.0),           # -100% ROI (total loss)
            (1000, 0, float('inf')),  # Infinite ROI (zero cost)
            (-100, 500, None),        # Invalid (negative revenue)
            (1000, -500, None),       # Invalid (negative cost)
        ]
        
        for revenue, cost, expected in test_cases:
            with self.subTest(revenue=revenue, cost=cost):
                result = self.roi_calculator.calculate_roi(revenue, cost)
                
                if expected is None or expected == float('inf'):
                    self.assertEqual(result, expected)
                else:
                    self.assertAlmostEqual(result, expected, places=2)
        
        # Test zero cost with include_zero_cost=False
        self.assertIsNone(
            self.roi_calculator.calculate_roi(1000, 0, include_zero_cost=False)
        )
    
    def test_update_source_cost(self):
        """Test updating source costs."""
        # Update existing source
        self.roi_calculator.update_source_cost("website", 1200.0)
        self.assertEqual(self.roi_calculator.get_source_cost("Website"), 1200.0)
        
        # Add new source
        self.roi_calculator.update_source_cost("facebook", 800.0)
        self.assertEqual(self.roi_calculator.get_source_cost("facebook"), 800.0)
        
        # Check persistence
        with open(self.cost_file, 'r') as f:
            saved_data = json.load(f)
        
        self.assertEqual(saved_data["sources"]["website"]["monthly_cost"], 1200.0)
        self.assertEqual(saved_data["sources"]["facebook"]["monthly_cost"], 800.0)
        self.assertEqual(len(saved_data["sources"]["website"]["history"]), 2)
    
    def test_process_dataframe(self):
        """Test processing a DataFrame for ROI metrics."""
        # Process the sample data
        result = self.roi_calculator.process_dataframe(
            self.sample_data,
            source_col="LeadSource",
            revenue_col="TotalGross"
        )
        
        # Verify structure
        expected_columns = [
            'LeadSource', 'TotalRevenue', 'AvgRevenue', 'LeadCount',
            'MonthlyCost', 'CostPerLead', 'ROI', 'ROIPercentage'
        ]
        self.assertListEqual(list(result.columns), expected_columns)
        
        # Verify lead source normalization
        self.assertListEqual(
            sorted(result['LeadSource'].unique().tolist()),
            ['autotrader', 'cargurus', 'other', 'website']
        )
        
        # Check metric calculations
        website_row = result[result['LeadSource'] == 'website'].iloc[0]
        self.assertEqual(website_row['LeadCount'], 5)
        self.assertEqual(website_row['TotalRevenue'], 5800)
        self.assertEqual(website_row['MonthlyCost'], 1200.0)  # Updated value
        self.assertAlmostEqual(website_row['ROI'], (5800 - 1200) / 1200, places=2)
        
        # Check weekly calculation
        weekly_result = self.roi_calculator.process_dataframe(
            self.sample_data,
            source_col="LeadSource",
            revenue_col="TotalGross",
            weekly=True
        )
        
        website_weekly = weekly_result[weekly_result['LeadSource'] == 'website'].iloc[0]
        self.assertLess(website_weekly['WeeklyCost'], 1200.0)  # Should be weekly value
    
    def test_roi_summary(self):
        """Test generating ROI summary metrics."""
        # Process the data first
        processed = self.roi_calculator.process_dataframe(
            self.sample_data,
            source_col="LeadSource",
            revenue_col="TotalGross"
        )
        
        # Get summary
        summary = self.roi_calculator.get_roi_summary(processed)
        
        # Check key metrics
        self.assertIn('total_cost', summary)
        self.assertIn('total_revenue', summary)
        self.assertIn('total_leads', summary)
        self.assertIn('overall_roi', summary)
        self.assertIn('overall_roi_percentage', summary)
        self.assertIn('top_performers', summary)
        self.assertIn('bottom_performers', summary)
        
        # Verify calculations
        self.assertEqual(summary['total_revenue'], self.sample_data['TotalGross'].sum())
        self.assertEqual(summary['total_leads'], len(self.sample_data))
        self.assertEqual(len(summary['top_performers']), min(3, len(processed)))
        self.assertEqual(len(summary['bottom_performers']), min(3, len(processed)))
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Empty DataFrame
        empty_df = pd.DataFrame()
        empty_result = self.roi_calculator.process_dataframe(empty_df)
        self.assertTrue(empty_result.empty)
        
        empty_summary = self.roi_calculator.get_roi_summary(empty_result)
        self.assertIn('error', empty_summary)
        
        # Missing columns
        bad_df = pd.DataFrame({"WrongColumn": [1, 2, 3]})
        bad_result = self.roi_calculator.process_dataframe(bad_df)
        self.assertTrue(bad_result.empty)
        
        # Zero cost source
        zero_cost_df = pd.DataFrame({
            "LeadSource": ["Email", "Email"],
            "TotalGross": [1000, 1500]
        })
        zero_result = self.roi_calculator.process_dataframe(zero_cost_df)
        self.assertEqual(zero_result['MonthlyCost'].iloc[0], 0)
        self.assertEqual(zero_result['ROI'].iloc[0], float('inf'))


class TestROISchema(unittest.TestCase):
    """Test the lead source ROI schema definition."""
    
    def test_schema_structure(self):
        """Test the schema has the correct structure."""
        schema = create_lead_source_roi_schema()
        
        # Check required fields
        required_fields = ['id', 'name', 'description', 'columns']
        for field in required_fields:
            with self.subTest(field=field):
                self.assertIn(field, schema)
        
        # Check column definitions
        expected_columns = [
            'LeadSource', 'LeadDate', 'LeadCount', 
            'LeadCost', 'Revenue', 'Closed'
        ]
        column_names = [col['name'] for col in schema['columns']]
        
        for column in expected_columns:
            with self.subTest(column=column):
                self.assertIn(column, column_names)
        
        # Check business rules
        lead_source_column = next(
            (col for col in schema['columns'] if col['name'] == 'LeadSource'),
            None
        )
        self.assertIsNotNone(lead_source_column)
        self.assertIn('business_rules', lead_source_column)
        self.assertTrue(len(lead_source_column['business_rules']) > 0)


if __name__ == '__main__':
    unittest.main()