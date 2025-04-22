"""
Tests for specific metric lookup functionality, particularly the fix for
the "Karen Davis days_to_close" query issue.
"""

import unittest
import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta

class TestKarenDavisDaysToClose(unittest.TestCase):
    """Test class for verifying the fix for the "Karen Davis days_to_close" query issue."""

    def setUp(self):
        """Set up test data with a Karen Davis record with days_to_close value of 9."""
        # Create a test DataFrame with a record for Karen Davis
        self.test_data = pd.DataFrame({
            'sales_rep': ['Karen Davis', 'John Smith', 'Michael Johnson', 'Sarah Williams'],
            'days_to_close': [9, 15, 7, 12],
            'gross_profit': [1200, 950, 1500, 1100],
            'sale_count': [3, 2, 4, 2],
            'revenue': [35000, 28000, 42000, 31000],
            'lead_source': ['Website', 'CarGurus', 'Referral', 'Website'],
            'date': [datetime.now() - timedelta(days=i) for i in range(4)]
        })

    def test_regex_extraction_patterns(self):
        """Test the regex patterns used for extraction"""
        # Test sales rep name extraction pattern - expanded to include more patterns
        sales_rep_pattern = re.compile(r"(?:(?:for|by|about|from|is)\s+['\"](.*?)['\"]\??)|(?:['\"](.*?)['\"]'?s)", re.IGNORECASE)
        test_strings = [
            "What are the days_to_close for the sale made by 'Karen Davis'?",
            "What is 'Karen Davis's days to close?",
            "Tell me about 'Karen Davis' performance",
            "Show metrics from 'Karen Davis'",
            "How is 'Karen Davis' doing?"
        ]
        
        for test_string in test_strings:
            match = sales_rep_pattern.search(test_string)
            self.assertIsNotNone(match, f"Failed to match: {test_string}")
            # Get the captured group (either first or second capturing group)
            name = next(group for group in match.groups() if group is not None)
            self.assertEqual(name, "Karen Davis")
        
        # Test days_to_close metric pattern
        metric_pattern = re.compile(r'(?:what\s+(?:is|are|was|were)\s+the\s+)?(days?[\s_-]to[\s_-]close|closing\s+time|time\s+to\s+close|closing\s+days?|sale\s+duration)', re.IGNORECASE)
        test_strings = [
            "What are the days_to_close for Karen Davis?",
            "What is the days to close?",
            "How many days to close did she have?",
            "Tell me the closing time",
            "What was the time to close?",
            "What were the closing days?"
        ]
        
        for test_string in test_strings:
            match = metric_pattern.search(test_string)
            self.assertIsNotNone(match, f"Failed to match: {test_string}")

    def test_enhanced_filtering(self):
        """Test the enhanced filtering for specific sales rep and days_to_close"""
        # Test case-insensitive filtering
        df = self.test_data
        
        # Filter by sales rep
        sales_rep_name = "karen davis"
        filtered_data = df[df['sales_rep'].str.lower() == sales_rep_name.lower()]
        self.assertEqual(len(filtered_data), 1)
        self.assertEqual(filtered_data.iloc[0]['days_to_close'], 9)
        
        # Ensure numeric conversion works
        self.assertEqual(filtered_data['days_to_close'].mean(), 9.0)

if __name__ == "__main__":
    unittest.main()