"""Unit tests for insight feed functionality."""

import unittest
from datetime import datetime
from src.watchdog_ai.ui.pages.insight_feed_page import (
    find_similar_insights,
    export_insight_as_json,
    export_insights_as_csv
)

class TestInsightFeed(unittest.TestCase):
    """Test cases for insight feed functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_insights = [
            {
                "id": "INS-1",
                "timestamp": datetime.now(),
                "priority": "Critical",
                "dealership_name": "Test Motors",
                "summary": "Sales performance dropped significantly",
                "tags": ["sales", "performance", "critical"],
                "kpi_metrics": {"Sales Volume": {"value": 100, "delta": -20}},
            },
            {
                "id": "INS-2",
                "timestamp": datetime.now(),
                "priority": "Recommended",
                "dealership_name": "Test Motors",
                "summary": "Inventory levels below target",
                "tags": ["inventory", "stock"],
                "kpi_metrics": {"Inventory Turn": {"value": 45, "delta": -5}},
            },
            {
                "id": "INS-3",
                "timestamp": datetime.now(),
                "priority": "Critical",
                "dealership_name": "Test Motors",
                "summary": "Monthly sales target missed",
                "tags": ["sales", "targets", "monthly"],
                "kpi_metrics": {"Sales Volume": {"value": 80, "delta": -30}},
            }
        ]

    def test_find_similar_insights_by_tags(self):
        """Test finding similar insights based on tags."""
        current_insight = self.sample_insights[0]  # Sales performance insight
        similar = find_similar_insights(current_insight, self.sample_insights)
        
        # Should find INS-3 (sales-related) but not INS-2 (inventory-related)
        self.assertEqual(len(similar), 1)
        self.assertEqual(similar[0]["id"], "INS-3")

    def test_find_similar_insights_by_content(self):
        """Test finding similar insights based on content/summary."""
        current_insight = {
            "id": "INS-4",
            "summary": "Sales team performance review needed",
            "tags": ["review"],
            "kpi_metrics": {"Performance": {"value": 75, "delta": -5}}
        }
        
        similar = find_similar_insights(current_insight, self.sample_insights)
        # Should find sales-related insights
        self.assertTrue(any(i["id"] in ["INS-1", "INS-3"] for i in similar))

    def test_find_similar_insights_by_kpi(self):
        """Test finding similar insights based on KPI metrics."""
        current_insight = {
            "id": "INS-4",
            "summary": "Review needed",
            "kpi_metrics": {"Sales Volume": {"value": 90, "delta": -10}}
        }
        
        similar = find_similar_insights(current_insight, self.sample_insights)
        # Should find insights with Sales Volume KPI
        self.assertTrue(len(similar) >= 2)
        self.assertTrue(all("Sales Volume" in i["kpi_metrics"] for i in similar))

    def test_export_insight_as_json(self):
        """Test JSON export of a single insight."""
        insight = self.sample_insights[0]
        json_str = export_insight_as_json(insight)
        
        # Verify JSON contains key fields
        self.assertIn('"id": "INS-1"', json_str)
        self.assertIn('"priority": "Critical"', json_str)
        self.assertIn('"dealership_name": "Test Motors"', json_str)

    def test_export_insights_as_csv(self):
        """Test CSV export of multiple insights."""
        csv_str = export_insights_as_csv(self.sample_insights)
        
        # Verify CSV contains headers and data
        self.assertIn("id,timestamp,priority,dealership_name", csv_str)
        self.assertIn("INS-1", csv_str)
        self.assertIn("INS-2", csv_str)
        self.assertIn("INS-3", csv_str)

    def test_find_similar_insights_empty_input(self):
        """Test similarity function with empty inputs."""
        # Test with empty current insight
        self.assertEqual(find_similar_insights({}, self.sample_insights), [])
        
        # Test with empty insight list
        self.assertEqual(find_similar_insights(self.sample_insights[0], []), [])

    def test_find_similar_insights_max_results(self):
        """Test max_similar parameter in similarity function."""
        current_insight = self.sample_insights[0]
        max_similar = 1
        similar = find_similar_insights(
            current_insight, 
            self.sample_insights, 
            max_similar=max_similar
        )
        self.assertEqual(len(similar), max_similar)

if __name__ == '__main__':
    unittest.main() 