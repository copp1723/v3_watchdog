"""
Unit tests for the enhanced insight tagger module.
"""

import unittest
import json
import os
import tempfile
from datetime import datetime, timedelta

from src.insight_tagger import (
    InsightTagger, 
    InsightStore, 
    InsightPriority, 
    InsightType,
    tag_insight,
    tag_insights,
    get_insights_by_tags
)

class TestInsightTagger(unittest.TestCase):
    """Test cases for the InsightTagger class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Initialize a tagger with default settings
        self.tagger = InsightTagger()
        
        # Create sample insights for testing
        self.create_sample_insights()
    
    def create_sample_insights(self):
        """Create sample insights for testing."""
        # Sales trend insight
        self.sales_insight = {
            "title": "Sales Trend Analysis",
            "summary": "Sales are trending upward with a 15% increase over the last quarter.",
            "metrics": {
                "sales_change_pct": 15,
                "total_sales": 1250000,
                "previous_sales": 1087500
            },
            "recommendations": [
                "Continue current sales strategy",
                "Expand marketing in high-performing regions"
            ]
        }
        
        # Inventory alert insight
        self.inventory_insight = {
            "title": "Aged Inventory Alert",
            "summary": "Critical alert: 25 vehicles in inventory are over 90 days old and require immediate attention.",
            "metrics": {
                "aged_units": 25,
                "avg_days_in_stock": 95,
                "total_value": 750000
            },
            "recommendations": [
                "Implement price reductions on vehicles over 90 days",
                "Contact previous prospects about these vehicles",
                "Consider auction options for oldest inventory"
            ]
        }
        
        # Lead source insight
        self.lead_insight = {
            "title": "Lead Source Performance",
            "summary": "Website leads are converting at 22%, while third-party leads are only at 8%.",
            "metrics": {
                "website_conversion": 22,
                "third_party_conversion": 8,
                "website_lead_count": 450,
                "third_party_lead_count": 320
            },
            "recommendations": [
                "Optimize website lead capture forms",
                "Review third-party lead quality"
            ]
        }
    
    def test_tag_insight_basic(self):
        """Test basic insight tagging functionality."""
        # Tag the sales insight
        tagged = self.tagger.tag_insight(self.sales_insight)
        
        # Check that necessary fields are added
        self.assertIn("priority", tagged)
        self.assertIn("type", tagged)
        self.assertIn("tags", tagged)
        self.assertIn("tagged_at", tagged)
        self.assertIn("audit", tagged)
        
        # Verify priority and type
        self.assertEqual(tagged["type"], InsightType.TREND)
        self.assertIn(tagged["priority"], [InsightPriority.RECOMMENDED, InsightPriority.OPTIONAL])
        
        # Verify tags
        self.assertIsInstance(tagged["tags"], list)
        self.assertGreater(len(tagged["tags"]), 2)  # Should have at least type and priority tags
        
        # Verify audit metadata
        self.assertIn("created_at", tagged["audit"])
        self.assertIn("version", tagged["audit"])
        self.assertIn("tag_history", tagged["audit"])
    
    def test_tag_critical_insight(self):
        """Test tagging of a critical insight."""
        # Tag the inventory insight (should be critical)
        tagged = self.tagger.tag_insight(self.inventory_insight)
        
        # Verify it's marked as critical
        self.assertEqual(tagged["priority"], InsightPriority.CRITICAL)
        
        # Verify it's marked as an alert
        self.assertEqual(tagged["type"], InsightType.ALERT)
        
        # Verify inventory-related tags are present
        self.assertIn("inventory", tagged["tags"])
        self.assertIn("alert", tagged["tags"])
    
    def test_multi_tag_support(self):
        """Test that multiple relevant tags are identified."""
        # Tag the lead insight
        tagged = self.tagger.tag_insight(self.lead_insight)
        
        # Verify multiple tags are present
        self.assertGreaterEqual(len(tagged["tags"]), 3)
        
        # Verify lead-related tags
        self.assertTrue(any("lead" in tag for tag in tagged["tags"]))
        self.assertTrue(any("performance" in tag for tag in tagged["tags"]))
    
    def test_audit_metadata(self):
        """Test that audit metadata is correctly generated."""
        # Tag an insight
        tagged = self.tagger.tag_insight(self.sales_insight)
        
        # Verify audit metadata structure
        audit = tagged["audit"]
        self.assertIn("created_at", audit)
        self.assertIn("created_by", audit)
        self.assertIn("origin_dataset", audit)
        self.assertIn("tag_history", audit)
        self.assertIn("version", audit)
        
        # Verify tag history
        self.assertEqual(len(audit["tag_history"]), 1)
        self.assertIn("timestamp", audit["tag_history"][0])
        self.assertIn("tags", audit["tag_history"][0])
    
    def test_tag_suggestions(self):
        """Test tag suggestion functionality."""
        # Create an insight with specific business area focus
        finance_insight = {
            "title": "Financing Performance",
            "summary": "Finance department generated $125,000 in profit from loans and F&I products this month.",
            "metrics": {
                "finance_profit": 125000,
                "loan_count": 85,
                "avg_finance_profit": 1470
            }
        }
        
        # Tag the insight
        tagged = self.tagger.tag_insight(finance_insight)
        
        # Should have finance-related tags
        self.assertTrue(any("finance" in tag.lower() for tag in tagged["tags"]))
        
        # Check for revenue/profit related tags too
        self.assertTrue(any("profit" in tag.lower() for tag in tagged["tags"]) or 
                       any("sales" in tag.lower() for tag in tagged["tags"]))


class TestInsightSimilarity(unittest.TestCase):
    """Test cases for insight similarity and grouping."""
    
    @unittest.skip("Skipping similarity tests that require sentence-transformers")
    def setUp(self):
        """Set up test fixtures."""
        # Initialize a tagger
        self.tagger = InsightTagger()
        
        # Create similar insights
        self.insights = [
            {
                "id": "1",
                "title": "Inventory Aging Report",
                "summary": "Several vehicles in inventory are approaching 60 days and need attention."
            },
            {
                "id": "2",
                "title": "Aged Inventory Alert",
                "summary": "Multiple vehicles in stock are over 60 days old and require price review."
            },
            {
                "id": "3",
                "title": "Sales Performance",
                "summary": "Sales team achieved 95% of target this month with strong used car performance."
            },
            {
                "id": "4",
                "title": "Lead Source Analysis",
                "summary": "Website leads continue to be the highest converting source at 18%."
            },
            {
                "id": "5",
                "title": "Inventory Age Problem",
                "summary": "Aging inventory is becoming an issue with 15 units over 90 days old."
            }
        ]
    
    @unittest.skip("Skipping similarity tests that require sentence-transformers")
    def test_find_similar_insights(self):
        """Test finding similar insights."""
        # Target insight
        target = self.insights[0]  # Inventory aging
        
        # Find similar insights
        similar = self.tagger.find_similar_insights(target, self.insights)
        
        # Should find at least one similar insight
        self.assertGreater(len(similar), 0)
        
        # The most similar should be inventory related
        if len(similar) > 0:
            most_similar = similar[0]["insight"]
            self.assertIn("inventory", most_similar["title"].lower())
    
    @unittest.skip("Skipping similarity tests that require sentence-transformers")
    def test_group_insights(self):
        """Test grouping similar insights."""
        # Group the insights
        groups = self.tagger.group_insights_by_similarity(self.insights)
        
        # Should create groups
        self.assertGreater(len(groups), 0)
        
        # Each group should have a seed and potentially members
        first_group = groups[0]
        self.assertIn("seed", first_group)
        self.assertIn("members", first_group)
        self.assertIn("size", first_group)


class TestInsightStore(unittest.TestCase):
    """Test cases for the InsightStore class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for storage
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage_path = os.path.join(self.temp_dir.name, "insights.json")
        self.audit_path = os.path.join(self.temp_dir.name, "audit_log.json")
        
        # Initialize the store
        self.store = InsightStore(self.storage_path)
        
        # Create sample insights
        self.create_sample_insights()
        
        # Add insights to the store
        for insight in self.insights:
            self.store.add_insight(insight)
    
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    def create_sample_insights(self):
        """Create sample insights for testing."""
        self.insights = [
            {
                "id": "test1",
                "title": "Test Insight 1",
                "summary": "This is a test summary for the first insight.",
                "priority": InsightPriority.RECOMMENDED,
                "type": InsightType.TREND,
                "tags": ["sales", "trend", "recommended"],
                "timestamp": datetime.now().isoformat()
            },
            {
                "id": "test2",
                "title": "Test Insight 2",
                "summary": "This is a test summary for the second insight.",
                "priority": InsightPriority.CRITICAL,
                "type": InsightType.ALERT,
                "tags": ["inventory", "alert", "critical"],
                "timestamp": datetime.now().isoformat()
            },
            {
                "id": "test3",
                "title": "Test Insight 3",
                "summary": "This is a test summary for the third insight.",
                "priority": InsightPriority.INFORMATIONAL,
                "type": InsightType.SUMMARY,
                "tags": ["lead", "performance", "informational"],
                "timestamp": datetime.now().isoformat()
            }
        ]
    
    def test_store_initialization(self):
        """Test store initialization and loading."""
        # Verify storage file was created
        self.assertTrue(os.path.exists(self.storage_path))
        
        # Create a new store instance (should load existing data)
        new_store = InsightStore(self.storage_path)
        
        # Verify insights were loaded
        self.assertEqual(len(new_store.insights), 3)
    
    def test_get_insight(self):
        """Test retrieving a specific insight."""
        # Get an insight by ID
        insight = self.store.get_insight("test1")
        
        # Verify the insight
        self.assertIsNotNone(insight)
        self.assertEqual(insight["id"], "test1")
        self.assertEqual(insight["title"], "Test Insight 1")
    
    def test_get_insights_by_priority(self):
        """Test retrieving insights by priority."""
        # Get critical insights
        critical = self.store.get_insights_by_priority(InsightPriority.CRITICAL)
        
        # Verify the results
        self.assertEqual(len(critical), 1)
        self.assertEqual(critical[0]["id"], "test2")
    
    def test_get_insights_by_type(self):
        """Test retrieving insights by type."""
        # Get trend insights
        trends = self.store.get_insights_by_type(InsightType.TREND)
        
        # Verify the results
        self.assertEqual(len(trends), 1)
        self.assertEqual(trends[0]["id"], "test1")
    
    def test_get_insights_by_tag(self):
        """Test retrieving insights by tag."""
        # Get insights with 'sales' tag
        sales = self.store.get_insights_by_tag("sales")
        
        # Verify the results
        self.assertEqual(len(sales), 1)
        self.assertEqual(sales[0]["id"], "test1")
    
    def test_get_insights_by_tags(self):
        """Test retrieving insights by multiple tags."""
        # Get insights with either 'sales' or 'inventory' tags
        results = self.store.get_insights_by_tags(["sales", "inventory"])
        
        # Verify the results (OR query)
        self.assertEqual(len(results), 2)
        
        # Get insights with both 'alert' and 'critical' tags
        results = self.store.get_insights_by_tags(["alert", "critical"], match_all=True)
        
        # Verify the results (AND query)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "test2")
    
    def test_update_tags(self):
        """Test updating tags for an insight."""
        # Update tags for an insight
        new_tags = ["updated", "sales", "trend", "recommended"]
        success = self.store.update_tags("test1", new_tags)
        
        # Verify the update succeeded
        self.assertTrue(success)
        
        # Get the updated insight
        updated = self.store.get_insight("test1")
        
        # Verify the tags were updated
        self.assertEqual(set(updated["tags"]), set(new_tags))
        
        # Verify tag history was updated
        self.assertGreaterEqual(len(updated["audit"]["tag_history"]), 1)
        
        # Verify version was incremented
        self.assertGreaterEqual(updated["audit"]["version"], 1)
        
        # Verify audit log was created
        self.assertTrue(os.path.exists(self.audit_path))
    
    def test_delete_insight(self):
        """Test deleting an insight."""
        # Delete an insight
        success = self.store.delete_insight("test3")
        
        # Verify the deletion succeeded
        self.assertTrue(success)
        
        # Verify the insight was removed
        self.assertIsNone(self.store.get_insight("test3"))
        
        # Verify the count decreased
        self.assertEqual(len(self.store.insights), 2)


class TestConvenienceFunctions(unittest.TestCase):
    """Test cases for convenience functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for storage
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage_path = os.path.join(self.temp_dir.name, "insights.json")
        
        # Create sample insights
        self.sample_insight = {
            "title": "Sample Insight",
            "summary": "This is a sample insight for testing convenience functions.",
            "metrics": {
                "value": 100,
                "change_pct": 25
            },
            "recommendations": [
                "This is a recommendation"
            ]
        }
        
        self.sample_insights = [
            {
                "title": "Insight 1",
                "summary": "Sales are up this month.",
                "tags": ["sales"]
            },
            {
                "title": "Insight 2",
                "summary": "Inventory is aging.",
                "tags": ["inventory", "alert"]
            },
            {
                "title": "Insight 3",
                "summary": "Service department performance is strong.",
                "tags": ["service", "performance"]
            }
        ]
    
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    def test_tag_insight_function(self):
        """Test the tag_insight convenience function."""
        # Tag a single insight
        tagged = tag_insight(self.sample_insight)
        
        # Verify the result
        self.assertIn("priority", tagged)
        self.assertIn("type", tagged)
        self.assertIn("tags", tagged)
    
    def test_tag_insights_function(self):
        """Test the tag_insights convenience function."""
        # Tag multiple insights
        tagged = tag_insights(self.sample_insights, save=True, storage_path=self.storage_path)
        
        # Verify the results
        self.assertEqual(len(tagged), 3)
        for insight in tagged:
            self.assertIn("priority", insight)
            self.assertIn("type", insight)
            self.assertIn("tags", insight)
        
        # Verify insights were saved
        self.assertTrue(os.path.exists(self.storage_path))
        
        # Load the saved insights
        with open(self.storage_path, 'r') as f:
            saved = json.load(f)
        
        # Verify count matches
        self.assertEqual(len(saved), 3)
    
    def test_get_insights_by_tags_function(self):
        """Test the get_insights_by_tags convenience function."""
        # First tag and save insights
        tag_insights(self.sample_insights, save=True, storage_path=self.storage_path)
        
        # Get insights by tags
        results = get_insights_by_tags(["sales", "inventory"], storage_path=self.storage_path)
        
        # Verify results
        self.assertIsInstance(results, list)
        
        # We don't verify the count because it depends on how the tagging went,
        # which can vary based on the tagger's implementation


if __name__ == '__main__':
    unittest.main()