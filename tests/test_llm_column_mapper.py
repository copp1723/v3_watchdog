"""
Tests for the LLM-powered column mapper.
"""

import unittest
import os
import tempfile
import shutil
import json
from datetime import datetime
from unittest.mock import patch, MagicMock

from src.watchdog_ai.utils.llm_column_mapper import LLMColumnMapper, MappingCache
from src.watchdog_ai.utils.adaptive_schema import SchemaProfile, SchemaColumn

class TestMappingCache(unittest.TestCase):
    """Test cases for MappingCache."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.cache = MappingCache(cache_dir=self.test_dir)
        
        # Sample data
        self.source_cols = ["sale_date", "total_gross"]
        self.schema = SchemaProfile(
            id="test_schema",
            name="Test Schema",
            role="general_manager",
            columns=[
                SchemaColumn(
                    name="SaleDate",
                    display_name="Sale Date",
                    description="Date of sale",
                    data_type="date"
                )
            ]
        )
        self.test_data = {
            "mappings": {
                "sale_date": {
                    "target": "SaleDate",
                    "confidence": 0.95
                }
            }
        }
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)
    
    def test_cache_operations(self):
        """Test basic cache operations."""
        # Set cache entry
        self.cache.set(self.source_cols, self.schema, self.test_data)
        
        # Get cache entry
        cached = self.cache.get(self.source_cols, self.schema)
        self.assertIsNotNone(cached)
        self.assertEqual(cached["mappings"]["sale_date"]["target"], "SaleDate")
        
        # Check cache file exists
        key = self.cache._compute_key(self.source_cols, self.schema)
        cache_file = os.path.join(self.test_dir, f"{key}.json")
        self.assertTrue(os.path.exists(cache_file))
    
    def test_cache_expiration(self):
        """Test cache entry expiration."""
        # Set cache entry
        self.cache.set(self.source_cols, self.schema, self.test_data)
        
        # Modify timestamp to simulate expiration
        key = self.cache._compute_key(self.source_cols, self.schema)
        cache_file = os.path.join(self.test_dir, f"{key}.json")
        
        with open(cache_file, 'r') as f:
            data = json.load(f)
        
        # Set timestamp to old date
        data["timestamp"] = "2020-01-01T00:00:00"
        
        with open(cache_file, 'w') as f:
            json.dump(data, f)
        
        # Try to get expired entry
        cached = self.cache.get(self.source_cols, self.schema)
        self.assertIsNone(cached)
        
        # Check that expired file was removed
        self.assertFalse(os.path.exists(cache_file))


class TestLLMColumnMapper(unittest.TestCase):
    """Test cases for LLMColumnMapper."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.mapper = LLMColumnMapper(cache_dir=self.test_dir)
        
        # Sample schema
        self.schema = SchemaProfile(
            id="test_schema",
            name="Test Schema",
            role="general_manager",
            columns=[
                SchemaColumn(
                    name="SaleDate",
                    display_name="Sale Date",
                    description="Date of sale",
                    data_type="date",
                    aliases=["date", "sale_date"]
                ),
                SchemaColumn(
                    name="TotalGross",
                    display_name="Total Gross",
                    description="Total gross profit",
                    data_type="float",
                    aliases=["gross", "total_gross"]
                )
            ]
        )
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)
    
    def test_mapping_confidence(self):
        """Test confidence calculation."""
        # Exact match
        conf = self.mapper.get_mapping_confidence("SaleDate", self.schema.columns[0])
        self.assertEqual(conf, 1.0)
        
        # Alias match
        conf = self.mapper.get_mapping_confidence("sale_date", self.schema.columns[0])
        self.assertGreaterEqual(conf, 0.9)
        
        # No match
        conf = self.mapper.get_mapping_confidence("unknown", self.schema.columns[0])
        self.assertLess(conf, self.mapper.confidence_threshold)
    
    @patch('src.watchdog_ai.utils.llm_column_mapper.LLMEngine')
    async def test_llm_suggestions(self, mock_llm):
        """Test getting suggestions from LLM."""
        # Mock LLM response
        mock_response = {
            "mappings": {
                "sale_date": {
                    "target": "SaleDate",
                    "confidence": 0.95,
                    "reason": "Exact match with alias"
                },
                "gross": {
                    "target": "TotalGross",
                    "confidence": 0.9,
                    "reason": "Alias match"
                }
            }
        }
        mock_llm.return_value.generate_completion.return_value = json.dumps(mock_response)
        
        # Get suggestions
        suggestions = await self.mapper.get_llm_suggestions(
            ["sale_date", "gross"],
            self.schema
        )
        
        self.assertIn("sale_date", suggestions)
        self.assertEqual(suggestions["sale_date"][0]["target"], "SaleDate")
        self.assertIn("gross", suggestions)
        self.assertEqual(suggestions["gross"][0]["target"], "TotalGross")
    
    def test_learned_mappings(self):
        """Test learned mapping persistence."""
        # Add learned mapping
        self.mapper.persist_learned_mapping(
            "custom_date",
            "SaleDate",
            0.85
        )
        
        # Check confidence using learned mapping
        conf = self.mapper.get_mapping_confidence("custom_date", self.schema.columns[0])
        self.assertEqual(conf, 0.85)

if __name__ == '__main__':
    unittest.main()