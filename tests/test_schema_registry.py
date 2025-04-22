"""
Tests for the multi-tenant schema registry.
"""

import os
import json
import shutil
import unittest
from datetime import datetime
from src.validators.registry import SchemaRegistry

class TestSchemaRegistry(unittest.TestCase):
    """Test the schema registry functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = "test_schemas"
        self.registry = SchemaRegistry(base_dir=self.test_dir)
        
        # Sample schema for testing
        self.sample_schema = {
            "id": "test_schema",
            "name": "Test Schema",
            "columns": [
                {
                    "name": "sale_date",
                    "display_name": "Sale Date",
                    "data_type": "datetime"
                }
            ]
        }
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_create_schema(self):
        """Test creating a new schema."""
        schema_id = self.registry.create_schema("dealer1", self.sample_schema)
        
        self.assertEqual(schema_id, "test_schema")
        
        # Verify schema was saved
        saved_schema = self.registry.get_schema("dealer1", schema_id)
        self.assertEqual(saved_schema["name"], "Test Schema")
        self.assertEqual(saved_schema["version"], "1.0.0")
        self.assertTrue("created_at" in saved_schema)
    
    def test_update_schema(self):
        """Test updating an existing schema."""
        # Create initial schema
        schema_id = self.registry.create_schema("dealer1", self.sample_schema)
        
        # Update schema
        updated_schema = self.sample_schema.copy()
        updated_schema["name"] = "Updated Schema"
        
        self.registry.update_schema("dealer1", schema_id, updated_schema)
        
        # Verify update
        saved_schema = self.registry.get_schema("dealer1", schema_id)
        self.assertEqual(saved_schema["name"], "Updated Schema")
        self.assertEqual(saved_schema["version"], "1.0.1")
    
    def test_delete_schema(self):
        """Test deleting a schema."""
        # Create schema
        schema_id = self.registry.create_schema("dealer1", self.sample_schema)
        
        # Delete schema
        self.registry.delete_schema("dealer1", schema_id)
        
        # Verify deletion
        with self.assertRaises(FileNotFoundError):
            self.registry.get_schema("dealer1", schema_id)
    
    def test_list_schemas(self):
        """Test listing schemas for a dealership."""
        # Create multiple schemas
        self.registry.create_schema("dealer1", self.sample_schema)
        
        schema2 = self.sample_schema.copy()
        schema2["id"] = "test_schema_2"
        self.registry.create_schema("dealer1", schema2)
        
        # List schemas
        schemas = self.registry.list_schemas("dealer1")
        
        self.assertEqual(len(schemas), 2)
        self.assertEqual(schemas[0]["id"], "test_schema_2")
    
    def test_schema_history(self):
        """Test getting schema version history."""
        # Create and update schema
        schema_id = self.registry.create_schema("dealer1", self.sample_schema)
        
        updated_schema = self.sample_schema.copy()
        updated_schema["name"] = "Updated Schema"
        self.registry.update_schema("dealer1", schema_id, updated_schema)
        
        # Get history
        history = self.registry.get_schema_history("dealer1", schema_id)
        
        self.assertEqual(len(history), 1)  # Original version in backup
        self.assertEqual(history[0]["version"], "1.0.0")
    
    def test_schema_isolation(self):
        """Test schema isolation between dealerships."""
        # Create schema for dealer1
        self.registry.create_schema("dealer1", self.sample_schema)
        
        # Verify dealer2 can't access it
        with self.assertRaises(FileNotFoundError):
            self.registry.get_schema("dealer2", "test_schema")
        
        # Create same schema ID for dealer2
        self.registry.create_schema("dealer2", self.sample_schema)
        
        # Verify both exist independently
        schema1 = self.registry.get_schema("dealer1", "test_schema")
        schema2 = self.registry.get_schema("dealer2", "test_schema")
        
        self.assertNotEqual(schema1["created_at"], schema2["created_at"])

if __name__ == '__main__':
    unittest.main()