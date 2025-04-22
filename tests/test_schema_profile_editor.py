"""
Tests for the Schema Profile Editor.
"""

import unittest
import os
import tempfile
import shutil
import pandas as pd
from datetime import datetime

from src.watchdog_ai.utils.schema_profile_editor import SchemaProfileEditor
from src.watchdog_ai.utils.adaptive_schema import SchemaProfile, SchemaColumn, ExecRole

class TestSchemaProfileEditor(unittest.TestCase):
    """Test cases for SchemaProfileEditor."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directories
        self.test_dir = tempfile.mkdtemp()
        self.profiles_dir = os.path.join(self.test_dir, "profiles")
        self.cache_dir = os.path.join(self.test_dir, "cache")
        
        # Create editor instance
        self.editor = SchemaProfileEditor(
            profiles_dir=self.profiles_dir,
            cache_dir=self.cache_dir
        )
        
        # Create sample profile
        self.sample_profile = SchemaProfile(
            id="test_profile",
            name="Test Profile",
            description="Test profile for unit tests",
            role="general_manager",
            columns=[
                SchemaColumn(
                    name="test_col",
                    display_name="Test Column",
                    description="A test column",
                    data_type="string",
                    visibility="public"
                )
            ],
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)
    
    def test_save_and_load_profile(self):
        """Test saving and loading a profile."""
        # Save profile
        result = self.editor.save_profile(self.sample_profile)
        self.assertTrue(result)
        
        # Load profile
        loaded = self.editor.load_profile("test_profile")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.id, self.sample_profile.id)
        self.assertEqual(loaded.name, self.sample_profile.name)
        self.assertEqual(len(loaded.columns), 1)
        self.assertEqual(loaded.columns[0].name, "test_col")
    
    def test_validate_profile(self):
        """Test profile validation."""
        # Valid profile
        result = self.editor.validate_profile(self.sample_profile)
        self.assertTrue(result["is_valid"])
        self.assertEqual(len(result["errors"]), 0)
        
        # Invalid profile (missing required fields)
        invalid_profile = SchemaProfile(
            id="",
            name="",
            description="",
            role="",
            columns=[]
        )
        result = self.editor.validate_profile(invalid_profile)
        self.assertFalse(result["is_valid"])
        self.assertGreater(len(result["errors"]), 0)
    
    def test_preview_validation(self):
        """Test preview validation with sample data."""
        # Create sample DataFrame
        df = pd.DataFrame({
            "test_col": ["value1", "value2"],
            "unmapped_col": [1, 2]
        })
        
        # Get preview
        preview = self.editor.preview_validation(self.sample_profile, df)
        
        self.assertTrue(preview["success"])
        self.assertEqual(preview["sample_rows"], 2)
        self.assertIn("test_col", preview["original_columns"])
        self.assertIn("unmapped_col", preview["original_columns"])
    
    def test_import_export_profile(self):
        """Test profile import/export functionality."""
        # Export profile
        exported_json = self.editor.export_profile(self.sample_profile, 'json')
        self.assertIsNotNone(exported_json)
        
        exported_yaml = self.editor.export_profile(self.sample_profile, 'yaml')
        self.assertIsNotNone(exported_yaml)
        
        # Import JSON profile
        imported_json = self.editor.import_profile(exported_json, 'json')
        self.assertIsNotNone(imported_json)
        self.assertEqual(imported_json.id, self.sample_profile.id)
        
        # Import YAML profile
        imported_yaml = self.editor.import_profile(exported_yaml, 'yaml')
        self.assertIsNotNone(imported_yaml)
        self.assertEqual(imported_yaml.id, self.sample_profile.id)
    
    def test_duplicate_profile(self):
        """Test profile duplication."""
        # Duplicate profile
        new_profile = self.editor.duplicate_profile(self.sample_profile, "test_profile_copy")
        self.assertIsNotNone(new_profile)
        self.assertEqual(new_profile.id, "test_profile_copy")
        self.assertTrue(new_profile.name.startswith("Copy of"))
        self.assertEqual(len(new_profile.columns), len(self.sample_profile.columns))
    
    def test_delete_profile(self):
        """Test profile deletion."""
        # Save profile first
        self.editor.save_profile(self.sample_profile)
        
        # Delete profile
        result = self.editor.delete_profile("test_profile")
        self.assertTrue(result)
        
        # Try to load deleted profile
        loaded = self.editor.load_profile("test_profile")
        self.assertIsNone(loaded)
        
        # Try to delete non-existent profile
        result = self.editor.delete_profile("non_existent")
        self.assertFalse(result)

if __name__ == '__main__':
    unittest.main()