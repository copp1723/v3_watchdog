"""
Tests for the metadata module.

This module tests the metadata generation functionality that extracts and
formats information about uploaded files and their contents.
"""

import pytest
import pandas as pd
import numpy as np
import json
import os
import tempfile
from datetime import datetime
from typing import Dict, List, Any

# Import the module to test
try:
    from v2_watchdog_ai.metadata import (
        generate_metadata,
        metadata_to_json,
        save_metadata,
        generate_and_save_metadata,
        generate_file_hash
    )
except ImportError:
    # Define empty functions with the same signatures for testing
    def generate_metadata(df, source_file_name, source_file_path=None): return {}
    def metadata_to_json(metadata, pretty=True): return "{}"
    def save_metadata(metadata, output_path): pass
    def generate_and_save_metadata(df, source_file_name, source_file_path=None, output_dir=None): return {}
    def generate_file_hash(filepath): return ""


class TestMetadataGeneration:
    """Tests for metadata generation from DataFrames."""
    
    def test_basic_metadata_structure(self, clean_csv_data):
        """Test the basic structure of generated metadata."""
        # Arrange
        df = clean_csv_data
        source_file = "test_file.csv"
        
        # Act
        metadata = generate_metadata(df, source_file)
        
        # Assert
        assert isinstance(metadata, dict)
        assert metadata.get('source_file_name') == source_file
        assert 'upload_timestamp' in metadata
        assert metadata.get('rows') == len(df)
        assert metadata.get('columns') == len(df.columns)
        assert metadata.get('column_names') == list(df.columns)
    
    def test_column_types_included(self, clean_csv_data):
        """Test that column types are included in metadata."""
        # Arrange
        df = clean_csv_data
        source_file = "test_file.csv"
        
        # Act
        metadata = generate_metadata(df, source_file)
        
        # Assert
        assert 'column_types' in metadata
        assert isinstance(metadata['column_types'], dict)
        assert len(metadata['column_types']) == len(df.columns)
        assert all(col in metadata['column_types'] for col in df.columns)
    
    def test_missing_values_tracking(self):
        """Test that missing values are tracked correctly."""
        # Arrange
        df = pd.DataFrame({
            'col1': [1, None, 3],
            'col2': ['a', 'b', None],
            'col3': [1.1, 2.2, 3.3]
        })
        source_file = "test_file.csv"
        
        # Act
        metadata = generate_metadata(df, source_file)
        
        # Assert
        assert 'missing_values' in metadata
        assert metadata['missing_values']['col1'] == 1
        assert metadata['missing_values']['col2'] == 1
        assert metadata['missing_values']['col3'] == 0
        
        # Should also have percentage
        assert 'missing_percentage' in metadata
        assert metadata['missing_percentage']['col1'] == pytest.approx(33.33, abs=0.01)
    
    def test_numeric_stats(self):
        """Test that numeric statistics are included for numeric columns."""
        # Arrange
        df = pd.DataFrame({
            'int_col': [1, 2, 3, 4, 5],
            'float_col': [1.1, 2.2, 3.3, 4.4, 5.5],
            'str_col': ['a', 'b', 'c', 'd', 'e']
        })
        source_file = "test_file.csv"
        
        # Act
        metadata = generate_metadata(df, source_file)
        
        # Assert
        assert 'numeric_stats' in metadata
        assert 'int_col' in metadata['numeric_stats']
        assert 'float_col' in metadata['numeric_stats']
        assert 'str_col' not in metadata['numeric_stats']
        
        # Check specific stats
        int_stats = metadata['numeric_stats']['int_col']
        assert int_stats['min'] == 1
        assert int_stats['max'] == 5
        assert int_stats['mean'] == 3
        assert int_stats['median'] == 3
    
    def test_with_file_path(self, sample_clean_csv_path):
        """Test metadata generation with a file path."""
        # Arrange
        df = pd.read_csv(sample_clean_csv_path)
        source_file = os.path.basename(sample_clean_csv_path)
        
        # Act
        metadata = generate_metadata(df, source_file, sample_clean_csv_path)
        
        # Assert
        assert 'file_size_bytes' in metadata
        assert metadata['file_size_bytes'] > 0
        assert 'file_extension' in metadata
        assert metadata['file_extension'] == '.csv'
        assert 'file_hash' in metadata
        assert isinstance(metadata['file_hash'], str)
        assert len(metadata['file_hash']) > 0


class TestMetadataFormatting:
    """Tests for metadata formatting and serialization."""
    
    def test_json_serialization(self, sample_metadata):
        """Test serialization of metadata to JSON."""
        # Arrange
        metadata = sample_metadata
        
        # Act
        json_str = metadata_to_json(metadata)
        
        # Assert
        assert isinstance(json_str, str)
        # Should be valid JSON
        json_obj = json.loads(json_str)
        assert isinstance(json_obj, dict)
        
        # Check if keys are preserved
        assert json_obj.get('source_file_name') == metadata.get('source_file_name')
        assert json_obj.get('rows') == metadata.get('rows')
    
    def test_pretty_json_formatting(self, sample_metadata):
        """Test pretty formatting of JSON output."""
        # Arrange
        metadata = sample_metadata
        
        # Act
        pretty_json = metadata_to_json(metadata, pretty=True)
        compact_json = metadata_to_json(metadata, pretty=False)
        
        # Assert
        assert len(pretty_json) > len(compact_json)
        assert '\n' in pretty_json
        assert '\n' not in compact_json
    
    def test_handle_non_serializable_types(self):
        """Test handling of non-serializable types in metadata."""
        # Arrange
        metadata = {
            'datetime': datetime(2023, 4, 15, 10, 30, 45),
            'numpy_int': np.int64(42),
            'numpy_float': np.float64(3.14),
            'numpy_array': np.array([1, 2, 3])
        }
        
        # Act
        json_str = metadata_to_json(metadata)
        
        # Assert
        assert isinstance(json_str, str)
        # Should be valid JSON
        json_obj = json.loads(json_str)
        assert isinstance(json_obj, dict)
        
        # Check conversions
        assert '2023-04-15' in json_obj['datetime']
        assert json_obj['numpy_int'] == 42
        assert json_obj['numpy_float'] == 3.14
        assert json_obj['numpy_array'] == [1, 2, 3]


class TestMetadataSaving:
    """Tests for saving metadata to files."""
    
    def test_save_metadata(self, sample_metadata, tmp_path):
        """Test saving metadata to a JSON file."""
        # Arrange
        metadata = sample_metadata
        output_path = os.path.join(tmp_path, "metadata.json")
        
        # Act
        save_metadata(metadata, output_path)
        
        # Assert
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0
        
        # Check if file contains valid JSON
        with open(output_path, 'r') as f:
            loaded_json = json.load(f)
        
        assert loaded_json.get('source_file_name') == metadata.get('source_file_name')
    
    def test_generate_and_save(self, clean_csv_data, tmp_path):
        """Test generating and saving metadata in one operation."""
        # Arrange
        df = clean_csv_data
        source_file = "test_file.csv"
        output_dir = tmp_path
        
        # Act
        result = generate_and_save_metadata(df, source_file, output_dir=output_dir)
        
        # Assert
        assert isinstance(result, dict)
        expected_path = os.path.join(output_dir, "test_file_metadata.json")
        assert os.path.exists(expected_path)


class TestFileHash:
    """Tests for file hash generation."""
    
    def test_hash_generation(self, sample_clean_csv_path):
        """Test generation of file hash."""
        # Act
        hash_value = generate_file_hash(sample_clean_csv_path)
        
        # Assert
        assert isinstance(hash_value, str)
        assert len(hash_value) > 0
        
        # Same file should produce same hash
        second_hash = generate_file_hash(sample_clean_csv_path)
        assert hash_value == second_hash
    
    def test_different_files_different_hashes(self, sample_clean_csv_path, sample_messy_csv_path):
        """Test that different files produce different hashes."""
        # Act
        hash1 = generate_file_hash(sample_clean_csv_path)
        hash2 = generate_file_hash(sample_messy_csv_path)
        
        # Assert
        assert hash1 != hash2
