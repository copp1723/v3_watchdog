"""
Tests for the schema profile management system.
"""

import pytest
import os
import json
from datetime import datetime
from unittest.mock import Mock, patch

from src.validators.schema_manager import SchemaProfileManager
from src.validators.column_mapper import ColumnMapper
import pandas as pd

@pytest.fixture
def temp_config_dir(tmpdir):
    """Create a temporary config directory."""
    return str(tmpdir)

@pytest.fixture
def schema_manager(temp_config_dir):
    """Create a schema manager instance."""
    manager = SchemaProfileManager(config_dir=temp_config_dir)
    # Create default profile
    manager.create_default_profile()
    return manager

@pytest.fixture
def sample_profile():
    """Create a sample schema profile."""
    return {
        "id": "test",
        "name": "Test Profile",
        "description": "Test schema profile",
        "role": "test",
        "columns": [
            {
                "name": "sale_date",
                "display_name": "Sale Date",
                "description": "Date of sale",
                "data_type": "datetime",
                "visibility": "public",
                "format": "%Y-%m-%d",
                "aliases": ["date", "transaction_date"]
            },
            {
                "name": "gross_profit",
                "display_name": "Gross Profit",
                "description": "Total gross profit",
                "data_type": "float",
                "visibility": "public",
                "format": "${:,.2f}",
                "aliases": ["gross", "total_gross"]
            }
        ]
    }

@pytest.fixture
def sample_data():
    """Create sample DataFrame for testing."""
    return pd.DataFrame({
        'transaction_date': ['2023-01-01', '2023-01-02'],
        'total_gross': [1000, 2000],
        'other_column': ['A', 'B']
    })

def test_schema_manager_initialization(schema_manager, temp_config_dir):
    """Test schema manager initialization."""
    assert os.path.exists(temp_config_dir)
    assert schema_manager.config_dir == temp_config_dir

def test_save_and_load_profile(schema_manager, sample_profile):
    """Test saving and loading a profile."""
    # Save profile
    schema_manager.save_profile("test", sample_profile)
    
    # Load profile
    loaded = schema_manager.load_profile("test")
    
    # Check core fields
    assert loaded['id'] == sample_profile['id']
    assert loaded['name'] == sample_profile['name']
    assert len(loaded['columns']) == len(sample_profile['columns'])
    
    # Check timestamps
    assert 'created_at' in loaded
    assert 'updated_at' in loaded

def test_list_profiles(schema_manager, sample_profile):
    """Test listing available profiles."""
    # Save multiple profiles with matching IDs
    profile1 = sample_profile.copy()
    profile1['id'] = "test1"  # Match the filename
    schema_manager.save_profile("test1", profile1)
    
    profile2 = sample_profile.copy()
    profile2['id'] = "test2"  # Match the filename
    schema_manager.save_profile("test2", profile2)
    
    # List profiles
    profiles = schema_manager.list_profiles()
    
    # Should include default profile plus our two test profiles
    assert len(profiles) == 3
    assert any(p['id'] == "test1" for p in profiles)
    assert any(p['id'] == "test2" for p in profiles)
    assert any(p['id'] == "default" for p in profiles)

def test_invalid_profile_structure(schema_manager):
    """Test validation of profile structure."""
    invalid_profile = {
        "id": "invalid",
        "name": "Invalid Profile"
        # Missing required fields
    }
    
    with pytest.raises(ValueError) as e:
        schema_manager.save_profile("invalid", invalid_profile)
    
    assert "Missing required field" in str(e.value)

def test_column_mapper_basic(schema_manager, sample_profile, sample_data):
    """Test basic column mapping functionality."""
    # Save test profile
    schema_manager.save_profile("test", sample_profile)
    
    # Create mapper
    mapper = ColumnMapper(schema_manager)
    
    # Map columns
    mapped_df, results = mapper.map_columns(sample_data, "test")
    
    # Check mappings
    assert "sale_date" in mapped_df.columns
    assert "gross_profit" in mapped_df.columns
    assert results['mapped']['sale_date'] == 'transaction_date'
    assert results['mapped']['gross_profit'] == 'total_gross'
    assert 'other_column' in results['unmapped']

def test_column_mapper_learning(schema_manager, sample_data):
    """Test column mapper learning capabilities."""
    mapper = ColumnMapper(schema_manager)
    
    # Teach a mapping
    mapper.learn_mapping("transaction_date", "sale_date", confidence=100)
    
    # Create a new DataFrame with learned column
    new_data = pd.DataFrame({
        'transaction_date': ['2023-01-03', '2023-01-04']
    })
    
    # Map columns using default profile
    mapped_df, results = mapper.map_columns(new_data)
    
    # Check that learned mapping was used
    assert "sale_date" in mapped_df.columns
    assert results['mapped']['sale_date'] == 'transaction_date'
    assert results['confidence_scores']['sale_date'] == 100

def test_fuzzy_matching(schema_manager, sample_profile):
    """Test fuzzy matching for column names."""
    mapper = ColumnMapper(schema_manager)
    
    # Create data with slightly different column names
    df = pd.DataFrame({
        'trans_date': ['2023-01-01'],
        'gross_prof': [1000]
    })
    
    # Save test profile
    schema_manager.save_profile("test", sample_profile)
    
    # Map columns
    mapped_df, results = mapper.map_columns(df, "test")
    
    # Check fuzzy matches
    assert "sale_date" in mapped_df.columns
    assert "gross_profit" in mapped_df.columns
    assert results['confidence_scores']['sale_date'] >= 80  # Above threshold
    assert results['confidence_scores']['gross_profit'] >= 80

@pytest.mark.skip("UI tests need more mocking")
def test_profile_editor_state(schema_manager, sample_profile):
    """Test profile editor state management."""
    from src.watchdog_ai.ui.pages.schema_profile_editor import SchemaProfileEditor
    
    # Save test profile
    schema_manager.save_profile("test", sample_profile)
    
    # Create editor
    editor = SchemaProfileEditor(schema_manager)
    
    # Mock streamlit session state
    with patch('streamlit.session_state', {'columns': []}):
        # Add a column
        new_column = {
            "name": "test_column",
            "display_name": "Test Column",
            "data_type": "string",
            "visibility": "public"
        }
        
        # Mock form submission
        with patch('streamlit.form') as mock_form:
            mock_form.return_value.__enter__.return_value = None
            mock_form.return_value.__exit__.return_value = None
            
            # Add column to session state
            st.session_state.columns.append(new_column)
            
            # Verify column was added
            assert len(st.session_state.columns) == 1
            assert st.session_state.columns[0]['name'] == "test_column"

@pytest.mark.skip("UI tests need more mocking")
def test_profile_preview(schema_manager, sample_profile, sample_data):
    """Test profile preview functionality."""
    from src.watchdog_ai.ui.pages.schema_profile_editor import SchemaProfileEditor
    
    # Save test profile
    schema_manager.save_profile("test", sample_profile)
    
    # Create editor
    editor = SchemaProfileEditor(schema_manager)
    
    # Mock file upload
    mock_file = Mock()
    mock_file.name = "test.csv"
    
    with patch('pandas.read_csv', return_value=sample_data):
        with patch('streamlit.file_uploader', return_value=mock_file):
            # Preview mapping
            editor._render_preview()
            
            # Verify mapping results were displayed
            assert "sale_date" in mapped_df.columns
            assert "gross_profit" in mapped_df.columns