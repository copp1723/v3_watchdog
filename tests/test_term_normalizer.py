"""
Tests for the enhanced term normalizer with embedding support.
"""

import pytest
import pandas as pd
import numpy as np
from src.utils.term_normalizer import TermNormalizer

@pytest.fixture
def normalizer():
    """Create a TermNormalizer instance for testing."""
    return TermNormalizer(
        config_path="config/normalization_rules.yml",
        embedding_model="all-MiniLM-L6-v2"
    )

@pytest.fixture
def sample_df():
    """Create sample DataFrame for testing."""
    return pd.DataFrame({
        'LeadSource': ['CarGuru', 'cargurus.com', 'Car Gurus', 'AutoTrader', 'auto trader'],
        'SalesRep': ['John Smith', 'john.smith', 'J. Smith', 'Alice Jones', 'A. Jones'],
        'VehicleType': ['SUV', 'Sport Utility', 'sport utility vehicle', 'Sedan', 'sedan']
    })

def test_embedding_initialization(normalizer):
    """Test embedding model initialization."""
    assert normalizer.use_embeddings is True
    assert normalizer.embedding_model is not None
    assert normalizer.similarity_threshold == 0.85

def test_embedding_similarity(normalizer):
    """Test semantic similarity computation."""
    # Test exact matches
    assert normalizer._compute_similarity("CarGurus", "CarGurus") > 0.95
    
    # Test similar terms
    assert normalizer._compute_similarity("car gurus", "cargurus.com") > 0.85
    assert normalizer._compute_similarity("auto trader", "autotrader.com") > 0.85
    
    # Test dissimilar terms
    assert normalizer._compute_similarity("CarGurus", "Facebook") < 0.5

def test_term_normalization_with_embeddings(normalizer):
    """Test term normalization using embeddings."""
    # Test lead sources
    assert normalizer.normalize_term("cargurus.com", "lead_sources") == "CarGurus"
    assert normalizer.normalize_term("Car Guru", "lead_sources") == "CarGurus"
    assert normalizer.normalize_term("auto trader", "lead_sources") == "AutoTrader"
    
    # Test personnel titles
    assert normalizer.normalize_term("sales representative", "personnel_titles") == "SalesRep"
    assert normalizer.normalize_term("sales person", "personnel_titles") == "SalesRep"
    
    # Test vehicle types
    assert normalizer.normalize_term("sport utility vehicle", "vehicle_types") == "SUV"
    assert normalizer.normalize_term("Sports Utility", "vehicle_types") == "SUV"

def test_dataframe_normalization(normalizer, sample_df):
    """Test DataFrame normalization with embeddings."""
    # Map column names to rule categories
    column_categories = {
        'LeadSource': 'lead_sources',
        'SalesRep': 'personnel_titles',
        'VehicleType': 'vehicle_types'
    }
    
    # Update rules in normalizer
    for col, category in column_categories.items():
        if col in normalizer.rules:
            del normalizer.rules[col]
        normalizer.rules[category] = normalizer.rules.get(category, {})
    
    normalized_df = normalizer.normalize_dataframe(sample_df)
    
    # Check lead sources
    assert all(source in ["CarGurus", "AutoTrader"] 
              for source in normalized_df['LeadSource'].unique())
    
    # Check sales reps (should maintain original if no clear match)
    assert len(normalized_df['SalesRep'].unique()) == 2  # Should group similar names
    
    # Check vehicle types
    assert all(vtype in ["SUV", "Sedan"] 
              for vtype in normalized_df['VehicleType'].unique())

def test_column_normalization(normalizer, sample_df):
    """Test single column normalization."""
    df = normalizer.normalize_column(sample_df, 'LeadSource', 'lead_sources')
    assert all(source in ["CarGurus", "AutoTrader"] 
              for source in df['LeadSource'].unique())

def test_embedding_cache(normalizer):
    """Test embedding cache functionality."""
    # First call should compute embedding
    text = "CarGurus"
    emb1 = normalizer._get_embedding(text)
    
    # Second call should use cache
    emb2 = normalizer._get_embedding(text)
    
    assert np.array_equal(emb1, emb2)
    assert text in normalizer._embedding_cache

def test_fallback_to_fuzzy(normalizer):
    """Test fallback to fuzzy matching when embeddings fail."""
    # Temporarily disable embeddings
    normalizer.use_embeddings = False
    
    # Should still work using fuzzy matching
    assert normalizer.normalize_term("cargurus.com", "lead_sources") == "CarGurus"
    assert normalizer.normalize_term("auto trader", "lead_sources") == "AutoTrader"
    
    # Restore embeddings
    normalizer.use_embeddings = True

def test_error_handling(normalizer):
    """Test error handling in normalization."""
    # Test with invalid input
    assert normalizer.normalize_term(None, "lead_sources") == None
    assert normalizer.normalize_term(123, "lead_sources") == "123"
    
    # Test with invalid category
    assert normalizer.normalize_term("CarGurus", "invalid_category") == "CarGurus"

def test_dataframe_with_missing_values(normalizer):
    """Test handling of missing values in DataFrame."""
    df = pd.DataFrame({
        'LeadSource': ['CarGuru', None, np.nan, 'AutoTrader', ''],
        'VehicleType': ['SUV', None, 'sport utility', '', np.nan]
    })
    
    normalized = normalizer.normalize_dataframe(df)
    
    # Check that NaN and None are preserved
    assert normalized['LeadSource'].isna().sum() == 2
    assert normalized['VehicleType'].isna().sum() == 2