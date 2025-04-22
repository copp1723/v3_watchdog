"""
Tests for the adaptive schema validation system.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.utils.adaptive_schema import (
    AdaptiveSchema,
    MappingStore,
    ConfidenceCalculator,
    MappingSuggestion,
    MappingHistory
)

@pytest.fixture
def sample_data():
    """Create sample DataFrame with various column names."""
    return pd.DataFrame({
        'total_gross': [1000.0, 2000.0, 3000.0],
        'source_of_lead': ['Website', 'Walk-in', 'CarGurus'],
        'transaction_date': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'vehicle_id': ['1HGCM82633A123456', '5TFBW5F13AX123457', 'WBAGH83576D123458'],
        'employee_name': ['John Doe', 'Jane Smith', 'Bob Wilson']
    })

@pytest.fixture
def mapping_store():
    """Create MappingStore instance with in-memory storage."""
    return MappingStore(redis_url=None)

def test_confidence_calculation():
    """Test confidence score calculation."""
    calc = ConfidenceCalculator()
    
    # Test exact match
    confidence = calc.calculate_confidence(
        "total_gross",
        "total_gross",
        {"expected_type": "float64", "actual_type": "float64"}
    )
    assert confidence == 1.0
    
    # Test fuzzy match
    confidence = calc.calculate_confidence(
        "total_gross_profit",
        "total_gross",
        {"expected_type": "float64", "actual_type": "float64"}
    )
    assert 0.7 <= confidence <= 1.0
    
    # Test low confidence match
    confidence = calc.calculate_confidence(
        "completely_different",
        "total_gross",
        {"expected_type": "float64", "actual_type": "object"}
    )
    assert confidence < 0.5

def test_mapping_suggestions(sample_data):
    """Test generation of mapping suggestions."""
    schema = AdaptiveSchema()
    result, suggestions = schema.validate_with_suggestions(sample_data)
    
    # Should suggest mappings for non-standard column names
    assert len(suggestions) > 0
    
    # Check suggestion properties
    for suggestion in suggestions:
        assert isinstance(suggestion, MappingSuggestion)
        assert 0.0 <= suggestion.confidence <= 1.0
        assert suggestion.original_name in sample_data.columns
        assert suggestion.canonical_name in schema.schema["required_columns"]

def test_pattern_matching(sample_data):
    """Test pattern-based confidence boosting."""
    schema = AdaptiveSchema()
    
    # Test VIN pattern
    vin_suggestion = schema._suggest_mapping(sample_data, "vin")
    assert vin_suggestion is not None
    assert vin_suggestion.original_name == "vehicle_id"
    assert vin_suggestion.confidence > 0.8
    
    # Test date pattern
    date_suggestion = schema._suggest_mapping(sample_data, "sale_date")
    assert date_suggestion is not None
    assert date_suggestion.original_name == "transaction_date"
    assert date_suggestion.confidence > 0.8

def test_mapping_persistence(mapping_store):
    """Test saving and retrieving mappings."""
    # Save a mapping
    mapping_store.save_mapping(
        "total_gross_profit",
        "total_gross",
        0.9,
        {"expected_type": "float64", "actual_type": "float64"}
    )
    
    # Retrieve mapping history
    history = mapping_store.get_mapping_history("total_gross_profit")
    assert isinstance(history, MappingHistory)
    assert len(history.mappings) == 1
    assert history.mappings[0]["confidence"] == 0.9

def test_learning_from_confirmation(sample_data):
    """Test learning from user confirmations."""
    schema = AdaptiveSchema()
    
    # Initial suggestion
    result, suggestions = schema.validate_with_suggestions(sample_data)
    initial_suggestion = next(s for s in suggestions if s.original_name == "source_of_lead")
    initial_confidence = initial_suggestion.confidence
    
    # Confirm the mapping
    schema.learn_from_confirmation(
        "source_of_lead",
        "lead_source",
        confirmed=True,
        context={"expected_type": "object", "actual_type": "object"}
    )
    
    # Get new suggestion
    result, suggestions = schema.validate_with_suggestions(sample_data)
    new_suggestion = next(s for s in suggestions if s.original_name == "source_of_lead")
    
    # Confidence should increase after confirmation
    assert new_suggestion.confidence > initial_confidence

def test_validation_with_learned_mappings(sample_data):
    """Test validation using previously learned mappings."""
    schema = AdaptiveSchema()
    
    # First validation
    result1, _ = schema.validate_with_suggestions(sample_data)
    
    # Learn some mappings
    schema.learn_from_confirmation(
        "source_of_lead", "lead_source", True,
        {"expected_type": "object", "actual_type": "object"}
    )
    schema.learn_from_confirmation(
        "transaction_date", "sale_date", True,
        {"expected_type": "datetime64", "actual_type": "object"}
    )
    
    # Second validation
    result2, suggestions2 = schema.validate_with_suggestions(sample_data)
    
    # Should have fewer missing required columns
    assert len(result2.missing_required) < len(result1.missing_required)
    
    # Should have fewer suggestions
    assert len(suggestions2) < len(result1.missing_required)

def test_confidence_decay():
    """Test confidence score decay over time."""
    calc = ConfidenceCalculator()
    
    # Create old context
    old_context = {
        "expected_type": "float64",
        "actual_type": "float64",
        "historical_confidence": 0.9,
        "last_used": (datetime.now() - timedelta(days=30)).isoformat()
    }
    
    # Create recent context
    recent_context = {
        "expected_type": "float64",
        "actual_type": "float64",
        "historical_confidence": 0.9,
        "last_used": datetime.now().isoformat()
    }
    
    # Old mapping should have lower confidence
    old_confidence = calc.calculate_confidence("total_gross", "total_gross", old_context)
    recent_confidence = calc.calculate_confidence("total_gross", "total_gross", recent_context)
    
    assert old_confidence < recent_confidence

def test_error_handling(sample_data):
    """Test error handling in adaptive schema."""
    schema = AdaptiveSchema()
    
    # Test with invalid DataFrame
    invalid_df = pd.DataFrame()
    with pytest.raises(ValidationError):
        schema.validate_with_suggestions(invalid_df)
    
    # Test with invalid column types
    invalid_types_df = sample_data.copy()
    invalid_types_df["total_gross"] = "not a number"
    result, suggestions = schema.validate_with_suggestions(invalid_types_df)
    assert not result.is_valid
    assert "total_gross" in result.type_errors