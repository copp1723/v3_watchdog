"""
Tests for the enhanced data validation system.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from src.watchdog_ai.utils.validation import (
    ValidationRule,
    RequiredRule,
    TypeRule,
    RangeRule,
    PatternRule,
    DataValidator,
    validate_string,
    validate_number,
    validate_date,
    validate_file,
    validate_dataframe,
    sanitize_html,
    sanitize_sql,
    sanitize_filename
)

@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'vin': ['1HGCM82633A004352', 'invalid', '5YJSA1E11GF151205'],
        'sale_date': ['2024-01-01', '2024-13-01', '2024-02-01'],
        'total_gross': [1000, -500, 'invalid'],
        'lead_source': ['CarGurus', None, 'Website']
    })

def test_required_rule():
    """Test RequiredRule validation."""
    rule = RequiredRule("test")
    assert rule.validate("value")[0] is True
    assert rule.validate("")[0] is True
    assert rule.validate(None)[0] is False
    assert rule.validate(np.nan)[0] is False

def test_type_rule():
    """Test TypeRule validation."""
    numeric_rule = TypeRule("numeric")
    assert numeric_rule.validate("123")[0] is True
    assert numeric_rule.validate("abc")[0] is False
    assert numeric_rule.validate(None)[0] is True  # Nulls are allowed
    
    date_rule = TypeRule("datetime")
    assert date_rule.validate("2024-01-01")[0] is True
    assert date_rule.validate("invalid")[0] is False
    
    bool_rule = TypeRule("boolean")
    assert bool_rule.validate("1")[0] is True
    assert bool_rule.validate("0")[0] is True
    assert bool_rule.validate("2")[0] is False

def test_range_rule():
    """Test RangeRule validation."""
    rule = RangeRule(min_val=0, max_val=100)
    assert rule.validate(50)[0] is True
    assert rule.validate(-1)[0] is False
    assert rule.validate(101)[0] is False
    assert rule.validate(None)[0] is True  # Nulls are allowed
    
    # Test with only min_val
    min_rule = RangeRule(min_val=0)
    assert min_rule.validate(1000)[0] is True
    assert min_rule.validate(-1)[0] is False
    
    # Test with only max_val
    max_rule = RangeRule(max_val=100)
    assert max_rule.validate(-50)[0] is True
    assert max_rule.validate(101)[0] is False

def test_pattern_rule():
    """Test PatternRule validation."""
    # Test VIN pattern
    vin_rule = PatternRule(r"^[A-HJ-NPR-Z0-9]{17}$")
    assert vin_rule.validate("1HGCM82633A004352")[0] is True
    assert vin_rule.validate("invalid")[0] is False
    assert vin_rule.validate(None)[0] is True  # Nulls are allowed
    
    # Test email pattern
    email_rule = PatternRule(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
    assert email_rule.validate("test@example.com")[0] is True
    assert email_rule.validate("invalid-email")[0] is False
    
    # Test phone pattern
    phone_rule = PatternRule(r"^\d{3}-\d{3}-\d{4}$")
    assert phone_rule.validate("123-456-7890")[0] is True
    assert phone_rule.validate("12345")[0] is False

def test_data_validator_initialization():
    """Test DataValidator initialization and rule setup."""
    validator = DataValidator()
    
    # Check default rules were set up
    assert "vin" in validator.column_rules
    assert "sale_date" in validator.column_rules
    assert "total_gross" in validator.column_rules
    assert "lead_source" in validator.column_rules
    
    # Check rule types
    vin_rules = validator.column_rules["vin"]
    assert any(isinstance(rule, RequiredRule) for rule in vin_rules)
    assert any(isinstance(rule, PatternRule) for rule in vin_rules)

def test_data_validator_with_sample_data(sample_df):
    """Test DataValidator with sample data."""
    validator = DataValidator()
    results = validator.validate_dataframe(sample_df)
    
    # Check overall structure
    assert isinstance(results, dict)
    assert 'valid' in results
    assert 'errors' in results
    assert 'warnings' in results
    assert 'stats' in results
    
    # Check validation results
    assert results['valid'] is False  # Should fail due to invalid values
    assert len(results['errors']) > 0
    assert 'total_rows' in results['stats']
    assert results['stats']['total_rows'] == 3

def test_data_validator_column_stats(sample_df):
    """Test column statistics generation."""
    validator = DataValidator()
    results = validator.validate_dataframe(sample_df)
    
    # Check column stats
    assert 'column_stats' in results['stats']
    for col in sample_df.columns:
        if col in validator.column_rules:
            assert col in results['stats']['column_stats']
            stats = results['stats']['column_stats'][col]
            assert 'null_count' in stats
            assert 'unique_count' in stats
            assert 'sample_values' in stats

def test_validate_string():
    """Test string validation function."""
    # Basic validation
    assert validate_string("test")[0] is True
    assert validate_string("")[0] is True
    
    # Length validation
    assert validate_string("test", min_length=2)[0] is True
    assert validate_string("a", min_length=2)[0] is False
    assert validate_string("test", max_length=3)[0] is False
    
    # Pattern validation
    assert validate_string("123", pattern=r"^\d+$")[0] is True
    assert validate_string("abc", pattern=r"^\d+$")[0] is False
    
    # Type validation
    with pytest.raises(TypeError):
        validate_string(123)

def test_validate_number():
    """Test number validation function."""
    # Basic validation
    assert validate_number("123")[0] is True
    assert validate_number(123)[0] is True
    assert validate_number("abc")[0] is False
    
    # Range validation
    assert validate_number(50, min_val=0, max_val=100)[0] is True
    assert validate_number(-1, min_val=0)[0] is False
    assert validate_number(101, max_val=100)[0] is False
    
    # Float validation
    assert validate_number("123.45")[0] is True
    assert validate_number(-123.45)[0] is True

def test_validate_date():
    """Test date validation function."""
    # Valid dates
    assert validate_date("2024-01-01")[0] is True
    assert validate_date("2024/01/01")[0] is True
    assert validate_date(datetime.now())[0] is True
    
    # Invalid dates
    assert validate_date("invalid")[0] is False
    assert validate_date("2024-13-01")[0] is False
    assert validate_date("")[0] is False

class MockFile:
    """Mock file object for testing."""
    def __init__(self, name):
        self.name = name

def test_validate_file():
    """Test file validation function."""
    # Valid files
    assert validate_file(MockFile("test.csv"))[0] is True
    assert validate_file(MockFile("test.xlsx"))[0] is True
    assert validate_file(MockFile("test.xls"))[0] is True
    
    # Invalid files
    assert validate_file(MockFile("test.txt"))[0] is False
    assert validate_file(MockFile(""))[0] is False
    assert validate_file(None)[0] is False

def test_validate_dataframe():
    """Test DataFrame validation function."""
    # Valid DataFrame
    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    assert validate_dataframe(df)[0] is True
    
    # Empty DataFrame
    assert validate_dataframe(pd.DataFrame())[0] is False
    
    # Missing required columns
    assert validate_dataframe(df, required_columns=['a', 'c'])[0] is False
    assert validate_dataframe(df, required_columns=['a', 'b'])[0] is True

def test_sanitization_functions():
    """Test sanitization utility functions."""
    # HTML sanitization
    assert sanitize_html("<p>test</p>") == "test"
    assert sanitize_html('<script>alert("xss")</script>') == 'alert("xss")'
    
    # SQL sanitization
    assert sanitize_sql("'; DROP TABLE users; --") == " DROP TABLE users "
    assert sanitize_sql("normal text") == "normal text"
    
    # Filename sanitization
    assert sanitize_filename("test.txt") == "test.txt"
    assert sanitize_filename("test/file.txt") == "test_file.txt"
    assert sanitize_filename("test<>:file.txt") == "test___file.txt"

def test_validation_with_large_dataset():
    """Test validation performance with a large dataset."""
    # Create a large DataFrame
    large_df = pd.DataFrame({
        'vin': ['1HGCM82633A004352'] * 10000,
        'sale_date': ['2024-01-01'] * 10000,
        'total_gross': [1000] * 10000,
        'lead_source': ['CarGurus'] * 10000
    })
    
    validator = DataValidator()
    results = validator.validate_dataframe(large_df)
    
    assert results['valid'] is True
    assert results['stats']['total_rows'] == 10000
    assert len(results['errors']) == 0

def test_validation_error_messages():
    """Test validation error message clarity."""
    validator = DataValidator()
    
    # Test with invalid data
    df = pd.DataFrame({
        'vin': ['invalid_vin'],
        'total_gross': ['not_a_number'],
        'sale_date': ['invalid_date']
    })
    
    results = validator.validate_dataframe(df)
    
    # Check error message clarity
    for error in results['errors']:
        assert isinstance(error, dict)
        assert 'column' in error
        assert 'rule' in error
        assert 'invalid_rows' in error
        assert len(error['invalid_rows']) > 0
        
        # Check error message format
        for invalid_row in error['invalid_rows']:
            assert 'row' in invalid_row
            assert 'value' in invalid_row
            assert 'message' in invalid_row
            assert isinstance(invalid_row['message'], str)
            assert len(invalid_row['message']) > 0

if __name__ == '__main__':
    pytest.main([__file__])