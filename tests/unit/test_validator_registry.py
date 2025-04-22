"""
Tests for the validator registry module.
"""

import pytest
import pandas as pd
import numpy as np
from src.validators.validator_registry import (
    get_validator_classes,
    get_validators,
    get_validator_by_name,
    get_available_validator_names
)
from src.validators.base_validator import BaseValidator, BaseRule
from src.validators.financial_validator import FinancialValidator
from src.validators.customer_validator import CustomerValidator

# Test fixtures
@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing validators."""
    return pd.DataFrame({
        "Gross_Profit": [1000, -100, 500, 0, 2000],
        "Lead_Source": ["Website", None, "Google", "", "Facebook"],
        "Salesperson": ["John", "Jane", "Bob", None, "Alice"],
        "SaleDate": ["2023-01-01", "2023-02-15", "invalid-date", None, "2023-05-30"],
        "VIN": ["1HGCM82633A123456", "5TDZA23C13S012345", None, "", "WBAFG01070L123456"]
    })

@pytest.fixture
def mock_validator_instance():
    """Create a mock validator instance for testing."""
    class MockValidator(BaseValidator):
        def __init__(self, data=None):
            super().__init__(data)
            self._name = "Mock Validator"
            self._description = "A mock validator for testing"
        
        def validate(self):
            return []
            
        def get_name(self):
            return self._name
            
        def get_description(self):
            return self._description
    
    return MockValidator()

@pytest.fixture
def base_rule_instance():
    """Create a test BaseRule instance."""
    return BaseRule(
        id="test_rule",
        name="Test Rule",
        description="A test rule for validation",
        enabled=True,
        severity="Medium",
        category="Test",
        column_mapping={"test_column": "TestColumn"},
        threshold_value=100,
        threshold_operator=">"
    )


def test_base_rule():
    """Test that BaseRule can be instantiated with parameters."""
    rule = BaseRule(
        id="test_rule",
        name="Test Rule",
        description="A test rule for validation",
        enabled=True,
        severity="Medium",
        category="Test",
        column_mapping={"test_column": "TestColumn"},
        threshold_value=100,
        threshold_operator=">"
    )
    
    # Check that attributes are set correctly
    assert rule.id == "test_rule"
    assert rule.name == "Test Rule"
    assert rule.description == "A test rule for validation"
    assert rule.enabled is True
    assert rule.severity == "Medium"
    assert rule.category == "Test"
    assert rule.column_mapping == {"test_column": "TestColumn"}
    assert rule.threshold_value == 100
    assert rule.threshold_operator == ">"


def test_get_validator_classes():
    """Test that all validator classes are discovered."""
    validator_classes = get_validator_classes()
    
    # Check that we have a list of classes
    assert isinstance(validator_classes, list)
    assert len(validator_classes) >= 2  # We should have at least our 2 validators
    
    # Check that all items are validator classes
    for cls in validator_classes:
        assert issubclass(cls, BaseValidator)
    
    # Check that our known validators are included
    assert FinancialValidator in validator_classes
    assert CustomerValidator in validator_classes


def test_get_validators():
    """Test that all validators are instantiated."""
    validators = get_validators()
    
    # Check that we have a list of validator instances
    assert isinstance(validators, list)
    assert len(validators) >= 2  # We should have at least our 2 validators
    
    # Check that all items are validator instances
    for validator in validators:
        assert isinstance(validator, BaseValidator)
    
    # Check that our known validators are included
    validator_classes = [type(v) for v in validators]
    assert FinancialValidator in validator_classes
    assert CustomerValidator in validator_classes
    
    # Check that each validator has a name and description
    for validator in validators:
        assert validator.get_name()
        assert validator.get_description()


def test_get_validator_by_name():
    """Test retrieving a validator by name."""
    # Get a financial validator
    financial_validator = get_validator_by_name("Financial Validator")
    assert financial_validator is not None
    assert isinstance(financial_validator, FinancialValidator)
    
    # Get a customer validator
    customer_validator = get_validator_by_name("Customer Validator")
    assert customer_validator is not None
    assert isinstance(customer_validator, CustomerValidator)
    
    # Try to get a non-existent validator
    nonexistent_validator = get_validator_by_name("Nonexistent Validator")
    assert nonexistent_validator is None


def test_get_available_validator_names():
    """Test retrieving the list of available validator names."""
    validator_names = get_available_validator_names()
    
    # Check that we have a list of strings
    assert isinstance(validator_names, list)
    assert len(validator_names) >= 2  # We should have at least our 2 validators
    assert all(isinstance(name, str) for name in validator_names)
    
    # Check that our known validators are included
    assert "Financial Validator" in validator_names
    assert "Customer Validator" in validator_names


def test_validator_registration_consistency():
    """Test that all validators are consistently registered."""
    # Get all validators and their names
    validators = get_validators()
    validator_names = get_available_validator_names()
    
    # Check that the number of validators matches the number of names
    assert len(validators) == len(validator_names)
    
    # Check that each validator's name is in the list of names
    for validator in validators:
        assert validator.get_name() in validator_names


def test_validator_rule_creation():
    """Test that validators can create rules properly."""
    # Create a financial validator
    financial_validator = FinancialValidator()
    
    # Test that rules are created
    rules = financial_validator._create_financial_rules()
    assert isinstance(rules, list)
    assert len(rules) > 0
    
    # Check that rules have proper attributes
    for rule in rules:
        assert isinstance(rule, BaseRule)
        assert hasattr(rule, 'id')
        assert hasattr(rule, 'name')
        assert hasattr(rule, 'description')
        assert hasattr(rule, 'enabled')


def test_mock_validator_methods(mock_validator_instance):
    """Test that the mock validator works as expected."""
    validator = mock_validator_instance
    
    # Test name and description methods
    assert validator.get_name() == "Mock Validator"
    assert validator.get_description() == "A mock validator for testing"
    
    # Test validate method
    assert validator.validate() == []


def test_validator_with_sample_data(sample_dataframe):
    """Test that validators can work with sample data."""
    # Create validators with sample data
    financial_validator = FinancialValidator(sample_dataframe)
    customer_validator = CustomerValidator(sample_dataframe)
    
    # Check that data was assigned
    assert financial_validator.data is not None
    assert customer_validator.data is not None
    
    # Get validator names
    assert financial_validator.get_name() == "Financial Validator"
    assert customer_validator.get_name() == "Customer Validator"
