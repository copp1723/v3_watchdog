"""
Tests for the validator registry module.
"""

import pytest
from src.validators.validator_registry import (
    get_validator_classes,
    get_validators,
    get_validator_by_name,
    get_available_validator_names
)
from src.validators.base_validator import BaseValidator
from src.validators.financial_validator import FinancialValidator
from src.validators.customer_validator import CustomerValidator


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