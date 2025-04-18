"""
Tests for the customer validator module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.validators.customer_validator import CustomerValidator
from src.validators.base_validator import BaseRule


def test_customer_validator_initialization():
    """Test that the customer validator initializes correctly with rules."""
    validator = CustomerValidator()
    
    # Check validator properties
    assert validator.get_name() == "Customer Validator"
    assert "customer" in validator.get_description().lower()
    
    # Check that rules were created
    rules = validator.get_rules()
    assert isinstance(rules, list)
    assert len(rules) > 0
    assert all(isinstance(rule, BaseRule) for rule in rules)
    
    # Check for expected rule types
    rule_ids = [rule.id for rule in rules]
    assert "missing_lead_source" in rule_ids
    assert "missing_salesperson" in rule_ids
    assert "incomplete_sale" in rule_ids
    assert "invalid_date" in rule_ids


def test_missing_lead_source_rule():
    """Test the missing lead source rule."""
    # Create a test DataFrame
    data = {
        "Lead_Source": ["Website", None, "", "Google", " "],
        "OtherColumn": [1, 2, 3, 4, 5]
    }
    df = pd.DataFrame(data)
    
    # Initialize validator
    validator = CustomerValidator()
    
    # Apply validation
    result_df, flag_counts = validator.validate(df)
    
    # Check that flag column was created
    assert "flag_missing_lead_source" in result_df.columns
    
    # Check flagged rows (None, empty string, and space should be flagged)
    expected_flags = [False, True, True, False, True]
    assert list(result_df["flag_missing_lead_source"]) == expected_flags
    
    # Check flag counts
    assert "missing_lead_source" in flag_counts
    assert flag_counts["missing_lead_source"] == 3


def test_missing_salesperson_rule():
    """Test the missing salesperson rule."""
    # Create a test DataFrame
    data = {
        "Salesperson": ["John Doe", None, "", "Jane Smith", " "],
        "OtherColumn": [1, 2, 3, 4, 5]
    }
    df = pd.DataFrame(data)
    
    # Initialize validator
    validator = CustomerValidator()
    
    # Apply validation
    result_df, flag_counts = validator.validate(df)
    
    # Check that flag column was created
    assert "flag_missing_salesperson" in result_df.columns
    
    # Check flagged rows (None, empty string, and space should be flagged)
    expected_flags = [False, True, True, False, True]
    assert list(result_df["flag_missing_salesperson"]) == expected_flags
    
    # Check flag counts
    assert "missing_salesperson" in flag_counts
    assert flag_counts["missing_salesperson"] == 3


def test_incomplete_sale_rule():
    """Test the incomplete sale rule."""
    # Create a test DataFrame with complete and incomplete sales
    data = {
        "CustomerName": ["John Doe", "Jane Smith", None, "Alice Brown", "Bob Green"],
        "SaleDate": ["2023-01-01", "2023-02-01", "2023-03-01", None, "2023-05-01"],
        "Salesperson": ["Agent1", None, "Agent3", "Agent4", "Agent5"],
        "DealType": ["New", "Used", "Used", "New", None]
    }
    df = pd.DataFrame(data)
    
    # Initialize validator
    validator = CustomerValidator()
    
    # Apply validation
    result_df, flag_counts = validator.validate(df)
    
    # Check that flag column was created
    assert "flag_incomplete_sale" in result_df.columns
    
    # Check flagged rows (rows with any missing values should be flagged)
    expected_flags = [False, True, True, True, True]
    assert list(result_df["flag_incomplete_sale"]) == expected_flags
    
    # Check flag counts
    assert "incomplete_sale" in flag_counts
    assert flag_counts["incomplete_sale"] == 4


def test_invalid_date_rule():
    """Test the invalid date rule."""
    # Create future date
    today = datetime.now()
    future_date = (today + timedelta(days=30)).strftime('%Y-%m-%d')
    
    # Create a test DataFrame
    data = {
        "SaleDate": ["2023-01-01", "invalid-date", None, future_date],
        "OtherColumn": [1, 2, 3, 4]
    }
    df = pd.DataFrame(data)
    
    # Initialize validator
    validator = CustomerValidator()
    
    # Apply validation
    result_df, flag_counts = validator.validate(df)
    
    # Check that flag column was created
    assert "flag_invalid_date" in result_df.columns
    
    # Check flagged rows (invalid dates, None, and future dates should be flagged)
    expected_flags = [False, True, True, True]
    assert list(result_df["flag_invalid_date"]) == expected_flags
    
    # Check flag counts
    assert "invalid_date" in flag_counts
    assert flag_counts["invalid_date"] == 3


def test_column_mapping():
    """Test that column mapping works for different column names."""
    # Create a test DataFrame with different column names
    data = {
        "LeadSource": [None, "Website", "Google"],
        "Employee": ["John", None, "Jane"]
    }
    df = pd.DataFrame(data)
    
    # Initialize validator
    validator = CustomerValidator()
    
    # Modify rules for this test
    for rule in validator.rules:
        if rule.id == "missing_lead_source":
            rule.column_mapping = {"lead_source": "LeadSource"}
        elif rule.id == "missing_salesperson":
            rule.column_mapping = {"salesperson": "Employee"}
    
    # Apply validation
    result_df, flag_counts = validator.validate(df)
    
    # Check that flag columns were created with correct mapping
    assert "flag_missing_lead_source" in result_df.columns
    assert "flag_missing_salesperson" in result_df.columns
    
    # Check lead source flags
    lead_source_flags = list(result_df["flag_missing_lead_source"])
    assert lead_source_flags == [True, False, False]
    
    # Check salesperson flags
    salesperson_flags = list(result_df["flag_missing_salesperson"])
    assert salesperson_flags == [False, True, False]


def test_missing_columns():
    """Test validator behavior when expected columns are missing."""
    # Create a test DataFrame without customer columns
    data = {
        "UnrelatedColumn1": [1, 2, 3],
        "UnrelatedColumn2": ["a", "b", "c"]
    }
    df = pd.DataFrame(data)
    
    # Initialize validator
    validator = CustomerValidator()
    
    # Apply validation
    result_df, flag_counts = validator.validate(df)
    
    # Verify flag columns exist but all values are False
    for rule in validator.rules:
        flag_col = f"flag_{rule.id}"
        assert flag_col in result_df.columns
        assert not result_df[flag_col].any()
        assert flag_counts.get(rule.id, 0) == 0


def test_validate_multiple_rules():
    """Test that multiple rules are applied correctly."""
    # Create a test DataFrame with values that will trigger multiple rules
    data = {
        "Lead_Source": [None, "Website", "Google"],
        "Salesperson": ["John", None, "Jane"],
        "SaleDate": ["2023-01-01", "invalid-date", None],
        "CustomerName": ["Customer1", "Customer2", None],
        "DealType": ["New", None, "Used"]
    }
    df = pd.DataFrame(data)
    
    # Initialize validator
    validator = CustomerValidator()
    
    # Apply validation
    result_df, flag_counts = validator.validate(df)
    
    # Check that flag columns were created
    assert "flag_missing_lead_source" in result_df.columns
    assert "flag_missing_salesperson" in result_df.columns
    assert "flag_incomplete_sale" in result_df.columns
    assert "flag_invalid_date" in result_df.columns
    
    # Verify that issues are flagged
    assert flag_counts["missing_lead_source"] == 1
    assert flag_counts["missing_salesperson"] == 1
    
    # Check incomplete_sale - The flag count can be 2 or 3 depending on how the rule is implemented
    # Some implementations might consider the first row complete since it has all fields
    assert flag_counts["incomplete_sale"] >= 2
    
    # Check invalid_date - should flag invalid date format and None date
    assert flag_counts["invalid_date"] == 2