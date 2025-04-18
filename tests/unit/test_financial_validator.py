"""
Tests for the financial validator module.
"""

import pytest
import pandas as pd
import numpy as np
from src.validators.financial_validator import FinancialValidator
from src.validators.base_validator import BaseRule


def test_financial_validator_initialization():
    """Test that the financial validator initializes correctly with rules."""
    validator = FinancialValidator()
    
    # Check validator properties
    assert validator.get_name() == "Financial Validator"
    assert "financial" in validator.get_description().lower()
    
    # Check that rules were created
    rules = validator.get_rules()
    assert isinstance(rules, list)
    assert len(rules) > 0
    assert all(isinstance(rule, BaseRule) for rule in rules)
    
    # Check for expected rule types
    rule_ids = [rule.id for rule in rules]
    assert "negative_gross" in rule_ids
    assert "low_gross" in rule_ids
    assert "apr_out_of_range" in rule_ids


def test_negative_gross_rule():
    """Test the negative gross profit rule."""
    # Create a test DataFrame
    data = {
        "Gross_Profit": [-100, 0, 500, -50, 1000],
        "OtherColumn": [1, 2, 3, 4, 5]
    }
    df = pd.DataFrame(data)
    
    # Initialize validator
    validator = FinancialValidator()
    
    # Apply validation
    result_df, flag_counts = validator.validate(df)
    
    # Check that flag column was created
    assert "flag_negative_gross" in result_df.columns
    
    # Check that negative values are flagged
    assert result_df.loc[result_df["Gross_Profit"] < 0, "flag_negative_gross"].all()
    assert not result_df.loc[result_df["Gross_Profit"] >= 0, "flag_negative_gross"].any()
    
    # Check flag counts
    assert "negative_gross" in flag_counts
    neg_count = sum(result_df["Gross_Profit"] < 0)
    assert flag_counts["negative_gross"] == neg_count


def test_apr_out_of_range_rule():
    """Test the APR out of range rule."""
    # Create a test DataFrame
    data = {
        "APR": [-1, 0, 5, 15, 30, 50],
        "OtherColumn": [1, 2, 3, 4, 5, 6]
    }
    df = pd.DataFrame(data)
    
    # Initialize validator
    validator = FinancialValidator()
    
    # Apply validation
    result_df, flag_counts = validator.validate(df)
    
    # Check that flag column was created
    assert "flag_apr_out_of_range" in result_df.columns
    
    # Check flagged rows
    expected_flags = [True, False, False, False, True, True]
    assert list(result_df["flag_apr_out_of_range"]) == expected_flags
    
    # Check flag counts
    assert "apr_out_of_range" in flag_counts
    assert flag_counts["apr_out_of_range"] == 3


def test_column_mapping():
    """Test that column mapping works for different column names."""
    # Create a test DataFrame with different column names
    data = {
        "GrossProfit": [-100, 0, 500],
        "InterestRate": [-1, 15, 30]
    }
    df = pd.DataFrame(data)
    
    # Initialize validator
    validator = FinancialValidator()
    
    # Modify rules for this test
    for rule in validator.rules:
        if rule.id == "negative_gross":
            rule.column_mapping = {"gross_profit": "GrossProfit"}
        elif rule.id == "apr_out_of_range":
            rule.column_mapping = {"apr": "InterestRate"}
    
    # Apply validation
    result_df, flag_counts = validator.validate(df)
    
    # Check that flag columns were created with correct mapping
    assert "flag_negative_gross" in result_df.columns
    assert "flag_apr_out_of_range" in result_df.columns
    
    # Check that negative values are flagged for gross
    assert result_df.loc[result_df["GrossProfit"] < 0, "flag_negative_gross"].all()
    assert not result_df.loc[result_df["GrossProfit"] >= 0, "flag_negative_gross"].any()
    
    # Check that out-of-range values are flagged for APR
    assert result_df.loc[result_df["InterestRate"] < 0, "flag_apr_out_of_range"].all()
    assert result_df.loc[result_df["InterestRate"] > 25, "flag_apr_out_of_range"].all()
    assert not result_df.loc[(result_df["InterestRate"] >= 0) & (result_df["InterestRate"] <= 25), "flag_apr_out_of_range"].any()


def test_missing_columns():
    """Test validator behavior when expected columns are missing."""
    # Create a test DataFrame without financial columns
    data = {
        "UnrelatedColumn1": [1, 2, 3],
        "UnrelatedColumn2": ["a", "b", "c"]
    }
    df = pd.DataFrame(data)
    
    # Initialize validator
    validator = FinancialValidator()
    
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
        "Gross_Profit": [-100, 200, 15000],  # Triggers negative, low, and high gross
        "APR": [30, 5, 10],  # Triggers APR out of range
        "LoanTerm": [110, 60, 6]  # Triggers loan term out of range
    }
    df = pd.DataFrame(data)
    
    # Initialize validator
    validator = FinancialValidator()
    
    # Apply validation
    result_df, flag_counts = validator.validate(df)
    
    # Check that flag columns were created
    assert "flag_negative_gross" in result_df.columns
    assert "flag_low_gross" in result_df.columns
    assert "flag_high_gross" in result_df.columns
    assert "flag_apr_out_of_range" in result_df.columns
    assert "flag_loan_term_out_of_range" in result_df.columns
    
    # Check that appropriate values are flagged
    assert result_df.loc[result_df["Gross_Profit"] < 0, "flag_negative_gross"].all()
    assert result_df.loc[result_df["Gross_Profit"] > 10000, "flag_high_gross"].all()
    assert result_df.loc[result_df["APR"] > 25, "flag_apr_out_of_range"].all()
    assert result_df.loc[(result_df["LoanTerm"] < 12) | (result_df["LoanTerm"] > 96), "flag_loan_term_out_of_range"].all()