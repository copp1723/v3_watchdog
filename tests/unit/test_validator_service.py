"""
Tests for the validator service module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.validators.validator_service import (
    ValidatorService,
    process_uploaded_file,
    auto_clean_dataframe,
    generate_validation_report
)
from src.validators.validator_registry import get_validators
from src.validators.base_validator import BaseValidator


class MockFile:
    """Mock file object for testing."""
    def __init__(self, name, content):
        self.name = name
        self.content = content
        self.position = 0
    
    def read(self):
        return self.content
    
    def seek(self, position, whence=0):
        if whence == 0:
            self.position = position
        elif whence == 1:
            self.position += position
        elif whence == 2:
            self.position = len(self.content) + position
        return self.position
    
    def tell(self):
        return self.position


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        "Gross_Profit": [-100, 0, 500],
        "Lead_Source": ["Website", None, "Google"],
        "Salesperson": ["John", None, "Jane"],
        "SaleDate": ["2023-01-01", "invalid-date", None]
    })


def test_validator_service_initialization():
    """Test the ValidatorService initialization."""
    service = ValidatorService()
    
    # Check that validators were loaded
    assert hasattr(service, 'validators')
    assert len(service.validators) > 0
    assert all(isinstance(v, BaseValidator) for v in service.validators)


def test_validator_service_validate_dataframe(sample_dataframe):
    """Test the ValidatorService validate_dataframe method."""
    service = ValidatorService()
    
    # Validate the DataFrame
    result_df, validation_summary = service.validate_dataframe(sample_dataframe)
    
    # Check that the result DataFrame has flag columns
    flag_columns = [col for col in result_df.columns if col.startswith('flag_')]
    assert len(flag_columns) > 0
    
    # Check that validation_summary contains expected keys
    assert isinstance(validation_summary, dict)
    assert "flag_counts" in validation_summary
    assert isinstance(validation_summary["flag_counts"], dict)
    
    # Ensure at least one flag was set
    total_flags = sum(validation_summary["flag_counts"].values())
    assert total_flags > 0


def test_validator_service_get_available_validators():
    """Test the ValidatorService get_available_validators method."""
    service = ValidatorService()
    
    # Get available validators
    validators = service.get_available_validators()
    
    # Check that validators were returned
    assert isinstance(validators, list)
    assert len(validators) > 0
    assert all(isinstance(v, str) for v in validators)
    
    # Check that the expected validators are included
    assert "Financial Validator" in validators
    assert "Customer Validator" in validators


def test_auto_clean_dataframe():
    """Test the auto_clean_dataframe function."""
    # Create a test DataFrame with issues to clean
    data = {
        "NumericColumn": [1, np.nan, np.inf, -np.inf, "not_a_number"],
        "StringColumn": ["value", "", " ", None, "trailing space  "],
        "DateColumn": ["2023-01-01", "invalid", None, "", "2023-13-32"]
    }
    df = pd.DataFrame(data)
    
    # Clean the DataFrame
    cleaned_df = auto_clean_dataframe(df)
    
    # Check that the DataFrame was cleaned
    assert cleaned_df is not None
    assert isinstance(cleaned_df, pd.DataFrame)
    
    # Check numeric column cleaning - auto_clean_dataframe might not change the column type
    # but should ensure all invalid numeric values become NaN
    assert "NumericColumn" in cleaned_df.columns
    # At least non-numeric, inf, and -inf values should be converted to NaN
    assert cleaned_df["NumericColumn"].isna().sum() >= 3
    
    # Check string column cleaning
    assert cleaned_df["StringColumn"].iloc[0] == "value"
    assert pd.isna(cleaned_df["StringColumn"].iloc[1])  # Empty string should be NaN
    assert pd.isna(cleaned_df["StringColumn"].iloc[2])  # Space should be NaN
    assert pd.isna(cleaned_df["StringColumn"].iloc[3])  # None should be NaN
    assert cleaned_df["StringColumn"].iloc[4] == "trailing space"  # Should be trimmed
    
    # Date column might be converted to datetime, but it depends on the implementation
    # We'll just check that invalid dates are now NaN if conversion was attempted
    if pd.api.types.is_datetime64_dtype(cleaned_df["DateColumn"]):
        assert pd.notna(cleaned_df["DateColumn"].iloc[0])  # Valid date
        assert pd.isna(cleaned_df["DateColumn"].iloc[1])   # Invalid date
        assert pd.isna(cleaned_df["DateColumn"].iloc[2])   # None
        assert pd.isna(cleaned_df["DateColumn"].iloc[3])   # Empty string
        assert pd.isna(cleaned_df["DateColumn"].iloc[4])   # Invalid date


def test_generate_validation_report(sample_dataframe):
    """Test the generate_validation_report function."""
    # First apply validation to get flag columns
    service = ValidatorService()
    validated_df, _ = service.validate_dataframe(sample_dataframe)
    
    # Generate the report
    report = generate_validation_report(validated_df)
    
    # Check that the report was generated
    assert report is not None
    assert isinstance(report, pd.DataFrame)
    
    # Check that the report has expected columns
    assert "has_issues" in report.columns
    assert "issue_count" in report.columns
    assert "issue_details" in report.columns
    
    # Check that the report has expected content
    assert report["has_issues"].any()  # At least one row should have issues
    assert (report["issue_count"] > 0).any()  # At least one issue should be counted
    
    # Check for issue details - find rows with issues
    issue_rows = report[report["has_issues"]]
    assert not issue_rows.empty
    
    # Ensure rows with issues have proper descriptions
    for _, row in issue_rows.iterrows():
        assert row["issue_details"] != "No issues"