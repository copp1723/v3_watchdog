"""
Tests for the validator service module.
"""

import pytest
import pandas as pd
import io
from ..src.validators.validator_service import (
    process_uploaded_file,
    ValidatorService,
    FileValidationError,
    detect_file_type,
    load_dataframe
)
from ..src.validators.validation_profile import ValidationProfile

@pytest.fixture
def sample_csv_file():
    """Create a sample CSV file for testing."""
    data = {
        'VIN': ['1HGCM82633A123456', '5TFBW5F13AX123457'],
        'Make': ['Honda', 'Toyota'],
        'Model': ['Accord', 'Tundra'],
        'Year': [2019, 2020],
        'Sale_Date': ['2023-01-15', '2023-02-20'],
        'Sale_Price': [28500.00, 45750.00],
        'Cost': [25000.00, 40000.00],
        'Gross_Profit': [3500.00, 5750.00],
        'Lead_Source': ['Website', 'Google'],
        'Salesperson': ['John Smith', 'Jane Doe']
    }
    df = pd.DataFrame(data)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    csv_buffer.name = 'test.csv'  # Add name attribute to simulate uploaded file
    return csv_buffer

@pytest.fixture
def sample_validation_profile():
    """Create a sample validation profile for testing."""
    return ValidationProfile(
        id="test_profile",
        name="Test Profile",
        description="Test validation profile",
        created_at="2024-01-01T00:00:00",
        updated_at="2024-01-01T00:00:00",
        rules=[],
        is_default=True
    )

def test_detect_file_type_csv(sample_csv_file):
    """Test file type detection for CSV files."""
    assert detect_file_type(sample_csv_file) == 'csv'

def test_detect_file_type_invalid():
    """Test file type detection for invalid files."""
    class MockFile:
        name = 'test.txt'
    
    with pytest.raises(FileValidationError):
        detect_file_type(MockFile())

def test_load_dataframe_csv(sample_csv_file):
    """Test loading a CSV file into a DataFrame."""
    df = load_dataframe(sample_csv_file, 'csv')
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert 'VIN' in df.columns

def test_process_uploaded_file_csv_valid(sample_csv_file, sample_validation_profile):
    """Test processing a valid CSV file."""
    cleaned_df, summary, report_df = process_uploaded_file(
        sample_csv_file,
        selected_profile=sample_validation_profile
    )
    
    assert cleaned_df is not None
    assert isinstance(cleaned_df, pd.DataFrame)
    assert summary["status"] == "success"
    assert summary["total_rows"] == len(cleaned_df)
    assert isinstance(report_df, pd.DataFrame)

def test_process_uploaded_file_invalid():
    """Test processing an invalid file."""
    class MockInvalidFile:
        name = 'test.txt'
        
    cleaned_df, summary, report_df = process_uploaded_file(MockInvalidFile())
    
    assert cleaned_df is None
    assert summary["status"] == "error"
    assert "Unsupported file type" in summary["message"]
    assert report_df is None

def test_validator_service_initialization():
    """Test ValidatorService initialization."""
    service = ValidatorService()
    assert service.profiles_dir == "profiles"
    assert hasattr(service, 'active_profile')

def test_validator_service_process_file(sample_csv_file):
    """Test ValidatorService process_file method."""
    service = ValidatorService()
    cleaned_df, summary, report_df = service.process_file(sample_csv_file)
    
    assert cleaned_df is not None
    assert isinstance(cleaned_df, pd.DataFrame)
    assert summary["status"] == "success"
    assert isinstance(report_df, pd.DataFrame) 