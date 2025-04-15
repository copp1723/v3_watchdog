"""
Pytest configuration file for Watchdog AI testing.

This file contains shared fixtures and configuration for all test modules.
"""

import os
import sys
import pytest
import pandas as pd
from typing import Dict, List, Tuple, Any

# Add the src directory to the path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Paths for test assets
ASSETS_DIR = os.path.join(os.path.dirname(__file__), 'assets')

# Ensure assets directory exists
os.makedirs(ASSETS_DIR, exist_ok=True)


@pytest.fixture
def assets_dir():
    """Fixture that provides the path to the assets directory."""
    return ASSETS_DIR


@pytest.fixture
def clean_csv_data():
    """Fixture that provides a clean pandas DataFrame for testing."""
    # Sample automotive dealer data with clean formatting
    data = {
        'VIN': ['1HGCM82633A123456', '5TFBW5F13AX123457', 'WBAGH83576D123458'],
        'Make': ['Honda', 'Toyota', 'BMW'],
        'Model': ['Accord', 'Tundra', '7 Series'],
        'Year': [2019, 2020, 2018],
        'Sale_Date': ['2023-01-15', '2023-02-20', '2023-03-05'],
        'Sale_Price': [28500.00, 45750.00, 62000.00],
        'Cost': [25000.00, 40000.00, 55000.00],
        'Gross_Profit': [3500.00, 5750.00, 7000.00],
        'Lead_Source': ['Website', 'Walk-in', 'CarGurus'],
        'Salesperson': ['John Smith', 'Jane Doe', 'Bob Johnson']
    }
    
    df = pd.DataFrame(data)
    return df


@pytest.fixture
def messy_csv_data():
    """Fixture that provides a messy pandas DataFrame for testing."""
    # Sample messy automotive dealer data with inconsistent formatting and issues
    data = {
        'VIN #': ['1HGCM82633A123456', '5TFBW5F13AX123457', 'INVALID VIN'],
        'Vehicle Make': ['Honda', 'Toyota', 'BMW'],
        ' Model ': ['Accord', 'Tundra', '7 Series'],
        'Year  ': [2019, 'Twenty-Twenty', 2018],
        'Date of Sale': ['01/15/2023', '02/20/2023', None],
        'Price $': ['$28,500.00', '$45,750.00', '$62,000.00'],
        'Dealer Cost': ['$25,000.00', '$40,000.00', None],
        'Gross': ['$3,500.00', '$5,750.00', '$7,000.00'],
        'Source of Lead': ['Website', 'Walk-in', None],
        'Sales Person': ['John Smith', 'Jane Doe', 'Bob Johnson']
    }
    
    df = pd.DataFrame(data)
    return df


@pytest.fixture
def create_sample_csv(assets_dir):
    """Fixture that creates a sample CSV file and returns its path."""
    def _create_csv(filename, data):
        filepath = os.path.join(assets_dir, filename)
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        return filepath
    
    return _create_csv


@pytest.fixture
def sample_clean_csv_path(assets_dir, clean_csv_data):
    """Fixture that provides a path to a clean CSV file."""
    filepath = os.path.join(assets_dir, 'clean_file.csv')
    clean_csv_data.to_csv(filepath, index=False)
    return filepath


@pytest.fixture
def sample_messy_csv_path(assets_dir, messy_csv_data):
    """Fixture that provides a path to a messy CSV file."""
    filepath = os.path.join(assets_dir, 'messy_file.csv')
    messy_csv_data.to_csv(filepath, index=False)
    return filepath


@pytest.fixture
def sample_excel_path(assets_dir, clean_csv_data, messy_csv_data):
    """Fixture that provides a path to a sample Excel file with multiple sheets."""
    filepath = os.path.join(assets_dir, 'sample_excel.xlsx')
    
    with pd.ExcelWriter(filepath) as writer:
        clean_csv_data.to_excel(writer, sheet_name='Clean Data', index=False)
        messy_csv_data.to_excel(writer, sheet_name='Messy Data', index=False)
        
        # Create an empty sheet
        pd.DataFrame().to_excel(writer, sheet_name='Empty Sheet', index=False)
        
        # Create a sheet with header issues
        header_issue_df = clean_csv_data.copy()
        header_issue_df.columns = [f"Column{i}" for i in range(len(header_issue_df.columns))]
        header_issue_df.to_excel(writer, sheet_name='Header Issues', index=False)
    
    return filepath


@pytest.fixture
def empty_dataframe():
    """Fixture that provides an empty DataFrame."""
    return pd.DataFrame()


@pytest.fixture
def sample_metadata():
    """Fixture that provides a sample metadata dictionary."""
    return {
        "source_file_name": "test_file.csv",
        "upload_timestamp": "2023-04-15T10:30:45",
        "rows": 100,
        "columns": 10,
        "column_names": ["VIN", "Make", "Model", "Year", "Sale_Date", 
                         "Sale_Price", "Cost", "Gross_Profit", "Lead_Source", "Salesperson"],
        "file_size_bytes": 10240,
        "missing_values": {"VIN": 0, "Make": 2, "Model": 1},
        "numeric_stats": {
            "Year": {"min": 2018, "max": 2023, "mean": 2020.5},
            "Sale_Price": {"min": 15000, "max": 85000, "mean": 42500}
        }
    }
