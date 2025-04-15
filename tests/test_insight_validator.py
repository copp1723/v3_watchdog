"""
Tests for the insight_validator module.

This module tests the functionality of the insight_validator module which flags
dealership-specific issues in DataFrames and provides summaries of detected problems.
"""

import pytest
import pandas as pd
import numpy as np
import os
from typing import Dict, Any, List, Tuple

# Import the module to test
from src.validators.insight_validator import (
    flag_negative_gross,
    flag_missing_lead_source,
    flag_duplicate_vins,
    flag_missing_vins,
    flag_all_issues,
    summarize_flags,
    generate_flag_summary
)


class TestFlagNegativeGross:
    """Tests for the flag_negative_gross function."""
    
    def test_flag_negative_gross_basic(self):
        """Test basic functionality of flag_negative_gross."""
        # Arrange
        data = {
            'Gross_Profit': [1000, -500, 0, 300, -20]
        }
        df = pd.DataFrame(data)
        
        # Act
        result_df = flag_negative_gross(df)
        
        # Assert
        assert 'flag_negative_gross' in result_df.columns
        assert result_df['flag_negative_gross'].tolist() == [False, True, False, False, True]
    
    def test_flag_negative_gross_with_string_values(self):
        """Test flag_negative_gross with string values that should be converted to numeric."""
        # Arrange
        data = {
            'Gross_Profit': ['1000', '-500', '0', '300', '-20']
        }
        df = pd.DataFrame(data)
        
        # Act
        result_df = flag_negative_gross(df)
        
        # Assert
        assert 'flag_negative_gross' in result_df.columns
        assert result_df['flag_negative_gross'].tolist() == [False, True, False, False, True]
    
    def test_flag_negative_gross_with_alternative_column_name(self):
        """Test flag_negative_gross with an alternative column name."""
        # Arrange
        data = {
            'Gross': [1000, -500, 0, 300, -20]
        }
        df = pd.DataFrame(data)
        
        # Act
        result_df = flag_negative_gross(df, gross_col='Gross')
        
        # Assert
        assert 'flag_negative_gross' in result_df.columns
        assert result_df['flag_negative_gross'].tolist() == [False, True, False, False, True]
    
    def test_flag_negative_gross_with_missing_column(self):
        """Test flag_negative_gross when the gross column is missing."""
        # Arrange
        data = {
            'VIN': ['123', '456', '789'],
            'Sale_Price': [1000, 2000, 3000]
        }
        df = pd.DataFrame(data)
        
        # Act
        result_df = flag_negative_gross(df)
        
        # Assert
        assert 'flag_negative_gross' in result_df.columns
        assert not result_df['flag_negative_gross'].any()  # All should be False


class TestFlagMissingLeadSource:
    """Tests for the flag_missing_lead_source function."""
    
    def test_flag_missing_lead_source_basic(self):
        """Test basic functionality of flag_missing_lead_source."""
        # Arrange
        data = {
            'Lead_Source': ['Website', None, '', 'Walk-in', ' ']
        }
        df = pd.DataFrame(data)
        
        # Act
        result_df = flag_missing_lead_source(df)
        
        # Assert
        assert 'flag_missing_lead_source' in result_df.columns
        assert result_df['flag_missing_lead_source'].tolist() == [False, True, True, False, True]
    
    def test_flag_missing_lead_source_with_alternative_column_name(self):
        """Test flag_missing_lead_source with an alternative column name."""
        # Arrange
        data = {
            'LeadSource': ['Website', None, '', 'Walk-in', ' ']
        }
        df = pd.DataFrame(data)
        
        # Act
        result_df = flag_missing_lead_source(df, lead_source_col='LeadSource')
        
        # Assert
        assert 'flag_missing_lead_source' in result_df.columns
        assert result_df['flag_missing_lead_source'].tolist() == [False, True, True, False, True]
    
    def test_flag_missing_lead_source_with_missing_column(self):
        """Test flag_missing_lead_source when the lead source column is missing."""
        # Arrange
        data = {
            'VIN': ['123', '456', '789'],
            'Sale_Price': [1000, 2000, 3000]
        }
        df = pd.DataFrame(data)
        
        # Act
        result_df = flag_missing_lead_source(df)
        
        # Assert
        assert 'flag_missing_lead_source' in result_df.columns
        assert not result_df['flag_missing_lead_source'].any()  # All should be False


class TestFlagDuplicateVins:
    """Tests for the flag_duplicate_vins function."""
    
    def test_flag_duplicate_vins_basic(self):
        """Test basic functionality of flag_duplicate_vins."""
        # Arrange
        data = {
            'VIN': ['123', '123', '456', '789', '789', '012']
        }
        df = pd.DataFrame(data)
        
        # Act
        result_df = flag_duplicate_vins(df)
        
        # Assert
        assert 'flag_duplicate_vin' in result_df.columns
        assert result_df['flag_duplicate_vin'].tolist() == [True, True, False, True, True, False]
    
    def test_flag_duplicate_vins_with_alternative_column_name(self):
        """Test flag_duplicate_vins with an alternative column name."""
        # Arrange
        data = {
            'Vehicle_VIN': ['123', '123', '456', '789', '789', '012']
        }
        df = pd.DataFrame(data)
        
        # Act
        result_df = flag_duplicate_vins(df, vin_col='Vehicle_VIN')
        
        # Assert
        assert 'flag_duplicate_vin' in result_df.columns
        assert result_df['flag_duplicate_vin'].tolist() == [True, True, False, True, True, False]
    
    def test_flag_duplicate_vins_with_missing_column(self):
        """Test flag_duplicate_vins when the VIN column is missing."""
        # Arrange
        data = {
            'Lead_Source': ['Website', 'Walk-in', 'Referral'],
            'Sale_Price': [1000, 2000, 3000]
        }
        df = pd.DataFrame(data)
        
        # Act
        result_df = flag_duplicate_vins(df)
        
        # Assert
        assert 'flag_duplicate_vin' in result_df.columns
        assert not result_df['flag_duplicate_vin'].any()  # All should be False


class TestFlagMissingVins:
    """Tests for the flag_missing_vins function."""
    
    def test_flag_missing_vins_basic(self):
        """Test basic functionality of flag_missing_vins."""
        # Arrange
        data = {
            'VIN': ['1HGCM82633A123456', '', None, '789', 'INVALID', '5TFBW5F13AX123457']
        }
        df = pd.DataFrame(data)
        
        # Act
        result_df = flag_missing_vins(df)
        
        # Assert
        assert 'flag_missing_vin' in result_df.columns
        assert result_df['flag_missing_vin'].tolist() == [False, True, True, True, True, False]
    
    def test_flag_missing_vins_with_alternative_column_name(self):
        """Test flag_missing_vins with an alternative column name."""
        # Arrange
        data = {
            'Vehicle_VIN': ['1HGCM82633A123456', '', None, '789', 'INVALID', '5TFBW5F13AX123457']
        }
        df = pd.DataFrame(data)
        
        # Act
        result_df = flag_missing_vins(df, vin_col='Vehicle_VIN')
        
        # Assert
        assert 'flag_missing_vin' in result_df.columns
        assert result_df['flag_missing_vin'].tolist() == [False, True, True, True, True, False]
    
    def test_flag_missing_vins_with_missing_column(self):
        """Test flag_missing_vins when the VIN column is missing."""
        # Arrange
        data = {
            'Lead_Source': ['Website', 'Walk-in', 'Referral'],
            'Sale_Price': [1000, 2000, 3000]
        }
        df = pd.DataFrame(data)
        
        # Act
        result_df = flag_missing_vins(df)
        
        # Assert
        assert 'flag_missing_vin' in result_df.columns
        assert not result_df['flag_missing_vin'].any()  # All should be False


class TestFlagAllIssues:
    """Tests for the flag_all_issues function."""
    
    def test_flag_all_issues_basic(self):
        """Test basic functionality of flag_all_issues."""
        # Arrange
        data = {
            'VIN': ['1HGCM82633A123456', '1HGCM82633A123456', '5TFBW5F13AX123457', '789', '', 'WBAGH83576D123458'],
            'Gross_Profit': [1000, -500, 300, -20, 750, 1200],
            'Lead_Source': ['Facebook', None, '', 'Google', 'Autotrader', 'Walk-in']
        }
        df = pd.DataFrame(data)
        
        # Act
        result_df = flag_all_issues(df)
        
        # Assert
        assert 'flag_negative_gross' in result_df.columns
        assert 'flag_missing_lead_source' in result_df.columns
        assert 'flag_duplicate_vin' in result_df.columns
        assert 'flag_missing_vin' in result_df.columns
        
        # Check specific flags
        assert result_df['flag_negative_gross'].sum() == 2  # Two negative gross entries
        assert result_df['flag_missing_lead_source'].sum() == 2  # Two missing lead sources
        assert result_df['flag_duplicate_vin'].sum() == 2  # One duplicate VIN (2 entries)
        assert result_df['flag_missing_vin'].sum() == 2  # Two invalid VINs
    
    def test_flag_all_issues_with_custom_column_names(self):
        """Test flag_all_issues with custom column names."""
        # Arrange
        data = {
            'Vehicle_VIN': ['1HGCM82633A123456', '1HGCM82633A123456', '5TFBW5F13AX123457', '789', '', 'WBAGH83576D123458'],
            'Gross': [1000, -500, 300, -20, 750, 1200],
            'Source': ['Facebook', None, '', 'Google', 'Autotrader', 'Walk-in']
        }
        df = pd.DataFrame(data)
        
        # Act
        result_df = flag_all_issues(
            df, 
            gross_col='Gross', 
            lead_source_col='Source', 
            vin_col='Vehicle_VIN'
        )
        
        # Assert
        assert 'flag_negative_gross' in result_df.columns
        assert 'flag_missing_lead_source' in result_df.columns
        assert 'flag_duplicate_vin' in result_df.columns
        assert 'flag_missing_vin' in result_df.columns
        
        # Check specific flags
        assert result_df['flag_negative_gross'].sum() == 2  # Two negative gross entries
        assert result_df['flag_missing_lead_source'].sum() == 2  # Two missing lead sources
        assert result_df['flag_duplicate_vin'].sum() == 2  # One duplicate VIN (2 entries)
        assert result_df['flag_missing_vin'].sum() == 2  # Two invalid VINs


class TestSummarizeFlags:
    """Tests for the summarize_flags function."""
    
    def test_summarize_flags_basic(self):
        """Test basic functionality of summarize_flags."""
        # Arrange
        data = {
            'VIN': ['1HGCM82633A123456', '1HGCM82633A123456', '5TFBW5F13AX123457', '789', '', 'WBAGH83576D123458'],
            'Gross_Profit': [1000, -500, 300, -20, 750, 1200],
            'Lead_Source': ['Facebook', None, '', 'Google', 'Autotrader', 'Walk-in']
        }
        df = pd.DataFrame(data)
        flagged_df = flag_all_issues(df)
        
        # Act
        summary = summarize_flags(flagged_df)
        
        # Assert
        assert 'total_records' in summary
        assert 'total_issues' in summary
        assert 'issue_summary' in summary
        assert 'percentage_clean' in summary
        
        assert summary['total_records'] == 6
        assert summary['total_issues'] > 0
        assert summary['percentage_clean'] < 100.0
        
        assert 'negative_gross_count' in summary
        assert 'missing_lead_source_count' in summary
        assert 'duplicate_vins_count' in summary
        assert 'missing_vins_count' in summary
        
        assert summary['negative_gross_count'] == 2
        assert summary['missing_lead_source_count'] == 2
        assert summary['duplicate_vins_count'] == 2
        assert summary['missing_vins_count'] == 2
    
    def test_summarize_flags_with_clean_data(self):
        """Test summarize_flags with clean data that has no issues."""
        # Arrange
        data = {
            'VIN': ['1HGCM82633A123456', '5TFBW5F13AX123457', 'WBAGH83576D123458'],
            'Gross_Profit': [1000, 300, 1200],
            'Lead_Source': ['Facebook', 'Google', 'Walk-in']
        }
        df = pd.DataFrame(data)
        flagged_df = flag_all_issues(df)
        
        # Act
        summary = summarize_flags(flagged_df)
        
        # Assert
        assert summary['total_records'] == 3
        assert summary['total_issues'] == 0
        assert summary['percentage_clean'] == 100.0
        
        assert summary['negative_gross_count'] == 0
        assert summary['missing_lead_source_count'] == 0
        assert summary['duplicate_vins_count'] == 0
        assert summary['missing_vins_count'] == 0
    
    def test_summarize_flags_with_no_flag_columns(self):
        """Test summarize_flags with a DataFrame that has no flag columns."""
        # Arrange
        data = {
            'VIN': ['123', '456', '789'],
            'Gross_Profit': [1000, 2000, 3000],
            'Lead_Source': ['Website', 'Walk-in', 'Referral']
        }
        df = pd.DataFrame(data)
        
        # Act
        summary = summarize_flags(df)
        
        # Assert
        assert summary['total_records'] == 3
        assert summary['total_issues'] == 0
        assert summary['percentage_clean'] == 100.0
        assert summary['issue_summary'] == {}


class TestGenerateFlagSummary:
    """Tests for the generate_flag_summary function."""
    
    def test_generate_flag_summary_basic(self):
        """Test basic functionality of generate_flag_summary."""
        # Arrange
        data = {
            'VIN': ['1HGCM82633A123456', '1HGCM82633A123456', '5TFBW5F13AX123457', '789', '', 'WBAGH83576D123458'],
            'Gross_Profit': [1000, -500, 300, -20, 750, 1200],
            'Lead_Source': ['Facebook', None, '', 'Google', 'Autotrader', 'Walk-in']
        }
        df = pd.DataFrame(data)
        flagged_df = flag_all_issues(df)
        
        # Act
        markdown_summary = generate_flag_summary(flagged_df)
        
        # Assert
        assert isinstance(markdown_summary, str)
        assert len(markdown_summary) > 0
        assert "# Data Quality Report" in markdown_summary
        assert "## Issue Summary" in markdown_summary
        assert "## Recommendations" in markdown_summary
        
        # Check for issue types in the summary
        assert "Negative Gross" in markdown_summary
        assert "Missing Lead Source" in markdown_summary
        assert "Duplicate Vin" in markdown_summary
        assert "Missing Vin" in markdown_summary
    
    def test_generate_flag_summary_with_clean_data(self):
        """Test generate_flag_summary with clean data that has no issues."""
        # Arrange
        data = {
            'VIN': ['1HGCM82633A123456', '5TFBW5F13AX123457', 'WBAGH83576D123458'],
            'Gross_Profit': [1000, 300, 1200],
            'Lead_Source': ['Facebook', 'Google', 'Walk-in']
        }
        df = pd.DataFrame(data)
        flagged_df = flag_all_issues(df)
        
        # Act
        markdown_summary = generate_flag_summary(flagged_df)
        
        # Assert
        assert isinstance(markdown_summary, str)
        assert len(markdown_summary) > 0
        assert "# Data Quality Report" in markdown_summary
        assert "## Issue Summary" in markdown_summary
        assert "No issues detected in the dataset." in markdown_summary


class TestIntegrationScenarios:
    """Integration tests for insight_validator module."""
    
    @pytest.fixture
    def clean_file_df(self):
        """Fixture that provides a clean DataFrame for testing."""
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
        return pd.DataFrame(data)
    
    @pytest.fixture
    def dirty_file_df(self):
        """Fixture that provides a dirty DataFrame for testing."""
        data = {
            'VIN': ['1HGCM82633A123456', '1HGCM82633A123456', '5TFBW5F13AX123457', '789', '', 'WBAGH83576D123458'],
            'Make': ['Honda', 'Honda', 'Toyota', 'Chevrolet', 'Ford', 'BMW'],
            'Model': ['Accord', 'Accord', 'Tundra', 'Malibu', 'F-150', '7 Series'],
            'Year': [2019, 2019, 2020, 2018, 2021, 2018],
            'Sale_Date': ['2023-01-15', '2023-02-10', '2023-02-20', '2023-03-01', '2023-03-15', '2023-03-05'],
            'Sale_Price': [28500.00, 27000.00, 45750.00, 22000.00, 35000.00, 62000.00],
            'Cost': [25000.00, 28000.00, 40000.00, 20000.00, 32000.00, 55000.00],
            'Gross_Profit': [3500.00, -1000.00, 5750.00, 2000.00, 3000.00, 7000.00],
            'Lead_Source': ['Website', None, '', 'Google', 'Autotrader', 'Walk-in'],
            'Salesperson': ['John Smith', 'Jane Doe', 'Jane Doe', 'Bob Johnson', 'John Smith', 'Bob Johnson']
        }
        return pd.DataFrame(data)
    
    def test_clean_file_scenario(self, clean_file_df):
        """Test the insight_validator with a clean file."""
        # Act
        flagged_df = flag_all_issues(clean_file_df)
        summary = summarize_flags(flagged_df)
        
        # Assert
        assert summary['total_records'] == 3
        assert summary['total_issues'] == 0
        assert summary['percentage_clean'] == 100.0
        
        assert summary['negative_gross_count'] == 0
        assert summary['missing_lead_source_count'] == 0
        assert summary['duplicate_vins_count'] == 0
        assert summary['missing_vins_count'] == 0
    
    def test_dirty_file_scenario(self, dirty_file_df):
        """Test the insight_validator with a dirty file that has multiple issues."""
        # Act
        flagged_df = flag_all_issues(dirty_file_df)
        summary = summarize_flags(flagged_df)
        
        # Assert
        assert summary['total_records'] == 6
        assert summary['total_issues'] > 0
        assert summary['percentage_clean'] < 100.0
        
        assert summary['negative_gross_count'] == 1
        assert summary['missing_lead_source_count'] == 2
        assert summary['duplicate_vins_count'] == 2
        assert summary['missing_vins_count'] == 2
    
    def test_missing_vins_scenario(self, clean_file_df):
        """Test the insight_validator with a file that has missing VINs."""
        # Arrange
        df = clean_file_df.copy()
        df.at[1, 'VIN'] = ''  # Make one VIN empty
        
        # Act
        flagged_df = flag_all_issues(df)
        summary = summarize_flags(flagged_df)
        
        # Assert
        assert summary['total_records'] == 3
        assert summary['total_issues'] == 1
        assert summary['percentage_clean'] < 100.0
        
        assert summary['missing_vins_count'] == 1
    
    def test_overlapping_vins_scenario(self, clean_file_df):
        """Test the insight_validator with a file that has overlapping VINs."""
        # Arrange
        df = clean_file_df.copy()
        df.at[1, 'VIN'] = df.at[0, 'VIN']  # Make second VIN same as first
        
        # Act
        flagged_df = flag_all_issues(df)
        summary = summarize_flags(flagged_df)
        
        # Assert
        assert summary['total_records'] == 3
        assert summary['total_issues'] == 2
        assert summary['percentage_clean'] < 100.0
        
        assert summary['duplicate_vins_count'] == 2
