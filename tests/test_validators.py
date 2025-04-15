"""
Test suite for the validation system.

This module contains tests for the validation profile system, insight validator,
and validator service components.
"""

import os
import sys
import pandas as pd
import unittest

# Add parent directory to path so we can import the validators package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.validators.validation_profile import (
    ValidationProfile,
    ValidationRule,
    get_available_profiles,
    apply_validation_profile,
    create_default_profile
)

from src.validators.insight_validator import (
    flag_all_issues,
    summarize_flags,
    flag_negative_gross,
    flag_missing_lead_source,
    flag_duplicate_vins,
    flag_missing_vins
)

from src.validators.validator_service import (
    ValidatorService
)


class TestValidationProfile(unittest.TestCase):
    """Test the validation profile system."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a test data directory
        self.test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_profiles")
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Create sample data
        self.sample_data = {
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
        self.df = pd.DataFrame(self.sample_data)
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up test profiles
        for filename in os.listdir(self.test_dir):
            if filename.endswith('.json'):
                os.remove(os.path.join(self.test_dir, filename))
    
    def test_create_default_profile(self):
        """Test creating a default profile."""
        profile = create_default_profile()
        
        self.assertEqual(profile.id, "default")
        self.assertEqual(profile.name, "Default Profile")
        self.assertTrue(profile.is_default)
        self.assertGreater(len(profile.rules), 0)
    
    def test_save_and_load_profile(self):
        """Test saving and loading a profile."""
        profile = create_default_profile()
        profile_path = profile.save(self.test_dir)
        
        loaded_profile = ValidationProfile.load(profile_path)
        
        self.assertEqual(profile.id, loaded_profile.id)
        self.assertEqual(profile.name, loaded_profile.name)
        self.assertEqual(len(profile.rules), len(loaded_profile.rules))
    
    def test_get_available_profiles(self):
        """Test getting available profiles."""
        # Create some profiles
        default_profile = create_default_profile()
        default_profile.save(self.test_dir)
        
        # Create a custom profile
        custom_profile = ValidationProfile(
            id="custom",
            name="Custom Profile",
            description="A custom profile",
            created_at="2023-01-01T00:00:00",
            updated_at="2023-01-01T00:00:00",
            rules=[],
            is_default=False
        )
        custom_profile.save(self.test_dir)
        
        # Get available profiles
        profiles = get_available_profiles(self.test_dir)
        
        self.assertEqual(len(profiles), 2)
        self.assertEqual(profiles[0].id, "default")  # Default should be first
        self.assertEqual(profiles[1].id, "custom")
    
    def test_apply_validation_profile(self):
        """Test applying a validation profile."""
        profile = create_default_profile()
        
        validated_df, flag_counts = apply_validation_profile(self.df, profile)
        
        # Check that flag columns were added
        self.assertIn("flag_negative_gross", validated_df.columns)
        self.assertIn("flag_missing_lead_source", validated_df.columns)
        self.assertIn("flag_duplicate_vin", validated_df.columns)
        self.assertIn("flag_missing_vin", validated_df.columns)
        
        # Check that flags were correctly applied
        self.assertEqual(flag_counts["negative_gross"], 1)  # One negative gross
        self.assertEqual(flag_counts["missing_lead_source"], 2)  # Two missing lead sources
        self.assertEqual(flag_counts["duplicate_vin"], 2)  # Two duplicate VINs
        self.assertEqual(flag_counts["missing_vin"], 2)  # Two missing/invalid VINs


class TestInsightValidator(unittest.TestCase):
    """Test the insight validator."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample data
        self.sample_data = {
            'VIN': ['1HGCM82633A123456', '1HGCM82633A123456', '5TFBW5F13AX123457', '789', '', 'WBAGH83576D123458'],
            'Gross_Profit': [3500.00, -1000.00, 5750.00, 2000.00, 3000.00, 7000.00],
            'Lead_Source': ['Website', None, '', 'Google', 'Autotrader', 'Walk-in']
        }
        self.df = pd.DataFrame(self.sample_data)
    
    def test_flag_negative_gross(self):
        """Test flagging negative gross values."""
        result_df = flag_negative_gross(self.df)
        
        self.assertIn("flag_negative_gross", result_df.columns)
        self.assertEqual(result_df["flag_negative_gross"].sum(), 1)
    
    def test_flag_missing_lead_source(self):
        """Test flagging missing lead sources."""
        result_df = flag_missing_lead_source(self.df)
        
        self.assertIn("flag_missing_lead_source", result_df.columns)
        self.assertEqual(result_df["flag_missing_lead_source"].sum(), 2)
    
    def test_flag_duplicate_vins(self):
        """Test flagging duplicate VINs."""
        result_df = flag_duplicate_vins(self.df)
        
        self.assertIn("flag_duplicate_vin", result_df.columns)
        self.assertEqual(result_df["flag_duplicate_vin"].sum(), 2)
    
    def test_flag_missing_vins(self):
        """Test flagging missing VINs."""
        result_df = flag_missing_vins(self.df)
        
        self.assertIn("flag_missing_vin", result_df.columns)
        self.assertEqual(result_df["flag_missing_vin"].sum(), 2)
    
    def test_flag_all_issues(self):
        """Test flagging all issues at once."""
        result_df = flag_all_issues(self.df)
        
        self.assertIn("flag_negative_gross", result_df.columns)
        self.assertIn("flag_missing_lead_source", result_df.columns)
        self.assertIn("flag_duplicate_vin", result_df.columns)
        self.assertIn("flag_missing_vin", result_df.columns)
    
    def test_summarize_flags(self):
        """Test summarizing flags."""
        flagged_df = flag_all_issues(self.df)
        summary = summarize_flags(flagged_df)
        
        self.assertIn("total_records", summary)
        self.assertIn("total_issues", summary)
        self.assertIn("percentage_clean", summary)
        self.assertIn("negative_gross_count", summary)
        self.assertIn("missing_lead_source_count", summary)
        self.assertIn("duplicate_vins_count", summary)
        self.assertIn("missing_vins_count", summary)


class TestValidatorService(unittest.TestCase):
    """Test the validator service."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a test data directory
        self.test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_profiles")
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Create sample data
        self.sample_data = {
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
        self.df = pd.DataFrame(self.sample_data)
        
        # Create a validator service
        self.validator = ValidatorService(profiles_dir=self.test_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up test profiles
        for filename in os.listdir(self.test_dir):
            if filename.endswith('.json'):
                os.remove(os.path.join(self.test_dir, filename))
    
    def test_validator_initialization(self):
        """Test initializing the validator service."""
        self.assertIsNotNone(self.validator.active_profile)
        self.assertGreater(len(self.validator.profiles), 0)
    
    def test_validate_dataframe(self):
        """Test validating a DataFrame with the service."""
        validated_df, summary = self.validator.validate_dataframe(self.df)
        
        self.assertIn("flag_negative_gross", validated_df.columns)
        self.assertIn("flag_missing_lead_source", validated_df.columns)
        self.assertIn("flag_duplicate_vin", validated_df.columns)
        self.assertIn("flag_missing_vin", validated_df.columns)
        
        self.assertIn("total_records", summary)
        self.assertIn("total_issues", summary)
        self.assertIn("percentage_clean", summary)
    
    def test_auto_clean_dataframe(self):
        """Test auto-cleaning a DataFrame."""
        validated_df, _ = self.validator.validate_dataframe(self.df)
        
        cleaned_df = self.validator.auto_clean_dataframe(validated_df)
        
        # Check that flag columns are removed
        self.assertNotIn("flag_negative_gross", cleaned_df.columns)
        self.assertNotIn("flag_missing_lead_source", cleaned_df.columns)
        self.assertNotIn("flag_duplicate_vin", cleaned_df.columns)
        self.assertNotIn("flag_missing_vin", cleaned_df.columns)
        
        # Check that negative gross values are fixed
        self.assertTrue((cleaned_df["Gross_Profit"] >= 0).all())
        
        # Check that missing lead sources are fixed
        lead_source_col = cleaned_df["Lead_Source"]
        self.assertFalse(lead_source_col.isna().any() or (lead_source_col == "").any())
        
        # Check that duplicate VINs are reduced
        self.assertEqual(cleaned_df["VIN"].value_counts().max(), 1)


if __name__ == "__main__":
    unittest.main()