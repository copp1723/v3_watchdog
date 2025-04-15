"""
Validator Service for Watchdog AI.

This module provides a unified service for validating data using validation profiles.
It integrates the validation profile system with the insight validator and provides
a clean API for processing uploaded files and validating data.
"""

import os
import pandas as pd
import streamlit as st
from typing import Dict, Any, List, Optional, Tuple, Callable, Union
from datetime import datetime

from .validation_profile import (
    ValidationProfile, 
    get_available_profiles,
    apply_validation_profile,
    render_profile_editor,
    create_default_profile
)

from .insight_validator import (
    summarize_flags,
    generate_flag_summary
)


class ValidatorService:
    """
    Service for validating data using validation profiles.
    
    This class integrates the validation profile system with the insight validator
    and provides a clean API for validating data.
    """
    
    def __init__(self, profiles_dir: str = "profiles"):
        """
        Initialize the validator service.
        
        Args:
            profiles_dir: Directory containing validation profiles
        """
        self.profiles_dir = profiles_dir
        self._ensure_profiles_dir()
        self.profiles = get_available_profiles(profiles_dir)
        self.active_profile = next((p for p in self.profiles if p.is_default), None)
        if not self.active_profile and self.profiles:
            self.active_profile = self.profiles[0]
        elif not self.active_profile:
            self.active_profile = create_default_profile()
            self.active_profile.save(profiles_dir)
            self.profiles = [self.active_profile]
    
    def _ensure_profiles_dir(self):
        """Ensure the profiles directory exists."""
        os.makedirs(self.profiles_dir, exist_ok=True)
    
    def validate_dataframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Validate a DataFrame using the active profile.
        
        Args:
            df: The DataFrame to validate
            
        Returns:
            Tuple of (DataFrame with flag columns, validation summary)
        """
        if not self.active_profile:
            return df, {"error": "No active validation profile"}
        
        validated_df, flag_counts = apply_validation_profile(df, self.active_profile)
        validation_summary = summarize_flags(validated_df)
        
        return validated_df, validation_summary
    
    def render_profile_selection(self, on_profile_change: Optional[Callable[[ValidationProfile], None]] = None) -> ValidationProfile:
        """
        Render a profile selection UI and return the selected profile.
        
        Args:
            on_profile_change: Optional callback function to execute when the profile changes
            
        Returns:
            The selected ValidationProfile
        """
        selected_profile = render_profile_editor(self.profiles_dir, on_profile_change)
        self.active_profile = selected_profile
        return selected_profile
    
    def get_validation_summary_markdown(self, df: pd.DataFrame) -> str:
        """
        Generate a markdown summary of validation results.
        
        Args:
            df: DataFrame with flag columns
            
        Returns:
            Markdown string with validation summary
        """
        return generate_flag_summary(df)
    
    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean a DataFrame by removing flag columns.
        
        Args:
            df: DataFrame with flag columns
            
        Returns:
            Cleaned DataFrame without flag columns
        """
        return df.loc[:, ~df.columns.str.startswith('flag_')]
    
    def auto_clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Automatically clean a DataFrame by applying common cleaning operations.
        
        Args:
            df: DataFrame with flag columns
            
        Returns:
            Cleaned DataFrame
        """
        cleaned_df = df.copy()
        
        # Use column name detection to find relevant columns
        flag_columns = [col for col in df.columns if col.startswith('flag_')]
        
        # Process each potential issue
        
        # 1. Fix negative gross values
        if 'flag_negative_gross' in flag_columns:
            # Find gross column(s)
            gross_cols = [col for col in df.columns if 'gross' in col.lower() and 'flag' not in col.lower()]
            if gross_cols:
                for col in gross_cols:
                    cleaned_df.loc[cleaned_df[col] < 0, col] = 0
        
        # 2. Fix missing lead sources
        if 'flag_missing_lead_source' in flag_columns:
            # Find lead source column(s)
            lead_cols = [col for col in df.columns if ('lead' in col.lower() or 'source' in col.lower()) and 'flag' not in col.lower()]
            if lead_cols:
                for col in lead_cols:
                    cleaned_df.loc[cleaned_df[col].isna() | (cleaned_df[col] == ''), col] = 'Unknown'
        
        # 3. Fix duplicate VINs
        if 'flag_duplicate_vin' in flag_columns:
            # Find VIN and date columns
            vin_cols = [col for col in df.columns if 'vin' in col.lower() and 'flag' not in col.lower()]
            date_cols = [col for col in df.columns if any(date_term in col.lower() for date_term in ['date', 'time', 'day', 'month', 'year']) and 'flag' not in col.lower()]
            
            if vin_cols and date_cols:
                vin_col = vin_cols[0]
                date_col = date_cols[0]
                
                # Convert date column to datetime if needed
                if not pd.api.types.is_datetime64_dtype(cleaned_df[date_col]):
                    try:
                        cleaned_df[date_col] = pd.to_datetime(cleaned_df[date_col])
                    except:
                        pass
                
                # Sort by date and keep last entry for each VIN
                cleaned_df = cleaned_df.sort_values(date_col).drop_duplicates(subset=[vin_col], keep='last')
        
        # 4. Mark records with missing VINs for review
        if 'flag_missing_vin' in flag_columns:
            cleaned_df['needs_vin_review'] = df['flag_missing_vin']
        
        # Remove flag columns
        cleaned_df = self.clean_dataframe(cleaned_df)
        
        return cleaned_df


def process_uploaded_file(file, 
                         profiles_dir: str = "profiles", 
                         apply_auto_cleaning: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any], ValidatorService]:
    """
    Process an uploaded file through validation.
    
    This function handles file upload, validation, and cleaning in one streamlined process.
    
    Args:
        file: The uploaded file object (from st.file_uploader)
        profiles_dir: Directory containing validation profiles
        apply_auto_cleaning: Whether to automatically apply cleaning operations
        
    Returns:
        Tuple of (DataFrame, validation summary, validator service)
    """
    # Create a validator service
    validator = ValidatorService(profiles_dir)
    
    # Load the file into a DataFrame
    try:
        df = pd.read_csv(file)
    except Exception as e:
        # If CSV parsing fails, try Excel
        try:
            df = pd.read_excel(file)
        except Exception as e2:
            return None, {"error": f"Failed to load file: {str(e)}, {str(e2)}"}, validator
    
    # Validate the DataFrame
    validated_df, validation_summary = validator.validate_dataframe(df)
    
    # Apply auto-cleaning if requested
    if apply_auto_cleaning:
        validated_df = validator.auto_clean_dataframe(validated_df)
    
    # Store the validated data in the session state
    if "validated_data" not in st.session_state:
        st.session_state["validated_data"] = {}
    
    # Generate a unique key for this upload
    upload_key = f"upload_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    # Store both the validated DataFrame and the validation summary
    st.session_state["validated_data"][upload_key] = {
        "df": validated_df,
        "summary": validation_summary,
        "timestamp": datetime.now().isoformat(),
        "cleaned": apply_auto_cleaning
    }
    
    # Set the current upload as the active upload
    st.session_state["active_upload"] = upload_key
    
    return validated_df, validation_summary, validator


def render_data_validation_interface(df: pd.DataFrame, 
                                    validator: ValidatorService,
                                    on_continue: Optional[Callable[[pd.DataFrame], None]] = None) -> Tuple[pd.DataFrame, bool]:
    """
    Render a complete data validation interface.
    
    This function renders a UI for validating data, selecting profiles, and
    continuing to the next step (insight generation).
    
    Args:
        df: The DataFrame to validate
        validator: The validator service
        on_continue: Optional callback function to execute when the "Continue to Insights" button is clicked
        
    Returns:
        Tuple of (potentially cleaned DataFrame, boolean indicating if cleaning was performed)
    """
    st.subheader("🔍 Data Validation")
    
    # Create tabs for profile selection and validation results
    tab1, tab2 = st.tabs(["Validation Profile", "Validation Results"])
    
    with tab1:
        st.write("Select or create a validation profile to use for this dataset:")
        
        # Render profile selection UI
        def handle_profile_change(profile):
            # Re-validate the data with the new profile
            validated_df, _ = validator.validate_dataframe(df)
            
            # Update the session state with the new validated data
            if "active_upload" in st.session_state and "validated_data" in st.session_state:
                upload_key = st.session_state["active_upload"]
                if upload_key in st.session_state["validated_data"]:
                    st.session_state["validated_data"][upload_key]["df"] = validated_df
                    st.session_state["validated_data"][upload_key]["profile"] = profile.id
        
        validator.render_profile_selection(handle_profile_change)
    
    with tab2:
        # Get the validated data
        validated_df, validation_summary = validator.validate_dataframe(df)
        
        # Display overview stats in a clean layout
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Total Records", 
                value=validation_summary['total_records']
            )
        
        with col2:
            st.metric(
                label="Records with Issues", 
                value=validation_summary['total_issues'],
                delta=f"{100 - validation_summary['percentage_clean']:.1f}% of data",
                delta_color="inverse"
            )
            
        with col3:
            st.metric(
                label="Data Quality Score", 
                value=f"{validation_summary['percentage_clean']:.1f}%",
                delta=None if validation_summary['percentage_clean'] > 95 else f"{95 - validation_summary['percentage_clean']:.1f}% below target",
                delta_color="normal" if validation_summary['percentage_clean'] > 95 else "inverse"
            )
        
        # Display the validation results
        st.write("---")
        
        if validation_summary['total_issues'] > 0:
            # Initialize variables for handling cleaning
            cleaned_df = validated_df.copy()
            cleaning_applied = False
            
            # Show expanders for each issue type
            
            # 1. Negative Gross
            if 'negative_gross_count' in validation_summary and validation_summary['negative_gross_count'] > 0:
                with st.expander(f"📌 Negative Gross Profit ({validation_summary['negative_gross_count']} records)", expanded=True):
                    st.markdown(f"""
                    **Issue**: {validation_summary['negative_gross_count']} records ({validation_summary['negative_gross_count']/validation_summary['total_records']*100:.1f}% of data) 
                    have negative gross profit values, which may indicate pricing errors or special circumstances.
                    """)
                    
                    # Show sample of problematic records
                    if st.checkbox("Show affected records", key="show_negative_gross"):
                        st.dataframe(validated_df[validated_df['flag_negative_gross']], use_container_width=True)
                    
                    # Add option to fix negative gross by setting to 0
                    if st.checkbox("📝 Convert negative gross values to zero", key="fix_negative_gross"):
                        # Find gross column(s)
                        gross_cols = [col for col in validated_df.columns if 'gross' in col.lower() and 'flag' not in col.lower()]
                        if gross_cols:
                            for col in gross_cols:
                                cleaned_df.loc[cleaned_df[col] < 0, col] = 0
                            st.success(f"Negative values in {', '.join(gross_cols)} will be converted to zero when cleaned.")
                            cleaning_applied = True
            
            # 2. Missing Lead Source
            if 'missing_lead_source_count' in validation_summary and validation_summary['missing_lead_source_count'] > 0:
                with st.expander(f"📌 Missing Lead Sources ({validation_summary['missing_lead_source_count']} records)", expanded=True):
                    st.markdown(f"""
                    **Issue**: {validation_summary['missing_lead_source_count']} records ({validation_summary['missing_lead_source_count']/validation_summary['total_records']*100:.1f}% of data) 
                    are missing lead source information, which prevents accurate marketing ROI analysis.
                    """)
                    
                    # Show sample of problematic records
                    if st.checkbox("Show affected records", key="show_missing_lead"):
                        st.dataframe(validated_df[validated_df['flag_missing_lead_source']], use_container_width=True)
                    
                    # Add option to set a default lead source
                    if st.checkbox("📝 Set missing lead sources to 'Unknown'", key="fix_missing_lead"):
                        # Find lead source column(s)
                        lead_cols = [col for col in validated_df.columns if ('lead' in col.lower() or 'source' in col.lower()) and 'flag' not in col.lower()]
                        if lead_cols:
                            for col in lead_cols:
                                cleaned_df.loc[cleaned_df[col].isna() | (cleaned_df[col] == ''), col] = 'Unknown'
                            st.success(f"Missing values in {', '.join(lead_cols)} will be set to 'Unknown' when cleaned.")
                            cleaning_applied = True
            
            # 3. Duplicate VINs
            if 'duplicate_vins_count' in validation_summary and validation_summary['duplicate_vins_count'] > 0:
                with st.expander(f"📌 Duplicate VINs ({validation_summary['duplicate_vins_count']} records)", expanded=True):
                    st.markdown(f"""
                    **Issue**: {validation_summary['duplicate_vins_count']} records ({validation_summary['duplicate_vins_count']/validation_summary['total_records']*100:.1f}% of data) 
                    have duplicate VIN numbers, which may indicate data entry errors or multiple transactions on the same vehicle.
                    """)
                    
                    # Show sample of problematic records
                    if st.checkbox("Show affected records", key="show_dup_vins"):
                        st.dataframe(validated_df[validated_df['flag_duplicate_vin']], use_container_width=True)
                    
                    # Add option to keep only the latest transaction for each VIN
                    if st.checkbox("📝 Keep only the latest transaction for each VIN", key="fix_dup_vins"):
                        # Find VIN and date columns
                        vin_cols = [col for col in validated_df.columns if 'vin' in col.lower() and 'flag' not in col.lower()]
                        date_cols = [col for col in validated_df.columns if any(date_term in col.lower() for date_term in ['date', 'time', 'day', 'month', 'year']) and 'flag' not in col.lower()]
                        
                        if vin_cols and date_cols:
                            vin_col = vin_cols[0]
                            date_col = date_cols[0]
                            
                            # Convert date column to datetime if needed
                            if not pd.api.types.is_datetime64_dtype(cleaned_df[date_col]):
                                try:
                                    cleaned_df[date_col] = pd.to_datetime(cleaned_df[date_col])
                                except:
                                    st.error(f"Could not convert {date_col} to date format.")
                                    
                            # Sort by date and keep last entry for each VIN
                            cleaned_df = cleaned_df.sort_values(date_col).drop_duplicates(subset=[vin_col], keep='last')
                            st.success(f"Duplicate VINs will be resolved by keeping only the latest transaction when cleaned.")
                            cleaning_applied = True
            
            # 4. Missing/Invalid VINs
            if 'missing_vins_count' in validation_summary and validation_summary['missing_vins_count'] > 0:
                with st.expander(f"📌 Missing/Invalid VINs ({validation_summary['missing_vins_count']} records)", expanded=True):
                    st.markdown(f"""
                    **Issue**: {validation_summary['missing_vins_count']} records ({validation_summary['missing_vins_count']/validation_summary['total_records']*100:.1f}% of data) 
                    have missing or invalid VIN numbers, which complicates inventory tracking and reporting.
                    """)
                    
                    # Show sample of problematic records
                    if st.checkbox("Show affected records", key="show_missing_vins"):
                        st.dataframe(validated_df[validated_df['flag_missing_vin']], use_container_width=True)
                    
                    # Add option to flag for manual review
                    if st.checkbox("📝 Mark records with missing VINs for review", key="fix_missing_vins"):
                        # Add a review flag column
                        cleaned_df['needs_vin_review'] = validated_df['flag_missing_vin']
                        st.success(f"Records with missing/invalid VINs will be marked for review when cleaned.")
                        cleaning_applied = True
            
            # Add the cleanup button and continue to insights button
            st.write("---")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                if cleaning_applied:
                    st.info("Cleaning operations are ready to be applied.")
                else:
                    st.info("No cleaning operations selected.")
            
            with col2:
                clean_button = st.button("✅ Apply Data Cleaning", 
                                        type="primary" if cleaning_applied else "secondary",
                                        disabled=not cleaning_applied)
            
            # Handle clean button click
            if clean_button and cleaning_applied:
                st.success("Data cleaned successfully!")
                
                # Remove flag columns from the cleaned data
                cleaned_df = cleaned_df.loc[:, ~cleaned_df.columns.str.startswith('flag_')]
                
                # Update the session state
                if "active_upload" in st.session_state and "validated_data" in st.session_state:
                    upload_key = st.session_state["active_upload"]
                    if upload_key in st.session_state["validated_data"]:
                        st.session_state["validated_data"][upload_key]["df"] = cleaned_df
                        st.session_state["validated_data"][upload_key]["cleaned"] = True
                
                # Show the Continue to Insights button
                st.write("---")
                continue_button = st.button("🚀 Continue to Insights Generation", type="primary")
                
                if continue_button and on_continue:
                    on_continue(cleaned_df)
                
                return cleaned_df, True
        else:
            st.success("✨ No data quality issues detected. Data is ready for analysis!")
            
            # Show the Continue to Insights button
            st.write("---")
            continue_button = st.button("🚀 Continue to Insights Generation", type="primary")
            
            if continue_button and on_continue:
                on_continue(validated_df)
    
    return df, False


if __name__ == "__main__":
    # Sample code for testing
    import streamlit as st
    
    st.set_page_config(page_title="Validator Service Demo", layout="wide")
    st.title("Watchdog AI - Validator Service Demo")
    
    # File upload section
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv", "xlsx", "xls"])
    
    if uploaded_file:
        # Process the uploaded file
        with st.spinner("Processing file..."):
            df, summary, validator = process_uploaded_file(uploaded_file, "test_profiles")
        
        if df is not None:
            # Display the validation interface
            def on_continue_callback(cleaned_df):
                st.session_state["cleaned_data"] = cleaned_df
                st.success("Data ready for insights generation!")
                st.balloons()
            
            render_data_validation_interface(df, validator, on_continue_callback)
        else:
            st.error(f"Error: {summary.get('error', 'Unknown error')}")
    else:
        st.info("Please upload a CSV or Excel file to begin.")