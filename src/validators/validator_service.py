"""
Validator Service for Watchdog AI.

This module provides a unified service for validating data using validation profiles.
It integrates the validation profile system with the insight validator and provides
a clean API for processing uploaded files and validating data.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable
import streamlit as st
from datetime import datetime
from io import BytesIO
import hashlib # Import hashlib for caching key

from src.validators.validation_profile import (
    ValidationProfile,
    get_available_profiles,
    apply_validation_profile
)

from src.validators.insight_validator import (
    summarize_flags,
    generate_flag_summary
)

from src.utils.errors import ValidationError, ProcessingError, handle_error
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

class FileValidationError(ValidationError):
    """Custom exception for file validation errors."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)

def detect_file_type(file) -> str:
    """
    Detect the type of the uploaded file with error handling.
    """
    try:
        logger.info(f"Detecting file type for: {file.name}")
        
        if file.name.endswith('.csv'):
            return 'csv'
        elif file.name.endswith(('.xlsx', '.xls')):
            return 'excel'
        else:
            raise FileValidationError(
                f"Unsupported file type: {file.name}",
                details={"filename": file.name, "supported_types": [".csv", ".xlsx", ".xls"]}
            )
    except Exception as e:
        error_response = handle_error(e)
        logger.error(f"File type detection failed: {error_response}")
        raise

def load_dataframe(file, file_type: str) -> pd.DataFrame:
    """
    Load a file into a pandas DataFrame with error handling.
    """
    try:
        logger.info(f"Loading {file_type} file: {file.name}")
        
        if file_type == 'csv':
            # Check if file is empty
            file_content = file.read()
            if not file_content.strip():
                raise FileValidationError("The uploaded CSV file is empty")
            
            # Reset file pointer
            file.seek(0)
            
            # Try different encodings
            encodings = ['utf-8', 'latin1', 'iso-8859-1']
            for encoding in encodings:
                try:
                    file.seek(0)
                    df = pd.read_csv(file, encoding=encoding)
                    logger.info(f"Successfully loaded CSV with encoding: {encoding}")
                    return df
                except UnicodeDecodeError:
                    logger.warning(f"Failed to load CSV with encoding: {encoding}")
                    continue
                    
            raise FileValidationError(
                "Could not decode CSV file with supported encodings",
                details={"attempted_encodings": encodings}
            )
        else:
            # For Excel files
            try:
                df = pd.read_excel(file, engine='openpyxl')
                logger.info("Successfully loaded Excel file")
                return df
            except Exception as e:
                raise FileValidationError(
                    f"Error reading Excel file: {str(e)}",
                    details={"original_error": str(e)}
                )
    except Exception as e:
        error_response = handle_error(e)
        logger.error(f"File loading failed: {error_response}")
        raise

# --- In-memory Cache for File Processing --- 
# Simple dictionary cache. Key: (filename, file_hash), Value: (dataframe, summary, report)
# Note: This cache is process-specific and not persistent.
file_processing_cache = {}

def process_uploaded_file(
    file,
    selected_profile: Optional[ValidationProfile] = None,
    apply_auto_cleaning: bool = False
) -> Tuple[Optional[pd.DataFrame], Dict[str, Any], Optional[pd.DataFrame], Optional[Dict[str, str]]]:
    """
    Process an uploaded file through validation and optional cleaning.
    
    Args:
        file: The uploaded file object
        selected_profile: Optional validation profile to use
        apply_auto_cleaning: Whether to apply automatic cleaning
        
    Returns:
        Tuple containing:
        - Optional[pd.DataFrame]: The processed DataFrame or None if failed
        - Dict[str, Any]: Summary including validation results and any warnings/errors
        - Optional[pd.DataFrame]: Detailed validation report or None if failed
        - Optional[Dict[str, str]]: Schema information or None if failed
    """
    summary = {
        "status": "pending",
        "message": "",
        "timestamp": datetime.now().isoformat(),
        "filename": getattr(file, 'name', 'unknown'),
        "total_rows": 0,
        "passed_rows": 0,
        "failed_rows": 0,
        "flag_counts": {},
        "profile_used": selected_profile.id if selected_profile else "None"
    }
    
    try:
        # Validate file object
        if file is None:
            raise FileValidationError("No file provided")
        if not hasattr(file, 'read'):
            raise FileValidationError("Invalid file object - missing read method")
            
        # Check file size
        file.seek(0, 2)  # Seek to end
        file_size = file.tell()
        file.seek(0)  # Reset to start
        
        if file_size == 0:
            raise FileValidationError("File is empty")
        
        max_size = 100 * 1024 * 1024  # 100MB limit
        if file_size > max_size:
            raise FileValidationError(f"File too large. Maximum size is {max_size/1024/1024}MB")
        
        # Detect file type and load DataFrame
        try:
            file_type = detect_file_type(file)
            df = load_dataframe(file, file_type)
        except Exception as e:
            raise FileValidationError(f"Failed to load file: {str(e)}")
            
        # Basic DataFrame validation
        if df.empty:
            raise FileValidationError("DataFrame is empty after loading")
        if df.columns.empty:
            raise FileValidationError("DataFrame has no columns")
            
        # Store schema information
        schema_info = {col: str(dtype) for col, dtype in df.dtypes.items()}
        
        # Apply validation profile if provided
        if selected_profile:
            try:
                validated_df, flag_counts = apply_validation_profile(df, selected_profile)
                summary["flag_counts"] = flag_counts
            except Exception as e:
                print(f"[ERROR] Validation profile application failed: {e}")
                # Don't fail completely - continue with original DataFrame
                validated_df = df
                summary["load_warning"] = f"Validation rules could not be applied: {str(e)}"
        else:
            validated_df = df
            
        # Apply auto-cleaning if requested
        if apply_auto_cleaning:
            try:
                validated_df = auto_clean_dataframe(validated_df)
                summary["message"] = "File processed and auto-cleaned successfully"
            except Exception as e:
                print(f"[WARN] Auto-cleaning failed: {e}")
                summary["load_warning"] = f"Auto-cleaning could not be applied: {str(e)}"
        
        # Generate validation report
        try:
            validation_report = generate_validation_report(validated_df)
        except Exception as e:
            print(f"[ERROR] Failed to generate validation report: {e}")
            validation_report = None
            summary["load_warning"] = f"Could not generate validation report: {str(e)}"
        
        # Update summary
        summary.update({
            "status": "success",
            "total_rows": len(validated_df),
            "passed_rows": len(validated_df) - sum(flag_counts.values()) if flag_counts else len(validated_df),
            "failed_rows": sum(flag_counts.values()) if flag_counts else 0
        })
        
        return validated_df, summary, validation_report, schema_info
        
    except FileValidationError as e:
        summary.update({
            "status": "error",
            "message": str(e)
        })
        return None, summary, None, None
        
    except Exception as e:
        summary.update({
            "status": "error",
            "message": f"Unexpected error: {str(e)}"
        })
        print(f"[ERROR] Unexpected error in process_uploaded_file: {e}")
        return None, summary, None, None

class ValidatorService:
    """Service for validating data using validation profiles."""
    
    def __init__(self, profiles_dir: str = "profiles"):
        """Initialize the validator service."""
        self.profiles_dir = profiles_dir
        self._ensure_profiles_dir()
        self.profiles = get_available_profiles(profiles_dir)
        self.active_profile = next((p for p in self.profiles if p.is_default), None)
        if not self.active_profile and self.profiles:
            self.active_profile = self.profiles[0]
    
    def _ensure_profiles_dir(self):
        """Ensure the profiles directory exists."""
        os.makedirs(self.profiles_dir, exist_ok=True)
    
    def process_file(self, file) -> Tuple[Optional[pd.DataFrame], Dict[str, Any], Optional[pd.DataFrame]]:
        """Process a file using the active profile."""
        return process_uploaded_file(file, self.active_profile)
    
    def validate_dataframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Validate a DataFrame using the active profile."""
        if not self.active_profile:
            return df, {"error": "No active validation profile"}
        
        validated_df, flag_counts = apply_validation_profile(df, self.active_profile)
        validation_summary = summarize_flags(validated_df)
        
        return validated_df, validation_summary


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

def auto_clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform automatic cleaning operations on a DataFrame.
    
    Args:
        df: Input DataFrame to clean
        
    Returns:
        Cleaned DataFrame
    """
    # Make a copy to avoid modifying the original
    cleaned = df.copy()
    
    # Remove completely empty rows
    cleaned = cleaned.dropna(how='all')
    
    # Remove completely empty columns
    cleaned = cleaned.dropna(axis=1, how='all')
    
    # Strip whitespace from string columns
    string_columns = cleaned.select_dtypes(include=['object']).columns
    for col in string_columns:
        cleaned[col] = cleaned[col].str.strip() if cleaned[col].dtype == 'object' else cleaned[col]
    
    # Convert empty strings to NaN
    cleaned = cleaned.replace(r'^\s*$', np.nan, regex=True)
    
    # Handle numeric columns
    numeric_columns = cleaned.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_columns:
        # Replace inf/-inf with NaN
        cleaned[col] = cleaned[col].replace([np.inf, -np.inf], np.nan)
        
        # Remove any non-numeric strings that might have gotten in
        cleaned[col] = pd.to_numeric(cleaned[col], errors='coerce')
    
    # Handle date columns
    date_columns = [col for col in cleaned.columns if 'date' in col.lower()]
    for col in date_columns:
        if cleaned[col].dtype == 'object':
            try:
                cleaned[col] = pd.to_datetime(cleaned[col], errors='coerce')
            except Exception as e:
                print(f"[WARN] Failed to convert {col} to datetime: {e}")
    
    return cleaned

def generate_validation_report(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Generate a detailed validation report from a DataFrame with flag columns.
    
    Args:
        df: DataFrame with flag columns
        
    Returns:
        DataFrame containing validation report or None if no flags found
    """
    # Get all flag columns
    flag_columns = [col for col in df.columns if col.startswith('flag_')]
    if not flag_columns:
        return None
        
    # Create report DataFrame
    report = df.copy()
    
    # Add has_issues column
    report['has_issues'] = report[flag_columns].any(axis=1)
    
    # Add issue_count column
    report['issue_count'] = report[flag_columns].sum(axis=1)
    
    # Add issue_details column
    def get_issue_details(row):
        issues = []
        for col in flag_columns:
            if row[col]:
                # Convert flag column name to readable message
                issue = col.replace('flag_', '').replace('_', ' ').title()
                issues.append(issue)
        return '; '.join(issues) if issues else 'No issues'
    
    report['issue_details'] = report.apply(get_issue_details, axis=1)
    
    return report

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