"""
Data Validation Component for Watchdog AI.
Provides UI components for validating and cleaning data.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional, Tuple

def render_validation_summary(df: pd.DataFrame, summary: Dict[str, Any]) -> None:
    """
    Render a summary of validation results.
    
    Args:
        df: The DataFrame being validated
        summary: Validation summary dictionary
    """
    st.markdown("### Data Validation Summary")
    
    # Display metrics in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Total Records",
            len(df),
            help="Total number of records in the dataset"
        )
    
    with col2:
        clean_rate = summary.get("clean_rate", 100)
        st.metric(
            "Clean Records",
            f"{clean_rate:.1f}%",
            help="Percentage of records with no validation issues"
        )
    
    with col3:
        issue_count = summary.get("total_issues", 0)
        st.metric(
            "Issues Found",
            issue_count,
            help="Number of validation issues found",
            delta_color="inverse"
        )
    
    # Show detailed validation results if there are issues
    if issue_count > 0:
        with st.expander("View Validation Details", expanded=True):
            # Display issues by type
            st.markdown("#### Issues by Type")
            for issue_type, count in summary.get("issues_by_type", {}).items():
                if count > 0:
                    st.markdown(f"- **{issue_type}**: {count} records")
            
            # Display sample of problematic records
            st.markdown("#### Sample Issues")
            if "problem_records" in summary:
                st.dataframe(summary["problem_records"].head(), use_container_width=True)

def render_data_cleaning(df: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
    """
    Render data cleaning options and return cleaned DataFrame.
    
    Args:
        df: DataFrame to clean
        
    Returns:
        Tuple of (cleaned DataFrame, whether cleaning was applied)
    """
    st.markdown("### Data Cleaning")
    
    cleaned_df = df.copy()
    cleaning_applied = False
    
    # Cleaning options
    with st.expander("Data Cleaning Options", expanded=True):
        # Handle missing values
        st.markdown("#### Handle Missing Values")
        missing_strategy = st.selectbox(
            "Strategy for missing values:",
            ["Keep as is", "Remove rows", "Fill with default values"]
        )
        
        if missing_strategy == "Remove rows":
            cleaned_df = cleaned_df.dropna()
            cleaning_applied = True
            st.success("Rows with missing values will be removed")
        elif missing_strategy == "Fill with default values":
            # Fill numeric columns with 0
            numeric_cols = cleaned_df.select_dtypes(include=['int64', 'float64']).columns
            cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(0)
            
            # Fill string columns with "Unknown"
            string_cols = cleaned_df.select_dtypes(include=['object']).columns
            cleaned_df[string_cols] = cleaned_df[string_cols].fillna("Unknown")
            
            cleaning_applied = True
            st.success("Missing values filled with defaults")
        
        # Handle duplicates
        st.markdown("#### Handle Duplicates")
        if st.checkbox("Remove duplicate records"):
            cleaned_df = cleaned_df.drop_duplicates()
            cleaning_applied = True
            st.success("Duplicate records will be removed")
    
    return cleaned_df, cleaning_applied

def render_data_validation(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Render the complete data validation interface.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Tuple of (validated/cleaned DataFrame, validation summary)
    """
    # Initialize validation summary
    summary = {
        "total_records": len(df),
        "clean_rate": 100.0,
        "total_issues": 0,
        "issues_by_type": {}
    }
    
    # Perform basic validation
    # Check for missing values
    missing_counts = df.isnull().sum()
    missing_cols = missing_counts[missing_counts > 0]
    if not missing_cols.empty:
        summary["issues_by_type"]["Missing Values"] = missing_cols.sum()
        summary["total_issues"] += missing_cols.sum()
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        summary["issues_by_type"]["Duplicate Records"] = duplicates
        summary["total_issues"] += duplicates
    
    # Update clean rate
    if summary["total_issues"] > 0:
        summary["clean_rate"] = ((len(df) - summary["total_issues"]) / len(df)) * 100
        
        # Get problematic records for display
        problem_mask = df.isnull().any(axis=1) | df.duplicated()
        summary["problem_records"] = df[problem_mask]
    
    # Render validation summary
    render_validation_summary(df, summary)
    
    # Render cleaning options if there are issues
    if summary["total_issues"] > 0:
        cleaned_df, cleaning_applied = render_data_cleaning(df)
        if cleaning_applied:
            return cleaned_df, summary
    
    return df, summary