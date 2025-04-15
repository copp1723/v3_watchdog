"""
Insight Validator for Watchdog AI.

This module provides functionality to review cleaned DataFrames and flag common
dealership-specific red flags such as negative gross, missing lead sources, and
duplicate VINs. The module also provides summary statistics about detected issues.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import markdown


def flag_negative_gross(df: pd.DataFrame, gross_col: str = 'Gross_Profit') -> pd.DataFrame:
    """
    Flags rows where Gross is negative and adds a 'flag_negative_gross' column.
    
    Args:
        df: The input DataFrame
        gross_col: The name of the column containing gross profit information
        
    Returns:
        DataFrame with added flag column
    """
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    # Check if the gross column exists
    if gross_col not in result_df.columns:
        # Try to find a column that might contain gross information
        potential_cols = [col for col in result_df.columns if 'gross' in col.lower()]
        if potential_cols:
            gross_col = potential_cols[0]
        else:
            # If no gross column found, return the original DataFrame with an empty flag column
            result_df['flag_negative_gross'] = False
            return result_df
    
    # Ensure the gross column is numeric
    if not pd.api.types.is_numeric_dtype(result_df[gross_col]):
        # Try to convert to numeric, coercing errors to NaN
        result_df[gross_col] = pd.to_numeric(result_df[gross_col], errors='coerce')
    
    # Flag rows with negative gross
    result_df['flag_negative_gross'] = result_df[gross_col] < 0
    
    return result_df


def flag_missing_lead_source(df: pd.DataFrame, lead_source_col: str = 'Lead_Source') -> pd.DataFrame:
    """
    Flags rows where LeadSource is null or empty.
    
    Args:
        df: The input DataFrame
        lead_source_col: The name of the column containing lead source information
        
    Returns:
        DataFrame with added flag column
    """
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    # Check if the lead source column exists
    if lead_source_col not in result_df.columns:
        # Try to find a column that might contain lead source information
        potential_cols = [col for col in result_df.columns if 'lead' in col.lower() or 'source' in col.lower()]
        if potential_cols:
            lead_source_col = potential_cols[0]
        else:
            # If no lead source column found, return the original DataFrame with an empty flag column
            result_df['flag_missing_lead_source'] = False
            return result_df
    
    # Flag rows with missing or empty lead source
    result_df['flag_missing_lead_source'] = (
        result_df[lead_source_col].isna() | 
        (result_df[lead_source_col] == '') | 
        (result_df[lead_source_col].astype(str).str.strip() == '')
    )
    
    return result_df


def flag_duplicate_vins(df: pd.DataFrame, vin_col: str = 'VIN') -> pd.DataFrame:
    """
    Flags duplicate VINs and appends a column noting it.
    
    Args:
        df: The input DataFrame
        vin_col: The name of the column containing VIN information
        
    Returns:
        DataFrame with added flag column
    """
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    # Check if the VIN column exists
    if vin_col not in result_df.columns:
        # Try to find a column that might contain VIN information
        potential_cols = [col for col in result_df.columns if 'vin' in col.lower()]
        if potential_cols:
            vin_col = potential_cols[0]
        else:
            # If no VIN column found, return the original DataFrame with an empty flag column
            result_df['flag_duplicate_vin'] = False
            return result_df
    
    # Count occurrences of each VIN
    vin_counts = result_df[vin_col].value_counts()
    
    # Flag rows with duplicate VINs
    result_df['flag_duplicate_vin'] = result_df[vin_col].map(lambda x: vin_counts.get(x, 0) > 1)
    
    return result_df


def flag_missing_vins(df: pd.DataFrame, vin_col: str = 'VIN') -> pd.DataFrame:
    """
    Flags rows where the VIN is missing or invalid.
    
    Args:
        df: The input DataFrame
        vin_col: The name of the column containing VIN information
        
    Returns:
        DataFrame with added flag column
    """
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    # Check if the VIN column exists
    if vin_col not in result_df.columns:
        # Try to find a column that might contain VIN information
        potential_cols = [col for col in result_df.columns if 'vin' in col.lower()]
        if potential_cols:
            vin_col = potential_cols[0]
        else:
            # If no VIN column found, return the original DataFrame with an empty flag column
            result_df['flag_missing_vin'] = False
            return result_df
    
    # Flag rows with missing or invalid VINs (basic check for length and pattern)
    result_df['flag_missing_vin'] = (
        result_df[vin_col].isna() | 
        (result_df[vin_col] == '') | 
        (result_df[vin_col].astype(str).str.strip() == '') |
        (~result_df[vin_col].astype(str).str.match(r'^[A-HJ-NPR-Z0-9]{17}$'))
    )
    
    return result_df


def flag_all_issues(df: pd.DataFrame, 
                   gross_col: str = 'Gross_Profit',
                   lead_source_col: str = 'Lead_Source', 
                   vin_col: str = 'VIN') -> pd.DataFrame:
    """
    Applies all flag functions to the DataFrame at once.
    
    Args:
        df: The input DataFrame
        gross_col: The name of the column containing gross profit information
        lead_source_col: The name of the column containing lead source information
        vin_col: The name of the column containing VIN information
        
    Returns:
        DataFrame with all flag columns added
    """
    result_df = df.copy()
    
    # Apply all flag functions
    result_df = flag_negative_gross(result_df, gross_col)
    result_df = flag_missing_lead_source(result_df, lead_source_col)
    result_df = flag_duplicate_vins(result_df, vin_col)
    result_df = flag_missing_vins(result_df, vin_col)
    
    return result_df


def summarize_flags(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Returns a summary of flagged issues in the DataFrame.
    
    Args:
        df: The input DataFrame with flag columns
        
    Returns:
        A dictionary with issue counts and percentages
    """
    # Ensure the DataFrame has the necessary flag columns
    required_flags = [
        'flag_negative_gross', 
        'flag_missing_lead_source', 
        'flag_duplicate_vin',
        'flag_missing_vin'
    ]
    
    # Filter for flags that actually exist in the DataFrame
    existing_flags = [flag for flag in required_flags if flag in df.columns]
    
    # If none of the expected flags exist, return an empty summary
    if not existing_flags:
        return {
            'total_records': len(df),
            'total_issues': 0,
            'issue_summary': {},
            'percentage_clean': 100.0
        }
    
    # Calculate issue counts
    issue_counts = {flag: int(df[flag].sum()) for flag in existing_flags}
    
    # Calculate total records with at least one issue
    if existing_flags:
        total_issues = int(df[existing_flags].any(axis=1).sum())
    else:
        total_issues = 0
    
    # Calculate percentage of clean records
    total_records = len(df)
    percentage_clean = 100.0 if total_records == 0 else 100.0 * (total_records - total_issues) / total_records
    
    # Create the summary dictionary
    summary = {
        'total_records': total_records,
        'total_issues': total_issues,
        'issue_summary': issue_counts,
        'percentage_clean': round(percentage_clean, 2)
    }
    
    # Add specific issue counts if they exist
    if 'flag_negative_gross' in df.columns:
        summary['negative_gross_count'] = int(df['flag_negative_gross'].sum())
    
    if 'flag_missing_lead_source' in df.columns:
        summary['missing_lead_source_count'] = int(df['flag_missing_lead_source'].sum())
    
    if 'flag_duplicate_vin' in df.columns:
        summary['duplicate_vins_count'] = int(df['flag_duplicate_vin'].sum())
    
    if 'flag_missing_vin' in df.columns:
        summary['missing_vins_count'] = int(df['flag_missing_vin'].sum())
    
    return summary


def generate_flag_summary(df: pd.DataFrame) -> str:
    """
    Generates a markdown summary of flagged issues in the DataFrame.
    
    Args:
        df: The input DataFrame with flag columns
        
    Returns:
        A markdown string summarizing the issues found
    """
    # Get the summary statistics
    summary = summarize_flags(df)
    
    # Create the markdown summary
    md = "# Data Quality Report\n\n"
    md += f"**Total Records:** {summary['total_records']}\n\n"
    md += f"**Records with Issues:** {summary['total_issues']} ({100 - summary['percentage_clean']:.2f}%)\n\n"
    md += f"**Clean Records:** {summary['total_records'] - summary['total_issues']} ({summary['percentage_clean']:.2f}%)\n\n"
    
    md += "## Issue Summary\n\n"
    
    issue_summary = summary.get('issue_summary', {})
    
    if not issue_summary:
        md += "No issues detected in the dataset.\n\n"
    else:
        md += "| Issue Type | Count | Percentage |\n"
        md += "|------------|-------|------------|\n"
        
        for flag, count in issue_summary.items():
            # Convert flag_name to a more readable format
            readable_flag = flag.replace('flag_', '').replace('_', ' ').title()
            percentage = 100.0 * count / summary['total_records'] if summary['total_records'] > 0 else 0
            md += f"| {readable_flag} | {count} | {percentage:.2f}% |\n"
    
    md += "\n## Recommendations\n\n"
    
    # Add recommendations based on the issues found
    if issue_summary.get('flag_negative_gross', 0) > 0:
        md += "- **Negative Gross:** Review pricing strategy and cost allocation. Transactions with negative gross should be examined for errors or special circumstances.\n\n"
    
    if issue_summary.get('flag_missing_lead_source', 0) > 0:
        md += "- **Missing Lead Source:** Improve lead tracking protocols. Missing lead sources prevent accurate marketing ROI analysis.\n\n"
    
    if issue_summary.get('flag_duplicate_vin', 0) > 0:
        md += "- **Duplicate VINs:** Investigate duplicate VIN entries. This may indicate data entry errors or legitimate multiple transactions on the same vehicle.\n\n"
    
    if issue_summary.get('flag_missing_vin', 0) > 0:
        md += "- **Missing/Invalid VINs:** Implement VIN validation at the point of data entry. Invalid VINs complicate inventory tracking and reporting.\n\n"
    
    return md


def highlight_flagged_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a styled DataFrame with highlighted issues for visualization.
    
    Args:
        df: The input DataFrame with flag columns
        
    Returns:
        A styled DataFrame with highlighted issues
    """
    # Get flag columns
    flag_cols = [col for col in df.columns if col.startswith('flag_')]
    
    # Create a styled DataFrame
    styled_df = df.style
    
    # Define a function to highlight rows with issues
    def highlight_issues(row):
        styles = [''] * len(row)
        
        for flag_col in flag_cols:
            if row[flag_col]:
                # Get the column name being flagged (e.g., Gross_Profit for flag_negative_gross)
                flagged_col = flag_col.replace('flag_', '').replace('_', ' ')
                
                # Find the corresponding data column
                if 'negative_gross' in flag_col and 'Gross_Profit' in df.columns:
                    col_idx = list(row.index).index('Gross_Profit')
                    styles[col_idx] = 'background-color: #ffcccc'
                
                elif 'missing_lead_source' in flag_col and 'Lead_Source' in df.columns:
                    col_idx = list(row.index).index('Lead_Source')
                    styles[col_idx] = 'background-color: #ffffcc'
                
                elif ('duplicate_vin' in flag_col or 'missing_vin' in flag_col) and 'VIN' in df.columns:
                    col_idx = list(row.index).index('VIN')
                    styles[col_idx] = 'background-color: #ffcccc'
        
        return styles
    
    # Apply the styling
    styled_df = styled_df.apply(highlight_issues, axis=1)
    
    return styled_df


if __name__ == "__main__":
    # Sample dirty DataFrame for testing
    data = {
        'VIN': ['123', '123', '456', '789', '', 'ABC'],
        'Gross_Profit': [1000, -500, 300, -20, 750, 1200],
        'Lead_Source': ['Facebook', None, '', 'Google', 'Autotrader', 'Walk-in']
    }
    
    df = pd.DataFrame(data)
    
    # Apply all flags
    flagged_df = flag_all_issues(df)
    
    # Print the flagged DataFrame
    print("Flagged DataFrame:")
    print(flagged_df)
    
    # Get the summary
    summary = summarize_flags(flagged_df)
    print("\nSummary:")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # Generate markdown summary
    md_summary = generate_flag_summary(flagged_df)
    print("\nMarkdown Summary:")
    print(md_summary)
