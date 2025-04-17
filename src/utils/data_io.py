"""
Data I/O utilities for Watchdog AI.
Provides cached functions for loading and processing data.
"""

import pandas as pd
import streamlit as st
from typing import Dict, Any, Optional, Tuple
import logging
import re
from .data_normalization import normalize_columns, get_supported_aliases
from .errors import ValidationError

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    'LeadSource',
    'TotalGross',
    'VIN',
    'SaleDate',
    'SalePrice'
]

def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and normalize the DataFrame schema.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Normalized DataFrame with canonical column names
        
    Raises:
        ValidationError if required columns are missing
    """
    # Normalize column names
    df = normalize_columns(df)
    
    # Check for required columns
    expected = set(REQUIRED_COLUMNS)
    missing = expected - set(df.columns)
    
    if missing:
        # Get the original column names for better error messages
        found_cols = list(df.columns)
        
        error_msg = (
            f"Missing required columns: {', '.join(sorted(missing))}\n"
            f"Found columns: {', '.join(found_cols)}"
        )
        
        raise ValidationError(error_msg)
        
    return df

@st.cache_data
def load_data(uploaded_file) -> pd.DataFrame:
    """
    Load data from uploaded file with caching.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        Loaded DataFrame
        
    Raises:
        ValueError if required columns are missing
    """
    try:
        # Read file based on extension
        if uploaded_file.name.lower().endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Show the user what columns were found
        found_cols = list(df.columns)
        
        try:
            # Normalize column names and validate schema
            df = validate_schema(df)
            return df
            
        except ValidationError as e:
            # Get supported aliases for missing columns
            aliases = get_supported_aliases()
            missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
            missing_details = []
            
            for col in missing_cols:
                if col in aliases:
                    supported = [f"'{alias}'" for alias in aliases[col]]
                    missing_details.append(f"{col} (accepts: {', '.join(supported)})")
                else:
                    missing_details.append(col)
            
            error_msg = (
                f"Your file is missing some required columns. Found these columns: {', '.join(found_cols)}\n\n"
                f"Missing required columns: {', '.join(missing_details)}\n\n"
                "Please ensure your file has these columns with any of the supported names."
            )
            raise ValueError(error_msg)
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

@st.cache_data
def compute_lead_gross(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute lead source gross metrics with caching.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with lead source metrics
    """
    try:
        # Group by lead source and calculate metrics
        lead_gross = df.groupby('LeadSource').agg({
            'TotalGross': ['sum', 'mean', 'count']
        }).reset_index()
        
        # Flatten column names
        lead_gross.columns = ['LeadSource', 'TotalGross', 'AvgGross', 'Count']
        
        # Sort by total gross descending
        lead_gross = lead_gross.sort_values('TotalGross', ascending=False)
        
        return lead_gross
        
    except Exception as e:
        logger.error(f"Error computing lead gross: {str(e)}")
        raise

@st.cache_data
def validate_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Validate DataFrame schema and data quality with caching.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Tuple of (validated DataFrame, validation summary)
    """
    try:
        validation_summary = {
            'total_records': len(df),
            'missing_values': {},
            'invalid_values': {}
        }
        
        # Check for missing values
        for col in REQUIRED_COLUMNS:
            missing = df[col].isna().sum()
            if missing > 0:
                validation_summary['missing_values'][col] = missing
        
        # Check for invalid values
        validation_summary['invalid_values']['negative_gross'] = (
            (df['TotalGross'] < 0).sum()
        )
        
        validation_summary['invalid_values']['empty_lead_source'] = (
            ((df['LeadSource'].isna()) | (df['LeadSource'] == '')).sum()
        )
        
        validation_summary['invalid_values']['invalid_vin'] = (
            ((df['VIN'].str.len() != 17) | (df['VIN'].isna())).sum()
        )
        
        # Calculate overall quality score
        total_issues = (
            sum(validation_summary['missing_values'].values()) +
            sum(validation_summary['invalid_values'].values())
        )
        validation_summary['quality_score'] = max(
            0, 100 - (total_issues / len(df) * 100)
        )
        
        return df, validation_summary
        
    except Exception as e:
        logger.error(f"Error validating data: {str(e)}")
        raise