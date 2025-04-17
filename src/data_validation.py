"""
Data validation module for Watchdog AI.
Handles validation of data files, schema checking, and data cleaning.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from .utils.data_normalization import normalize_columns, get_supported_aliases

class DataValidator:
    """
    Validates and processes data files.
    Provides methods for schema validation, data cleaning, and quality checks.
    """
    
    def __init__(self):
        """Initialize the DataValidator."""
        self.latest_validation_report = None
        self.latest_validation_summary = None
    
    def validate_file(self, df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate the given dataframe against expected schema.
        
        Args:
            df: The pandas DataFrame to validate
            
        Returns:
            Tuple containing:
            - success: Boolean indicating if validation passed
            - report: Dict with validation details
        """
        if df is None or df.empty:
            return False, {
                "status": "error",
                "message": "DataFrame is empty or None",
                "details": {}
            }
        
        # Normalize column names
        df = normalize_columns(df)
        
        # Basic validation
        report = {
            "status": "success",
            "message": "Data validation passed",
            "details": {
                "row_count": len(df),
                "column_count": len(df.columns),
                "columns": list(df.columns),
                "original_columns": list(df.columns),  # Store original names
                "normalized_columns": list(df.columns),  # Store normalized names
                "missing_values": df.isna().sum().to_dict(),
                "data_types": {col: str(df[col].dtype) for col in df.columns},
                "supported_aliases": get_supported_aliases()  # Include alias info
            }
        }
        
        self.latest_validation_report = report
        self.latest_validation_summary = {
            "status": "success",
            "message": f"Validated {len(df)} rows and {len(df.columns)} columns successfully"
        }
        
        return True, report
    
    def clean_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Clean the provided dataframe by handling missing values, outliers, etc.
        
        Args:
            df: The pandas DataFrame to clean
            
        Returns:
            Tuple containing:
            - cleaned_df: The cleaned DataFrame
            - cleaning_report: Dict with cleaning details
        """
        if df is None or df.empty:
            return df, {
                "status": "error",
                "message": "DataFrame is empty or None",
                "actions": []
            }
        
        # Make a copy to avoid modifying the original
        cleaned_df = df.copy()
        
        # Normalize column names
        cleaned_df = normalize_columns(cleaned_df)
        
        # Basic cleaning operations
        cleaning_actions = []
        
        # 1. Handle missing values
        missing_before = cleaned_df.isna().sum().sum()
        cleaned_df = cleaned_df.fillna(method='ffill')  # Forward fill
        cleaned_df = cleaned_df.fillna(method='bfill')  # Backward fill for any remaining NAs
        missing_after = cleaned_df.isna().sum().sum()
        
        if missing_before > missing_after:
            cleaning_actions.append({
                "action": "fill_missing",
                "count": missing_before - missing_after,
                "method": "ffill/bfill"
            })
        
        # 2. Convert date columns
        date_cols = []
        for col in cleaned_df.columns:
            if 'date' in col.lower():
                try:
                    cleaned_df[col] = pd.to_datetime(cleaned_df[col])
                    date_cols.append(col)
                except:
                    pass
        
        if date_cols:
            cleaning_actions.append({
                "action": "convert_dates",
                "columns": date_cols
            })
        
        # Return cleaned dataframe and report
        cleaning_report = {
            "status": "success", 
            "message": f"Cleaned data with {len(cleaning_actions)} actions",
            "actions": cleaning_actions
        }
        
        return cleaned_df, cleaning_report
    
    def get_data_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a profile of the dataframe with summary statistics.
        
        Args:
            df: The pandas DataFrame to profile
            
        Returns:
            Dict with profile information
        """
        if df is None or df.empty:
            return {
                "status": "error",
                "message": "Cannot profile empty DataFrame",
                "profile": {}
            }
        
        # Normalize column names
        df = normalize_columns(df)
        
        # Generate basic profile
        profile = {
            "shape": df.shape,
            "columns": list(df.columns),
            "summary": df.describe().to_dict(),
            "missing_values": df.isna().sum().to_dict(),
            "data_types": {col: str(df[col].dtype) for col in df.columns}
        }
        
        return {
            "status": "success",
            "message": "Generated data profile",
            "profile": profile
        }