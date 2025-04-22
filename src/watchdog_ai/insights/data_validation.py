"""
Data validation and quality checks for insight generation.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

def check_data_quality(df: pd.DataFrame, required_columns: list = None) -> Tuple[bool, Dict[str, Any]]:
    """
    Check data quality and return validation results.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Returns:
        Tuple of (is_valid, validation_info)
    """
    validation_info = {
        "is_valid": True,
        "issues": [],
        "metrics": {},
        "valid_data_percentage": 100.0
    }
    
    if df is None or df.empty:
        validation_info["is_valid"] = False
        validation_info["issues"].append("No data provided")
        return False, validation_info
    
    # Check required columns
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            validation_info["is_valid"] = False
            validation_info["issues"].append(f"Missing required columns: {missing_cols}")
    
    # Check for completely empty columns
    empty_cols = df.columns[df.isna().all()].tolist()
    if empty_cols:
        validation_info["issues"].append(f"Completely empty columns: {empty_cols}")
    
    # Calculate missing value percentages
    missing_percentages = (df.isna().sum() / len(df) * 100).round(2)
    validation_info["metrics"]["missing_percentages"] = missing_percentages.to_dict()
    
    # Check numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        # Calculate valid data percentage
        valid_count = df[col].notna().sum()
        valid_percentage = (valid_count / len(df)) * 100
        validation_info["metrics"][f"{col}_valid_percentage"] = valid_percentage
        
        # Check for all zeros or very low variance
        if valid_count > 0:
            non_zero = df[col].dropna().astype(float).abs().sum()
            if non_zero == 0:
                validation_info["issues"].append(f"Column {col} contains all zeros")
            elif df[col].std() < 0.0001:
                validation_info["issues"].append(f"Column {col} has very low variance")
    
    # Calculate overall valid data percentage
    total_cells = df.size
    valid_cells = total_cells - df.isna().sum().sum()
    validation_info["valid_data_percentage"] = (valid_cells / total_cells) * 100
    
    # Set validity threshold
    if validation_info["valid_data_percentage"] < 50:
        validation_info["is_valid"] = False
        validation_info["issues"].append("Less than 50% valid data overall")
    
    return validation_info["is_valid"], validation_info