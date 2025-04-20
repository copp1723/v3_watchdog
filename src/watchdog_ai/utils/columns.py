"""
Column utilities for data analysis.
"""

from typing import Optional, List
import pandas as pd

def find_metric_column(df: pd.DataFrame, hint: Optional[str] = None) -> Optional[str]:
    """Find a metric column in a DataFrame."""
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if hint:
        for col in numeric_cols:
            if hint.lower() in col.lower():
                return col
    return numeric_cols[0] if len(numeric_cols) > 0 else None

def find_category_column(df: pd.DataFrame, hint: Optional[str] = None) -> Optional[str]:
    """Find a category column in a DataFrame."""
    category_cols = df.select_dtypes(include=['object', 'category']).columns
    if hint:
        for col in category_cols:
            if hint.lower() in col.lower():
                return col
    return category_cols[0] if len(category_cols) > 0 else None