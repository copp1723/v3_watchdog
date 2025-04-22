"""
Common data transformation utilities for Watchdog AI.
"""
from typing import Optional, Any, Dict, Union, List
import logging
import pandas as pd
import numpy as np

from watchdog_ai.core.constants import (
    COLUMN_MAPPINGS,
    NAN_WARNING_THRESHOLD,
    NAN_SEVERE_THRESHOLD
)

logger = logging.getLogger(__name__)

def find_matching_column(df: pd.DataFrame, column_key: Union[str, List[str]]) -> Optional[str]:
    """
    Find a matching column name in the DataFrame using predefined mappings or direct search.

    This function can work in two modes:
    1. If column_key is a string, it looks up possible alternative names in COLUMN_MAPPINGS.
    2. If column_key is a list, it treats the list as candidate column names to search for.

    Args:
        df: Input DataFrame
        column_key: Either a key to look up in COLUMN_MAPPINGS or a list of column names to try

    Returns:
        Actual column name if found, None otherwise
    """
    # Handle case where column_key is a list of candidate names
    if isinstance(column_key, list):
        for col_name in column_key:
            if col_name in df.columns:
                return col_name
        return None

    # Handle case where column_key is a string to look up in COLUMN_MAPPINGS
    if column_key not in COLUMN_MAPPINGS:
        logger.warning(f"No mapping found for column key: {column_key}")
        return None

    for col_name in COLUMN_MAPPINGS[column_key]:
        if col_name in df.columns:
            return col_name
    return None


def normalize_boolean_column(series: pd.Series) -> pd.Series:
    """
    Normalize various boolean-like values to 1 or 0.
    
    Args:
        series: Input series containing boolean-like values
        
    Returns:
        Series with values normalized to 1 (True) or 0 (False)
    """
    def to_binary(val: Any) -> int:
        if pd.isna(val):
            return 0
        if isinstance(val, (int, float)):
            return int(val != 0)
        if isinstance(val, str):
            return int(val.strip().lower() in ['1', 'true', 'yes', 'sold', 'y', 't', 'true'])
        if isinstance(val, bool):
            return int(val)
        return 0
    
    return series.apply(to_binary)


def clean_numeric_data(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Clean numeric data by removing currency symbols and commas.

    Args:
        df: Input DataFrame
        column: Column to clean

    Returns:
        DataFrame with cleaned numeric column
    """
    df = df.copy()
    df[column] = pd.to_numeric(
        df[column].astype(str).str.replace(r'[\$,]', '', regex=True),
        errors='coerce'
    )
    return df


def analyze_data_quality(df: pd.DataFrame, column: str) -> Dict[str, Any]:
    """
    Analyze data quality for a specific column.

    Args:
        df: Input DataFrame
        column: Column to analyze

    Returns:
        Dictionary with quality metrics
    """
    total_rows = len(df)
    nan_count = df[column].isna().sum()
    nan_percentage = (nan_count / total_rows * 100) if total_rows > 0 else 0

    return {
        'total_rows': total_rows,
        'nan_count': nan_count,
        'nan_percentage': nan_percentage,
        'warning_level': (
            'severe' if nan_percentage >= NAN_SEVERE_THRESHOLD * 100
            else 'warning' if nan_percentage >= NAN_WARNING_THRESHOLD * 100
            else 'normal'
        )
    }


def format_metric_value(value: float, metric_name: str) -> str:
    """
    Format a metric value based on its type.

    Args:
        value: Value to format
        metric_name: Name of the metric

    Returns:
        Formatted string representation
    """
    metric_lower = metric_name.lower()

    if any(indicator in metric_lower for indicator in [
        'price', 'cost', 'profit', 'revenue'
    ]):
        return f"${value:,.2f}"
    elif any(indicator in metric_lower for indicator in [
        'percent', 'percentage', 'rate'
    ]):
        return f"{value:.1f}%"
    else:
        return f"{value:,.0f}"


def get_error_response(
    error_type: str,
    error_details: str = ""
) -> Dict[str, Any]:
    """
    Generate a standardized error response.

    Args:
        error_type: Type of error
        error_details: Optional error details

    Returns:
        Standardized error response dictionary
    """
    return {
        "summary": (f"⚠️ {error_details}"
                    if error_details else "⚠️ An error occurred."),
        "metrics": {},
        "breakdown": [],
        "recommendations": [],
        "confidence": "low",
        "error_type": error_type
    }
