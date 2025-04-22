"""
Utility functions for DataFrame operations with comprehensive validation and error handling.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple
import logging
from datetime import datetime
import traceback

logger = logging.getLogger(__name__)

class DataFrameError(Exception):
    """Custom exception for DataFrame operations."""
    pass

class DataFrameUtils:
    """Utility class for robust DataFrame operations with comprehensive validation."""
    
    NUMERIC_TYPES = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    MAX_UNIQUE_CATEGORICAL = 1000  # Maximum unique values for categorical columns
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_cols: Optional[List[str]] = None) -> Tuple[bool, List[str]]:
        """
        Comprehensive DataFrame validation with detailed error reporting.
        
        Args:
            df: DataFrame to validate
            required_cols: Optional list of required column names
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_issues)
        """
        issues = []
        try:
            # Basic DataFrame checks
            if df is None:
                issues.append("DataFrame is None")
                return False, issues
                
            if not isinstance(df, pd.DataFrame):
                issues.append(f"Input is not a DataFrame (got {type(df)})")
                return False, issues
                
            if df.empty:
                issues.append("DataFrame is empty")
                return False, issues
                
            if len(df.columns) == 0:
                issues.append("DataFrame has no columns")
                return False, issues
                
            # Check for required columns
            if required_cols:
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    issues.append(f"Missing required columns: {', '.join(missing_cols)}")
                
            # Check for duplicate column names
            if len(df.columns) != len(set(df.columns)):
                issues.append("DataFrame contains duplicate column names")
                
            # Check for all-null columns
            null_cols = [col for col in df.columns if df[col].isnull().all()]
            if null_cols:
                issues.append(f"Columns with all null values: {', '.join(null_cols)}")
                
            # Check for mixed data types
            mixed_type_cols = []
            for col in df.columns:
                if df[col].dtype == 'object':
                    unique_types = {type(x) for x in df[col].dropna()}
                    if len(unique_types) > 1:
                        mixed_type_cols.append(col)
            if mixed_type_cols:
                issues.append(f"Columns with mixed data types: {', '.join(mixed_type_cols)}")
                
            return len(issues) == 0, issues
            
        except Exception as e:
            issues.append(f"Validation error: {str(e)}")
            logger.error(f"DataFrame validation error: {traceback.format_exc()}")
            return False, issues
            
    @staticmethod
    def get_column_stats(df: pd.DataFrame, column: str, sample_size: Optional[int] = 1000) -> Dict[str, Any]:
        """
        Get comprehensive statistics for a DataFrame column with error handling.
        
        Args:
            df: Source DataFrame
            column: Column name to analyze
            sample_size: Optional sample size for large datasets
            
        Returns:
            Dict containing column statistics
        """
        try:
            # Validate inputs
            is_valid, issues = DataFrameUtils.validate_dataframe(df)
            if not is_valid:
                raise DataFrameError(f"Invalid DataFrame: {'; '.join(issues)}")
                
            if not isinstance(column, str):
                raise DataFrameError(f"Column name must be string, got {type(column)}")
                
            if column not in df.columns:
                raise DataFrameError(f"Column '{column}' not found in DataFrame")
                
            # Initialize stats dictionary
            stats = {
                "column": column,
                "dtype": str(df[column].dtype),
                "row_count": len(df),
                "null_count": int(df[column].isnull().sum()),
                "null_percentage": float(df[column].isnull().mean() * 100),
                "unique_count": int(df[column].nunique()),
                "memory_usage": int(df[column].memory_usage(deep=True)),
                "timestamp": datetime.now().isoformat()
            }
            
            # Handle different data types
            if pd.api.types.is_numeric_dtype(df[column]):
                clean_series = df[column].dropna()
                if not clean_series.empty:
                    stats.update({
                        "min": float(clean_series.min()),
                        "max": float(clean_series.max()),
                        "mean": float(clean_series.mean()),
                        "median": float(clean_series.median()),
                        "std": float(clean_series.std()),
                        "skew": float(clean_series.skew()),
                        "kurtosis": float(clean_series.kurtosis()),
                        "zero_count": int((clean_series == 0).sum()),
                        "negative_count": int((clean_series < 0).sum())
                    })
                    
            elif pd.api.types.is_string_dtype(df[column]):
                clean_series = df[column].dropna()
                if not clean_series.empty:
                    # Sample for large datasets
                    if len(clean_series) > sample_size:
                        clean_series = clean_series.sample(sample_size)
                    
                    stats.update({
                        "empty_string_count": int((clean_series == "").sum()),
                        "min_length": int(clean_series.str.len().min()),
                        "max_length": int(clean_series.str.len().max()),
                        "avg_length": float(clean_series.str.len().mean()),
                        "contains_numbers": bool(clean_series.str.contains(r'\d').any()),
                        "contains_special_chars": bool(clean_series.str.contains(r'[^a-zA-Z0-9\s]').any())
                    })
                    
            elif pd.api.types.is_datetime64_any_dtype(df[column]):
                clean_series = df[column].dropna()
                if not clean_series.empty:
                    stats.update({
                        "min_date": clean_series.min().isoformat(),
                        "max_date": clean_series.max().isoformat(),
                        "date_range_days": int((clean_series.max() - clean_series.min()).days)
                    })
                    
            return stats
            
        except Exception as e:
            logger.error(f"Error getting column stats: {traceback.format_exc()}")
            return {
                "error": str(e),
                "column": column,
                "timestamp": datetime.now().isoformat()
            }
            
    @staticmethod
    def get_breakdown(
        df: pd.DataFrame, 
        group_by: Union[str, List[str]], 
        metric: str,
        agg_func: str = "count",
        top_n: Optional[int] = 10,
        min_group_size: int = 5,
        handle_nulls: bool = True,
        normalize: bool = False
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Get robust breakdown of data with comprehensive error handling and metadata.
        
        Args:
            df: Source DataFrame
            group_by: Column(s) to group by
            metric: Column to aggregate
            agg_func: Aggregation function to use
            top_n: Number of top results to return
            min_group_size: Minimum group size to include
            handle_nulls: Whether to handle null values
            normalize: Whether to normalize values to percentages
            
        Returns:
            Tuple[List[Dict[str, Any]], Dict[str, Any]]: (breakdown_data, metadata)
        """
        try:
            # Validate DataFrame
            is_valid, issues = DataFrameUtils.validate_dataframe(df)
            if not is_valid:
                raise DataFrameError(f"Invalid DataFrame: {'; '.join(issues)}")
                
            # Validate and standardize inputs
            if isinstance(group_by, str):
                group_by = [group_by]
            
            if not all(isinstance(col, str) for col in group_by):
                raise DataFrameError("All group_by columns must be strings")
                
            if not all(col in df.columns for col in group_by):
                missing = [col for col in group_by if col not in df.columns]
                raise DataFrameError(f"Columns not found: {', '.join(missing)}")
                
            # Validate aggregation function
            valid_aggs = ["count", "sum", "mean", "median", "min", "max", "std"]
            agg_func = agg_func.lower()
            if agg_func not in valid_aggs:
                raise DataFrameError(f"Invalid aggregation function. Must be one of: {', '.join(valid_aggs)}")
                
            # Handle null values if requested
            work_df = df.copy()
            if handle_nulls:
                for col in group_by:
                    work_df[col] = work_df[col].fillna("Unknown")
                if metric in work_df.columns:
                    work_df[metric] = pd.to_numeric(work_df[metric], errors='coerce')
                    
            # Perform groupby operation
            try:
                if agg_func == "count":
                    grouped = work_df.groupby(group_by).size().reset_index(name="value")
                else:
                    if metric not in work_df.columns:
                        raise DataFrameError(f"Metric column '{metric}' not found")
                    grouped = work_df.groupby(group_by)[metric].agg(agg_func).reset_index(name="value")
                    
                # Filter by minimum group size
                if min_group_size > 1:
                    counts = work_df.groupby(group_by).size().reset_index(name="group_size")
                    grouped = grouped.merge(counts, on=group_by)
                    grouped = grouped[grouped["group_size"] >= min_group_size]
                    grouped = grouped.drop("group_size", axis=1)
                    
                # Normalize if requested
                if normalize and len(grouped) > 0:
                    total = grouped["value"].sum()
                    if total > 0:
                        grouped["value"] = grouped["value"] / total * 100
                        
                # Sort and limit results
                grouped = grouped.sort_values("value", ascending=False)
                if top_n and len(grouped) > top_n:
                    grouped = grouped.head(top_n)
                    
                # Convert to list of dicts
                result = []
                for _, row in grouped.iterrows():
                    item = {"value": float(row["value"])}  # Ensure value is float
                    for col in group_by:
                        item["category" if len(group_by) == 1 else col] = str(row[col])  # Ensure categories are strings
                    result.append(item)
                    
                # Generate metadata
                metadata = {
                    "total_groups": len(grouped),
                    "total_records": len(df),
                    "aggregation_function": agg_func,
                    "normalized": normalize,
                    "min_group_size": min_group_size,
                    "null_handled": handle_nulls,
                    "timestamp": datetime.now().isoformat()
                }
                
                return result, metadata
                
            except Exception as e:
                raise DataFrameError(f"Error in groupby operation: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error getting breakdown: {traceback.format_exc()}")
            return [], {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            } 