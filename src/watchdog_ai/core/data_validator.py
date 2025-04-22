"""
Data validation framework for Watchdog AI.

This module provides utilities for validating data quality and structure,
including checks for required columns, data types, missing values,
duplicates, and outliers.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import logging
import pandas as pd
import numpy as np

from watchdog_ai.core.constants import (
    DEFAULT_REQUIRED_COLUMNS,
    DEFAULT_COLUMN_TYPES,
    NAN_WARNING_THRESHOLD,
    NAN_SEVERE_THRESHOLD
)
from watchdog_ai.core.data_utils import (
    analyze_data_quality,
    clean_numeric_data,
    find_matching_column
)

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    """
    Dataclass to hold validation results with severity levels.
    
    Attributes:
        overall_status: Overall status of validation ('normal', 'warning', or 'severe')
        issues: Dictionary of validation issues by check type
        statistics: Raw statistics about the validation checks
    """
    overall_status: str = "normal"  # normal, warning, severe
    issues: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    statistics: Dict[str, Any] = field(default_factory=dict)
    
    def add_issue(self, check_type: str, status: str, severity: str, 
                 details: str, suggestion: Optional[str] = None) -> None:
        """
        Add an issue to the validation report.
        
        Args:
            check_type: Type of check (e.g., 'required_columns', 'data_types')
            status: Status of the check ('pass' or 'fail')
            severity: Severity level ('normal', 'warning', or 'severe')
            details: Detailed description of the issue
            suggestion: Optional suggestion for fixing the issue
        """
        self.issues[check_type] = {
            "status": status,
            "severity": severity,
            "details": details,
            "suggestion": suggestion
        }
        
        # Update overall status based on severity
        if severity == "severe" and self.overall_status != "severe":
            self.overall_status = "severe"
        elif severity == "warning" and self.overall_status == "normal":
            self.overall_status = "warning"


def _determine_severity(value: float, warning_threshold: float, 
                       severe_threshold: float) -> str:
    """
    Determine severity level based on thresholds.
    
    Args:
        value: Value to check
        warning_threshold: Threshold for warning level
        severe_threshold: Threshold for severe level
        
    Returns:
        Severity level ('normal', 'warning', or 'severe')
    """
    if value >= severe_threshold:
        return "severe"
    elif value >= warning_threshold:
        return "warning"
    else:
        return "normal"


def validate_dataframe(df: pd.DataFrame) -> ValidationReport:
    """
    Validate a DataFrame for data quality and structure.
    
    Args:
        df: Input DataFrame to validate
        
    Returns:
        ValidationReport with validation results
    """
    report = ValidationReport()
    
    if df.empty:
        report.add_issue(
            "empty_dataframe", 
            "fail", 
            "severe", 
            "DataFrame is empty.",
            "Ensure data source is properly connected and contains records."
        )
        return report
    
    # Store basic statistics
    report.statistics["total_rows"] = len(df)
    report.statistics["total_columns"] = len(df.columns)
    
    # 1. Required columns check
    _validate_required_columns(df, report)
    
    # 2. Data type check
    _validate_data_types(df, report)
    
    # 3. NaN values check
    _validate_nan_values(df, report)
    
    # 4. Duplicates check
    _validate_duplicates(df, report)
    
    # 5. Outliers check (primarily for TotalGross)
    _validate_outliers(df, report)
    
    return report


def _validate_required_columns(df: pd.DataFrame, report: ValidationReport) -> None:
    """
    Validate that required columns are present in the DataFrame.
    
    Args:
        df: Input DataFrame
        report: ValidationReport to update
    """
    missing_columns = []
    misnamed_columns = {}
    
    for column in DEFAULT_REQUIRED_COLUMNS:
        if column in df.columns:
            continue
        
        # Try to find a matching column using mappings
        matched_column = find_matching_column(df, column.lower())
        if matched_column:
            misnamed_columns[column] = matched_column
        else:
            missing_columns.append(column)
    
    if missing_columns or misnamed_columns:
        details = ""
        suggestion = ""
        
        if missing_columns:
            details += f"Missing required columns: {', '.join(missing_columns)}. "
            suggestion += f"Ensure data includes these columns: {', '.join(missing_columns)}. "
        
        if misnamed_columns:
            details += f"Misnamed columns found: {misnamed_columns}. "
            suggestion += "Consider renaming columns to match expected format, or update column mappings. "
        
        severity = "severe" if missing_columns else "warning"
        
        report.add_issue(
            "required_columns",
            "fail",
            severity,
            details.strip(),
            suggestion.strip()
        )
    else:
        report.add_issue(
            "required_columns",
            "pass",
            "normal",
            "All required columns are present.",
            None
        )
    
    # Save statistics
    report.statistics["missing_columns"] = missing_columns
    report.statistics["misnamed_columns"] = misnamed_columns


def _validate_data_types(df: pd.DataFrame, report: ValidationReport) -> None:
    """
    Validate that columns have the correct data types.
    
    Args:
        df: Input DataFrame
        report: ValidationReport to update
    """
    type_mismatches = {}
    
    for column, expected_type in DEFAULT_COLUMN_TYPES.items():
        if column not in df.columns:
            continue  # Skip columns that don't exist (handled in required columns check)
        
        if expected_type == 'datetime64[ns]':
            try:
                # Check if conversion is possible
                pd.to_datetime(df[column], errors='raise')
            except (ValueError, TypeError):
                type_mismatches[column] = f"Expected {expected_type}, found non-convertible type"
        else:
            # For numeric columns, check if conversion is possible
            actual_type = df[column].dtype.name
            if expected_type == 'float64' and not pd.api.types.is_numeric_dtype(df[column]):
                try:
                    # Check if conversion is possible
                    pd.to_numeric(df[column], errors='raise')
                except (ValueError, TypeError):
                    type_mismatches[column] = f"Expected {expected_type}, found {actual_type}"
            elif actual_type != expected_type and not (
                    expected_type == 'float64' and actual_type == 'int64' or
                    expected_type == 'object' and actual_type in ['str', 'string']):
                type_mismatches[column] = f"Expected {expected_type}, found {actual_type}"
    
    if type_mismatches:
        details = f"Data type mismatches found in {len(type_mismatches)} columns: {type_mismatches}"
        suggestion = (
            "Convert columns to expected types: \n" +
            "\n".join([f"- {col}: Convert to {type}" for col, type in DEFAULT_COLUMN_TYPES.items() 
                      if col in type_mismatches])
        )
        
        severity = "severe" if "SaleDate" in type_mismatches or "TotalGross" in type_mismatches else "warning"
        
        report.add_issue(
            "data_types",
            "fail",
            severity,
            details,
            suggestion
        )
    else:
        report.add_issue(
            "data_types",
            "pass",
            "normal",
            "All columns have the correct data type.",
            None
        )
    
    # Save statistics
    report.statistics["type_mismatches"] = type_mismatches


def _validate_nan_values(df: pd.DataFrame, report: ValidationReport) -> None:
    """
    Validate that columns don't have too many NaN values.
    
    Args:
        df: Input DataFrame
        report: ValidationReport to update
    """
    column_nan_stats = {}
    severe_columns = []
    warning_columns = []
    
    for column in DEFAULT_REQUIRED_COLUMNS:
        if column not in df.columns:
            continue  # Skip columns that don't exist (handled in required columns check)
        
        quality_stats = analyze_data_quality(df, column)
        column_nan_stats[column] = quality_stats
        
        nan_percentage = quality_stats["nan_percentage"] / 100.0  # Convert from percentage to fraction
        
        if nan_percentage >= NAN_SEVERE_THRESHOLD:
            severe_columns.append(column)
        elif nan_percentage >= NAN_WARNING_THRESHOLD:
            warning_columns.append(column)
    
    if severe_columns or warning_columns:
        details = ""
        suggestion = ""
        
        if severe_columns:
            details += f"Severe: Columns with more than {NAN_SEVERE_THRESHOLD*100}% missing values: {', '.join(severe_columns)}. "
            suggestion += f"Critical columns with many missing values should be addressed: {', '.join(severe_columns)}. "
        
        if warning_columns:
            details += f"Warning: Columns with more than {NAN_WARNING_THRESHOLD*100}% missing values: {', '.join(warning_columns)}. "
            suggestion += "Consider imputing or filtering rows with missing values. "
        
        severity = "severe" if severe_columns else "warning"
        
        report.add_issue(
            "nan_values",
            "fail",
            severity,
            details.strip(),
            suggestion.strip()
        )
    else:
        report.add_issue(
            "nan_values",
            "pass",
            "normal",
            "All required columns have acceptable levels of missing values.",
            None
        )
    
    # Save statistics
    report.statistics["column_nan_stats"] = column_nan_stats
    report.statistics["severe_nan_columns"] = severe_columns
    report.statistics["warning_nan_columns"] = warning_columns


def _validate_duplicates(df: pd.DataFrame, report: ValidationReport) -> None:
    """
    Validate that the DataFrame doesn't have too many duplicate rows.
    
    Args:
        df: Input DataFrame
        report: ValidationReport to update
    """
    # For duplicate detection, focus on required columns
    present_required_columns = [col for col in DEFAULT_REQUIRED_COLUMNS if col in df.columns]
    
    if not present_required_columns:
        # Can't check duplicates without required columns
        report.add_issue(
            "duplicates",
            "skip",
            "warning",
            "Cannot check for duplicates because required columns are missing.",
            "Ensure required columns are present in the data."
        )
        return
    
    # Count duplicates
    duplicate_count = df.duplicated(subset=present_required_columns).sum()
    duplicate_percentage = duplicate_count / len(df) if len(df) > 0 else 0
    
    # Custom thresholds for duplicates
    duplicate_warning_threshold = 0.01  # 1%
    duplicate_severe_threshold = 0.05   # 5%
    
    severity = _determine_severity(
        duplicate_percentage, 
        duplicate_warning_threshold, 
        duplicate_severe_threshold
    )
    
    # Save statistics
    report.statistics["duplicate_count"] = duplicate_count
    report.statistics["duplicate_percentage"] = duplicate_percentage * 100  # as percentage
    
    if duplicate_count > 0:
        details = f"Found {duplicate_count} duplicate rows ({duplicate_percentage:.2%})."
        suggestion = "Consider removing duplicate entries or investigating why they exist."
        
        report.add_issue(
            "duplicates",
            "fail",
            severity,
            details,
            suggestion
        )
    else:
        report.add_issue(
            "duplicates",
            "pass",
            "normal",
            "No duplicate rows found.",
            None
        )


def _validate_outliers(df: pd.DataFrame, report: ValidationReport) -> None:
    """
    Validate that numeric columns don't have too many outliers.
    
    Args:
        df: Input DataFrame
        report: ValidationReport to update
    """
    outlier_counts = {}
    outlier_percentages = {}
    total_outliers = 0
    
    # Focus on TotalGross for outlier detection
    if 'TotalGross' in df.columns:
        # Clean and prepare numeric data
        clean_df = clean_numeric_data(df, 'TotalGross')
        
        # Remove NaN values for outlier detection
        clean_series = clean_df['TotalGross'].dropna()
        
        if len(clean_series) > 10:  # Only check outliers if we have enough data
            q1 = clean_series.quantile(0.25)
            q3 = clean_series.quantile(0.75)
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = clean_series[(clean_series < lower_bound) | (clean_series > upper_bound)]
            outlier_count = len(outliers)
            outlier_percentage = outlier_count / len(clean_series) if len(clean_series) > 0 else 0
            
            # Custom thresholds for outliers
            outlier_warning_threshold = 0.01  # 1%
            outlier_severe_threshold = 0.05   # 5%
            
            severity = _determine_severity(
                outlier_percentage, 
                outlier_warning_threshold, 
                outlier_severe_threshold
            )
            
            outlier_counts['TotalGross'] = outlier_count
            outlier_percentages['TotalGross'] = outlier_percentage * 100  # as percentage
            total_outliers += outlier_count
            
            if outlier_count > 0:
                details = (
                    f"Found {outlier_count} outliers ({outlier_percentage:.2%}) in TotalGross. "
                    f"Values outside range: [{lower_bound:.2f}, {upper_bound:.2f}]"
                )
                suggestion = (
                    "Consider investigating extreme values in TotalGross. "
                    "They may represent data entry errors or special cases."
                )
                report.add_issue(
                    "outliers",
                    "fail",
                    severity,
                    details,
                    suggestion
                )
            else:
                report.add_issue(
                    "outliers",
                    "pass",
                    "normal",
                    "No outliers found in TotalGross.",
                    None
                )
        else:
            report.add_issue(
                "outliers",
                "skip",
                "normal",
                "Not enough data points to perform outlier detection.",
                "Ensure there are more than 10 valid rows for reliable outlier detection."
            )
    else:
        report.add_issue(
            "outliers",
            "skip",
            "warning",
            "TotalGross column not found. Cannot perform outlier detection.",
            "Ensure TotalGross column is present in the data."
        )
    
    # Save statistics
    report.statistics["outlier_counts"] = outlier_counts
    report.statistics["outlier_percentages"] = outlier_percentages
    report.statistics["total_outliers"] = total_outliers


# Module exports
__all__ = ["ValidationReport", "validate_dataframe"]
