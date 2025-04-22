"""
Enhanced validation system for data quality and schema validation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import re
import logging

logger = logging.getLogger(__name__)

class ValidationRule:
    """Base class for validation rules."""
    
    def __init__(self, name: str, severity: str = "error"):
        self.name = name
        self.severity = severity
    
    def validate(self, value: Any) -> Tuple[bool, str]:
        """
        Validate a value against the rule.
        
        Args:
            value: Value to validate
            
        Returns:
            Tuple of (is_valid, message)
        """
        raise NotImplementedError("Subclasses must implement validate()")

class RequiredRule(ValidationRule):
    """Rule to check if a value is required (not null)."""
    
    def validate(self, value: Any) -> Tuple[bool, str]:
        is_valid = pd.notna(value)
        return is_valid, "" if is_valid else "Value is required"

class TypeRule(ValidationRule):
    """Rule to check value type."""
    
    def __init__(self, expected_type: str, severity: str = "error"):
        super().__init__(f"type_{expected_type}", severity)
        self.expected_type = expected_type
    
    def validate(self, value: Any) -> Tuple[bool, str]:
        if pd.isna(value):
            return True, ""  # Skip type validation for null values
            
        try:
            if self.expected_type == "numeric":
                pd.to_numeric(value)
            elif self.expected_type == "datetime":
                pd.to_datetime(value)
            elif self.expected_type == "boolean":
                val = str(value).lower()
                if val not in ['0', '1', 'true', 'false']:
                    return False, "Value must be a boolean (0/1 or true/false)"
            return True, ""
        except:
            return False, f"Value must be of type {self.expected_type}"

class RangeRule(ValidationRule):
    """Rule to check if a numeric value is within a range."""
    
    def __init__(self, min_val: Optional[float] = None, max_val: Optional[float] = None, severity: str = "error"):
        super().__init__("range", severity)
        self.min_val = min_val
        self.max_val = max_val
    
    def validate(self, value: Any) -> Tuple[bool, str]:
        if pd.isna(value):
            return True, ""
            
        try:
            num_val = pd.to_numeric(value)
            if self.min_val is not None and num_val < self.min_val:
                return False, f"Value must be >= {self.min_val}"
            if self.max_val is not None and num_val > self.max_val:
                return False, f"Value must be <= {self.max_val}"
            return True, ""
        except:
            return False, "Value must be numeric"

class PatternRule(ValidationRule):
    """Rule to check if a string matches a pattern."""
    
    def __init__(self, pattern: str, severity: str = "error"):
        super().__init__("pattern", severity)
        self.pattern = re.compile(pattern)
    
    def validate(self, value: Any) -> Tuple[bool, str]:
        if pd.isna(value):
            return True, ""
            
        try:
            str_val = str(value)
            is_valid = bool(self.pattern.match(str_val))
            return is_valid, "" if is_valid else f"Value must match pattern {self.pattern.pattern}"
        except:
            return False, "Value must be string"

class DataValidator:
    """Validates data quality and schema compliance."""
    
    def __init__(self):
        """Initialize the validator."""
        self.column_rules: Dict[str, List[ValidationRule]] = {}
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Set up default validation rules."""
        # VIN validation
        self.add_column_rules("vin", [
            RequiredRule("vin_required"),
            PatternRule(r"^[A-HJ-NPR-Z0-9]{17}$", severity="warning")
        ])
        
        # Date validation
        date_columns = ["sale_date"]
        for col in date_columns:
            self.add_column_rules(col, [
                RequiredRule(f"{col}_required"),
                TypeRule("datetime")
            ])
        
        # Numeric validation
        numeric_columns = ["total_gross"]
        for col in numeric_columns:
            self.add_column_rules(col, [
                RequiredRule(f"{col}_required"),
                TypeRule("numeric")
            ])
        
        # Lead source validation
        self.add_column_rules("lead_source", [
            RequiredRule("lead_source_required")
        ])
    
    def add_column_rules(self, column: str, rules: List[ValidationRule]):
        """
        Add validation rules for a column.
        
        Args:
            column: Column name
            rules: List of ValidationRule objects
        """
        self.column_rules[column] = rules
    
    def _generate_column_stats(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """Generate statistics for a column."""
        return {
            'null_count': int(df[column].isnull().sum()) if column in df else 0,
            'unique_count': int(df[column].nunique()) if column in df else 0,
            'sample_values': df[column].dropna().head(3).tolist() if column in df else []
        }
    
    def validate_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate a DataFrame against all rules.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'valid': True,
            'errors': [],
            'warnings': [],
            'stats': {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'missing_columns': [],
                'column_stats': {}
            }
        }
        
        # Generate stats for all columns with rules
        for col in self.column_rules.keys():
            results['stats']['column_stats'][col] = self._generate_column_stats(df, col)
        
        # Check for missing columns
        missing_columns = []
        for required_col in self.column_rules.keys():
            if required_col not in df.columns:
                missing_columns.append(required_col)
        
        if missing_columns:
            results['stats']['missing_columns'] = missing_columns
            results['valid'] = False
            results['errors'].append({
                'type': 'missing_columns',
                'columns': missing_columns,
                'message': f"Missing required columns: {', '.join(missing_columns)}",
                'column': missing_columns[0],
                'rule': 'required_columns',
                'invalid_rows': [{
                    'row': 0,
                    'value': 'MISSING',
                    'message': f"Column {missing_columns[0]} is required"
                }]
            })
        
        # Validate each column
        for col in df.columns:
            if col in self.column_rules:
                # Apply rules
                for rule in self.column_rules[col]:
                    invalid_rows = []
                    
                    # Validate each value
                    for idx, value in df[col].items():
                        is_valid, message = rule.validate(value)
                        if not is_valid:
                            invalid_rows.append({
                                'row': int(idx),
                                'value': str(value),
                                'message': message
                            })
                    
                    # Record validation failures
                    if invalid_rows:
                        issue = {
                            'type': 'validation_error',
                            'column': col,
                            'rule': rule.name,
                            'invalid_rows': invalid_rows,
                            'message': f"Validation failed for column '{col}' with rule '{rule.name}'"
                        }
                        
                        if rule.severity == "error":
                            results['valid'] = False
                            results['errors'].append(issue)
                        else:
                            results['warnings'].append(issue)
        
        return results

def validate_string(value: str, min_length: int = 0, max_length: Optional[int] = None, pattern: Optional[str] = None) -> Tuple[bool, str]:
    """
    Validate a string value.
    
    Args:
        value: String to validate
        min_length: Minimum length
        max_length: Maximum length (optional)
        pattern: Regex pattern (optional)
        
    Returns:
        Tuple of (is_valid, message)
    """
    if not isinstance(value, str):
        raise TypeError("Value must be a string")
    
    if len(value) < min_length:
        return False, f"String length must be at least {min_length}"
    
    if max_length and len(value) > max_length:
        return False, f"String length must be at most {max_length}"
    
    if pattern and not re.match(pattern, value):
        return False, f"String must match pattern: {pattern}"
    
    return True, ""

def validate_number(value: Any, min_val: Optional[float] = None, max_val: Optional[float] = None) -> Tuple[bool, str]:
    """
    Validate a numeric value.
    
    Args:
        value: Number to validate
        min_val: Minimum value (optional)
        max_val: Maximum value (optional)
        
    Returns:
        Tuple of (is_valid, message)
    """
    try:
        num_val = float(value)
    except:
        return False, "Value must be numeric"
    
    if min_val is not None and num_val < min_val:
        return False, f"Value must be >= {min_val}"
    
    if max_val is not None and num_val > max_val:
        return False, f"Value must be <= {max_val}"
    
    return True, ""

def validate_date(value: Any) -> Tuple[bool, str]:
    """
    Validate a date value.
    
    Args:
        value: Date to validate
        
    Returns:
        Tuple of (is_valid, message)
    """
    if pd.isna(value) or value == "":
        return False, "Date value cannot be empty"
        
    try:
        pd.to_datetime(value)
        return True, ""
    except:
        return False, "Value must be a valid date"

def validate_file(file) -> Tuple[bool, str]:
    """
    Validate a file object.
    
    Args:
        file: File object to validate
        
    Returns:
        Tuple of (is_valid, message)
    """
    if not hasattr(file, 'name'):
        return False, "Invalid file object"
    
    if not file.name:
        return False, "File must have a name"
    
    if not file.name.lower().endswith(('.csv', '.xlsx', '.xls')):
        return False, "File must be CSV or Excel format"
    
    return True, ""

def validate_dataframe(df: pd.DataFrame, required_columns: Optional[List[str]] = None) -> Tuple[bool, str]:
    """
    Validate a pandas DataFrame.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names (optional)
        
    Returns:
        Tuple of (is_valid, message)
    """
    if df is None or df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing = set(required_columns) - set(df.columns)
        if missing:
            return False, f"Missing required columns: {', '.join(missing)}"
    
    return True, ""

def sanitize_html(value: str) -> str:
    """
    Sanitize HTML content.
    
    Args:
        value: String to sanitize
        
    Returns:
        Sanitized string
    """
    return re.sub(r'<[^>]*?>', '', value)

def sanitize_sql(value: str) -> str:
    """
    Sanitize SQL content.
    
    Args:
        value: String to sanitize
        
    Returns:
        Sanitized string
    """
    return re.sub(r'[\'";\-]', '', value)

def sanitize_filename(value: str) -> str:
    """
    Sanitize a filename.
    
    Args:
        value: Filename to sanitize
        
    Returns:
        Sanitized filename
    """
    return re.sub(r'[^\w\-_\.]', '_', value)