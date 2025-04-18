"""
Input validation and sanitization for Watchdog AI.
"""

import re
import os
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union, Set
from pathlib import Path
from datetime import datetime
import json
import logging

from .errors import ValidationError
from .logging_config import get_logger

logger = get_logger(__name__)

# --- Added Canonical Columns and Aliases ---
CANONICAL_COLUMNS = [
    "LeadSource", "LeadSource Category", "DealNumber", "SellingPrice",
    "FrontGross", "BackGross", "Total Gross", "SalesRepName", "SplitSalesRep",
    "VehicleYear", "VehicleMake", "VehicleModel", "VehicleStockNumber", "VehicleVIN",
    "Sale_Date" # Adding Sale_Date based on test data
]

COLUMN_ALIASES = {
    # LeadSource
    "leadsource": "LeadSource",
    "lead source": "LeadSource",
    "lead_source": "LeadSource",
    # LeadSource Category
    "leadsource category": "LeadSource Category",
    "lead_source_category": "LeadSource Category",
    "category": "LeadSource Category",
    # DealNumber
    "dealnumber": "DealNumber",
    "deal number": "DealNumber",
    "deal_number": "DealNumber",
    "deal id": "DealNumber",
    # SellingPrice
    "sellingprice": "SellingPrice",
    "selling price": "SellingPrice",
    "sale price": "SellingPrice",
    "sale_price": "SellingPrice",
    "price": "SellingPrice",
    # FrontGross
    "frontgross": "FrontGross",
    "front gross": "FrontGross",
    "front_gross": "FrontGross",
    "frontend gross": "FrontGross",
    # BackGross
    "backgross": "BackGross",
    "back gross": "BackGross",
    "back_gross": "BackGross",
    "backend gross": "BackGross",
    "f&i gross": "BackGross",
    # Total Gross
    "total gross": "Total Gross",
    "total_gross": "Total Gross",
    "gross": "Total Gross",
    "totalgross": "Total Gross", # Added based on test_data.csv
    # SalesRepName
    "salesrepname": "SalesRepName",
    "sales rep name": "SalesRepName",
    "sales_rep_name": "SalesRepName",
    "salesperson": "SalesRepName",
    "sales rep": "SalesRepName",
    "sales_rep": "SalesRepName",
    # SplitSalesRep
    "splitsalesrep": "SplitSalesRep",
    "split sales rep": "SplitSalesRep",
    "split_sales_rep": "SplitSalesRep",
    "split salesperson": "SplitSalesRep",
    # VehicleYear
    "vehicleyear": "VehicleYear",
    "vehicle year": "VehicleYear",
    "vehicle_year": "VehicleYear",
    "year": "VehicleYear",
    # VehicleMake
    "vehiclemake": "VehicleMake",
    "vehicle make": "VehicleMake",
    "vehicle_make": "VehicleMake",
    "make": "VehicleMake",
    # VehicleModel
    "vehiclemodel": "VehicleModel",
    "vehicle model": "VehicleModel",
    "vehicle_model": "VehicleModel",
    "model": "VehicleModel",
    # VehicleStockNumber
    "vehiclestocknumber": "VehicleStockNumber",
    "vehicle stock number": "VehicleStockNumber",
    "vehicle_stock_number": "VehicleStockNumber",
    "stock number": "VehicleStockNumber",
    "stock_number": "VehicleStockNumber",
    "stock #": "VehicleStockNumber",
    # VehicleVIN
    "vehiclevin": "VehicleVIN",
    "vehicle vin": "VehicleVIN",
    "vehicle_vin": "VehicleVIN",
    "vin": "VehicleVIN",
    # Sale_Date
    "sale_date": "Sale_Date",
    "sale date": "Sale_Date",
    "saledate": "Sale_Date",
    "date": "Sale_Date",
    "deal date": "Sale_Date",
}

def normalize_column_name(name: str) -> str:
    """Convert column name to a standardized format for matching aliases."""
    if not isinstance(name, str):
        return "" # Return empty string for non-string headers
    return re.sub(r'[^a-z0-9]', '', name.lower().strip())

def normalize_and_alias_columns(columns: List[str]) -> Dict[str, str]:
    """Normalize column names and map them to canonical names using aliases."""
    normalized_map = {}
    processed_normalized = set() # Track normalized names already processed

    for original_col in columns:
        normalized_col = normalize_column_name(original_col)
        if not normalized_col or normalized_col in processed_normalized:
            continue # Skip empty, non-string, or duplicate normalized names

        canonical_name = COLUMN_ALIASES.get(normalized_col)
        if canonical_name:
            # Avoid overwriting if multiple original cols map to the same canonical
            if canonical_name not in normalized_map.values():
                 normalized_map[original_col] = canonical_name
            processed_normalized.add(normalized_col)
        elif original_col in CANONICAL_COLUMNS:
             # If the original name is already canonical and not aliased
            if original_col not in normalized_map.values():
                normalized_map[original_col] = original_col
            processed_normalized.add(normalized_col)
            
    return normalized_map

class InputValidator:
    """Validates and sanitizes user input."""
    
    def __init__(self):
        """Initialize the input validator."""
        # Common patterns for validation
        self.patterns = {
            'vin': r'^[A-HJ-NPR-Z0-9]{17}$',
            'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'phone': r'^\+?1?\d{9,15}$',
            'date': r'^\d{4}-\d{2}-\d{2}$',
            'currency': r'^\$?\d+(\.\d{2})?$'
        }
        
        # Maximum file size (100MB)
        self.max_file_size = 100 * 1024 * 1024
    
    def validate_string(self, value: str, pattern: str = None, 
                       min_length: int = None, max_length: int = None) -> str:
        """
        Validate and sanitize a string value.
        
        Args:
            value: String to validate
            pattern: Optional regex pattern to match
            min_length: Optional minimum length
            max_length: Optional maximum length
            
        Returns:
            Sanitized string value
            
        Raises:
            ValidationError if validation fails
        """
        if not isinstance(value, str):
            raise ValidationError(
                "Invalid string value",
                details={"value": str(value), "type": str(type(value))}
            )
        
        # Strip whitespace
        value = value.strip()
        
        # Check length
        if min_length is not None and len(value) < min_length:
            raise ValidationError(
                f"String too short (minimum {min_length} characters)",
                details={"value": value, "length": len(value)}
            )
        
        if max_length is not None and len(value) > max_length:
            raise ValidationError(
                f"String too long (maximum {max_length} characters)",
                details={"value": value, "length": len(value)}
            )
        
        # Check pattern
        if pattern:
            if pattern in self.patterns:
                pattern = self.patterns[pattern]
            if not re.match(pattern, value):
                raise ValidationError(
                    "String does not match required pattern",
                    details={"value": value, "pattern": pattern}
                )
        
        return value
    
    def validate_number(self, value: Union[int, float], 
                       min_value: Union[int, float] = None,
                       max_value: Union[int, float] = None) -> Union[int, float]:
        """
        Validate a numeric value.
        
        Args:
            value: Number to validate
            min_value: Optional minimum value
            max_value: Optional maximum value
            
        Returns:
            Validated number
            
        Raises:
            ValidationError if validation fails
        """
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValidationError(
                "Invalid numeric value",
                details={"value": str(value)}
            )
        
        if min_value is not None and value < min_value:
            raise ValidationError(
                f"Number too small (minimum {min_value})",
                details={"value": value}
            )
        
        if max_value is not None and value > max_value:
            raise ValidationError(
                f"Number too large (maximum {max_value})",
                details={"value": value}
            )
        
        return value
    
    def validate_date(self, value: str) -> str:
        """
        Validate a date string.
        
        Args:
            value: Date string to validate
            
        Returns:
            Validated date string
            
        Raises:
            ValidationError if validation fails
        """
        try:
            # Try to parse as ISO format
            datetime.fromisoformat(value)
            return value
        except ValueError:
            raise ValidationError(
                "Invalid date format (use YYYY-MM-DD)",
                details={"value": value}
            )
    
    def validate_file(self, file: Any) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate an uploaded file.
        
        Args:
            file: File-like object to validate
            
        Returns:
            Tuple of (is_valid, details)
            
        Raises:
            ValidationError if validation fails
        """
        if not hasattr(file, 'read'):
            raise ValidationError(
                "Invalid file object",
                details={"type": str(type(file))}
            )
        
        # Check file size
        file.seek(0, 2)  # Seek to end
        size = file.tell()
        file.seek(0)  # Reset position
        
        if size > self.max_file_size:
            raise ValidationError(
                f"File too large (maximum {self.max_file_size/1024/1024}MB)",
                details={"size": size}
            )
        
        # Check file type
        content_start = file.read(2048)
        file.seek(0)  # Reset position
        
        # Simple file type check based on content
        if content_start.startswith(b'PK'):
            mime = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        elif content_start.startswith(b'\xD0\xCF\x11\xE0'):
            mime = 'application/vnd.ms-excel'
        else:
            # Assume CSV if it looks like text
            try:
                content_start.decode('utf-8')
                mime = 'text/csv'
            except UnicodeDecodeError:
                raise ValidationError(
                    "Invalid file type",
                    details={"mime": "unknown"}
                )
        
        # Generate file hash
        import hashlib
        sha256 = hashlib.sha256()
        file.seek(0)
        for chunk in iter(lambda: file.read(8192), b''):
            sha256.update(chunk)
        file_hash = sha256.hexdigest()
        file.seek(0)  # Reset position
        
        return True, {
            "size": size,
            "mime": mime,
            "hash": file_hash
        }
    
    def validate_dataframe(self, df: pd.DataFrame, 
                          required_columns: List[str] = None,
                          column_types: Dict[str, str] = None) -> pd.DataFrame:
        """
        Validate a pandas DataFrame, including schema normalization.
        
        Args:
            df: DataFrame to validate
            required_columns: Optional list of required canonical column names
            column_types: Optional dict mapping canonical column names to expected types
            
        Returns:
            Validated DataFrame with canonical column names
            
        Raises:
            ValidationError if validation fails
        """
        if df is None or df.empty:
            raise ValidationError("DataFrame is empty")
        
        # --- Modified Section ---
        original_columns = list(df.columns)
        rename_mapping = normalize_and_alias_columns(original_columns)
        
        # Rename columns in the DataFrame
        df = df.rename(columns=rename_mapping)
        
        # Get the set of canonical columns present after renaming
        found_canonical_columns = set(df.columns) & set(CANONICAL_COLUMNS)

        # Check required columns against the canonical columns found
        if required_columns:
            required_set = set(required_columns)
            missing = required_set - found_canonical_columns
            if missing:
                raise ValidationError(
                    "Missing required columns after normalization",
                    details={
                        "missing_canonical": sorted(list(missing)),
                        "found_canonical": sorted(list(found_canonical_columns)),
                        "original_columns": original_columns
                    }
                )
        # --- End Modified Section ---

        # Check column types (using canonical names)
        if column_types:
            for col, expected_type in column_types.items():
                if col not in df.columns: # Now checks against potentially renamed columns
                    continue
                    
                actual_type = df[col].dtype.name
                if actual_type != expected_type:
                    try:
                        # Attempt to convert
                        df[col] = df[col].astype(expected_type)
                    except:
                        raise ValidationError(
                            f"Invalid type for column {col}",
                            details={
                                "column": col,
                                "expected": expected_type,
                                "actual": actual_type
                            }
                        )
        
        return df
    
    def sanitize_html(self, value: str) -> str:
        """
        Sanitize HTML content.
        
        Args:
            value: HTML string to sanitize
            
        Returns:
            Sanitized HTML string
        """
        import bleach
        
        # Define allowed tags and attributes
        allowed_tags = [
            'p', 'br', 'strong', 'em', 'u', 'h1', 'h2', 'h3',
            'h4', 'h5', 'h6', 'ul', 'ol', 'li', 'span', 'div',
            'table', 'thead', 'tbody', 'tr', 'th', 'td'
        ]
        
        allowed_attrs = {
            '*': ['class', 'style'],
            'a': ['href', 'title'],
            'img': ['src', 'alt', 'title']
        }
        
        # Sanitize the HTML
        return bleach.clean(
            value,
            tags=allowed_tags,
            attributes=allowed_attrs,
            strip=True
        )
    
    def sanitize_sql(self, value: str) -> str:
        """
        Sanitize a string for SQL injection prevention.
        
        Args:
            value: String to sanitize
            
        Returns:
            Sanitized string
        """
        # Remove common SQL injection patterns
        sql_patterns = [
            r'--',           # SQL comment
            r';',            # Statement terminator
            r'\/\*|\*\/',   # Block comments
            r'union\s+all',  # UNION ALL
            r'union\s+select', # UNION SELECT
            r'drop\s+table', # DROP TABLE
            r'delete\s+from', # DELETE FROM
            r'insert\s+into', # INSERT INTO
            r'xp_cmdshell',  # SQL Server command shell
        ]
        
        sanitized = value
        for pattern in sql_patterns:
            sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
        
        return sanitized
    
    def sanitize_filename(self, filename: str) -> str:
        """
        Sanitize a filename.
        
        Args:
            filename: Filename to sanitize
            
        Returns:
            Sanitized filename
        """
        # Remove or replace potentially dangerous characters
        filename = re.sub(r'[^\w\-_\. ]', '', filename)
        
        # Remove spaces
        filename = filename.replace(' ', '_')
        
        # Ensure it's not too long
        if len(filename) > 255:
            name, ext = os.path.splitext(filename)
            filename = name[:255-len(ext)] + ext
        
        return filename


# Global validator instance
validator = InputValidator()