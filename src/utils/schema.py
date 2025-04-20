"""
Schema validation module for Watchdog AI.
Defines required data structure and validates uploaded files.
"""

from typing import Dict, List, Set, Tuple, Optional, Any
import pandas as pd
from dataclasses import dataclass
import logging
from datetime import datetime
from .errors import ValidationError
from .data_normalization import normalize_column_name, COLUMN_ALIASES

logger = logging.getLogger(__name__)

# Core schema definition
DATASET_SCHEMA = {
    'required_columns': {
        'gross': {
            'type': float,
            'aliases': ['gross', 'total_gross', 'front_gross', 'gross_profit'],
            'description': 'Total gross profit for the deal',
            'validation': lambda x: x is not None
        },
        'lead_source': {
            'type': str,
            'aliases': ['lead_source', 'leadsource', 'source', 'lead_type'],
            'description': 'Source of the lead',
            'validation': lambda x: x is not None and str(x).strip() != ''
        },
        'sale_date': {
            'type': 'datetime',
            'aliases': ['date', 'sale_date', 'transaction_date', 'deal_date'],
            'description': 'Date of the sale',
            'validation': lambda x: pd.notna(x)
        }
    },
    'optional_columns': {
        'vin': {
            'type': str,
            'aliases': ['vin', 'vin_number', 'vehicle_id'],
            'description': 'Vehicle Identification Number',
            'validation': lambda x: x is None or (isinstance(x, str) and len(x) == 17)
        },
        'sales_rep': {
            'type': str,
            'aliases': ['sales_rep', 'salesperson', 'rep_name', 'employee'],
            'description': 'Sales representative name',
            'validation': None  # No specific validation
        }
    }
}

@dataclass
class ValidationResult:
    """Stores the results of schema validation."""
    is_valid: bool
    missing_required: List[str]
    type_errors: Dict[str, List[str]]
    validation_errors: Dict[str, List[str]]
    column_mapping: Dict[str, str]
    timestamp: str = datetime.now().isoformat()

class DatasetSchema:
    """
    Defines and validates the schema for automotive dealership datasets.
    Handles column aliases, type validation, and data coercion.
    """
    
    def __init__(self, schema: Dict[str, Dict] = None):
        """
        Initialize the schema validator.
        
        Args:
            schema: Optional custom schema definition. If None, uses DATASET_SCHEMA.
        """
        self.schema = schema or DATASET_SCHEMA
        self._validate_schema_definition()
        logger.info("Initialized DatasetSchema with %d required and %d optional columns",
                   len(self.schema['required_columns']),
                   len(self.schema['optional_columns']))
    
    def _validate_schema_definition(self):
        """Validate the schema definition itself."""
        required_keys = {'type', 'aliases', 'description'}
        
        for section in ['required_columns', 'optional_columns']:
            if section not in self.schema:
                raise ValueError(f"Schema must contain '{section}'")
            
            for col, config in self.schema[section].items():
                missing = required_keys - set(config.keys())
                if missing:
                    raise ValueError(f"Column '{col}' missing required keys: {missing}")
    
    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate a DataFrame against the schema.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            ValidationResult object containing validation details
        """
        # Initialize validation results
        missing_required = []
        type_errors = {}
        validation_errors = {}
        column_mapping = {}
        
        # First pass: Check for required columns and build column mapping
        df_cols_lower = {col.lower(): col for col in df.columns}
        
        for col_name, config in self.schema['required_columns'].items():
            # Try to find the column using aliases
            found_col = None
            for alias in config['aliases']:
                if alias.lower() in df_cols_lower:
                    found_col = df_cols_lower[alias.lower()]
                    break
            
            if found_col:
                column_mapping[col_name] = found_col
            else:
                missing_required.append(col_name)
        
        # If missing required columns, return early
        if missing_required:
            return ValidationResult(
                is_valid=False,
                missing_required=missing_required,
                type_errors={},
                validation_errors={},
                column_mapping=column_mapping
            )
        
        # Second pass: Validate types and custom validation rules
        for col_name, config in {**self.schema['required_columns'], 
                               **self.schema['optional_columns']}.items():
            if col_name in column_mapping:
                df_col = column_mapping[col_name]
                
                # Type validation and coercion
                try:
                    if config['type'] == 'datetime':
                        df[df_col] = pd.to_datetime(df[df_col], errors='coerce')
                    else:
                        df[df_col] = df[df_col].astype(config['type'])
                except (ValueError, TypeError) as e:
                    type_errors[col_name] = [str(e)]
                
                # Custom validation if specified
                if config.get('validation'):
                    invalid_rows = ~df[df_col].apply(config['validation'])
                    if invalid_rows.any():
                        validation_errors[col_name] = [
                            f"Invalid values in rows: {invalid_rows.sum()}"
                        ]
        
        return ValidationResult(
            is_valid=not (missing_required or type_errors or validation_errors),
            missing_required=missing_required,
            type_errors=type_errors,
            validation_errors=validation_errors,
            column_mapping=column_mapping
        )
    
    def standardize_dataframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Standardize a DataFrame to match the schema.
        
        Args:
            df: DataFrame to standardize
            
        Returns:
            Tuple of (standardized DataFrame, column mapping dictionary)
        """
        # Validate first
        validation = self.validate(df)
        if not validation.is_valid:
            raise ValidationError(
                "Cannot standardize invalid DataFrame",
                details={
                    'missing_required': validation.missing_required,
                    'type_errors': validation.type_errors,
                    'validation_errors': validation.validation_errors
                }
            )
        
        # Create a copy to avoid modifying the original
        std_df = df.copy()
        
        # Rename columns to canonical names
        reverse_mapping = {v: k for k, v in validation.column_mapping.items()}
        std_df = std_df.rename(columns=reverse_mapping)
        
        # Ensure proper types
        for col_name, config in {**self.schema['required_columns'], 
                               **self.schema['optional_columns']}.items():
            if col_name in std_df.columns:
                if config['type'] == 'datetime':
                    std_df[col_name] = pd.to_datetime(std_df[col_name], errors='coerce')
                else:
                    std_df[col_name] = std_df[col_name].astype(config['type'])
        
        return std_df, validation.column_mapping
    
    def get_column_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get detailed information about all columns in the schema.
        
        Returns:
            Dictionary with column information
        """
        info = {}
        for section in ['required_columns', 'optional_columns']:
            for col_name, config in self.schema[section].items():
                info[col_name] = {
                    'type': config['type'],
                    'aliases': config['aliases'],
                    'description': config['description'],
                    'required': section == 'required_columns'
                }
        return info

# Create a global instance for convenience
default_schema = DatasetSchema()

def validate_dataframe(df: pd.DataFrame) -> ValidationResult:
    """
    Convenience function to validate a DataFrame using the default schema.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        ValidationResult object
    """
    return default_schema.validate(df)