"""
Schema validation module for Watchdog AI.
Defines required data structure and validates uploaded files.
"""

from typing import Dict, List, Set, Tuple, Optional
import pandas as pd
from dataclasses import dataclass

# Required sheets and their columns
REQUIRED_SHEETS = {
    'sales': {'required': True, 'description': 'Sales transaction data'},
    'inventory': {'required': False, 'description': 'Current inventory status'},
    'leads': {'required': False, 'description': 'Lead tracking data'}
}

# Required columns per sheet with aliases
REQUIRED_COLUMNS = {
    'sales': {
        # Core required columns
        'gross': ['gross', 'total_gross', 'front_gross', 'gross_profit'],
        'lead_source': ['lead_source', 'leadsource', 'source', 'lead_type'],
        # Optional columns with fallbacks
        'date': {
            'aliases': ['date', 'sale_date', 'transaction_date', 'deal_date'],
            'required': False,
            'fallback': 'created_at'  # Will create this if missing
        },
        'sales_rep': {
            'aliases': ['sales_rep', 'salesperson', 'rep_name', 'employee'],
            'required': False,
            'fallback': 'Unknown'  # Will use this value if missing
        },
        'vin': {
            'aliases': ['vin', 'vin_number', 'vehicle_id'],
            'required': False,
            'fallback': None  # Optional with no fallback
        }
    },
    'inventory': {
        'vin': ['vin', 'vin_number', 'vehicle_id'],
        'days_in_stock': ['days_in_stock', 'age', 'days_on_lot'],
        'price': ['price', 'list_price', 'asking_price', 'msrp']
    },
    'leads': {
        'date': ['date', 'lead_date', 'inquiry_date'],
        'source': ['source', 'lead_source', 'origin', 'channel'],
        'status': ['status', 'lead_status', 'state']
    }
}

@dataclass
class SchemaValidationError(Exception):
    """Custom error for schema validation failures."""
    message: str
    missing_sheets: List[str] = None
    missing_columns: Dict[str, List[str]] = None
    
    def __str__(self) -> str:
        details = []
        if self.missing_sheets:
            details.append(f"Missing required sheets: {', '.join(self.missing_sheets)}")
        if self.missing_columns:
            for sheet, cols in self.missing_columns.items():
                details.append(f"Sheet '{sheet}' missing columns: {', '.join(cols)}")
        return f"{self.message}\n" + "\n".join(details)

def find_matching_column(df: pd.DataFrame, aliases: List[str]) -> Optional[str]:
    """
    Find a column in the DataFrame that matches any of the provided aliases.
    
    Args:
        df: DataFrame to search
        aliases: List of possible column names
        
    Returns:
        Matching column name or None if not found
    """
    # Convert all column names to lowercase for case-insensitive matching
    df_cols_lower = {col.lower(): col for col in df.columns}
    
    # Try exact matches first
    for alias in aliases:
        if alias.lower() in df_cols_lower:
            return df_cols_lower[alias.lower()]
    
    # Try partial matches
    for alias in aliases:
        for col_lower, col in df_cols_lower.items():
            if alias.lower() in col_lower or col_lower in alias.lower():
                return col
    
    return None

def validate_sheet_schema(df: pd.DataFrame, sheet_name: str) -> Tuple[bool, Dict[str, str], List[str]]:
    """
    Validate a single sheet against its required schema.
    
    Args:
        df: DataFrame to validate
        sheet_name: Name of the sheet for looking up requirements
        
    Returns:
        Tuple containing:
        - success: Boolean indicating if validation passed
        - column_map: Dictionary mapping required columns to found columns
        - missing: List of missing required columns
    """
    if sheet_name not in REQUIRED_COLUMNS:
        return True, {}, []  # No requirements defined
    
    column_map = {}
    missing = []
    
    # Check each required column
    for required_col, config in REQUIRED_COLUMNS[sheet_name].items():
        if isinstance(config, list):  # Simple required column with aliases
            aliases = config
            found_col = find_matching_column(df, aliases)
            if found_col:
                column_map[required_col] = found_col
            else:
                missing.append(required_col)
        elif isinstance(config, dict):  # Column with additional configuration
            aliases = config['aliases']
            found_col = find_matching_column(df, aliases)
            if found_col:
                column_map[required_col] = found_col
            elif config.get('required', True):  # Only add to missing if required
                missing.append(required_col)
            elif config.get('fallback') is not None:
                # Add fallback column if specified
                if isinstance(config['fallback'], str):
                    df[required_col] = config['fallback']
                    column_map[required_col] = required_col
    
    # For sales sheet, add created_at if date is missing
    if sheet_name == 'sales' and 'date' not in column_map:
        df['date'] = pd.Timestamp.now()
        column_map['date'] = 'date'
    
    return len(missing) == 0, column_map, missing

def validate_workbook_schema(sheets: Dict[str, pd.DataFrame]) -> Tuple[bool, Dict[str, Dict[str, str]], Dict[str, List[str]]]:
    """
    Validate all sheets in a workbook against their required schemas.
    
    Args:
        sheets: Dictionary mapping sheet names to DataFrames
        
    Returns:
        Tuple containing:
        - success: Boolean indicating if validation passed
        - column_maps: Dictionary mapping sheets to their column mappings
        - missing_columns: Dictionary mapping sheets to their missing columns
    """
    # Check for required sheets
    missing_required = [
        name for name, info in REQUIRED_SHEETS.items()
        if info['required'] and name not in sheets
    ]
    
    if missing_required:
        raise SchemaValidationError(
            "Missing required sheets",
            missing_sheets=missing_required
        )
    
    # Validate each sheet
    success = True
    column_maps = {}
    missing_columns = {}
    
    for sheet_name, df in sheets.items():
        if sheet_name in REQUIRED_COLUMNS:
            sheet_success, sheet_map, sheet_missing = validate_sheet_schema(df, sheet_name)
            success = success and sheet_success
            column_maps[sheet_name] = sheet_map
            if sheet_missing:
                missing_columns[sheet_name] = sheet_missing
    
    if not success:
        raise SchemaValidationError(
            "Missing required columns",
            missing_columns=missing_columns
        )
    
    return success, column_maps, missing_columns

def load_and_validate_file(file_path: str) -> Dict[str, pd.DataFrame]:
    """
    Load an Excel workbook or CSV file and validate its schema.
    
    Args:
        file_path: Path to the file to load
        
    Returns:
        Dictionary mapping sheet names to validated DataFrames
    """
    sheets = {}
    
    try:
        # Try loading as Excel first
        if file_path.endswith(('.xlsx', '.xls')):
            xl = pd.ExcelFile(file_path)
            for sheet_name in xl.sheet_names:
                sheets[sheet_name] = pd.read_excel(xl, sheet_name)
        else:
            # Assume CSV
            df = pd.read_csv(file_path)
            sheets['sales'] = df  # Treat CSV as sales sheet
        
        # Validate schema
        success, column_maps, missing = validate_workbook_schema(sheets)
        
        # Rename columns according to schema
        for sheet_name, mapping in column_maps.items():
            sheets[sheet_name] = sheets[sheet_name].rename(columns={v: k for k, v in mapping.items()})
        
        return sheets
        
    except SchemaValidationError as e:
        raise  # Re-raise schema validation errors
    except Exception as e:
        raise SchemaValidationError(f"Error loading file: {str(e)}")