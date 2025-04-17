"""
Data normalization utilities for Watchdog AI.
Handles column name standardization and data format normalization.
"""

import pandas as pd
import re
from typing import Dict, List, Set

# Define comprehensive column name mappings
COLUMN_ALIASES = {
    'SaleDate': ['sale_date', 'saledate', 'date', 'Sale_Date', 'SALE_DATE', 'SaleDate', 'sale date', 'Date', 'Sale Date', 'saleDate'],
    'SalePrice': ['sale_price', 'saleprice', 'price', 'Sale_Price', 'SALE_PRICE', 'SalePrice', 'sale price', 'Price', 'Sale Price', 'salePrice'],
    'VIN': ['vin', 'vehicle_vin', 'vehiclevin', 'VehicleVIN', 'VEHICLE_VIN', 'Vehicle_VIN', 'vehicle vin', 'VIN', 'Vehicle VIN', 'vehicleVIN'],
    'TotalGross': ['total_gross', 'totalgross', 'gross', 'Total_Gross', 'TOTAL_GROSS', 'Gross_Profit', 'gross profit', 'Gross', 'Total Gross', 'totalGross', 'Gross Profit'],
    'LeadSource': ['lead_source', 'leadsource', 'source', 'Lead_Source', 'LEAD_SOURCE', 'LeadSource', 'lead source', 'Source', 'Lead Source', 'leadSource'],
    'DealNumber': ['deal_number', 'dealnumber', 'deal', 'Deal_Number', 'DEAL_NUMBER', 'DealNumber', 'deal number', 'Deal', 'dealNumber'],
}

def normalize_column_name(column: str) -> str:
    """
    Normalize a single column name by removing spaces, underscores,
    and converting to standard format.
    
    Args:
        column: Original column name
        
    Returns:
        Normalized column name
    """
    # Convert to string and clean
    clean_name = str(column).strip()
    
    # First, check for exact matches in the aliases
    for standard_name, aliases in COLUMN_ALIASES.items():
        if clean_name in aliases or clean_name == standard_name:
            return standard_name
    
    # If no exact match, normalize the column name (remove spaces, underscores, lowercase)
    normalized = clean_name.lower().replace(' ', '').replace('_', '').replace('-', '')
    
    # Check if the normalized version matches any canonical name when normalized
    for standard_name, aliases in COLUMN_ALIASES.items():
        normalized_standard = standard_name.lower().replace(' ', '').replace('_', '')
        if normalized == normalized_standard:
            return standard_name
        
        # Check against normalized aliases too
        normalized_aliases = [alias.lower().replace(' ', '').replace('_', '') for alias in aliases]
        if normalized in normalized_aliases:
            return standard_name
    
    # If no match found, return the original column cleaned of spaces and special chars
    # But preserve the original case
    return clean_name.replace(' ', '').replace('_', '')

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize all column names in a DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with normalized column names
    """
    # Create a mapping of old to new column names
    column_mapping = {col: normalize_column_name(col) for col in df.columns}
    
    # Rename columns
    df = df.rename(columns=column_mapping)
    
    return df

def get_column_mapping(columns: List[str]) -> Dict[str, str]:
    """
    Get mapping of original column names to normalized names.
    
    Args:
        columns: List of original column names
        
    Returns:
        Dict mapping original names to normalized names
    """
    return {col: normalize_column_name(col) for col in columns}

def get_supported_aliases() -> Dict[str, List[str]]:
    """
    Get dictionary of all supported column name aliases.
    
    Returns:
        Dict mapping standard names to lists of supported aliases
    """
    return COLUMN_ALIASES