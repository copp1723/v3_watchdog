import pandas as pd
import numpy as np
import re
import logging

logger = logging.getLogger(__name__)

def is_all_sales_dataset(df: pd.DataFrame) -> bool:
    """
    Determines if a DataFrame represents a sales dataset by looking for common sales-related columns.
    
    Args:
        df: The DataFrame to check
        
    Returns:
        bool: True if the DataFrame appears to be a sales dataset
    """
    # Common sales-related column patterns
    sales_patterns = [
        r'sale', r'price', r'cost', r'amount', r'revenue', r'profit', 
        r'customer', r'client', r'lead', r'source', r'rep', r'agent'
    ]
    
    # Check if any column matches sales patterns
    for col in df.columns:
        col_lower = col.lower()
        if any(re.search(pattern, col_lower) for pattern in sales_patterns):
            logger.debug(f"Validated as sales dataset via column: {col}")
            return True
    
    return False

def find_column(df: pd.DataFrame, column_name: str) -> str:
    """
    Finds a column in a DataFrame that matches the given name, accounting for case and common variations.
    
    Args:
        df: The DataFrame to search
        column_name: The column name to find
        
    Returns:
        str: The actual column name if found, otherwise None
    """
    if not column_name:
        return None
        
    # Direct match
    if column_name in df.columns:
        return column_name
        
    # Case-insensitive match
    column_lower = column_name.lower()
    for col in df.columns:
        if col.lower() == column_lower:
            return col
            
    # Partial match
    for col in df.columns:
        if column_lower in col.lower() or col.lower() in column_lower:
            return col
            
    return None

def expand_variants(text: str) -> list:
    """
    Expands a text string into common variants to help with column name matching.
    
    Args:
        text: The text to expand
        
    Returns:
        list: A list of possible variants
    """
    if not text:
        return []
        
    variants = [text]
    
    # Common replacements
    replacements = {
        'price': ['cost', 'amount', 'value'],
        'cost': ['price', 'amount', 'value'],
        'amount': ['price', 'cost', 'value'],
        'value': ['price', 'cost', 'amount'],
        'sale': ['transaction', 'deal'],
        'transaction': ['sale', 'deal'],
        'deal': ['sale', 'transaction'],
        'customer': ['client', 'buyer'],
        'client': ['customer', 'buyer'],
        'buyer': ['customer', 'client'],
        'rep': ['representative', 'agent', 'salesperson'],
        'representative': ['rep', 'agent', 'salesperson'],
        'agent': ['rep', 'representative', 'salesperson'],
        'salesperson': ['rep', 'representative', 'agent'],
        'source': ['origin', 'lead_source'],
        'origin': ['source', 'lead_source'],
        'lead_source': ['source', 'origin'],
        'date': ['day', 'time'],
        'day': ['date', 'time'],
        'time': ['date', 'day'],
        'make': ['brand', 'manufacturer'],
        'brand': ['make', 'manufacturer'],
        'manufacturer': ['make', 'brand'],
        'model': ['type', 'variant'],
        'type': ['model', 'variant'],
        'variant': ['model', 'type'],
        'year': ['yr', 'y'],
        'yr': ['year', 'y'],
        'y': ['year', 'yr'],
        'profit': ['margin', 'earnings'],
        'margin': ['profit', 'earnings'],
        'earnings': ['profit', 'margin'],
        'days': ['d', 'day'],
        'd': ['days', 'day'],
        'day': ['days', 'd'],
        'close': ['closing', 'closed'],
        'closing': ['close', 'closed'],
        'closed': ['close', 'closing']
    }
    
    # Generate variants based on replacements
    for original, alternatives in replacements.items():
        if original in text.lower():
            for alt in alternatives:
                variant = text.lower().replace(original, alt)
                variants.append(variant)
                # Also add with underscore
                variants.append(variant.replace(' ', '_'))
                # Also add with camelCase
                variants.append(''.join(word.capitalize() for word in variant.split('_')))
    
    # Add common prefixes/suffixes
    prefixes = ['total_', 'avg_', 'average_', 'mean_', 'sum_', 'count_', 'max_', 'min_']
    suffixes = ['_total', '_avg', '_average', '_mean', '_sum', '_count', '_max', '_min']
    
    for prefix in prefixes:
        variants.append(prefix + text.lower())
        variants.append(prefix + text.lower().replace(' ', '_'))
    
    for suffix in suffixes:
        variants.append(text.lower() + suffix)
        variants.append(text.lower().replace(' ', '_') + suffix)
    
    return list(set(variants))  # Remove duplicates

def validate_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure numeric columns are properly typed."""
    df = df.copy()  # Avoid modifying original
    
    # Common numeric column patterns
    numeric_patterns = [
        r'price', r'cost', r'profit', r'revenue', r'sales', r'amount',
        r'quantity', r'count', r'number', r'total', r'average', r'avg',
        r'days', r'time', r'duration', r'age', r'score', r'rating'
    ]
    
    for col in df.columns:
        # Check if column name matches any numeric pattern
        if any(re.search(pattern, col.lower()) for pattern in numeric_patterns):
            if not pd.api.types.is_numeric_dtype(df[col]):
                try:
                    # First try direct conversion
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # If that fails, try cleaning the data
                    if df[col].isna().any():
                        # Remove currency symbols and commas
                        cleaned = df[col].astype(str).str.replace('$', '', regex=False)
                        cleaned = cleaned.str.replace(',', '', regex=False)
                        cleaned = cleaned.str.replace(' ', '', regex=False)
                        
                        # Remove any remaining non-numeric characters except decimal points and minus signs
                        cleaned = cleaned.str.replace(r'[^\d.-]', '', regex=True)
                        
                        # Convert to numeric
                        df[col] = pd.to_numeric(cleaned, errors='coerce')
                        
                        # Log any remaining non-numeric values
                        if df[col].isna().any():
                            non_numeric = df[df[col].isna()][col].unique()
                            logger.warning(f"Could not convert {len(non_numeric)} values in column {col} to numeric: {non_numeric[:5]}")
                    
                    logger.info(f"Successfully converted column {col} to numeric")
                except Exception as e:
                    logger.error(f"Error converting column {col} to numeric: {str(e)}")
    
    # Special handling for known numeric columns
    known_numeric_cols = {
        'sold_price': ['price', 'sale_price', 'selling_price'],
        'profit': ['profit', 'gross_profit', 'net_profit'],
        'days_to_close': ['days_to_close', 'close_time', 'days_to_sale'],
        'close_time': ['close_time', 'days_to_close', 'days_to_sale']
    }
    
    for target_col, variants in known_numeric_cols.items():
        for variant in variants:
            if variant in df.columns:
                try:
                    # Clean and convert
                    cleaned = df[variant].astype(str).str.replace('$', '', regex=False)
                    cleaned = cleaned.str.replace(',', '', regex=False)
                    cleaned = cleaned.str.replace(' ', '', regex=False)
                    cleaned = cleaned.str.replace(r'[^\d.-]', '', regex=True)
                    
                    df[variant] = pd.to_numeric(cleaned, errors='coerce')
                    logger.info(f"Successfully converted known numeric column {variant}")
                except Exception as e:
                    logger.error(f"Error converting known numeric column {variant}: {str(e)}")
    
    return df 