"""
Column finder utility for flexible column name matching.
Uses RapidFuzz for improved performance and accuracy.
"""

from typing import List, Dict, Optional, Tuple
import re
import hashlib
import streamlit as st
from rapidfuzz import fuzz, process
import pandas as pd

# Common aliases for different metric types
METRIC_ALIASES = {
    "gross": ["gross", "profit", "gp", "total_gross", "grossprofit", "gross profit", "total gross"],
    "price": ["price", "amount", "sale_price", "saleprice", "sale price", "list_price", "listprice"],
    "revenue": ["revenue", "rev", "total_revenue", "totalrevenue", "total revenue"],
    "cost": ["cost", "expense", "total_cost", "totalcost", "total cost"]
}

# Common aliases for category types
CATEGORY_ALIASES = {
    "rep": ["rep", "sales rep", "salesperson", "sales_rep", "salesrep", "representative", "sales person"],
    "source": [
        "source", "lead source", "leadsource", "lead_source", "channel", "origin",
        "lead_type", "leadtype", "lead type", "lead_channel", "leadchannel", "lead channel",
        "source_type", "sourcetype", "source type", "lead_origin", "leadorigin", "lead origin"
    ],
    "make": ["make", "manufacturer", "brand", "car_make", "carmake"],
    "model": ["model", "car_model", "carmodel", "vehicle_model", "vehiclemodel"]
}

# Common lead source variations
LEAD_SOURCE_VARIATIONS = {
    "cargurus": ["cargurus", "car gurus", "car guru", "car_gurus", "carguru"],
    "autotrader": ["autotrader", "auto trader", "auto_trader"],
    "cars.com": ["cars.com", "cars com", "carscom", "cars_com"],
    "facebook": ["facebook", "fb", "face book", "facebook marketplace"],
    "dealer_website": ["dealer website", "dealer site", "dealerwebsite", "dealer_website"],
    "walk_in": ["walk in", "walk-in", "walkin", "walk_in", "walk up"]
}

def _compute_hash(items: List[str]) -> str:
    """
    Compute a stable hash for a list of strings.
    
    Args:
        items: List of strings to hash
        
    Returns:
        Hash string
    """
    # Sort to ensure stable hash
    sorted_items = sorted(items)
    # Join with delimiter unlikely to appear in column names
    combined = "||".join(sorted_items)
    # Create hash
    return hashlib.sha256(combined.encode()).hexdigest()

def _get_cache_key(prompt: str, columns: List[str]) -> str:
    """
    Generate a cache key from prompt and columns.
    
    Args:
        prompt: Search prompt
        columns: List of column names
        
    Returns:
        Cache key string
    """
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:8]
    columns_hash = _compute_hash(columns)[:8]
    return f"col_finder_{prompt_hash}_{columns_hash}"

def normalize_column_name(name: str) -> str:
    """
    Normalize a column name for comparison.
    
    Args:
        name: Column name to normalize
        
    Returns:
        Normalized column name
    """
    # Convert to lowercase and remove special characters
    normalized = name.lower()
    normalized = re.sub(r'[^a-z0-9]', '', normalized)
    return normalized

def find_column(df_columns: List[str], candidates: List[str], alias_map: Optional[Dict[str, List[str]]] = None) -> Optional[str]:
    """
    Find the best matching column using RapidFuzz with caching.
    
    Args:
        df_columns: List of available DataFrame column names
        candidates: List of possible column names to match
        alias_map: Optional dictionary mapping canonical names to aliases
        
    Returns:
        Best matching column name or None if no match found
    """
    # Initialize cache in session state if needed
    if 'column_finder_cache' not in st.session_state:
        st.session_state.column_finder_cache = {}
    
    # Create cache key from candidates and columns
    cache_key = _get_cache_key("|".join(candidates), df_columns)
    
    # Check cache first
    if cache_key in st.session_state.column_finder_cache:
        return st.session_state.column_finder_cache[cache_key]
    
    # Normalize all DataFrame column names
    normalized_columns = {normalize_column_name(col): col for col in df_columns}
    
    # First try exact matches
    for candidate in candidates:
        if candidate in df_columns:
            st.session_state.column_finder_cache[cache_key] = candidate
            return candidate
        
        normalized_candidate = normalize_column_name(candidate)
        if normalized_candidate in normalized_columns:
            result = normalized_columns[normalized_candidate]
            st.session_state.column_finder_cache[cache_key] = result
            return result
    
    # Then try aliases if provided
    if alias_map:
        for canonical, aliases in alias_map.items():
            normalized_aliases = [normalize_column_name(alias) for alias in aliases]
            for col in df_columns:
                normalized_col = normalize_column_name(col)
                if normalized_col in normalized_aliases:
                    st.session_state.column_finder_cache[cache_key] = col
                    return col
    
    # Finally try fuzzy matching with RapidFuzz
    best_match = None
    best_score = 0
    
    for candidate in candidates:
        # Try token set ratio for each column
        for col in df_columns:
            # Use token set ratio which handles word order and partial matches well
            score = fuzz.token_set_ratio(candidate.lower(), col.lower())
            
            # Only consider matches above 80% threshold
            if score > 80 and score > best_score:
                best_match = col
                best_score = score
    
    # Cache and return the result
    st.session_state.column_finder_cache[cache_key] = best_match
    return best_match

def find_metric_column(df_columns: List[str], metric_type: str) -> Optional[str]:
    """
    Find a metric column of a specific type.
    
    Args:
        df_columns: List of DataFrame column names
        metric_type: Type of metric to find (e.g., 'gross', 'price')
        
    Returns:
        Matching column name or None
    """
    if metric_type not in METRIC_ALIASES:
        return None
    return find_column(df_columns, METRIC_ALIASES[metric_type], METRIC_ALIASES)

def find_category_column(df_columns: List[str], category_type: str) -> Optional[str]:
    """
    Find a category column of a specific type.
    
    Args:
        df_columns: List of DataFrame column names
        category_type: Type of category to find (e.g., 'rep', 'source')
        
    Returns:
        Matching column name or None
    """
    if category_type not in CATEGORY_ALIASES:
        return None
    return find_column(df_columns, CATEGORY_ALIASES[category_type], CATEGORY_ALIASES)

def normalize_lead_source(source: str) -> str:
    """
    Normalize a lead source value to a standard form.
    
    Args:
        source: Lead source value to normalize
        
    Returns:
        Normalized lead source value
    """
    if pd.isna(source):
        return "Unknown"
    
    source_lower = str(source).lower().strip()
    
    # Check each variation group
    for canonical, variations in LEAD_SOURCE_VARIATIONS.items():
        if any(variation in source_lower for variation in variations):
            return canonical.title().replace('_', ' ')
    
    # If no match found, title case and clean up
    return source.title().replace('_', ' ')