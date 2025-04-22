"""
Column utilities for data analysis.
"""

from typing import Optional, List
import pandas as pd
import hashlib
import streamlit as st

def normalize_column_name(name: str) -> str:
    """Normalize a column name by removing special characters and converting to lowercase."""
    # Remove currency symbols and parentheses
    name = name.replace('€', '').replace('$', '').replace('(', '').replace(')', '')
    # Remove spaces and convert to lowercase
    return ''.join(name.lower().split())

def _compute_hash(items: List[str]) -> str:
    """Compute a stable hash for a list of strings."""
    # Sort to ensure consistent hash regardless of order
    sorted_items = sorted(items)
    # Create hash
    hash_obj = hashlib.sha256()
    for item in sorted_items:
        hash_obj.update(item.encode())
    return hash_obj.hexdigest()

def _get_cache_key(prompt: str, columns: List[str]) -> str:
    """Generate a cache key for column finding results."""
    return _compute_hash([prompt.lower()] + [col.lower() for col in columns])

def find_column(columns: List[str], hints: List[str], threshold: float = 0.8) -> Optional[str]:
    """Find a column that best matches the given hints."""
    # Check cache first
    cache_key = _get_cache_key(hints[0], columns)
    if 'column_finder_cache' in st.session_state:
        if cache_key in st.session_state.column_finder_cache:
            return st.session_state.column_finder_cache[cache_key]
    
    # Initialize cache if needed
    if 'column_finder_cache' not in st.session_state:
        st.session_state.column_finder_cache = {}
    
    # Try exact matches first
    for hint in hints:
        for col in columns:
            if hint.lower() in col.lower():
                st.session_state.column_finder_cache[cache_key] = col
                return col
    
    # No match found
    st.session_state.column_finder_cache[cache_key] = None
    return None

def find_metric_column(columns: List[str], hint: str) -> Optional[str]:
    """Find a metric column that matches the hint."""
    return find_column(columns, [hint])

def find_category_column(columns: List[str], hint: str) -> Optional[str]:
    """Find a category column that matches the hint."""
    return find_column(columns, [hint])