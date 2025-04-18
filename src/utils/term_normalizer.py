"""
Term normalization module for Watchdog AI.

Provides functionality for normalizing common term variations based on YAML rules.
"""

import yaml
import os
import pandas as pd
import logging
import sentry_sdk
from typing import Dict, List, Any, Optional, Union
from functools import lru_cache
from rapidfuzz import fuzz

logger = logging.getLogger(__name__)

def load_normalization_rules(config_path="config/normalization_rules.yml"):
    with open(config_path, "r") as f:
        rules = yaml.safe_load(f)
    fuzzy_threshold = rules.get("fuzzy_threshold", 0.8)
    # Remove fuzzy_threshold key to leave only category definitions
    categories = {k: v for k, v in rules.items() if k != "fuzzy_threshold"}
    return fuzzy_threshold, categories

@lru_cache(maxsize=1024)
def normalize_term(term, category, fuzzy_threshold, rules):
    term_lower = term.lower().strip()
    # Exact match: Check canonical and synonyms
    for canonical, synonyms in rules.get(category, {}).items():
        if term_lower == canonical.lower():
            return canonical
        for syn in synonyms:
            if term_lower == syn.lower():
                return canonical

    # Fuzzy match: iterate over canonical and synonyms
    best_match = None
    best_score = 0
    for canonical, synonyms in rules.get(category, {}).items():
        score = fuzz.token_set_ratio(term_lower, canonical.lower()) / 100.0
        if score > best_score:
            best_score = score
            best_match = canonical
        for syn in synonyms:
            score = fuzz.token_set_ratio(term_lower, syn.lower()) / 100.0
            if score > best_score:
                best_score = score
                best_match = canonical

    if best_score >= fuzzy_threshold:
        # Correct Sentry instrumentation
        sentry_sdk.set_tag("normalization_step", "fuzzy_match")
        sentry_sdk.set_tag("matched_to", best_match)
        sentry_sdk.set_tag("score", best_score)
        sentry_sdk.capture_message("Fuzzy match performed", level="info")
        return best_match

    # No match: return original term
    return term

class TermNormalizer:
    """
    Normalizes common term variations based on rules defined in a YAML file.
    """
    
    def __init__(self, config_path="config/normalization_rules.yml", default_columns=None):
        """
        Initialize the term normalizer with rules from the YAML file.
        
        Args:
            config_path: Path to the YAML rules file. If None, uses the default.
            default_columns: List of default columns to normalize. If None, uses all categories.
        """
        # Default rules file path at project root config directory
        if config_path is None:
            module_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(module_dir, os.pardir, os.pardir))
            config_path = os.path.join(project_root, 'config', 'normalization_rules.yml')
        self.config_path = config_path
        self.fuzzy_threshold, self.rules = load_normalization_rules(config_path)
        # Default to the category keys from the YAML if not provided
        if default_columns is None:
            self.default_columns = list(self.rules.keys())
        else:
            self.default_columns = default_columns
        logger.info(f"Loaded {len(self.rules)} rules from {config_path}")
        
    def normalize(self, df, columns=None):
        """
        Normalize specified columns in a DataFrame.
        
        Args:
            df: DataFrame to process
            columns: List of column names to normalize. If None, uses self.default_columns.
            
        Returns:
            DataFrame with normalized values
        """
        if columns is None:
            columns = self.default_columns
        df_norm = df.copy()
        for col in columns:
            if col in df_norm.columns and col in self.rules:
                df_norm[col] = df_norm[col].apply(lambda x: normalize_term(str(x), col, self.fuzzy_threshold, self.rules))
        return df_norm
    
    def normalize_column(self, df: pd.DataFrame, column: str, inplace: bool = False) -> pd.DataFrame:
        """
        Normalize values in a specific column of a DataFrame.
        
        Args:
            df: DataFrame to process
            column: Column name to normalize
            inplace: Whether to modify the original DataFrame
            
        Returns:
            DataFrame with normalized values in the specified column
        """
        if not inplace:
            df = df.copy()
            
        if column not in df.columns:
            logger.warning(f"Column '{column}' not found in DataFrame, skipping normalization")
            return df
            
        # Normalize cell values
        sentry_sdk.set_tag("normalization_rules_version", self.rules_version)
        sentry_sdk.set_tag("normalization_step", "cell_value")
        df[column] = df[column].apply(lambda x: normalize_term(str(x), column, self.fuzzy_threshold, self.rules))
        
        # Log unique values before and after normalization
        unique_values = df[column].dropna().unique()
        logger.info(f"Normalized '{column}' column. Now contains {len(unique_values)} unique values")
        
        return df
    
    def normalize_dataframe(self, df: pd.DataFrame, columns: Optional[List[str]] = None,
                          inplace: bool = False) -> pd.DataFrame:
        """
        Normalize specified columns in a DataFrame.
        
        Args:
            df: DataFrame to process
            columns: List of column names to normalize. If None, attempts to normalize
                     known columns like LeadSource.
            inplace: Whether to modify the original DataFrame
            
        Returns:
            DataFrame with normalized values
        """
        if not inplace:
            df = df.copy()
            
        if not columns:
            # Default columns that are typically normalized
            default_normalizable = [
                'LeadSource', 'Lead_Source', 'lead_source', 'Source',
                'SalesRep', 'Sales_Rep', 'sales_rep', 'Salesperson',
                'VehicleType', 'Vehicle_Type', 'vehicle_type'
            ]
            # Use columns that exist in the DataFrame
            columns = [col for col in default_normalizable if col in df.columns]
            
        if not columns:
            logger.warning(f"No normalizable columns found in DataFrame, skipping normalization")
            return df
            
        # Normalize each column
        for column in columns:
            df = self.normalize_column(df, column, inplace=True)
            
        return df

    def rename_columns(self, df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
        """
        Rename any columns in the DataFrame based on normalization rules.
        """
        sentry_sdk.set_tag("normalization_rules_version", self.rules_version)
        sentry_sdk.set_tag("normalization_step", "column_rename")
        if not inplace:
            df = df.copy()
        mapping = {}
        for col in df.columns:
            key = col.strip().lower()
            if key in self.rules:
                mapping[col] = self.rules[key]
        if mapping:
            df = df.rename(columns=mapping)
        return df

# Create a global instance for easy access
normalizer = TermNormalizer()

def normalize(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Normalize DataFrame headers and cell values based on normalization rules."""
    # First rename columns, then normalize values
    df = normalizer.rename_columns(df)
    df = normalizer.normalize(df, columns)
    return df

# Backward compatibility alias for the old normalize_terms function name
normalize_terms = normalize