"""
Data Normalization Module for Watchdog AI.

This module provides functionality for normalizing data according to
schema profiles, including column mapping, data type conversion, and
value standardization.
"""

import os
import re
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union, Set
from datetime import datetime
import dateutil.parser
import Levenshtein
from .adaptive_schema import AdaptiveSchema, SchemaProfile, SchemaColumn

logger = logging.getLogger(__name__)

# Constants
DEFAULT_STRING_MATCH_THRESHOLD = 0.75
DEFAULT_SYNONYM_MATCH_THRESHOLD = 0.8
CURRENCY_PATTERN = r'^\s*[\$£€¥]?\s*([0-9,]+(\.[0-9]+)?)\s*$'
PERCENTAGE_PATTERN = r'^\s*([0-9,]+(\.[0-9]+)?)\s*%\s*$'
DATE_FORMATS = [
    '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%m-%d-%Y', '%d-%m-%Y',
    '%Y/%m/%d', '%b %d, %Y', '%d %b %Y', '%B %d, %Y', '%d %B %Y'
]

class ColumnMapping:
    """Represents a mapping between source and target columns."""
    
    def __init__(self, source_column: str, target_column: str, 
                confidence: float, reason: str = None):
        """
        Initialize a column mapping.
        
        Args:
            source_column: Original column name from the data source
            target_column: Target column name in the schema
            confidence: Confidence score for the mapping (0.0 to 1.0)
            reason: Reason for the mapping
        """
        self.source_column = source_column
        self.target_column = target_column
        self.confidence = confidence
        self.reason = reason or "String similarity match"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert mapping to dictionary."""
        return {
            "source_column": self.source_column,
            "target_column": self.target_column,
            "confidence": self.confidence,
            "reason": self.reason
        }


class MappingSuggestion:
    """Suggestion for handling unmapped or problematic columns."""
    
    def __init__(self, column: str, suggestion_type: str, 
                suggestion: str, reason: str, confidence: float = None):
        """
        Initialize a mapping suggestion.
        
        Args:
            column: Column name
            suggestion_type: Type of suggestion ('rename', 'drop', 'cast', etc.)
            suggestion: The suggested action
            reason: Reason for the suggestion
            confidence: Confidence in the suggestion (0.0 to 1.0)
        """
        self.column = column
        self.suggestion_type = suggestion_type
        self.suggestion = suggestion
        self.reason = reason
        self.confidence = confidence
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert suggestion to dictionary."""
        return {
            "column": self.column,
            "suggestion_type": self.suggestion_type,
            "suggestion": self.suggestion,
            "reason": self.reason,
            "confidence": self.confidence
        }


class ColumnMapper:
    """Maps source columns to schema columns using fuzzy matching and synonyms."""
    
    def __init__(self, schema: Optional[AdaptiveSchema] = None, 
                synonym_file: Optional[str] = None):
        """
        Initialize the column mapper.
        
        Args:
            schema: AdaptiveSchema instance
            synonym_file: Path to synonym JSON file
        """
        self.schema = schema
        self.synonyms = self._load_synonyms(synonym_file)
        self.string_match_threshold = DEFAULT_STRING_MATCH_THRESHOLD
        self.synonym_match_threshold = DEFAULT_SYNONYM_MATCH_THRESHOLD
    
    def _load_synonyms(self, synonym_file: Optional[str]) -> Dict[str, List[str]]:
        """
        Load column synonyms from a JSON file.
        
        Args:
            synonym_file: Path to synonym JSON file
            
        Returns:
            Dictionary of canonical names to lists of synonyms
        """
        synonyms = {}
        
        if synonym_file and os.path.exists(synonym_file):
            try:
                with open(synonym_file, 'r') as f:
                    synonyms = json.load(f)
                logger.info(f"Loaded {len(synonyms)} synonym sets from {synonym_file}")
            except Exception as e:
                logger.error(f"Error loading synonyms: {str(e)}")
        
        # Add some default automotive synonyms if not present
        if 'vehicle_make' not in synonyms:
            synonyms['vehicle_make'] = [
                'make', 'car_make', 'manufacturer', 'brand', 'veh_make', 'mfr'
            ]
        
        if 'vehicle_model' not in synonyms:
            synonyms['vehicle_model'] = [
                'model', 'car_model', 'veh_model', 'mdl'
            ]
        
        if 'sale_date' not in synonyms:
            synonyms['sale_date'] = [
                'date', 'sold_date', 'purchase_date', 'transaction_date', 'deal_date'
            ]
        
        return synonyms
    
    def _normalize_column_name(self, name: str) -> str:
        """
        Normalize a column name for comparison.
        
        Args:
            name: Column name
            
        Returns:
            Normalized column name
        """
        if not name:
            return ""
        
        # Convert to lowercase
        normalized = name.lower()
        
        # Remove special characters
        normalized = re.sub(r'[^a-z0-9_]', '_', normalized)
        
        # Replace multiple underscores with single underscore
        normalized = re.sub(r'_+', '_', normalized)
        
        # Remove leading/trailing underscores
        normalized = normalized.strip('_')
        
        return normalized
    
    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """
        Calculate string similarity using normalized Levenshtein distance.
        
        Args:
            str1: First string
            str2: Second string
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        if not str1 or not str2:
            return 0.0
        
        # Normalize strings
        str1_norm = self._normalize_column_name(str1)
        str2_norm = self._normalize_column_name(str2)
        
        if str1_norm == str2_norm:
            return 1.0
        
        # Calculate Levenshtein distance
        distance = Levenshtein.distance(str1_norm, str2_norm)
        max_len = max(len(str1_norm), len(str2_norm))
        
        if max_len == 0:
            return 0.0
        
        # Convert distance to similarity (0.0 to 1.0)
        similarity = 1.0 - (distance / max_len)
        
        return similarity
    
    def _check_synonym_match(self, source_column: str, target_column: str) -> Tuple[bool, float]:
        """
        Check if source column matches a synonym of the target column.
        
        Args:
            source_column: Source column name
            target_column: Target column name
            
        Returns:
            Tuple of (is_match, confidence)
        """
        # Normalize column names
        source_norm = self._normalize_column_name(source_column)
        target_norm = self._normalize_column_name(target_column)
        
        # Check direct match
        if source_norm == target_norm:
            return True, 1.0
        
        # Check if target is in synonyms dictionary
        if target_norm in self.synonyms:
            for synonym in self.synonyms[target_norm]:
                synonym_norm = self._normalize_column_name(synonym)
                
                # Check exact synonym match
                if source_norm == synonym_norm:
                    return True, 0.95
                
                # Check fuzzy synonym match
                similarity = self._calculate_string_similarity(source_norm, synonym_norm)
                if similarity >= self.synonym_match_threshold:
                    return True, similarity * 0.9  # Slightly reduce confidence for fuzzy synonym
        
        # No synonym match
        return False, 0.0
    
    def map_columns(self, source_columns: List[str], 
                   schema_profile: SchemaProfile) -> Tuple[Dict[str, str], List[str], List[MappingSuggestion]]:
        """
        Map source columns to schema columns.
        
        Args:
            source_columns: List of source column names
            schema_profile: Schema profile to map against
            
        Returns:
            Tuple of (column_mapping, unmapped_columns, suggestions)
        """
        column_mapping = {}
        unmapped_columns = []
        suggestions = []
        
        # Get all schema columns
        schema_columns = schema_profile.columns
        schema_column_names = [col.name for col in schema_columns]
        
        # Track already mapped target columns to avoid duplicates
        mapped_targets = set()
        
        # First pass: Look for exact matches and synonym matches
        for source_col in source_columns:
            matched = False
            
            for schema_col in schema_columns:
                # Skip already mapped targets
                if schema_col.name in mapped_targets:
                    continue
                
                # Check exact match
                if self._normalize_column_name(source_col) == self._normalize_column_name(schema_col.name):
                    column_mapping[source_col] = schema_col.name
                    mapped_targets.add(schema_col.name)
                    matched = True
                    break
                
                # Check synonym match
                is_synonym, confidence = self._check_synonym_match(source_col, schema_col.name)
                if is_synonym:
                    column_mapping[source_col] = schema_col.name
                    mapped_targets.add(schema_col.name)
                    matched = True
                    break
                
                # Check against aliases
                for alias in schema_col.aliases:
                    if self._normalize_column_name(source_col) == self._normalize_column_name(alias):
                        column_mapping[source_col] = schema_col.name
                        mapped_targets.add(schema_col.name)
                        matched = True
                        break
                    
                    # Check fuzzy match against aliases
                    similarity = self._calculate_string_similarity(source_col, alias)
                    if similarity >= self.string_match_threshold:
                        column_mapping[source_col] = schema_col.name
                        mapped_targets.add(schema_col.name)
                        matched = True
                        break
                
                if matched:
                    break
            
            if not matched:
                unmapped_columns.append(source_col)
        
        # Second pass: Try fuzzy matching for remaining columns
        for source_col in unmapped_columns[:]:
            best_match = None
            best_score = 0.0
            best_target = None
            
            for schema_col in schema_columns:
                # Skip already mapped targets
                if schema_col.name in mapped_targets:
                    continue
                
                # Check string similarity with schema column name
                similarity = self._calculate_string_similarity(source_col, schema_col.name)
                if similarity > best_score:
                    best_score = similarity
                    best_match = schema_col
                    best_target = schema_col.name
                
                # Check string similarity with aliases
                for alias in schema_col.aliases:
                    similarity = self._calculate_string_similarity(source_col, alias)
                    if similarity > best_score:
                        best_score = similarity
                        best_match = schema_col
                        best_target = schema_col.name
            
            # If we found a good match
            if best_match and best_score >= self.string_match_threshold:
                column_mapping[source_col] = best_target
                mapped_targets.add(best_target)
                unmapped_columns.remove(source_col)
            else:
                # Generate a suggestion for this unmapped column
                suggestion = self._generate_mapping_suggestion(source_col, schema_columns, best_match, best_score)
                suggestions.append(suggestion)
        
        return column_mapping, unmapped_columns, suggestions
    
    def _generate_mapping_suggestion(self, source_col: str, 
                                   schema_columns: List[SchemaColumn],
                                   best_match: Optional[SchemaColumn],
                                   best_score: float) -> MappingSuggestion:
        """
        Generate a suggestion for handling an unmapped column.
        
        Args:
            source_col: Source column name
            schema_columns: List of schema columns
            best_match: Best matching schema column (if any)
            best_score: Best match score
            
        Returns:
            MappingSuggestion object
        """
        # If we have a decent match below threshold, suggest renaming
        if best_match and best_score >= 0.5:
            return MappingSuggestion(
                column=source_col,
                suggestion_type="rename",
                suggestion=best_match.name,
                reason=f"Column name similar to '{best_match.name}' (score: {best_score:.2f})",
                confidence=best_score
            )
        
        # Check if it looks like a date column
        if any(date_term in source_col.lower() for date_term in ['date', 'time', 'day', 'month', 'year']):
            date_cols = [col for col in schema_columns if col.data_type in ['date', 'datetime']]
            if date_cols:
                return MappingSuggestion(
                    column=source_col,
                    suggestion_type="cast_and_map",
                    suggestion=date_cols[0].name,
                    reason=f"Column name suggests it contains date/time data",
                    confidence=0.7
                )
        
        # Check if it looks like a numeric column
        if any(num_term in source_col.lower() for num_term in ['amount', 'price', 'cost', 'total', 'sum', 'count', 'quantity']):
            numeric_cols = [col for col in schema_columns if col.data_type in ['float', 'int', 'integer', 'number']]
            if numeric_cols:
                return MappingSuggestion(
                    column=source_col,
                    suggestion_type="cast_and_map",
                    suggestion=numeric_cols[0].name,
                    reason=f"Column name suggests it contains numeric data",
                    confidence=0.6
                )
        
        # Default: suggest dropping or keeping as-is
        return MappingSuggestion(
            column=source_col,
            suggestion_type="drop",
            suggestion="",
            reason="Column not recognized in schema",
            confidence=0.5
        )

    def generate_column_mappings(self, df: pd.DataFrame, 
                               schema_profile: SchemaProfile) -> List[ColumnMapping]:
        """
        Generate column mappings for a DataFrame.
        
        Args:
            df: Source DataFrame
            schema_profile: Schema profile to map against
            
        Returns:
            List of ColumnMapping objects
        """
        source_columns = df.columns.tolist()
        mappings = []
        
        # Map columns
        column_map, unmapped, suggestions = self.map_columns(source_columns, schema_profile)
        
        # Create ColumnMapping objects
        for source_col, target_col in column_map.items():
            # Determine confidence and reason
            confidence = 1.0
            reason = "Exact match"
            
            # If not exact match, recalculate similarity
            if self._normalize_column_name(source_col) != self._normalize_column_name(target_col):
                # Check synonym match
                is_synonym, syn_confidence = self._check_synonym_match(source_col, target_col)
                if is_synonym:
                    confidence = syn_confidence
                    reason = "Synonym match"
                else:
                    # Must be a fuzzy match
                    confidence = self._calculate_string_similarity(source_col, target_col)
                    reason = "Fuzzy string match"
                    
                    # Check if matched on alias
                    schema_col = schema_profile.get_column_by_name(target_col)
                    if schema_col:
                        for alias in schema_col.aliases:
                            alias_similarity = self._calculate_string_similarity(source_col, alias)
                            if alias_similarity > confidence:
                                confidence = alias_similarity
                                reason = f"Matched on alias '{alias}'"
            
            mappings.append(ColumnMapping(source_col, target_col, confidence, reason))
        
        return mappings


class DataNormalizer:
    """Normalizes data according to schema rules."""
    
    def __init__(self, schema: Optional[AdaptiveSchema] = None):
        """
        Initialize the data normalizer.
        
        Args:
            schema: AdaptiveSchema instance
        """
        self.schema = schema
    
    def normalize_column_names(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Normalize column names to a standard format.
        
        Args:
            df: Source DataFrame
            
        Returns:
            Tuple of (normalized DataFrame, column_mapping)
        """
        if df.empty:
            return df, {}
        
        # Create a copy to avoid modifying the original
        result = df.copy()
        column_mapping = {}
        
        # Normalize each column name
        for col in df.columns:
            # Convert to lowercase
            normalized = str(col).lower()
            
            # Replace spaces and special characters with underscores
            normalized = re.sub(r'[^a-z0-9_]', '_', normalized)
            
            # Replace multiple underscores with single underscore
            normalized = re.sub(r'_+', '_', normalized)
            
            # Remove leading/trailing underscores
            normalized = normalized.strip('_')
            
            # Avoid duplicate column names
            if normalized in column_mapping.values():
                i = 1
                while f"{normalized}_{i}" in column_mapping.values():
                    i += 1
                normalized = f"{normalized}_{i}"
            
            # Record the mapping
            column_mapping[col] = normalized
        
        # Rename columns
        result.columns = [column_mapping[col] for col in df.columns]
        
        return result, column_mapping
    
    def convert_currency_columns(self, df: pd.DataFrame, 
                               column_name: str) -> Tuple[pd.DataFrame, bool]:
        """
        Convert currency strings to float values.
        
        Args:
            df: Source DataFrame
            column_name: Column to convert
            
        Returns:
            Tuple of (modified DataFrame, whether conversion was performed)
        """
        if df.empty or column_name not in df.columns:
            return df, False
        
        # Create a copy to avoid modifying the original
        result = df.copy()
        
        # Check if column contains currency values
        if pd.api.types.is_numeric_dtype(result[column_name]):
            # Already numeric
            return result, False
        
        try:
            # Try to convert strings like "$1,234.56" to floats
            def convert_currency(val):
                if pd.isna(val):
                    return np.nan
                
                val_str = str(val)
                match = re.match(CURRENCY_PATTERN, val_str)
                if match:
                    # Extract the numeric part and remove commas
                    return float(match.group(1).replace(',', ''))
                
                try:
                    return float(val_str.replace('$', '').replace(',', '').strip())
                except:
                    return np.nan
            
            # Check if at least 50% of non-null values match the currency pattern
            non_null_values = result[column_name].dropna()
            if len(non_null_values) == 0:
                return result, False
            
            matches = sum(1 for val in non_null_values if re.match(CURRENCY_PATTERN, str(val)))
            match_ratio = matches / len(non_null_values)
            
            if match_ratio >= 0.5:
                result[column_name] = result[column_name].apply(convert_currency)
                return result, True
        except Exception as e:
            logger.warning(f"Error converting currency column {column_name}: {str(e)}")
        
        return result, False
    
    def convert_percentage_columns(self, df: pd.DataFrame, 
                                 column_name: str) -> Tuple[pd.DataFrame, bool]:
        """
        Convert percentage strings to float values.
        
        Args:
            df: Source DataFrame
            column_name: Column to convert
            
        Returns:
            Tuple of (modified DataFrame, whether conversion was performed)
        """
        if df.empty or column_name not in df.columns:
            return df, False
        
        # Create a copy to avoid modifying the original
        result = df.copy()
        
        # Check if column contains percentage values
        if pd.api.types.is_numeric_dtype(result[column_name]):
            # Already numeric
            return result, False
        
        try:
            # Try to convert strings like "12.34%" to floats
            def convert_percentage(val):
                if pd.isna(val):
                    return np.nan
                
                val_str = str(val)
                match = re.match(PERCENTAGE_PATTERN, val_str)
                if match:
                    # Extract the numeric part and remove commas
                    return float(match.group(1).replace(',', '')) / 100.0
                
                try:
                    if '%' in val_str:
                        return float(val_str.replace('%', '').replace(',', '').strip()) / 100.0
                    return float(val_str.replace(',', '').strip())
                except:
                    return np.nan
            
            # Check if at least 50% of non-null values match the percentage pattern
            non_null_values = result[column_name].dropna()
            if len(non_null_values) == 0:
                return result, False
            
            matches = sum(1 for val in non_null_values if '%' in str(val) or re.match(PERCENTAGE_PATTERN, str(val)))
            match_ratio = matches / len(non_null_values)
            
            if match_ratio >= 0.5:
                result[column_name] = result[column_name].apply(convert_percentage)
                return result, True
        except Exception as e:
            logger.warning(f"Error converting percentage column {column_name}: {str(e)}")
        
        return result, False
    
    def convert_date_columns(self, df: pd.DataFrame, 
                           column_name: str) -> Tuple[pd.DataFrame, bool]:
        """
        Convert date strings to datetime objects.
        
        Args:
            df: Source DataFrame
            column_name: Column to convert
            
        Returns:
            Tuple of (modified DataFrame, whether conversion was performed)
        """
        if df.empty or column_name not in df.columns:
            return df, False
        
        # Create a copy to avoid modifying the original
        result = df.copy()
        
        # Check if column already contains datetime values
        if pd.api.types.is_datetime64_dtype(result[column_name]):
            # Already datetime
            return result, False
        
        try:
            # Try to convert using pandas
            result[column_name] = pd.to_datetime(result[column_name], errors='coerce')
            
            # Check if conversion was successful
            null_ratio = result[column_name].isna().mean()
            if null_ratio < 0.5:
                return result, True
            
            # If too many NaN values, try different formats
            for fmt in DATE_FORMATS:
                try:
                    temp = pd.to_datetime(df[column_name], format=fmt, errors='coerce')
                    null_ratio = temp.isna().mean()
                    if null_ratio < 0.5:
                        result[column_name] = temp
                        return result, True
                except:
                    continue
            
            # If standard formats failed, try flexible parsing
            def parse_date(val):
                if pd.isna(val):
                    return np.nan
                try:
                    return dateutil.parser.parse(str(val))
                except:
                    return np.nan
            
            temp = df[column_name].apply(parse_date)
            null_ratio = pd.isna(temp).mean()
            if null_ratio < 0.5:
                result[column_name] = temp
                return result, True
        except Exception as e:
            logger.warning(f"Error converting date column {column_name}: {str(e)}")
        
        # If all methods failed, revert to original
        result[column_name] = df[column_name]
        return result, False
    
    def normalize_dataframe(self, df: pd.DataFrame, 
                          schema_profile: SchemaProfile) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Normalize a DataFrame according to schema rules.
        
        Args:
            df: Source DataFrame
            schema_profile: Schema profile to use
            
        Returns:
            Tuple of (normalized DataFrame, normalization_summary)
        """
        if df.empty:
            return df, {"empty_dataframe": True}
        
        # Start with a copy of the original
        result = df.copy()
        
        # Track changes
        normalization_summary = {
            "column_name_changes": {},
            "type_conversions": [],
            "value_standardizations": [],
            "original_shape": df.shape,
            "timestamp": datetime.now().isoformat()
        }
        
        # 1. Normalize column names
        result, column_mapping = self.normalize_column_names(result)
        normalization_summary["column_name_changes"] = column_mapping
        
        # 2. Apply data type conversions based on schema
        for col in schema_profile.columns:
            # Find matching column in DataFrame
            matching_cols = [df_col for df_col in result.columns 
                            if col.name == df_col or col.name == column_mapping.get(df_col, "")]
            
            if not matching_cols:
                continue
            
            df_col = matching_cols[0]
            
            # Apply conversions based on data type
            if col.data_type in ['float', 'double', 'numeric', 'number']:
                # Try currency conversion first
                temp_df, converted = self.convert_currency_columns(result, df_col)
                if converted:
                    result = temp_df
                    normalization_summary["type_conversions"].append({
                        "column": df_col,
                        "from_type": str(df[df_col].dtype),
                        "to_type": str(result[df_col].dtype),
                        "conversion": "currency_to_float"
                    })
                else:
                    # Try percentage conversion
                    temp_df, converted = self.convert_percentage_columns(result, df_col)
                    if converted:
                        result = temp_df
                        normalization_summary["type_conversions"].append({
                            "column": df_col,
                            "from_type": str(df[df_col].dtype),
                            "to_type": str(result[df_col].dtype),
                            "conversion": "percentage_to_float"
                        })
                    else:
                        # Try regular numeric conversion
                        try:
                            result[df_col] = pd.to_numeric(result[df_col], errors='coerce')
                            normalization_summary["type_conversions"].append({
                                "column": df_col,
                                "from_type": str(df[df_col].dtype),
                                "to_type": str(result[df_col].dtype),
                                "conversion": "to_numeric"
                            })
                        except:
                            pass
            
            elif col.data_type in ['integer', 'int']:
                try:
                    # Try to convert to integer
                    result[df_col] = pd.to_numeric(result[df_col], errors='coerce').astype('Int64')
                    normalization_summary["type_conversions"].append({
                        "column": df_col,
                        "from_type": str(df[df_col].dtype),
                        "to_type": str(result[df_col].dtype),
                        "conversion": "to_integer"
                    })
                except:
                    pass
            
            elif col.data_type in ['date', 'datetime', 'timestamp']:
                temp_df, converted = self.convert_date_columns(result, df_col)
                if converted:
                    result = temp_df
                    normalization_summary["type_conversions"].append({
                        "column": df_col,
                        "from_type": str(df[df_col].dtype),
                        "to_type": str(result[df_col].dtype),
                        "conversion": "to_datetime"
                    })
            
            elif col.data_type == 'boolean':
                try:
                    # Convert various boolean representations
                    result[df_col] = result[df_col].map({
                        'true': True, 'false': False,
                        'yes': True, 'no': False,
                        'y': True, 'n': False,
                        '1': True, '0': False,
                        1: True, 0: False,
                        'True': True, 'False': False
                    }).astype('boolean')
                    normalization_summary["type_conversions"].append({
                        "column": df_col,
                        "from_type": str(df[df_col].dtype),
                        "to_type": str(result[df_col].dtype),
                        "conversion": "to_boolean"
                    })
                except:
                    pass
        
        # Update summary with final shape
        normalization_summary["final_shape"] = result.shape
        
        return result, normalization_summary


class DataSchemaApplier:
    """
    Applies schema profiles to data, handling column mapping, normalization,
    and validation.
    """
    
    def __init__(self, profiles_dir: str = "config/schema_profiles"):
        """
        Initialize the schema applier.
        
        Args:
            profiles_dir: Directory containing schema profiles
        """
        self.profiles_dir = profiles_dir
        self.schema = AdaptiveSchema(profiles_dir=profiles_dir)
        self.mapper = ColumnMapper(self.schema)
        self.normalizer = DataNormalizer(self.schema)
        self.dealer_profiles = {}
        self.default_profile = None
        
        # Load default profile
        self._load_profiles()
    
    def _load_profiles(self) -> None:
        """Load schema profiles from disk."""
        # Ensure profiles directory exists
        if not os.path.exists(self.profiles_dir):
            logger.warning(f"Schema profiles directory {self.profiles_dir} does not exist")
            return
        
        # Load default profile
        default_paths = [
            os.path.join(self.profiles_dir, "default.json"),
            os.path.join(self.profiles_dir, "general.json"),
            os.path.join(self.profiles_dir, "generic.json")
        ]
        
        for path in default_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        data = json.load(f)
                    self.default_profile = SchemaProfile.from_dict(data)
                    logger.info(f"Loaded default schema profile from {path}")
                    break
                except Exception as e:
                    logger.error(f"Error loading default profile from {path}: {str(e)}")
        
        # Load dealer-specific profiles
        for filename in os.listdir(self.profiles_dir):
            if filename.endswith('.json') and filename not in ['default.json', 'general.json', 'generic.json']:
                try:
                    path = os.path.join(self.profiles_dir, filename)
                    with open(path, 'r') as f:
                        data = json.load(f)
                    
                    profile = SchemaProfile.from_dict(data)
                    self.dealer_profiles[profile.id] = profile
                    logger.info(f"Loaded dealer profile {profile.id} from {path}")
                except Exception as e:
                    logger.error(f"Error loading profile from {path}: {str(e)}")
    
    def get_profile(self, dealer_id: Optional[str] = None) -> SchemaProfile:
        """
        Get schema profile for a dealer.
        
        Args:
            dealer_id: Dealer ID
            
        Returns:
            SchemaProfile object (default if dealer-specific not found)
        """
        if dealer_id and dealer_id in self.dealer_profiles:
            return self.dealer_profiles[dealer_id]
        
        if self.default_profile:
            return self.default_profile
        
        # No profiles found, create minimal default
        logger.warning("No profiles found, creating minimal default")
        return SchemaProfile(
            id="default",
            name="Default Schema Profile",
            description="Minimal default schema profile",
            role="general_manager",
            columns=[]
        )
    
    def apply_schema(self, df: pd.DataFrame, 
                    dealer_id: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Apply schema to a DataFrame, performing column mapping and normalization.
        
        Args:
            df: Source DataFrame
            dealer_id: Dealer ID for profile selection
            
        Returns:
            Tuple of (processed DataFrame, processing_summary)
        """
        if df.empty:
            return df, {"error": "Empty DataFrame", "success": False}
        
        # Get profile for dealer
        profile = self.get_profile(dealer_id)
        
        processing_summary = {
            "profile_id": profile.id,
            "profile_name": profile.name,
            "timestamp": datetime.now().isoformat(),
            "original_shape": df.shape,
            "original_columns": df.columns.tolist(),
            "success": True
        }
        
        # 1. Generate column mappings
        mappings = self.mapper.generate_column_mappings(df, profile)
        processing_summary["column_mappings"] = [m.to_dict() for m in mappings]
        
        # Get mapping dictionary
        column_map = {m.source_column: m.target_column for m in mappings}
        
        # Identify unmapped columns
        unmapped_columns = [col for col in df.columns if col not in column_map]
        processing_summary["unmapped_columns"] = unmapped_columns
        
        # 2. Normalize the DataFrame
        normalized_df, normalization_summary = self.normalizer.normalize_dataframe(df, profile)
        processing_summary["normalization_summary"] = normalization_summary
        
        # 3. Apply column mappings
        if column_map:
            # Only map columns that exist in the normalized DataFrame
            valid_mappings = {source: target for source, target in column_map.items() 
                            if source in normalized_df.columns}
            
            # Rename columns according to mapping
            result = normalized_df.rename(columns=valid_mappings)
            
            # Add mapping info to summary
            processing_summary["applied_mappings"] = valid_mappings
            processing_summary["mapped_column_count"] = len(valid_mappings)
        else:
            result = normalized_df
            processing_summary["applied_mappings"] = {}
            processing_summary["mapped_column_count"] = 0
        
        # 4. Validate against schema
        validation_result = self._validate_against_schema(result, profile)
        processing_summary["validation_result"] = validation_result
        
        # Update final shape
        processing_summary["final_shape"] = result.shape
        processing_summary["final_columns"] = result.columns.tolist()
        
        return result, processing_summary
    
    def _validate_against_schema(self, df: pd.DataFrame, 
                              profile: SchemaProfile) -> Dict[str, Any]:
        """
        Validate a DataFrame against a schema profile.
        
        Args:
            df: DataFrame to validate
            profile: Schema profile
            
        Returns:
            Validation result dictionary
        """
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "missing_required_columns": [],
            "column_validations": {}
        }
        
        # Check for missing required columns
        schema_columns = profile.columns
        required_columns = [col.name for col in schema_columns if hasattr(col, 'required') and col.required]
        
        for col_name in required_columns:
            if col_name not in df.columns:
                validation_result["is_valid"] = False
                validation_result["missing_required_columns"].append(col_name)
                validation_result["errors"].append(f"Missing required column: {col_name}")
        
        # Validate each column's data type
        for col in schema_columns:
            if col.name in df.columns:
                col_validation = {"is_valid": True, "issues": []}
                
                # Check data type
                if col.data_type in ['float', 'double', 'numeric', 'number']:
                    if not pd.api.types.is_numeric_dtype(df[col.name]):
                        col_validation["is_valid"] = False
                        issue = f"Column should be numeric but has type {df[col.name].dtype}"
                        col_validation["issues"].append(issue)
                        validation_result["warnings"].append(issue)
                
                elif col.data_type in ['integer', 'int']:
                    if not pd.api.types.is_integer_dtype(df[col.name]):
                        col_validation["is_valid"] = False
                        issue = f"Column should be integer but has type {df[col.name].dtype}"
                        col_validation["issues"].append(issue)
                        validation_result["warnings"].append(issue)
                
                elif col.data_type in ['date', 'datetime', 'timestamp']:
                    if not pd.api.types.is_datetime64_dtype(df[col.name]):
                        col_validation["is_valid"] = False
                        issue = f"Column should be datetime but has type {df[col.name].dtype}"
                        col_validation["issues"].append(issue)
                        validation_result["warnings"].append(issue)
                
                # Check for completely null columns
                null_ratio = df[col.name].isna().mean()
                if null_ratio == 1.0:
                    col_validation["is_valid"] = False
                    issue = f"Column is completely null"
                    col_validation["issues"].append(issue)
                    validation_result["warnings"].append(f"Column {col.name} is completely null")
                elif null_ratio > 0.5:
                    issue = f"Column has {null_ratio:.1%} null values"
                    col_validation["issues"].append(issue)
                    validation_result["warnings"].append(f"Column {col.name} has {null_ratio:.1%} null values")
                
                # Apply business rules if any
                if hasattr(col, 'business_rules') and col.business_rules:
                    for rule in col.business_rules:
                        if rule['type'] == 'comparison':
                            operator = rule['operator']
                            threshold = rule['threshold']
                            
                            if operator == '>=' and df[col.name].min() < threshold:
                                col_validation["is_valid"] = False
                                issue = f"Values less than {threshold} found (min: {df[col.name].min()})"
                                col_validation["issues"].append(issue)
                                validation_result["warnings"].append(f"Column {col.name}: {issue}")
                            
                            elif operator == '<=' and df[col.name].max() > threshold:
                                col_validation["is_valid"] = False
                                issue = f"Values greater than {threshold} found (max: {df[col.name].max()})"
                                col_validation["issues"].append(issue)
                                validation_result["warnings"].append(f"Column {col.name}: {issue}")
                
                validation_result["column_validations"][col.name] = col_validation
        
        return validation_result


def normalize_dataframe(df: pd.DataFrame, 
                      dealer_id: Optional[str] = None,
                      profiles_dir: str = "config/schema_profiles") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Convenience function to normalize a DataFrame according to a schema profile.
    
    Args:
        df: DataFrame to normalize
        dealer_id: Dealer ID for profile selection
        profiles_dir: Directory containing schema profiles
        
    Returns:
        Tuple of (normalized DataFrame, processing_summary)
    """
    applier = DataSchemaApplier(profiles_dir)
    return applier.apply_schema(df, dealer_id)