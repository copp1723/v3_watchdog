"""
Enhanced column mapping engine with synonym-based suggestions.
"""

import logging
import json
import os
import re
from typing import Dict, List, Tuple, Any, Optional, Set
import pandas as pd
import Levenshtein
from difflib import get_close_matches
from datetime import datetime

from .adaptive_schema import SchemaProfile, SchemaColumn
from .data_lineage import DataLineage

logger = logging.getLogger(__name__)

class MappingSuggestion:
    """Represents a mapping suggestion for a column."""
    
    def __init__(self, source_column: str, target_column: str, 
                confidence: float, reason: str,
                alternatives: List[str] = None):
        """
        Initialize a mapping suggestion.
        
        Args:
            source_column: Original column name
            target_column: Suggested target column name
            confidence: Confidence score (0.0 to 1.0)
            reason: Reason for the suggestion
            alternatives: Alternative mapping suggestions
        """
        self.source_column = source_column
        self.target_column = target_column
        self.confidence = confidence
        self.reason = reason
        self.alternatives = alternatives or []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source_column": self.source_column,
            "target_column": self.target_column,
            "confidence": self.confidence,
            "reason": self.reason,
            "alternatives": self.alternatives
        }


class EnhancedColumnMapper:
    """
    Enhanced column mapper with synonym-based suggestions and learning capabilities.
    """
    
    def __init__(self, 
                synonym_file: Optional[str] = None,
                lineage: Optional[DataLineage] = None,
                learning_enabled: bool = True):
        """
        Initialize the enhanced column mapper.
        
        Args:
            synonym_file: Path to synonym dictionary file
            lineage: DataLineage instance for historical mappings
            learning_enabled: Whether to learn from past mappings
        """
        self.synonyms = self._load_synonyms(synonym_file)
        self.lineage = lineage
        self.learning_enabled = learning_enabled
        self.string_match_threshold = 0.75
        self.synonym_match_threshold = 0.8
        self.learned_mappings = {}
        
        # Load learned mappings from lineage if available
        if self.lineage and self.learning_enabled:
            self._load_learned_mappings()
    
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
        
        # Add default automotive synonyms
        self._add_default_synonyms(synonyms)
        
        return synonyms
    
    def _add_default_synonyms(self, synonyms: Dict[str, List[str]]) -> None:
        """
        Add default automotive industry synonyms if not present.
        
        Args:
            synonyms: Existing synonyms dictionary to update
        """
        default_synonyms = {
            # Vehicle information
            'vehicle_make': [
                'make', 'car_make', 'manufacturer', 'brand', 'veh_make', 'mfr', 'manuf',
                'carmake', 'auto_make', 'automobile_make', 'vehicle_manufacturer'
            ],
            'vehicle_model': [
                'model', 'car_model', 'veh_model', 'mdl', 'carmodel', 'auto_model',
                'automobile_model', 'vehicle_mdl', 'model_name'
            ],
            'vehicle_year': [
                'year', 'model_year', 'yr', 'veh_year', 'car_year', 'year_model',
                'manufacture_year', 'production_year', 'vehicle_yr'
            ],
            'vehicle_vin': [
                'vin', 'vin_number', 'vehicle_identification_number', 'vin_num',
                'vin_code', 'vehicle_id', 'vehicle_number', 'id_number'
            ],
            'vehicle_trim': [
                'trim', 'trim_level', 'series', 'grade', 'model_trim', 'vehicle_grade',
                'car_trim', 'trim_package', 'edition'
            ],
            'vehicle_color': [
                'color', 'exterior_color', 'ext_color', 'car_color', 'paint_color',
                'vehicle_paint', 'body_color', 'color_ext', 'paint'
            ],
            
            # Sales information
            'sale_date': [
                'date', 'sold_date', 'purchase_date', 'transaction_date', 'deal_date',
                'date_sold', 'date_of_sale', 'sales_date', 'closing_date', 'contract_date'
            ],
            'sale_price': [
                'price', 'selling_price', 'sold_price', 'transaction_price', 'deal_amount',
                'amount', 'final_price', 'sales_price', 'vehicle_price', 'contract_price'
            ],
            'customer_name': [
                'name', 'client_name', 'buyer_name', 'purchaser_name', 'customer',
                'client', 'buyer', 'purchaser', 'customer_full_name', 'client_full_name'
            ],
            'salesperson': [
                'sales_person', 'sales_rep', 'sales_associate', 'seller', 'sales_agent',
                'associate', 'rep', 'sales_consultant', 'sales_advisor'
            ],
            
            # Financial information
            'down_payment': [
                'down_pmt', 'downpayment', 'down', 'deposit', 'initial_payment',
                'cash_down', 'down_amount', 'initial_deposit', 'cash_deposit'
            ],
            'monthly_payment': [
                'monthly_pmt', 'monthly', 'payment', 'monthly_amount', 'monthly_installment',
                'installment', 'payment_amount', 'monthly_payment_amount'
            ],
            'interest_rate': [
                'rate', 'apr', 'annual_percentage_rate', 'finance_rate', 'loan_rate',
                'interest', 'finance_charge_rate', 'percentage_rate'
            ],
            'term_months': [
                'term', 'loan_term', 'finance_term', 'months', 'loan_months',
                'finance_months', 'loan_length', 'finance_length', 'term_length'
            ]
        }
        
        # Add default synonyms if not already present
        for key, values in default_synonyms.items():
            if key not in synonyms:
                synonyms[key] = values
            else:
                # Add any missing synonyms to existing entries
                existing = set(synonyms[key])
                for value in values:
                    if value not in existing:
                        synonyms[key].append(value)
    
    def _load_learned_mappings(self) -> None:
        """Load learned mappings from lineage history."""
        if not self.lineage:
            return
            
        try:
            # Get all vendor mappings
            all_mappings = []
            for vendor in ['upload', 'csv', 'manual']:
                mappings = self.lineage.get_vendor_column_mappings(vendor)
                all_mappings.extend(mappings)
            
            # Process mappings
            for mapping in all_mappings:
                source = mapping.get('source_id')
                target = mapping.get('target_id')
                confidence = mapping.get('metadata', {}).get('confidence', 0.0)
                
                if source and target and confidence >= 0.8:
                    if source not in self.learned_mappings:
                        self.learned_mappings[source] = {}
                    
                    # Count occurrences of each target mapping
                    if target not in self.learned_mappings[source]:
                        self.learned_mappings[source][target] = 1
                    else:
                        self.learned_mappings[source][target] += 1
            
            logger.info(f"Loaded {len(self.learned_mappings)} learned mappings from lineage")
            
        except Exception as e:
            logger.error(f"Error loading learned mappings: {str(e)}")
    
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
    
    def _check_synonym_match(self, source_column: str, target_column: str) -> Tuple[bool, float, str]:
        """
        Check if source column matches a synonym of the target column.
        
        Args:
            source_column: Source column name
            target_column: Target column name
            
        Returns:
            Tuple of (is_match, confidence, reason)
        """
        # Normalize column names
        source_norm = self._normalize_column_name(source_column)
        target_norm = self._normalize_column_name(target_column)
        
        # Check direct match
        if source_norm == target_norm:
            return True, 1.0, "Exact match"
        
        # Check if target is in synonyms dictionary
        if target_norm in self.synonyms:
            for synonym in self.synonyms[target_norm]:
                synonym_norm = self._normalize_column_name(synonym)
                
                # Check exact synonym match
                if source_norm == synonym_norm:
                    return True, 0.95, f"Exact synonym match: '{synonym}'"
                
                # Check fuzzy synonym match
                similarity = self._calculate_string_similarity(source_norm, synonym_norm)
                if similarity >= self.synonym_match_threshold:
                    return True, similarity * 0.9, f"Fuzzy synonym match: '{synonym}' ({similarity:.2f})"
        
        # Check learned mappings
        if source_norm in self.learned_mappings and target_norm in self.learned_mappings[source_norm]:
            count = self.learned_mappings[source_norm][target_norm]
            confidence = min(0.9, 0.7 + (count / 10) * 0.2)  # Increase confidence with frequency, max 0.9
            return True, confidence, f"Learned from {count} previous mappings"
        
        # No synonym match
        return False, 0.0, ""
    
    def _get_column_data_type(self, df: pd.DataFrame, column: str) -> str:
        """
        Detect the data type of a column based on its content.
        
        Args:
            df: DataFrame containing the column
            column: Column name
            
        Returns:
            Detected data type as string
        """
        if column not in df.columns:
            return "unknown"
            
        # Check if already numeric
        if pd.api.types.is_numeric_dtype(df[column]):
            if pd.api.types.is_integer_dtype(df[column]):
                return "integer"
            else:
                return "float"
                
        # Check if already datetime
        if pd.api.types.is_datetime64_dtype(df[column]):
            return "datetime"
            
        # Check if boolean
        if pd.api.types.is_bool_dtype(df[column]):
            return "boolean"
            
        # Sample values for type detection
        sample = df[column].dropna().head(10)
        if len(sample) == 0:
            return "unknown"
            
        # Try to convert to datetime
        try:
            pd.to_datetime(sample, errors='raise')
            return "datetime"
        except:
            pass
            
        # Try to convert to numeric
        try:
            # Check if values look like currency
            if all(str(val).strip().startswith('$') for val in sample if pd.notna(val) and str(val).strip()):
                return "currency"
                
            # Check if values look like percentages
            if all(str(val).strip().endswith('%') for val in sample if pd.notna(val) and str(val).strip()):
                return "percentage"
                
            # Try numeric conversion
            numeric_sample = pd.to_numeric(sample, errors='raise')
            if all(float(val).is_integer() for val in numeric_sample if pd.notna(val)):
                return "integer"
            else:
                return "float"
        except:
            pass
            
        # Check if looks like VIN
        if all(len(str(val)) == 17 for val in sample if pd.notna(val) and str(val).strip()):
            return "vin"
            
        # Default to string
        return "string"
    
    def _generate_data_type_suggestions(self, df: pd.DataFrame, 
                                      schema_columns: List[SchemaColumn]) -> Dict[str, List[str]]:
        """
        Generate data type based mapping suggestions.
        
        Args:
            df: Source DataFrame
            schema_columns: List of schema columns
            
        Returns:
            Dictionary mapping data types to lists of suggested columns
        """
        type_suggestions = {}
        
        # Group schema columns by data type
        schema_by_type = {}
        for col in schema_columns:
            if col.data_type not in schema_by_type:
                schema_by_type[col.data_type] = []
            schema_by_type[col.data_type].append(col.name)
        
        # Detect data types of source columns
        for column in df.columns:
            data_type = self._get_column_data_type(df, column)
            
            # Map detected types to schema types
            if data_type == "datetime" and "date" in schema_by_type:
                type_suggestions[column] = schema_by_type["date"]
            elif data_type == "datetime" and "datetime" in schema_by_type:
                type_suggestions[column] = schema_by_type["datetime"]
            elif data_type in ["float", "currency"] and "float" in schema_by_type:
                type_suggestions[column] = schema_by_type["float"]
            elif data_type in ["float", "currency"] and "number" in schema_by_type:
                type_suggestions[column] = schema_by_type["number"]
            elif data_type == "integer" and "integer" in schema_by_type:
                type_suggestions[column] = schema_by_type["integer"]
            elif data_type == "integer" and "int" in schema_by_type:
                type_suggestions[column] = schema_by_type["int"]
            elif data_type == "percentage" and "float" in schema_by_type:
                type_suggestions[column] = schema_by_type["float"]
            elif data_type == "vin" and any("vin" in col.lower() for col in schema_by_type.get("string", [])):
                type_suggestions[column] = [col for col in schema_by_type.get("string", []) if "vin" in col.lower()]
        
        return type_suggestions
    
    def map_columns(self, df: pd.DataFrame, 
                   schema_profile: SchemaProfile,
                   vendor: Optional[str] = None) -> Tuple[Dict[str, str], List[MappingSuggestion]]:
        """
        Map source columns to schema columns with enhanced suggestions.
        
        Args:
            df: Source DataFrame
            schema_profile: Schema profile to map against
            vendor: Optional vendor name for lineage tracking
            
        Returns:
            Tuple of (column_mapping, suggestions)
        """
        source_columns = df.columns.tolist()
        schema_columns = schema_profile.columns
        
        # Initialize results
        column_mapping = {}
        suggestions = []
        
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
                is_synonym, confidence, reason = self._check_synonym_match(source_col, schema_col.name)
                if is_synonym:
                    column_mapping[source_col] = schema_col.name
                    mapped_targets.add(schema_col.name)
                    matched = True
                    
                    # Track mapping in lineage if available
                    if self.lineage and vendor:
                        self.lineage.track_column_mapping(
                            source_column=source_col,
                            target_column=schema_col.name,
                            confidence=confidence,
                            vendor=vendor,
                            metadata={"reason": reason}
                        )
                    
                    break
                
                # Check against aliases
                for alias in schema_col.aliases:
                    if self._normalize_column_name(source_col) == self._normalize_column_name(alias):
                        column_mapping[source_col] = schema_col.name
                        mapped_targets.add(schema_col.name)
                        matched = True
                        
                        # Track mapping in lineage
                        if self.lineage and vendor:
                            self.lineage.track_column_mapping(
                                source_column=source_col,
                                target_column=schema_col.name,
                                confidence=0.9,
                                vendor=vendor,
                                metadata={"reason": f"Matched on alias '{alias}'"}
                            )
                        
                        break
                
                if matched:
                    break
        
        # Second pass: Try fuzzy matching for remaining columns
        unmapped_columns = [col for col in source_columns if col not in column_mapping]
        
        # Get data type suggestions
        type_suggestions = self._generate_data_type_suggestions(df, schema_columns)
        
        for source_col in unmapped_columns:
            best_matches = []
            
            # Check string similarity with schema column names
            for schema_col in schema_columns:
                # Skip already mapped targets
                if schema_col.name in mapped_targets:
                    continue
                
                # Check string similarity
                similarity = self._calculate_string_similarity(source_col, schema_col.name)
                if similarity >= self.string_match_threshold:
                    best_matches.append((schema_col.name, similarity, "String similarity"))
                
                # Check aliases
                for alias in schema_col.aliases:
                    similarity = self._calculate_string_similarity(source_col, alias)
                    if similarity >= self.string_match_threshold:
                        best_matches.append((schema_col.name, similarity, f"Similar to alias '{alias}'"))
            
            # Add data type based suggestions
            if source_col in type_suggestions:
                for target_col in type_suggestions[source_col]:
                    if target_col not in mapped_targets:
                        # Lower confidence for data type matches
                        best_matches.append((target_col, 0.7, "Data type match"))
            
            # Sort by confidence
            best_matches.sort(key=lambda x: x[1], reverse=True)
            
            if best_matches:
                # Use the best match
                best_target, best_score, best_reason = best_matches[0]
                
                # Create alternatives list from other matches
                alternatives = [match[0] for match in best_matches[1:4]]  # Up to 3 alternatives
                
                # Create suggestion
                suggestion = MappingSuggestion(
                    source_column=source_col,
                    target_column=best_target,
                    confidence=best_score,
                    reason=best_reason,
                    alternatives=alternatives
                )
                
                suggestions.append(suggestion)
            else:
                # No good matches found
                suggestion = MappingSuggestion(
                    source_column=source_col,
                    target_column="",
                    confidence=0.0,
                    reason="No matching column found",
                    alternatives=[]
                )
                
                suggestions.append(suggestion)
        
        return column_mapping, suggestions
    
    def apply_mapping_feedback(self, source_column: str, target_column: str, 
                             confidence: float = 0.9, vendor: str = "manual") -> None:
        """
        Apply user feedback on a mapping suggestion.
        
        Args:
            source_column: Source column name
            target_column: Correct target column name
            confidence: Confidence in the mapping
            vendor: Source of the mapping feedback
        """
        # Track in lineage if available
        if self.lineage:
            self.lineage.track_column_mapping(
                source_column=source_column,
                target_column=target_column,
                confidence=confidence,
                vendor=vendor,
                metadata={"reason": "User feedback"}
            )
        
        # Update learned mappings
        if self.learning_enabled:
            source_norm = self._normalize_column_name(source_column)
            target_norm = self._normalize_column_name(target_column)
            
            if source_norm not in self.learned_mappings:
                self.learned_mappings[source_norm] = {}
            
            if target_norm not in self.learned_mappings[source_norm]:
                self.learned_mappings[source_norm][target_norm] = 1
            else:
                self.learned_mappings[source_norm][target_norm] += 1
    
    def detect_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Auto-detect schema from DataFrame headers and content.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with detected schema information
        """
        schema_info = {
            "columns": [],
            "detected_types": {},
            "primary_key_candidates": [],
            "timestamp": datetime.now().isoformat()
        }
        
        # Analyze each column
        for column in df.columns:
            col_info = {
                "name": column,
                "data_type": self._get_column_data_type(df, column),
                "non_null_count": int(df[column].count()),
                "null_count": int(df[column].isnull().sum()),
                "unique_count": int(df[column].nunique()),
                "sample_values": df[column].dropna().head(3).tolist()
            }
            
            schema_info["columns"].append(col_info)
            schema_info["detected_types"][column] = col_info["data_type"]
            
            # Check if column could be a primary key
            if col_info["unique_count"] == len(df) and col_info["null_count"] == 0:
                schema_info["primary_key_candidates"].append(column)
        
        return schema_info
