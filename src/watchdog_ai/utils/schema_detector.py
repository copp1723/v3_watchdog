"""
Schema auto-detection system for CSV imports.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Set
from datetime import datetime
import re
import json
import os
from difflib import get_close_matches

from .adaptive_schema import SchemaProfile, SchemaColumn
from .enhanced_column_mapper import EnhancedColumnMapper

logger = logging.getLogger(__name__)

class SchemaDetector:
    """
    Automatically detects and maps CSV schemas to known profiles.
    Provides fuzzy matching and header detection capabilities.
    """
    
    def __init__(self, profiles_dir: str = "profiles",
                synonym_file: Optional[str] = None):
        """
        Initialize the schema detector.
        
        Args:
            profiles_dir: Directory containing schema profiles
            synonym_file: Path to synonym dictionary file
        """
        self.profiles_dir = profiles_dir
        self.synonym_file = synonym_file
        self.column_mapper = EnhancedColumnMapper(synonym_file=synonym_file)
        self.profiles = self._load_profiles()
    
    def _load_profiles(self) -> Dict[str, SchemaProfile]:
        """
        Load schema profiles from disk.
        
        Returns:
            Dictionary of profile ID to SchemaProfile
        """
        profiles = {}
        
        if not os.path.exists(self.profiles_dir):
            logger.warning(f"Profiles directory {self.profiles_dir} does not exist")
            return profiles
        
        for filename in os.listdir(self.profiles_dir):
            if filename.endswith('.json'):
                try:
                    with open(os.path.join(self.profiles_dir, filename), 'r') as f:
                        data = json.load(f)
                    
                    profile = SchemaProfile.from_dict(data)
                    profiles[profile.id] = profile
                    logger.info(f"Loaded schema profile: {profile.id}")
                except Exception as e:
                    logger.error(f"Error loading profile {filename}: {str(e)}")
        
        return profiles
    
    def _find_header_row(self, df: pd.DataFrame, max_rows: int = 10) -> int:
        """
        Find the most likely header row in a DataFrame.
        
        Args:
            df: DataFrame to analyze
            max_rows: Maximum number of rows to check
            
        Returns:
            Index of the most likely header row (0 if not found)
        """
        if len(df) <= 1:
            return 0
        
        # Check up to max_rows
        rows_to_check = min(max_rows, len(df))
        
        # Score each row as a potential header
        scores = []
        
        for i in range(rows_to_check):
            row = df.iloc[i]
            score = 0
            
            # Check if values look like column names
            for val in row:
                if pd.isna(val):
                    continue
                    
                val_str = str(val).strip()
                
                # Increase score for string values
                if isinstance(val, str):
                    score += 1
                
                # Increase score for values with underscores or spaces
                if '_' in val_str or ' ' in val_str:
                    score += 1
                
                # Increase score for values with mixed case (camelCase or PascalCase)
                if re.search(r'[a-z][A-Z]', val_str):
                    score += 1
                
                # Decrease score for numeric values
                try:
                    float(val_str)
                    score -= 1
                except:
                    pass
                
                # Increase score for common column name terms
                common_terms = ['id', 'name', 'date', 'time', 'price', 'cost', 
                               'total', 'count', 'number', 'code', 'status']
                for term in common_terms:
                    if term in val_str.lower():
                        score += 1
            
            scores.append(score)
        
        # Find the row with the highest score
        if not scores:
            return 0
            
        max_score = max(scores)
        if max_score <= 0:
            return 0  # No good header row found
            
        return scores.index(max_score)
    
    def _clean_header_row(self, df: pd.DataFrame, header_row: int) -> pd.DataFrame:
        """
        Clean and normalize the header row of a DataFrame.
        
        Args:
            df: DataFrame to process
            header_row: Index of the header row
            
        Returns:
            DataFrame with cleaned headers
        """
        if header_row == 0:
            # Header is already in the right place
            result = df.copy()
        else:
            # Set the header row
            result = pd.DataFrame(df.values[header_row:], columns=df.iloc[header_row])
        
        # Clean column names
        clean_columns = []
        for col in result.columns:
            if pd.isna(col):
                # Generate a name for empty columns
                clean_columns.append(f"column_{len(clean_columns)}")
            else:
                # Clean the column name
                clean_name = str(col).strip()
                
                # Replace spaces and special characters with underscores
                clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', clean_name)
                
                # Replace multiple underscores with a single underscore
                clean_name = re.sub(r'_+', '_', clean_name)
                
                # Remove leading/trailing underscores
                clean_name = clean_name.strip('_')
                
                # Ensure the name is not empty
                if not clean_name:
                    clean_name = f"column_{len(clean_columns)}"
                
                clean_columns.append(clean_name)
        
        # Handle duplicate column names
        seen_columns = set()
        for i, col in enumerate(clean_columns):
            if col in seen_columns:
                # Add a suffix to make the name unique
                j = 1
                while f"{col}_{j}" in seen_columns:
                    j += 1
                clean_columns[i] = f"{col}_{j}"
            
            seen_columns.add(clean_columns[i])
        
        # Set the cleaned column names
        result.columns = clean_columns
        
        return result
    
    def _detect_column_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Detect the data types of columns in a DataFrame.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary mapping column names to detected data types
        """
        column_types = {}
        
        for column in df.columns:
            column_types[column] = self.column_mapper._get_column_data_type(df, column)
        
        return column_types
    
    def _match_profile(self, df: pd.DataFrame, 
                     column_types: Dict[str, str]) -> List[Tuple[SchemaProfile, float]]:
        """
        Match a DataFrame to known schema profiles.
        
        Args:
            df: DataFrame to match
            column_types: Dictionary of column data types
            
        Returns:
            List of (profile, confidence) tuples, sorted by confidence
        """
        if not self.profiles:
            return []
        
        matches = []
        
        for profile_id, profile in self.profiles.items():
            # Calculate match score
            score = self._calculate_profile_match(df, column_types, profile)
            
            if score > 0.3:  # Minimum threshold for a potential match
                matches.append((profile, score))
        
        # Sort by score in descending order
        matches.sort(key=lambda x: x[1], reverse=True)
        
        return matches
    
    def _calculate_profile_match(self, df: pd.DataFrame, 
                               column_types: Dict[str, str],
                               profile: SchemaProfile) -> float:
        """
        Calculate how well a DataFrame matches a schema profile.
        
        Args:
            df: DataFrame to match
            column_types: Dictionary of column data types
            profile: Schema profile to match against
            
        Returns:
            Match confidence score (0.0 to 1.0)
        """
        # Get profile columns
        profile_columns = {col.name: col for col in profile.columns}
        
        # Count matches
        exact_matches = 0
        fuzzy_matches = 0
        type_matches = 0
        
        for col in df.columns:
            # Check for exact match
            if col in profile_columns:
                exact_matches += 1
                
                # Check if data type matches
                if column_types.get(col) == profile_columns[col].data_type:
                    type_matches += 1
                continue
            
            # Check for fuzzy match
            normalized_col = self.column_mapper._normalize_column_name(col)
            for profile_col in profile_columns.values():
                # Check against column name
                if self.column_mapper._normalize_column_name(profile_col.name) == normalized_col:
                    fuzzy_matches += 1
                    break
                
                # Check against aliases
                for alias in profile_col.aliases:
                    if self.column_mapper._normalize_column_name(alias) == normalized_col:
                        fuzzy_matches += 1
                        break
        
        # Calculate match score
        total_profile_columns = len(profile_columns)
        if total_profile_columns == 0:
            return 0.0
        
        # Weight exact matches more heavily
        match_score = (exact_matches * 1.0 + fuzzy_matches * 0.7) / total_profile_columns
        
        # Adjust score based on type matches
        if exact_matches > 0:
            type_score = type_matches / exact_matches
            match_score = match_score * 0.7 + type_score * 0.3
        
        return match_score
    
    def detect_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect the schema of a DataFrame.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with schema detection results
        """
        if df.empty:
            return {
                "success": False,
                "error": "Empty DataFrame",
                "dataframe": df
            }
        
        try:
            # Find header row
            header_row = self._find_header_row(df)
            
            # Clean and normalize headers
            cleaned_df = self._clean_header_row(df, header_row)
            
            # Detect column types
            column_types = self._detect_column_types(cleaned_df)
            
            # Match to known profiles
            profile_matches = self._match_profile(cleaned_df, column_types)
            
            # Generate schema info
            schema_info = self.column_mapper.detect_schema(cleaned_df)
            
            # Prepare result
            result = {
                "success": True,
                "header_row": header_row,
                "column_types": column_types,
                "schema_info": schema_info,
                "dataframe": cleaned_df
            }
            
            # Add profile matches if any
            if profile_matches:
                result["profile_matches"] = [
                    {
                        "profile_id": profile.id,
                        "profile_name": profile.name,
                        "confidence": confidence,
                        "role": profile.role.value if hasattr(profile.role, 'value') else profile.role
                    }
                    for profile, confidence in profile_matches
                ]
                
                # Add column mappings for best match
                best_profile, _ = profile_matches[0]
                column_mapping, suggestions = self.column_mapper.map_columns(
                    cleaned_df, best_profile
                )
                
                result["column_mapping"] = column_mapping
                result["mapping_suggestions"] = [s.to_dict() for s in suggestions]
            
            return result
            
        except Exception as e:
            logger.error(f"Error detecting schema: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "dataframe": df
            }
    
    def apply_detected_schema(self, df: pd.DataFrame, 
                            detection_result: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply a detected schema to a DataFrame.
        
        Args:
            df: Original DataFrame
            detection_result: Schema detection result
            
        Returns:
            DataFrame with applied schema
        """
        if not detection_result.get("success", False):
            return df
        
        # Use the cleaned DataFrame from detection
        result_df = detection_result.get("dataframe", df)
        
        # Apply column mapping if available
        column_mapping = detection_result.get("column_mapping")
        if column_mapping:
            # Rename columns according to mapping
            result_df = result_df.rename(columns=column_mapping)
        
        return result_df
