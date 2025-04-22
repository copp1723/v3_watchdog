"""
Column mapping engine with fuzzy matching and learning capabilities.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
from fuzzywuzzy import fuzz
import re
from datetime import datetime

logger = logging.getLogger(__name__)

class ColumnMapper:
    """Maps columns using fuzzy matching and learned patterns."""
    
    def __init__(self, schema_manager=None):
        """
        Initialize the column mapper.
        
        Args:
            schema_manager: Optional schema manager instance
        """
        self.schema_manager = schema_manager
        self.known_mappings: Dict[str, Dict[str, Any]] = {}
        self.confidence_threshold = 80  # Minimum confidence score for automatic mapping
    
    def map_columns(self, df: pd.DataFrame, profile_name: str = "default") -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Map DataFrame columns using the specified profile.
        
        Args:
            df: DataFrame to map
            profile_name: Name of the schema profile to use
            
        Returns:
            Tuple of (mapped DataFrame, mapping results)
        """
        # Load schema profile
        if self.schema_manager:
            profile = self.schema_manager.load_profile(profile_name)
        else:
            raise ValueError("Schema manager not configured")
        
        # Initialize results
        results = {
            "mapped": {},
            "unmapped": [],
            "confidence_scores": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Create new DataFrame for mapped columns
        mapped_df = pd.DataFrame(index=df.index)
        
        # Track unmapped columns
        unmapped = list(df.columns)
        
        # First pass: Try exact matches and known mappings
        for column in profile['columns']:
            target_name = column['name']
            aliases = column.get('aliases', [])
            
            # Check for exact matches
            matches = [col for col in df.columns if self._is_exact_match(col, target_name, aliases)]
            if matches:
                # Use the first match
                source_col = matches[0]
                mapped_df[target_name] = df[source_col]
                results["mapped"][target_name] = source_col
                results["confidence_scores"][target_name] = 100
                unmapped.remove(source_col)
                continue
            
            # Check known mappings
            if target_name in self.known_mappings:
                for source_col in df.columns:
                    if source_col in self.known_mappings[target_name]:
                        mapped_df[target_name] = df[source_col]
                        results["mapped"][target_name] = source_col
                        results["confidence_scores"][target_name] = self.known_mappings[target_name][source_col]['confidence']
                        unmapped.remove(source_col)
                        break
        
        # Second pass: Try fuzzy matching for remaining columns
        for column in profile['columns']:
            target_name = column['name']
            if target_name in results["mapped"]:
                continue
            
            # Find best fuzzy match
            best_match, confidence = self._find_best_fuzzy_match(
                target_name,
                unmapped,
                column.get('aliases', [])
            )
            
            if best_match and confidence >= self.confidence_threshold:
                mapped_df[target_name] = df[best_match]
                results["mapped"][target_name] = best_match
                results["confidence_scores"][target_name] = confidence
                unmapped.remove(best_match)
        
        # Record unmapped columns
        results["unmapped"] = unmapped
        
        # Copy any unmapped columns with their original names
        for col in unmapped:
            mapped_df[col] = df[col]
        
        return mapped_df, results
    
    def learn_mapping(self, source_col: str, target_col: str, confidence: float = 100) -> None:
        """
        Learn a new column mapping.
        
        Args:
            source_col: Original column name
            target_col: Target column name
            confidence: Confidence score for this mapping
        """
        if target_col not in self.known_mappings:
            self.known_mappings[target_col] = {}
        
        self.known_mappings[target_col][source_col] = {
            'confidence': confidence,
            'learned_at': datetime.now().isoformat(),
            'uses': 0
        }
    
    def _is_exact_match(self, source: str, target: str, aliases: List[str]) -> bool:
        """Check if source matches target or any aliases exactly."""
        normalized_source = self._normalize_column_name(source)
        normalized_target = self._normalize_column_name(target)
        normalized_aliases = [self._normalize_column_name(alias) for alias in aliases]
        
        return (normalized_source == normalized_target or 
                normalized_source in normalized_aliases)
    
    def _find_best_fuzzy_match(self, target: str, candidates: List[str],
                              aliases: List[str]) -> Tuple[Optional[str], float]:
        """
        Find the best fuzzy match for a target column.
        
        Args:
            target: Target column name
            candidates: List of candidate column names
            aliases: List of known aliases
            
        Returns:
            Tuple of (best match, confidence score)
        """
        best_match = None
        best_score = 0
        
        # Normalize target and aliases
        normalized_target = self._normalize_column_name(target)
        normalized_aliases = [self._normalize_column_name(alias) for alias in aliases]
        
        for candidate in candidates:
            normalized_candidate = self._normalize_column_name(candidate)
            
            # Calculate similarity scores
            scores = [
                fuzz.ratio(normalized_candidate, normalized_target),  # Basic similarity
                fuzz.partial_ratio(normalized_candidate, normalized_target),  # Partial match
                max([fuzz.ratio(normalized_candidate, alias) for alias in normalized_aliases], default=0)  # Alias match
            ]
            
            # Use the highest score
            score = max(scores)
            
            # Apply bonuses for common patterns
            score += self._calculate_pattern_bonus(normalized_candidate, normalized_target)
            
            # Cap at 100
            score = min(score, 100)
            
            if score > best_score:
                best_score = score
                best_match = candidate
        
        return best_match, best_score
    
    def _normalize_column_name(self, name: str) -> str:
        """Normalize a column name for comparison."""
        # Convert to lowercase
        name = name.lower()
        
        # Remove special characters and extra spaces
        name = re.sub(r'[^a-z0-9]', '', name)
        
        return name
    
    def _calculate_pattern_bonus(self, source: str, target: str) -> float:
        """
        Calculate bonus points for common patterns.
        
        Args:
            source: Source column name
            target: Target column name
            
        Returns:
            Bonus points to add to similarity score
        """
        bonus = 0
        
        # Common prefix/suffix patterns
        common_patterns = {
            'date': 5,
            'time': 5,
            'total': 5,
            'amount': 5,
            'price': 5,
            'cost': 5,
            'id': 3,
            'name': 3,
            'num': 3,
            'count': 3
        }
        
        for pattern, points in common_patterns.items():
            if pattern in source and pattern in target:
                bonus += points
        
        # Data type indicators
        type_patterns = {
            r'\d': 'number',
            r'^\d{4}-\d{2}-\d{2}': 'date',
            r'^\$': 'currency',
            r'%$': 'percentage'
        }
        
        for pattern, type_name in type_patterns.items():
            if re.search(pattern, source) and re.search(pattern, target):
                bonus += 5
        
        return bonus