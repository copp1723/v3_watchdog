"""
LLM-powered column mapping for Watchdog AI.

This module provides intelligent column mapping using LLM suggestions
with caching and confidence scoring.
"""

import os
import json
import hashlib
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path

from .adaptive_schema import SchemaProfile, SchemaColumn
from ..llm.llm_engine import LLMEngine

logger = logging.getLogger(__name__)

class MappingCache:
    """Cache for LLM mapping suggestions."""
    
    def __init__(self, cache_dir: str = ".cache/column_mapping",
                ttl_hours: int = 24):
        """
        Initialize the mapping cache.
        
        Args:
            cache_dir: Directory for cache files
            ttl_hours: Cache TTL in hours
        """
        self.cache_dir = cache_dir
        self.ttl = timedelta(hours=ttl_hours)
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # In-memory cache
        self.cache = {}
    
    def _compute_key(self, source_cols: List[str], 
                    target_schema: SchemaProfile) -> str:
        """Compute cache key from inputs."""
        # Sort columns for consistent hashing
        sorted_cols = sorted(source_cols)
        schema_id = target_schema.id
        
        # Combine and hash
        combined = f"{','.join(sorted_cols)}|{schema_id}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def get(self, source_cols: List[str], 
            target_schema: SchemaProfile) -> Optional[Dict[str, Any]]:
        """Get cached mapping if available and not expired."""
        key = self._compute_key(source_cols, target_schema)
        
        # Check memory cache first
        if key in self.cache:
            entry = self.cache[key]
            if datetime.now() - entry["timestamp"] < self.ttl:
                return entry["data"]
            else:
                del self.cache[key]
        
        # Check file cache
        cache_file = os.path.join(self.cache_dir, f"{key}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    entry = json.load(f)
                
                # Check expiration
                timestamp = datetime.fromisoformat(entry["timestamp"])
                if datetime.now() - timestamp < self.ttl:
                    # Update memory cache
                    self.cache[key] = {
                        "timestamp": timestamp,
                        "data": entry["data"]
                    }
                    return entry["data"]
                else:
                    # Remove expired cache file
                    os.remove(cache_file)
            except Exception as e:
                logger.error(f"Error reading cache file: {str(e)}")
        
        return None
    
    def set(self, source_cols: List[str], 
            target_schema: SchemaProfile,
            data: Dict[str, Any]) -> None:
        """Cache mapping data."""
        key = self._compute_key(source_cols, target_schema)
        timestamp = datetime.now()
        
        # Update memory cache
        self.cache[key] = {
            "timestamp": timestamp,
            "data": data
        }
        
        # Update file cache
        cache_file = os.path.join(self.cache_dir, f"{key}.json")
        try:
            with open(cache_file, 'w') as f:
                json.dump({
                    "timestamp": timestamp.isoformat(),
                    "data": data
                }, f)
        except Exception as e:
            logger.error(f"Error writing cache file: {str(e)}")


class LLMColumnMapper:
    """Intelligent column mapping using LLM suggestions."""
    
    def __init__(self, cache_dir: str = ".cache/column_mapping",
                confidence_threshold: float = 0.7):
        """
        Initialize the LLM column mapper.
        
        Args:
            cache_dir: Directory for caching mapping results
            confidence_threshold: Minimum confidence for automatic mapping
        """
        self.llm = LLMEngine()
        self.cache = MappingCache(cache_dir)
        self.confidence_threshold = confidence_threshold
        self.learned_mappings = {}
    
    def get_mapping_confidence(self, source_col: str,
                             target_col: SchemaColumn) -> float:
        """
        Calculate mapping confidence for a column pair.
        
        Args:
            source_col: Source column name
            target_col: Target schema column
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        # Check learned mappings first
        if source_col in self.learned_mappings:
            learned = self.learned_mappings[source_col]
            if learned["target"] == target_col.name:
                return learned["confidence"]
        
        # Calculate base confidence using column matching
        confidence = target_col.matches_query_term(source_col)
        
        # Boost confidence for exact matches
        if source_col.lower() == target_col.name.lower():
            confidence = 1.0
        elif source_col.lower() == target_col.display_name.lower():
            confidence = 0.95
        
        # Check aliases
        elif any(source_col.lower() == alias.lower() for alias in target_col.aliases):
            confidence = max(confidence, 0.9)
        
        return confidence
    
    async def get_llm_suggestions(self, source_cols: List[str],
                                target_schema: SchemaProfile) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get mapping suggestions from LLM.
        
        Args:
            source_cols: Source column names
            target_schema: Target schema profile
            
        Returns:
            Dictionary mapping source columns to suggested targets
        """
        # Check cache first
        cached = self.cache.get(source_cols, target_schema)
        if cached:
            return cached
        
        try:
            # Prepare prompt for LLM
            prompt = self._create_mapping_prompt(source_cols, target_schema)
            
            # Get LLM response
            response = await self.llm.generate_completion(prompt)
            
            # Parse suggestions
            suggestions = self._parse_llm_response(response, source_cols, target_schema)
            
            # Cache results
            self.cache.set(source_cols, target_schema, suggestions)
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Error getting LLM suggestions: {str(e)}")
            return {}
    
    def _create_mapping_prompt(self, source_cols: List[str],
                             target_schema: SchemaProfile) -> str:
        """Create prompt for LLM mapping request."""
        # Build schema description
        schema_desc = []
        for col in target_schema.columns:
            aliases = ", ".join(col.aliases) if col.aliases else "none"
            schema_desc.append(
                f"- {col.name} ({col.data_type}): {col.description}\n"
                f"  Aliases: {aliases}"
            )
        
        # Create prompt
        prompt = f"""Given these source column names:
{", ".join(source_cols)}

And this target schema:
{"".join(schema_desc)}

Suggest mappings between source columns and target schema columns.
For each source column, provide:
1. Best matching target column
2. Confidence score (0.0 to 1.0)
3. Reasoning for the match

Format as JSON with structure:
{{
    "mappings": {{
        "source_column": {{
            "target": "target_column",
            "confidence": 0.95,
            "reason": "Exact match with alias"
        }}
    }}
}}"""
        
        return prompt
    
    def _parse_llm_response(self, response: str,
                          source_cols: List[str],
                          target_schema: SchemaProfile) -> Dict[str, List[Dict[str, Any]]]:
        """Parse LLM response into mapping suggestions."""
        try:
            # Parse JSON response
            data = json.loads(response)
            
            # Validate and normalize suggestions
            suggestions = {}
            
            for source_col in source_cols:
                if source_col in data["mappings"]:
                    mapping = data["mappings"][source_col]
                    
                    # Validate target column exists
                    target_col = target_schema.get_column_by_name(mapping["target"])
                    if not target_col:
                        continue
                    
                    # Validate confidence
                    confidence = float(mapping["confidence"])
                    confidence = max(0.0, min(1.0, confidence))
                    
                    suggestions[source_col] = [{
                        "target": mapping["target"],
                        "confidence": confidence,
                        "reason": mapping.get("reason", "LLM suggestion")
                    }]
                    
                    # Add alternative suggestions if available
                    if "alternatives" in mapping:
                        for alt in mapping["alternatives"]:
                            alt_target = target_schema.get_column_by_name(alt["target"])
                            if alt_target:
                                alt_confidence = float(alt["confidence"])
                                suggestions[source_col].append({
                                    "target": alt["target"],
                                    "confidence": alt_confidence,
                                    "reason": alt.get("reason", "Alternative suggestion")
                                })
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {str(e)}")
            return {}
    
    def persist_learned_mapping(self, source: str,
                              target: str,
                              confidence: float) -> None:
        """
        Save a learned mapping.
        
        Args:
            source: Source column name
            target: Target column name
            confidence: Confidence score
        """
        self.learned_mappings[source] = {
            "target": target,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        }