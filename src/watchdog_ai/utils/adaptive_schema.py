"""
Adaptive Schema System for Watchdog AI.

This module provides a dynamic schema system that can adapt to different user roles,
preferences, and feedback. It loads schema profiles from disk and applies per-user
schema adjustments at runtime.
"""

import os
import json
import logging
import pandas as pd
from typing import Dict, List, Set, Optional, Any, Tuple, Union
from dataclasses import dataclass, field, fields
from enum import Enum
from datetime import datetime
from pathlib import Path

# Optional YAML support
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class MappingSuggestion:
    """Suggestion for mapping a column to a schema field."""
    source_column: str
    target_column: str
    confidence: float
    reason: str
    alternatives: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

# Default paths
DEFAULT_PROFILES_DIR = "profiles"
DEFAULT_USER_PROFILES_DIR = "user_profiles"
DEFAULT_ADJUSTMENTS_DIR = "schema_adjustments"

class ColumnVisibility(str, Enum):
    """Visibility level for schema columns."""
    PUBLIC = "public"     # Visible to all roles
    RESTRICTED = "restricted"  # Visible only to specific roles
    PRIVATE = "private"   # Visible only to highest authority roles

class MetricType(str, Enum):
    """Types of metrics available in the system."""
    FINANCIAL = "financial"
    SALES = "sales"
    INVENTORY = "inventory"
    MARKETING = "marketing"
    SERVICE = "service"
    CUSTOMER = "customer"
    OPERATIONAL = "operational"

class ExecRole(str, Enum):
    """Executive roles with distinct schema profiles."""
    GENERAL_MANAGER = "general_manager"
    GENERAL_SALES_MANAGER = "general_sales_manager"
    FINANCE_MANAGER = "finance_manager"
    SERVICE_MANAGER = "service_manager"
    MARKETING_MANAGER = "marketing_manager"
    INVENTORY_MANAGER = "inventory_manager"
    
    @classmethod
    def from_string(cls, role_str: str) -> 'ExecRole':
        """Convert string to ExecRole, with fallback to GENERAL_MANAGER."""
        try:
            return cls(role_str.lower())
        except ValueError:
            logger.warning(f"Unknown role '{role_str}', defaulting to GENERAL_MANAGER")
            return cls.GENERAL_MANAGER

@dataclass
class SchemaColumn:
    """Definition of a single column in the schema."""
    name: str
    display_name: str
    description: str
    data_type: str
    metric_type: Optional[str] = None
    visibility: str = "public"
    allowed_roles: List[str] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)
    format: Optional[str] = None
    units: Optional[str] = None
    aggregations: List[str] = field(default_factory=list)
    primary_groupings: List[str] = field(default_factory=list)
    related_columns: List[str] = field(default_factory=list)
    sample_queries: List[str] = field(default_factory=list)
    business_rules: List[Dict[str, Any]] = field(default_factory=list)
    
    def is_visible_to(self, role: Union[ExecRole, str]) -> bool:
        """Check if this column is visible to the specified role."""
        if isinstance(role, str):
            role = ExecRole.from_string(role)
            
        if self.visibility == ColumnVisibility.PUBLIC.value:
            return True
        
        if self.visibility == ColumnVisibility.PRIVATE.value and role == ExecRole.GENERAL_MANAGER:
            return True
            
        return role.value in self.allowed_roles
    
    def matches_query_term(self, term: str) -> float:
        """
        Calculate how well this column matches a query term.
        Returns a confidence score between 0.0 and 1.0.
        """
        # 1. Exact match with column name or display name
        if term.lower() == self.name.lower() or term.lower() == self.display_name.lower():
            return 1.0
            
        # 2. Check aliases
        if any(term.lower() == alias.lower() for alias in self.aliases):
            return 0.9
            
        # 3. Partial matches
        if term.lower() in self.name.lower() or term.lower() in self.display_name.lower():
            return 0.7
            
        # 4. Partial matches in aliases
        if any(term.lower() in alias.lower() for alias in self.aliases):
            return 0.6
            
        # 5. Check description
        if term.lower() in self.description.lower():
            return 0.5
            
        # 6. Check sample queries
        if any(term.lower() in query.lower() for query in self.sample_queries):
            return 0.4
            
        return 0.0
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SchemaColumn':
        """Create SchemaColumn from dictionary."""
        # Filter out any unexpected fields to avoid constructor errors
        filtered_data = {
            k: v for k, v in data.items() 
            if k in cls.__annotations__ or k in [f.name for f in fields(cls)]
        }
        return cls(**filtered_data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert SchemaColumn to dictionary."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "data_type": self.data_type,
            "metric_type": self.metric_type,
            "visibility": self.visibility,
            "allowed_roles": self.allowed_roles,
            "aliases": self.aliases,
            "format": self.format,
            "units": self.units,
            "aggregations": self.aggregations,
            "primary_groupings": self.primary_groupings,
            "related_columns": self.related_columns,
            "sample_queries": self.sample_queries,
            "business_rules": self.business_rules
        }

@dataclass
class SchemaProfile:
    """Schema profile definition with role-specific column visibility."""
    id: str
    name: str
    description: str
    role: Union[ExecRole, str]
    columns: List[SchemaColumn] = field(default_factory=list)
    default_metrics: List[str] = field(default_factory=list)
    default_dimensions: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def __post_init__(self):
        """Ensure role is an ExecRole enum."""
        if isinstance(self.role, str):
            self.role = ExecRole.from_string(self.role)
    
    def get_visible_columns(self, role: Optional[Union[ExecRole, str]] = None) -> List[SchemaColumn]:
        """Get all columns visible to the specified role (or profile's role if not specified)."""
        target_role = role if role is not None else self.role
        if isinstance(target_role, str):
            target_role = ExecRole.from_string(target_role)
            
        return [col for col in self.columns if col.is_visible_to(target_role)]
    
    def get_column_by_name(self, name: str) -> Optional[SchemaColumn]:
        """Get a column by its name."""
        for col in self.columns:
            if col.name.lower() == name.lower():
                return col
        return None
    
    def find_matching_columns(self, query_terms: List[str], threshold: float = 0.4) -> Dict[str, List[Tuple[SchemaColumn, float]]]:
        """
        Find columns that match the query terms.
        Returns a dictionary mapping query terms to lists of (column, confidence) tuples.
        """
        results = {}
        
        for term in query_terms:
            term_matches = []
            
            for col in self.get_visible_columns():
                confidence = col.matches_query_term(term)
                if confidence >= threshold:
                    term_matches.append((col, confidence))
            
            if term_matches:
                # Sort by confidence score
                term_matches.sort(key=lambda x: x[1], reverse=True)
                results[term] = term_matches
                
        return results
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SchemaProfile':
        """Create a SchemaProfile from a dictionary."""
        # Convert column dictionaries to SchemaColumn objects
        columns = []
        for col_data in data.get("columns", []):
            if isinstance(col_data, dict):
                columns.append(SchemaColumn.from_dict(col_data))
            elif isinstance(col_data, SchemaColumn):
                columns.append(col_data)
                
        # Create profile with basic attributes
        profile_data = {
            "id": data.get("id", ""),
            "name": data.get("name", ""),
            "description": data.get("description", ""),
            "role": data.get("role", "general_manager"),
            "columns": columns,
            "default_metrics": data.get("default_metrics", []),
            "default_dimensions": data.get("default_dimensions", []),
            "created_at": data.get("created_at", datetime.now().isoformat()),
            "updated_at": data.get("updated_at", datetime.now().isoformat())
        }
        
        return cls(**profile_data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert SchemaProfile to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "role": self.role.value,
            "columns": [col.to_dict() for col in self.columns],
            "default_metrics": self.default_metrics,
            "default_dimensions": self.default_dimensions,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }

@dataclass
class SchemaAdjustment:
    """Represents a user-specific schema adjustment."""
    user_id: str
    column_name: str
    adjustment_type: str  # "alias", "visibility", "relevance", etc.
    value: Any
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    source: str = "user_feedback"  # or "system_learning"
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "user_id": self.user_id,
            "column_name": self.column_name,
            "adjustment_type": self.adjustment_type,
            "value": self.value,
            "timestamp": self.timestamp,
            "source": self.source,
            "confidence": self.confidence
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SchemaAdjustment':
        """Create from dictionary representation."""
        return cls(
            user_id=data["user_id"],
            column_name=data["column_name"],
            adjustment_type=data["adjustment_type"],
            value=data["value"],
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            source=data.get("source", "user_feedback"),
            confidence=data.get("confidence", 1.0)
        )

class AdaptiveSchema:
    """Manages loading, storing, and applying schema profiles and adjustments."""
    
    def __init__(self, 
                profiles_dir: str = DEFAULT_PROFILES_DIR,
                user_profiles_dir: str = DEFAULT_USER_PROFILES_DIR,
                adjustments_dir: str = DEFAULT_ADJUSTMENTS_DIR):
        """Initialize the schema profile manager."""
        self.profiles_dir = profiles_dir
        self.user_profiles_dir = user_profiles_dir
        self.adjustments_dir = adjustments_dir
        
        # Ensure directories exist
        os.makedirs(profiles_dir, exist_ok=True)
        os.makedirs(user_profiles_dir, exist_ok=True)
        os.makedirs(adjustments_dir, exist_ok=True)
        
        # Cache for profiles and adjustments
        self.profiles = {}
        self.user_adjustments = {}
        
        # Load profiles
        self.load_profiles()
    
    def load_profiles(self) -> Dict[str, SchemaProfile]:
        """Load all schema profiles from the profiles directory."""
        profiles = {}
        
        if not os.path.exists(self.profiles_dir):
            logger.warning(f"Profiles directory {self.profiles_dir} does not exist")
            return profiles
            
        for filename in os.listdir(self.profiles_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(self.profiles_dir, filename)
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    profile = SchemaProfile.from_dict(data)
                    profiles[profile.id] = profile
                except Exception as e:
                    logger.error(f"Error loading profile from {file_path}: {e}")
            elif YAML_AVAILABLE and filename.endswith(('.yml', '.yaml')):
                file_path = os.path.join(self.profiles_dir, filename)
                try:
                    with open(file_path, 'r') as f:
                        data = yaml.safe_load(f)
                    profile = SchemaProfile.from_dict(data)
                    profiles[profile.id] = profile
                except Exception as e:
                    logger.error(f"Error loading profile from {file_path}: {e}")
                    
        return profiles
    
    def validate_with_suggestions(self, df: pd.DataFrame) -> Tuple[Dict[str, Any], List[MappingSuggestion]]:
        """
        Validate a DataFrame against the schema and provide mapping suggestions.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (validation result, list of suggestions)
        """
        # For now, return a simple validation result
        result = {
            "is_valid": True,
            "errors": []
        }
        suggestions = []
        
        # Example suggestion for testing
        if len(df.columns) > 0:
            suggestions.append(MappingSuggestion(
                source_column=df.columns[0],
                target_column="standard_name",
                confidence=0.8,
                reason="Column name matches standard schema",
                alternatives=["alt1", "alt2"],
                metadata={"type": "column_mapping"}
            ))
            
        return result, suggestions