"""
Adaptive Schema System for Watchdog AI.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class SchemaColumn:
    """Definition of a single column in the schema."""
    name: str
    display_name: str
    description: str
    data_type: str
    metric_type: Optional[str] = None
    visibility: str = "public"
    format: Optional[str] = None
    aliases: List[str] = None
    aggregations: List[str] = None
    business_rules: List[Dict[str, Any]] = None

    def __post_init__(self):
        """Initialize default values for lists."""
        self.aliases = self.aliases or []
        self.aggregations = self.aggregations or []
        self.business_rules = self.business_rules or []

@dataclass
class SchemaProfile:
    """Schema profile definition."""
    id: str
    name: str
    description: str
    role: str
    columns: List[SchemaColumn]
    default_metrics: List[str]
    default_dimensions: List[str]
    created_at: str
    updated_at: str

@dataclass
class SchemaAdjustment:
    """User-specific schema adjustment."""
    user_id: str
    column_name: str
    adjustment_type: str
    value: Any
    timestamp: str = None

    def __post_init__(self):
        """Set timestamp if not provided."""
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

class SchemaProfileManager:
    """Manages schema profiles and adjustments."""
    
    def __init__(self, profiles_dir: str = "config/schema_profiles"):
        """Initialize the schema profile manager."""
        self.profiles_dir = profiles_dir
        self.adjustments_dir = os.path.join(profiles_dir, "adjustments")
        
        # Create directories if they don't exist
        os.makedirs(self.profiles_dir, exist_ok=True)
        os.makedirs(self.adjustments_dir, exist_ok=True)
        
        # Load profiles
        self.profiles = self._load_profiles()
    
    def _load_profiles(self) -> Dict[str, SchemaProfile]:
        """Load all schema profiles."""
        profiles = {}
        
        for filename in os.listdir(self.profiles_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(self.profiles_dir, filename)
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        
                        # Convert columns to SchemaColumn objects
                        columns = []
                        for col_data in data.get('columns', []):
                            columns.append(SchemaColumn(**col_data))
                        
                        # Create profile
                        profile = SchemaProfile(
                            id=data['id'],
                            name=data['name'],
                            description=data['description'],
                            role=data['role'],
                            columns=columns,
                            default_metrics=data.get('default_metrics', []),
                            default_dimensions=data.get('default_dimensions', []),
                            created_at=data.get('created_at', datetime.now().isoformat()),
                            updated_at=data.get('updated_at', datetime.now().isoformat())
                        )
                        
                        profiles[profile.id] = profile
                        
                except Exception as e:
                    logger.error(f"Error loading profile from {file_path}: {e}")
        
        return profiles
    
    def get_profile(self, profile_id: str) -> Optional[SchemaProfile]:
        """Get a schema profile by ID."""
        return self.profiles.get(profile_id)
    
    def add_adjustment(self, adjustment: SchemaAdjustment) -> bool:
        """Add a schema adjustment for a user."""
        file_path = os.path.join(self.adjustments_dir, f"{adjustment.user_id}_adjustments.json")
        
        try:
            # Load existing adjustments
            adjustments = []
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    adjustments = json.load(f)
            
            # Add new adjustment
            adjustments.append({
                "user_id": adjustment.user_id,
                "column_name": adjustment.column_name,
                "adjustment_type": adjustment.adjustment_type,
                "value": adjustment.value,
                "timestamp": adjustment.timestamp
            })
            
            # Save adjustments
            with open(file_path, 'w') as f:
                json.dump(adjustments, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving adjustment: {e}")
            return False
    
    def get_adjusted_profile(self, base_profile_id: str, user_id: str) -> Optional[SchemaProfile]:
        """Get a profile with user adjustments applied."""
        # Get base profile
        base_profile = self.get_profile(base_profile_id)
        if not base_profile:
            return None
        
        # Load user adjustments
        adjustments = self._load_user_adjustments(user_id)
        if not adjustments:
            return base_profile
        
        # Create a copy of the profile
        adjusted_profile = SchemaProfile(
            id=f"{base_profile.id}_user_{user_id}",
            name=f"{base_profile.name} (Adjusted)",
            description=base_profile.description,
            role=base_profile.role,
            columns=base_profile.columns.copy(),
            default_metrics=base_profile.default_metrics.copy(),
            default_dimensions=base_profile.default_dimensions.copy(),
            created_at=base_profile.created_at,
            updated_at=datetime.now().isoformat()
        )
        
        # Apply adjustments
        for adj in adjustments:
            if adj["adjustment_type"] == "alias":
                # Find column and add alias
                for col in adjusted_profile.columns:
                    if col.name == adj["column_name"]:
                        if adj["value"] not in col.aliases:
                            col.aliases.append(adj["value"])
        
        return adjusted_profile
    
    def _load_user_adjustments(self, user_id: str) -> List[Dict[str, Any]]:
        """Load adjustments for a specific user."""
        file_path = os.path.join(self.adjustments_dir, f"{user_id}_adjustments.json")
        
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading adjustments: {e}")
        
        return []