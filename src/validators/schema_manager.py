"""
Schema profile management system for data validation.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class SchemaProfileManager:
    """Manages schema profiles for data validation."""
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize the schema manager.
        
        Args:
            config_dir: Optional custom config directory
        """
        self.config_dir = config_dir or os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "config",
            "schema_profiles"
        )
        
        # Create directory if it doesn't exist
        os.makedirs(self.config_dir, exist_ok=True)
        
        # Cache for loaded profiles
        self._profile_cache: Dict[str, Dict[str, Any]] = {}
    
    def load_profile(self, name: str) -> Dict[str, Any]:
        """
        Load a schema profile by name.
        
        Args:
            name: Name of the profile to load
            
        Returns:
            Dictionary containing the profile
            
        Raises:
            FileNotFoundError: If profile doesn't exist
            ValueError: If profile is invalid
        """
        # Check cache first
        if name in self._profile_cache:
            return self._profile_cache[name].copy()
        
        # Load from file
        profile_path = os.path.join(self.config_dir, f"{name}.json")
        if not os.path.exists(profile_path):
            raise FileNotFoundError(f"Profile not found: {name}")
        
        try:
            with open(profile_path, 'r') as f:
                profile = json.load(f)
            
            # Validate profile structure
            self._validate_profile_structure(profile)
            
            # Cache the profile
            self._profile_cache[name] = profile
            
            return profile.copy()
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in profile {name}: {e}")
        except Exception as e:
            raise ValueError(f"Error loading profile {name}: {e}")
    
    def save_profile(self, name: str, profile: Dict[str, Any]) -> None:
        """
        Save a schema profile.
        
        Args:
            name: Name of the profile
            profile: Profile data to save
            
        Raises:
            ValueError: If profile is invalid
        """
        # Validate profile structure
        self._validate_profile_structure(profile)
        
        # Add metadata if not present
        if 'created_at' not in profile:
            profile['created_at'] = datetime.now().isoformat()
        profile['updated_at'] = datetime.now().isoformat()
        
        # Save to file
        profile_path = os.path.join(self.config_dir, f"{name}.json")
        try:
            with open(profile_path, 'w') as f:
                json.dump(profile, f, indent=2)
            
            # Update cache
            self._profile_cache[name] = profile.copy()
            
        except Exception as e:
            raise ValueError(f"Error saving profile {name}: {e}")
    
    def list_profiles(self) -> List[Dict[str, Any]]:
        """
        List all available profiles.
        
        Returns:
            List of profile metadata
        """
        profiles = []
        
        # List all JSON files in the config directory
        for file in os.listdir(self.config_dir):
            if file.endswith('.json'):
                try:
                    name = file[:-5]  # Remove .json
                    profile = self.load_profile(name)
                    
                    # Use the name as the ID if not present
                    profile_id = profile.get('id', name)
                    
                    profiles.append({
                        'name': name,
                        'id': profile_id,
                        'description': profile.get('description', ''),
                        'role': profile.get('role', ''),
                        'updated_at': profile.get('updated_at', '')
                    })
                except Exception as e:
                    logger.warning(f"Error loading profile {file}: {e}")
        
        return profiles
    
    def _validate_profile_structure(self, profile: Dict[str, Any]) -> None:
        """
        Validate the structure of a schema profile.
        
        Args:
            profile: Profile to validate
            
        Raises:
            ValueError: If profile structure is invalid
        """
        required_fields = ['id', 'name', 'columns']
        for field in required_fields:
            if field not in profile:
                raise ValueError(f"Missing required field: {field}")
        
        if not isinstance(profile['columns'], list):
            raise ValueError("'columns' must be a list")
        
        for column in profile['columns']:
            if not isinstance(column, dict):
                raise ValueError("Each column must be a dictionary")
            
            required_column_fields = ['name', 'display_name', 'data_type']
            for field in required_column_fields:
                if field not in column:
                    raise ValueError(f"Column missing required field: {field}")
            
            # Validate business rules if present
            if 'business_rules' in column:
                if not isinstance(column['business_rules'], list):
                    raise ValueError("'business_rules' must be a list")
                
                for rule in column['business_rules']:
                    if not isinstance(rule, dict):
                        raise ValueError("Each business rule must be a dictionary")
                    
                    required_rule_fields = ['type', 'operator', 'threshold']
                    for field in required_rule_fields:
                        if field not in rule:
                            raise ValueError(f"Business rule missing required field: {field}")
    
    def create_default_profile(self) -> Dict[str, Any]:
        """
        Create a default schema profile.
        
        Returns:
            Default profile dictionary
        """
        default_profile = {
            "id": "default",
            "name": "Default Schema Profile",
            "description": "Basic validation rules for dealership data",
            "role": "default",
            "columns": [
                {
                    "name": "sale_date",
                    "display_name": "Sale Date",
                    "description": "Date of sale",
                    "data_type": "datetime",
                    "visibility": "public",
                    "format": "%Y-%m-%d",
                    "aliases": ["date", "transaction_date"],
                    "business_rules": [
                        {
                            "type": "comparison",
                            "operator": "<=",
                            "threshold": "today",
                            "message": "Sale date cannot be in the future",
                            "severity": "high"
                        }
                    ]
                },
                {
                    "name": "gross_profit",
                    "display_name": "Gross Profit",
                    "description": "Total gross profit",
                    "data_type": "float",
                    "visibility": "public",
                    "format": "${:,.2f}",
                    "aliases": ["gross", "total_gross"],
                    "business_rules": [
                        {
                            "type": "comparison",
                            "operator": ">=",
                            "threshold": 0,
                            "message": "Gross profit should not be negative",
                            "severity": "medium"
                        }
                    ]
                }
            ],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        self.save_profile("default", default_profile)
        return default_profile