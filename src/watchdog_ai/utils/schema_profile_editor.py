"""
Schema Profile Editor for Watchdog AI.

This module provides functionality for editing, validating, and previewing
schema profiles with live validation feedback.
"""

import os
import json
import yaml
import logging
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path

from .adaptive_schema import SchemaProfile, SchemaColumn, ExecRole
from .data_normalization import DataSchemaApplier

logger = logging.getLogger(__name__)

class SchemaProfileEditor:
    """Editor for schema profiles with live preview and validation."""
    
    def __init__(self, profiles_dir: str = "config/schema_profiles",
                cache_dir: str = ".cache/schema_profiles"):
        """
        Initialize the schema profile editor.
        
        Args:
            profiles_dir: Directory containing schema profiles
            cache_dir: Directory for caching preview results
        """
        self.profiles_dir = profiles_dir
        self.cache_dir = cache_dir
        self.current_profile = None
        self.preview_cache = {}
        self.schema_applier = DataSchemaApplier(profiles_dir)
        
        # Ensure directories exist
        os.makedirs(profiles_dir, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)
    
    def load_profile(self, profile_id: str) -> Optional[SchemaProfile]:
        """
        Load a schema profile by ID.
        
        Args:
            profile_id: Profile identifier
            
        Returns:
            SchemaProfile object or None if not found
        """
        # Try JSON first
        json_path = os.path.join(self.profiles_dir, f"{profile_id}.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                profile = SchemaProfile.from_dict(data)
                self.current_profile = profile
                return profile
            except Exception as e:
                logger.error(f"Error loading profile from {json_path}: {str(e)}")
        
        # Try YAML
        yaml_path = os.path.join(self.profiles_dir, f"{profile_id}.yml")
        if os.path.exists(yaml_path):
            try:
                with open(yaml_path, 'r') as f:
                    data = yaml.safe_load(f)
                profile = SchemaProfile.from_dict(data)
                self.current_profile = profile
                return profile
            except Exception as e:
                logger.error(f"Error loading profile from {yaml_path}: {str(e)}")
        
        return None
    
    def save_profile(self, profile: SchemaProfile, format: str = 'json') -> bool:
        """
        Save a schema profile.
        
        Args:
            profile: SchemaProfile to save
            format: Output format ('json' or 'yaml')
            
        Returns:
            True if successful
        """
        try:
            # Validate profile before saving
            validation_result = self.validate_profile(profile)
            if not validation_result["is_valid"]:
                logger.error(f"Invalid profile: {validation_result['errors']}")
                return False
            
            # Update timestamps
            profile.updated_at = datetime.now().isoformat()
            
            # Convert to dictionary
            data = profile.to_dict()
            
            # Save based on format
            if format.lower() == 'json':
                path = os.path.join(self.profiles_dir, f"{profile.id}.json")
                with open(path, 'w') as f:
                    json.dump(data, f, indent=2)
            else:
                path = os.path.join(self.profiles_dir, f"{profile.id}.yml")
                with open(path, 'w') as f:
                    yaml.safe_dump(data, f)
            
            logger.info(f"Saved profile {profile.id} to {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving profile: {str(e)}")
            return False
    
    def validate_profile(self, profile: SchemaProfile) -> Dict[str, Any]:
        """
        Validate a schema profile.
        
        Args:
            profile: SchemaProfile to validate
            
        Returns:
            Validation result dictionary
        """
        result = {
            "is_valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check required fields
        if not profile.id:
            result["errors"].append("Profile ID is required")
        if not profile.name:
            result["errors"].append("Profile name is required")
        if not profile.role:
            result["errors"].append("Profile role is required")
        
        # Validate role
        try:
            ExecRole.from_string(profile.role)
        except ValueError:
            result["errors"].append(f"Invalid role: {profile.role}")
        
        # Check columns
        column_names = set()
        for col in profile.columns:
            # Check required column fields
            if not col.name:
                result["errors"].append("Column name is required")
            if not col.data_type:
                result["errors"].append(f"Data type required for column {col.name}")
            
            # Check for duplicate names
            if col.name in column_names:
                result["errors"].append(f"Duplicate column name: {col.name}")
            column_names.add(col.name)
            
            # Validate visibility
            if col.visibility not in ['public', 'restricted', 'private']:
                result["errors"].append(f"Invalid visibility '{col.visibility}' for column {col.name}")
        
        # Update validity flag
        result["is_valid"] = len(result["errors"]) == 0
        
        return result
    
    def preview_validation(self, profile: SchemaProfile, 
                         sample_data: pd.DataFrame,
                         max_rows: int = 5) -> Dict[str, Any]:
        """
        Preview validation results for a profile using sample data.
        
        Args:
            profile: SchemaProfile to validate
            sample_data: Sample DataFrame
            max_rows: Maximum rows to process
            
        Returns:
            Validation preview results
        """
        if sample_data.empty:
            return {
                "success": False,
                "error": "Empty sample data"
            }
        
        # Limit sample size
        preview_df = sample_data.head(max_rows)
        
        try:
            # Apply schema and get results
            normalized_df, summary = self.schema_applier.apply_schema(preview_df, profile.id)
            
            # Enhance summary with preview-specific info
            preview_summary = {
                "success": True,
                "sample_rows": len(preview_df),
                "original_columns": preview_df.columns.tolist(),
                "normalized_columns": normalized_df.columns.tolist(),
                "normalization_summary": summary,
                "sample_data": {
                    "original": preview_df.to_dict(orient='records'),
                    "normalized": normalized_df.to_dict(orient='records')
                }
            }
            
            return preview_summary
            
        except Exception as e:
            logger.error(f"Error generating preview: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def export_profile(self, profile: SchemaProfile, 
                      format: str = 'json') -> Optional[str]:
        """
        Export a schema profile to string.
        
        Args:
            profile: SchemaProfile to export
            format: Output format ('json' or 'yaml')
            
        Returns:
            String representation of profile or None on error
        """
        try:
            data = profile.to_dict()
            
            if format.lower() == 'json':
                return json.dumps(data, indent=2)
            else:
                return yaml.safe_dump(data)
                
        except Exception as e:
            logger.error(f"Error exporting profile: {str(e)}")
            return None
    
    def import_profile(self, data: str, 
                      format: str = 'json') -> Optional[SchemaProfile]:
        """
        Import a schema profile from string.
        
        Args:
            data: Profile data string
            format: Input format ('json' or 'yaml')
            
        Returns:
            SchemaProfile object or None on error
        """
        try:
            # Parse data
            if format.lower() == 'json':
                profile_data = json.loads(data)
            else:
                profile_data = yaml.safe_load(data)
            
            # Create profile
            profile = SchemaProfile.from_dict(profile_data)
            
            # Validate
            validation_result = self.validate_profile(profile)
            if not validation_result["is_valid"]:
                logger.error(f"Invalid imported profile: {validation_result['errors']}")
                return None
            
            return profile
            
        except Exception as e:
            logger.error(f"Error importing profile: {str(e)}")
            return None
    
    def duplicate_profile(self, profile: SchemaProfile, 
                        new_id: str) -> Optional[SchemaProfile]:
        """
        Create a copy of a profile with a new ID.
        
        Args:
            profile: SchemaProfile to duplicate
            new_id: ID for the new profile
            
        Returns:
            New SchemaProfile object or None on error
        """
        try:
            # Create new profile data
            data = profile.to_dict()
            data["id"] = new_id
            data["name"] = f"Copy of {profile.name}"
            data["created_at"] = datetime.now().isoformat()
            data["updated_at"] = datetime.now().isoformat()
            
            # Create new profile
            new_profile = SchemaProfile.from_dict(data)
            
            return new_profile
            
        except Exception as e:
            logger.error(f"Error duplicating profile: {str(e)}")
            return None
    
    def delete_profile(self, profile_id: str) -> bool:
        """
        Delete a schema profile.
        
        Args:
            profile_id: Profile identifier
            
        Returns:
            True if successful
        """
        try:
            # Check both JSON and YAML
            json_path = os.path.join(self.profiles_dir, f"{profile_id}.json")
            yaml_path = os.path.join(self.profiles_dir, f"{profile_id}.yml")
            
            deleted = False
            
            if os.path.exists(json_path):
                os.remove(json_path)
                deleted = True
                
            if os.path.exists(yaml_path):
                os.remove(yaml_path)
                deleted = True
            
            if deleted:
                logger.info(f"Deleted profile {profile_id}")
                if self.current_profile and self.current_profile.id == profile_id:
                    self.current_profile = None
                return True
            else:
                logger.warning(f"Profile {profile_id} not found")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting profile: {str(e)}")
            return False