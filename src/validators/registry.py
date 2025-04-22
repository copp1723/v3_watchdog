"""
Multi-tenant schema registry with versioning and metadata tracking.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import shutil

logger = logging.getLogger(__name__)

class SchemaRegistry:
    """
    Multi-tenant schema registry that manages schema profiles scoped by dealership.
    Includes versioning, metadata tracking, and validation.
    """
    
    def __init__(self, base_dir: str = "config/schema_profiles"):
        """
        Initialize the schema registry.
        
        Args:
            base_dir: Base directory for schema storage
        """
        self.base_dir = base_dir
        self._ensure_directory()
        
        # Cache for loaded schemas
        self._schema_cache: Dict[str, Dict[str, Any]] = {}
        
    def _ensure_directory(self) -> None:
        """Ensure the schema directory structure exists."""
        os.makedirs(self.base_dir, exist_ok=True)
        
    def _get_dealership_dir(self, dealership_id: str) -> str:
        """Get the directory path for a dealership's schemas."""
        return os.path.join(self.base_dir, dealership_id)
        
    def _validate_schema(self, schema: Dict[str, Any]) -> None:
        """
        Validate schema structure and required fields.
        
        Args:
            schema: Schema to validate
            
        Raises:
            ValueError: If schema is invalid
        """
        required_fields = ['id', 'name', 'columns', 'version']
        for field in required_fields:
            if field not in schema:
                raise ValueError(f"Missing required field: {field}")
                
        if not isinstance(schema['columns'], list):
            raise ValueError("'columns' must be a list")
            
        for column in schema['columns']:
            if not isinstance(column, dict):
                raise ValueError("Each column must be a dictionary")
                
            required_column_fields = ['name', 'display_name', 'data_type']
            for field in required_column_fields:
                if field not in column:
                    raise ValueError(f"Column missing required field: {field}")
    
    def create_schema(self, dealership_id: str, schema: Dict[str, Any]) -> str:
        """
        Create a new schema for a dealership.
        
        Args:
            dealership_id: ID of the dealership
            schema: Schema definition
            
        Returns:
            ID of created schema
            
        Raises:
            ValueError: If schema is invalid
        """
        # Add metadata
        schema['created_at'] = datetime.now().isoformat()
        schema['updated_at'] = schema['created_at']
        schema['version'] = '1.0.0'
        
        # Validate schema
        self._validate_schema(schema)
        
        # Create dealership directory
        dealership_dir = self._get_dealership_dir(dealership_id)
        os.makedirs(dealership_dir, exist_ok=True)
        
        # Save schema
        schema_path = os.path.join(dealership_dir, f"{schema['id']}.json")
        with open(schema_path, 'w') as f:
            json.dump(schema, f, indent=2)
            
        # Update cache
        cache_key = f"{dealership_id}:{schema['id']}"
        self._schema_cache[cache_key] = schema.copy()
        
        return schema['id']
    
    def get_schema(self, dealership_id: str, schema_id: str) -> Dict[str, Any]:
        """
        Get a schema by ID.
        
        Args:
            dealership_id: ID of the dealership
            schema_id: ID of the schema
            
        Returns:
            Schema definition
            
        Raises:
            FileNotFoundError: If schema doesn't exist
        """
        # Check cache
        cache_key = f"{dealership_id}:{schema_id}"
        if cache_key in self._schema_cache:
            return self._schema_cache[cache_key].copy()
        
        # Load from file
        schema_path = os.path.join(self._get_dealership_dir(dealership_id), f"{schema_id}.json")
        if not os.path.exists(schema_path):
            raise FileNotFoundError(f"Schema not found: {schema_id}")
            
        with open(schema_path, 'r') as f:
            schema = json.load(f)
            
        # Update cache
        self._schema_cache[cache_key] = schema.copy()
        
        return schema
    
    def update_schema(self, dealership_id: str, schema_id: str, 
                     schema: Dict[str, Any], increment_version: bool = True) -> None:
        """
        Update an existing schema.
        
        Args:
            dealership_id: ID of the dealership
            schema_id: ID of the schema to update
            schema: Updated schema definition
            increment_version: Whether to increment version number
            
        Raises:
            FileNotFoundError: If schema doesn't exist
            ValueError: If schema is invalid
        """
        # Get existing schema
        existing = self.get_schema(dealership_id, schema_id)
        
        # Update metadata
        schema['created_at'] = existing['created_at']
        schema['updated_at'] = datetime.now().isoformat()
        
        if increment_version:
            # Increment version number
            major, minor, patch = existing['version'].split('.')
            schema['version'] = f"{major}.{minor}.{int(patch) + 1}"
        else:
            schema['version'] = existing['version']
        
        # Validate schema
        self._validate_schema(schema)
        
        # Save updated schema
        schema_path = os.path.join(self._get_dealership_dir(dealership_id), f"{schema_id}.json")
        
        # Create backup of existing schema
        backup_dir = os.path.join(self._get_dealership_dir(dealership_id), 'backups')
        os.makedirs(backup_dir, exist_ok=True)
        backup_path = os.path.join(
            backup_dir, 
            f"{schema_id}_v{existing['version']}_{existing['updated_at']}.json"
        )
        shutil.copy2(schema_path, backup_path)
        
        # Save new version
        with open(schema_path, 'w') as f:
            json.dump(schema, f, indent=2)
            
        # Update cache
        cache_key = f"{dealership_id}:{schema_id}"
        self._schema_cache[cache_key] = schema.copy()
    
    def delete_schema(self, dealership_id: str, schema_id: str) -> None:
        """
        Delete a schema.
        
        Args:
            dealership_id: ID of the dealership
            schema_id: ID of the schema to delete
            
        Raises:
            FileNotFoundError: If schema doesn't exist
        """
        schema_path = os.path.join(self._get_dealership_dir(dealership_id), f"{schema_id}.json")
        if not os.path.exists(schema_path):
            raise FileNotFoundError(f"Schema not found: {schema_id}")
            
        # Create backup before deletion
        backup_dir = os.path.join(self._get_dealership_dir(dealership_id), 'backups')
        os.makedirs(backup_dir, exist_ok=True)
        
        schema = self.get_schema(dealership_id, schema_id)
        backup_path = os.path.join(
            backup_dir, 
            f"{schema_id}_v{schema['version']}_{schema['updated_at']}_deleted.json"
        )
        shutil.copy2(schema_path, backup_path)
        
        # Delete schema
        os.remove(schema_path)
        
        # Remove from cache
        cache_key = f"{dealership_id}:{schema_id}"
        self._schema_cache.pop(cache_key, None)
    
    def list_schemas(self, dealership_id: str) -> List[Dict[str, Any]]:
        """
        List all schemas for a dealership.
        
        Args:
            dealership_id: ID of the dealership
            
        Returns:
            List of schema metadata
        """
        dealership_dir = self._get_dealership_dir(dealership_id)
        if not os.path.exists(dealership_dir):
            return []
            
        schemas = []
        for filename in os.listdir(dealership_dir):
            if filename.endswith('.json') and not filename.startswith('.'):
                try:
                    schema = self.get_schema(dealership_id, filename[:-5])
                    schemas.append({
                        'id': schema['id'],
                        'name': schema['name'],
                        'version': schema['version'],
                        'updated_at': schema['updated_at']
                    })
                except Exception as e:
                    logger.warning(f"Error loading schema {filename}: {e}")
                    
        return sorted(schemas, key=lambda x: x['updated_at'], reverse=True)
    
    def get_schema_history(self, dealership_id: str, schema_id: str) -> List[Dict[str, Any]]:
        """
        Get version history for a schema.
        
        Args:
            dealership_id: ID of the dealership
            schema_id: ID of the schema
            
        Returns:
            List of historical versions
        """
        backup_dir = os.path.join(self._get_dealership_dir(dealership_id), 'backups')
        if not os.path.exists(backup_dir):
            return []
            
        history = []
        for filename in os.listdir(backup_dir):
            if filename.startswith(f"{schema_id}_v"):
                try:
                    with open(os.path.join(backup_dir, filename), 'r') as f:
                        schema = json.load(f)
                    history.append({
                        'version': schema['version'],
                        'updated_at': schema['updated_at'],
                        'is_deleted': filename.endswith('_deleted.json')
                    })
                except Exception as e:
                    logger.warning(f"Error loading schema history {filename}: {e}")
                    
        return sorted(history, key=lambda x: x['updated_at'], reverse=True)