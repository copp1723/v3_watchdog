"""
Incremental update system for CSV data imports.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Set
from datetime import datetime
import hashlib
import json
import os

from .data_lineage import DataLineage

logger = logging.getLogger(__name__)

class IncrementalUpdater:
    """
    Handles incremental updates for CSV data imports based on VIN or timestamps.
    Identifies new, updated, and unchanged records between imports.
    """
    
    def __init__(self, lineage: Optional[DataLineage] = None,
                cache_dir: str = "data/cache"):
        """
        Initialize the incremental updater.
        
        Args:
            lineage: Optional DataLineage instance for tracking
            cache_dir: Directory to store cached data snapshots
        """
        self.lineage = lineage
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _generate_dataset_id(self, df: pd.DataFrame, dealer_id: Optional[str] = None) -> str:
        """
        Generate a unique identifier for a dataset based on its structure.
        
        Args:
            df: DataFrame to identify
            dealer_id: Optional dealer ID to include in the identifier
            
        Returns:
            Unique dataset identifier
        """
        # Create a signature based on column names and data types
        columns = sorted(df.columns.tolist())
        dtypes = {col: str(df[col].dtype) for col in columns}
        
        # Create a hash of the signature
        signature = {
            "columns": columns,
            "dtypes": dtypes,
            "dealer_id": dealer_id,
            "row_count": len(df)
        }
        
        # Generate a hash
        hash_obj = hashlib.md5(json.dumps(signature, sort_keys=True).encode())
        return hash_obj.hexdigest()
    
    def _get_cache_path(self, dataset_id: str) -> str:
        """
        Get the cache file path for a dataset.
        
        Args:
            dataset_id: Dataset identifier
            
        Returns:
            Path to the cache file
        """
        return os.path.join(self.cache_dir, f"{dataset_id}.parquet")
    
    def _save_to_cache(self, df: pd.DataFrame, dataset_id: str) -> None:
        """
        Save a DataFrame to the cache.
        
        Args:
            df: DataFrame to cache
            dataset_id: Dataset identifier
        """
        cache_path = self._get_cache_path(dataset_id)
        
        try:
            # Add metadata columns for tracking
            df_with_meta = df.copy()
            df_with_meta['_last_updated'] = datetime.now().isoformat()
            df_with_meta['_update_count'] = 1
            
            # Save to parquet format
            df_with_meta.to_parquet(cache_path, index=False)
            logger.info(f"Saved dataset {dataset_id} to cache")
        except Exception as e:
            logger.error(f"Error saving dataset to cache: {str(e)}")
    
    def _load_from_cache(self, dataset_id: str) -> Optional[pd.DataFrame]:
        """
        Load a DataFrame from the cache.
        
        Args:
            dataset_id: Dataset identifier
            
        Returns:
            Cached DataFrame or None if not found
        """
        cache_path = self._get_cache_path(dataset_id)
        
        if not os.path.exists(cache_path):
            return None
            
        try:
            df = pd.read_parquet(cache_path)
            logger.info(f"Loaded dataset {dataset_id} from cache")
            return df
        except Exception as e:
            logger.error(f"Error loading dataset from cache: {str(e)}")
            return None
    
    def _identify_key_column(self, df: pd.DataFrame) -> Optional[str]:
        """
        Identify a suitable key column for record matching.
        Prefers VIN, then unique ID fields, then timestamps.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Name of the key column or None if no suitable column found
        """
        # Look for VIN column
        vin_candidates = [col for col in df.columns if 'vin' in col.lower()]
        for col in vin_candidates:
            # Check if column has mostly unique values
            if df[col].nunique() > len(df) * 0.9:
                return col
        
        # Look for ID columns
        id_candidates = [col for col in df.columns if 'id' in col.lower() or '_id' in col.lower()]
        for col in id_candidates:
            # Check if column has mostly unique values
            if df[col].nunique() > len(df) * 0.9:
                return col
        
        # Look for timestamp columns
        date_candidates = [col for col in df.columns if any(term in col.lower() 
                                                         for term in ['date', 'time', 'timestamp'])]
        
        # Convert to datetime and check for uniqueness
        for col in date_candidates:
            try:
                # Try to convert to datetime
                if not pd.api.types.is_datetime64_dtype(df[col]):
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                
                # Check if column has mostly unique values
                if df[col].nunique() > len(df) * 0.7:  # Lower threshold for dates
                    return col
            except:
                continue
        
        # No suitable key column found
        return None
    
    def _get_record_hash(self, row: pd.Series, exclude_cols: List[str] = None) -> str:
        """
        Generate a hash for a record to detect changes.
        
        Args:
            row: DataFrame row
            exclude_cols: Columns to exclude from the hash
            
        Returns:
            Hash string representing the record content
        """
        exclude_cols = exclude_cols or []
        exclude_cols.extend(['_last_updated', '_update_count'])
        
        # Create a dictionary of values, excluding specified columns
        values = {col: str(val) for col, val in row.items() 
                 if col not in exclude_cols and not pd.isna(val)}
        
        # Generate hash
        hash_obj = hashlib.md5(json.dumps(values, sort_keys=True).encode())
        return hash_obj.hexdigest()
    
    def process_incremental_update(self, df: pd.DataFrame, 
                                 dealer_id: Optional[str] = None,
                                 key_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a DataFrame as an incremental update.
        
        Args:
            df: New DataFrame to process
            dealer_id: Optional dealer ID
            key_column: Optional key column name for record matching
            
        Returns:
            Dictionary with update statistics and the merged DataFrame
        """
        if df.empty:
            return {
                "success": False,
                "error": "Empty DataFrame",
                "dataframe": df
            }
        
        # Generate dataset ID
        dataset_id = self._generate_dataset_id(df, dealer_id)
        
        # Load previous version from cache
        previous_df = self._load_from_cache(dataset_id)
        
        # If no previous version, save current and return
        if previous_df is None:
            self._save_to_cache(df, dataset_id)
            return {
                "success": True,
                "is_first_import": True,
                "new_records": len(df),
                "updated_records": 0,
                "unchanged_records": 0,
                "dataframe": df
            }
        
        # Identify key column if not provided
        if not key_column:
            key_column = self._identify_key_column(df)
            
        if not key_column:
            logger.warning("No suitable key column found for incremental update")
            # Fallback to full replacement
            self._save_to_cache(df, dataset_id)
            return {
                "success": True,
                "is_first_import": False,
                "error": "No suitable key column found",
                "new_records": len(df),
                "updated_records": 0,
                "unchanged_records": 0,
                "dataframe": df
            }
        
        # Ensure key column exists in both DataFrames
        if key_column not in df.columns or key_column not in previous_df.columns:
            logger.warning(f"Key column '{key_column}' not found in both DataFrames")
            # Fallback to full replacement
            self._save_to_cache(df, dataset_id)
            return {
                "success": True,
                "is_first_import": False,
                "error": f"Key column '{key_column}' not found in both DataFrames",
                "new_records": len(df),
                "updated_records": 0,
                "unchanged_records": 0,
                "dataframe": df
            }
        
        # Process the incremental update
        try:
            # Get sets of keys
            current_keys = set(df[key_column].dropna().unique())
            previous_keys = set(previous_df[key_column].dropna().unique())
            
            # Identify new, updated, and unchanged records
            new_keys = current_keys - previous_keys
            existing_keys = current_keys.intersection(previous_keys)
            
            # Initialize counters
            new_count = len(new_keys)
            updated_count = 0
            unchanged_count = 0
            
            # Create a copy of the previous DataFrame for merging
            merged_df = previous_df.copy()
            
            # Add new records
            new_records = df[df[key_column].isin(new_keys)]
            
            # Process existing records for updates
            for key in existing_keys:
                current_record = df[df[key_column] == key].iloc[0]
                previous_record = previous_df[previous_df[key_column] == key].iloc[0]
                
                # Generate hashes to detect changes
                current_hash = self._get_record_hash(current_record, [key_column])
                previous_hash = self._get_record_hash(previous_record, [key_column])
                
                if current_hash != previous_hash:
                    # Record has changed
                    updated_count += 1
                    
                    # Update the record in the merged DataFrame
                    merged_df.loc[merged_df[key_column] == key, df.columns] = current_record
                    
                    # Update metadata
                    merged_df.loc[merged_df[key_column] == key, '_last_updated'] = datetime.now().isoformat()
                    merged_df.loc[merged_df[key_column] == key, '_update_count'] += 1
                else:
                    # Record unchanged
                    unchanged_count += 1
            
            # Add new records to merged DataFrame
            if not new_records.empty:
                # Add metadata columns
                new_records['_last_updated'] = datetime.now().isoformat()
                new_records['_update_count'] = 1
                
                # Append to merged DataFrame
                merged_df = pd.concat([merged_df, new_records], ignore_index=True)
            
            # Save updated DataFrame to cache
            self._save_to_cache(merged_df, dataset_id)
            
            # Track in lineage if available
            if self.lineage:
                self.lineage.track_data_transformation(
                    source_id=f"incremental_update_{dataset_id}",
                    target_id=f"merged_dataset_{dataset_id}",
                    transform_type="incremental_update",
                    metadata={
                        "dealer_id": dealer_id,
                        "key_column": key_column,
                        "new_records": new_count,
                        "updated_records": updated_count,
                        "unchanged_records": unchanged_count,
                        "timestamp": datetime.now().isoformat()
                    }
                )
            
            return {
                "success": True,
                "is_first_import": False,
                "key_column": key_column,
                "new_records": new_count,
                "updated_records": updated_count,
                "unchanged_records": unchanged_count,
                "dataframe": merged_df
            }
            
        except Exception as e:
            logger.error(f"Error processing incremental update: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "dataframe": df
            }
    
    def get_update_history(self, dataset_id: str) -> List[Dict[str, Any]]:
        """
        Get the update history for a dataset.
        
        Args:
            dataset_id: Dataset identifier
            
        Returns:
            List of update events
        """
        if not self.lineage:
            return []
            
        try:
            # Get transformation events from lineage
            events = self.lineage.get_column_lineage(f"incremental_update_{dataset_id}")
            
            # Format events
            history = []
            for event in events:
                if event.get('event_type') == 'data_transform' and event.get('metadata', {}).get('transform_type') == 'incremental_update':
                    history.append({
                        "timestamp": event.get('timestamp'),
                        "new_records": event.get('metadata', {}).get('new_records', 0),
                        "updated_records": event.get('metadata', {}).get('updated_records', 0),
                        "unchanged_records": event.get('metadata', {}).get('unchanged_records', 0)
                    })
            
            return sorted(history, key=lambda x: x['timestamp'], reverse=True)
            
        except Exception as e:
            logger.error(f"Error getting update history: {str(e)}")
            return []
