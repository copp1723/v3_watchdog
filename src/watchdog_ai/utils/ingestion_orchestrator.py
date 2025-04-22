"""
IngestionOrchestrator for Watchdog AI.

This module provides a central orchestrator for the ingestion pipeline,
coordinating file parsing, normalization, and validation.
"""

import os
import logging
import glob
import pandas as pd
import json
import time
from typing import Dict, List, Tuple, Any, Optional, Union, Set
from dataclasses import dataclass, field
from datetime import datetime
from .data_parser import DataParser, DataParserResult
from .data_normalization import DataSchemaApplier, normalize_dataframe
from .adaptive_schema import AdaptiveSchema
from .data_lineage import DataLineage

logger = logging.getLogger(__name__)

@dataclass
class IngestionResult:
    """Results from the ingestion process."""
    success: bool
    dataframe: Optional[pd.DataFrame] = None
    validation_summary: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    source_file: Optional[str] = None
    ingestion_time: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "success": self.success,
            "validation_summary": self.validation_summary,
            "error_message": self.error_message,
            "source_file": self.source_file,
            "ingestion_time": self.ingestion_time,
            "metadata": self.metadata,
            "data_shape": self.dataframe.shape if self.dataframe is not None else None,
            "columns": self.dataframe.columns.tolist() if self.dataframe is not None else []
        }
    
    def save_summary(self, output_path: str) -> None:
        """Save the ingestion summary to a JSON file."""
        try:
            with open(output_path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            logger.info(f"Saved ingestion summary to {output_path}")
        except Exception as e:
            logger.error(f"Error saving ingestion summary: {str(e)}")


class DMSConnector:
    """
    Connector stub for Dealer Management Systems.
    This is a placeholder implementation that will be expanded in the future.
    """
    
    def __init__(self, vendor: str, credentials_path: Optional[str] = None):
        """
        Initialize the DMS connector.
        
        Args:
            vendor: DMS vendor name
            credentials_path: Path to credentials file
        """
        self.vendor = vendor
        self.credentials_path = credentials_path
        self.credentials = self._load_credentials()
        
    def _load_credentials(self) -> Dict[str, Any]:
        """Load credentials from file."""
        if not self.credentials_path or not os.path.exists(self.credentials_path):
            logger.warning(f"No credentials file found for {self.vendor}")
            return {}
        
        try:
            with open(self.credentials_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading credentials: {str(e)}")
            return {}
    
    def test_connection(self) -> bool:
        """Test the connection to the DMS."""
        # Placeholder implementation
        logger.info(f"Testing connection to {self.vendor} DMS (placeholder)")
        return True
    
    def fetch_data(self, data_type: str, start_date: Optional[str] = None, 
                  end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch data from the DMS.
        
        Args:
            data_type: Type of data to fetch
            start_date: Start date for data range
            end_date: End date for data range
            
        Returns:
            DataFrame with fetched data
        """
        # Placeholder implementation
        logger.info(f"Fetching {data_type} data from {self.vendor} DMS (placeholder)")
        
        # Return empty DataFrame for now
        return pd.DataFrame()


class IngestionOrchestrator:
    """
    Orchestrates the entire ingestion pipeline from file upload to
    normalized and validated DataFrame.
    """
    
    def __init__(self, 
                schema_profiles_dir: str = "config/schema_profiles",
                output_dir: str = "data/ingestion_results",
                lineage_tracking: bool = True,
                redis_url: Optional[str] = None):
        """
        Initialize the ingestion orchestrator.
        
        Args:
            schema_profiles_dir: Directory containing schema profiles
            output_dir: Directory for saving ingestion results
            lineage_tracking: Whether to track data lineage
            redis_url: Redis URL for lineage storage
        """
        self.schema_profiles_dir = schema_profiles_dir
        self.output_dir = output_dir
        self.parser = DataParser()
        self.schema_applier = DataSchemaApplier(schema_profiles_dir)
        self.lineage_tracking = lineage_tracking
        
        # Initialize lineage tracking if enabled
        if lineage_tracking:
            self.lineage = DataLineage(redis_url)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def ingest_file(self, file_path: str, 
                   dealer_id: Optional[str] = None,
                   vendor: Optional[str] = None,
                   **kwargs) -> IngestionResult:
        """
        Ingest a single file through the pipeline.
        
        Args:
            file_path: Path to the file
            dealer_id: Dealer ID for schema profile selection
            vendor: Vendor name for lineage tracking
            **kwargs: Additional arguments for parser and normalizer
            
        Returns:
            IngestionResult object
        """
        start_time = time.time()
        
        try:
            # 1. Parse the file
            parse_result = self.parser.parse_file(file_path, **kwargs)
            
            if not parse_result.is_successful():
                return IngestionResult(
                    success=False,
                    error_message=parse_result.errors[0] if parse_result.errors else "Unknown error during parsing",
                    source_file=file_path,
                    validation_summary={"parsing_result": parse_result.get_summary()}
                )
            
            # 2. Apply schema normalization
            df, processing_summary = self.schema_applier.apply_schema(
                parse_result.dataframe, dealer_id
            )
            
            # 3. Track lineage if enabled
            if self.lineage_tracking and vendor:
                self._track_lineage(file_path, df, processing_summary, vendor)
            
            # 4. Create result
            result = IngestionResult(
                success=True,
                dataframe=df,
                source_file=file_path,
                validation_summary={
                    "parsing_result": parse_result.get_summary(),
                    "normalization_result": processing_summary
                },
                metadata={
                    "dealer_id": dealer_id,
                    "vendor": vendor,
                    "file_type": parse_result.file_type,
                    "ingestion_duration_sec": time.time() - start_time
                }
            )
            
            # 5. Save summary if output directory is specified
            if self.output_dir:
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                filename = os.path.basename(file_path)
                base_name = os.path.splitext(filename)[0]
                summary_path = os.path.join(
                    self.output_dir, 
                    f"{base_name}_ingestion_summary_{timestamp}.json"
                )
                result.save_summary(summary_path)
            
            return result
            
        except Exception as e:
            logger.error(f"Error ingesting file {file_path}: {str(e)}", exc_info=True)
            return IngestionResult(
                success=False,
                error_message=f"Error during ingestion: {str(e)}",
                source_file=file_path
            )
    
    def ingest_directory(self, directory_path: str, 
                       file_pattern: str = "*.*",
                       dealer_id: Optional[str] = None,
                       vendor: Optional[str] = None,
                       combine_results: bool = True,
                       **kwargs) -> Tuple[Optional[pd.DataFrame], List[IngestionResult]]:
        """
        Ingest all matching files in a directory.
        
        Args:
            directory_path: Path to the directory
            file_pattern: Glob pattern for matching files
            dealer_id: Dealer ID for schema profile selection
            vendor: Vendor name for lineage tracking
            combine_results: Whether to combine results into a single DataFrame
            **kwargs: Additional arguments for parser and normalizer
            
        Returns:
            Tuple of (combined DataFrame or None, list of IngestionResult objects)
        """
        if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
            logger.error(f"Directory not found: {directory_path}")
            return None, []
        
        # Find matching files
        pattern = os.path.join(directory_path, file_pattern)
        file_paths = glob.glob(pattern)
        
        if not file_paths:
            logger.warning(f"No files found matching pattern {pattern}")
            return None, []
        
        # Ingest each file
        results = []
        successful_dfs = []
        
        for file_path in file_paths:
            result = self.ingest_file(file_path, dealer_id, vendor, **kwargs)
            results.append(result)
            
            if result.success and result.dataframe is not None and not result.dataframe.empty:
                successful_dfs.append(result.dataframe)
        
        # Combine DataFrames if requested
        combined_df = None
        if combine_results and successful_dfs:
            try:
                combined_df = pd.concat(successful_dfs, ignore_index=True)
                
                # Log the combined result
                logger.info(f"Combined {len(successful_dfs)} DataFrames into one with shape {combined_df.shape}")
                
                # Track lineage for combined DataFrame
                if self.lineage_tracking and vendor:
                    self.lineage.track_data_transformation(
                        source_id=",".join(r.source_file for r in results if r.success),
                        target_id="combined_dataframe",
                        transform_type="concat",
                        metadata={
                            "dealer_id": dealer_id,
                            "vendor": vendor,
                            "file_count": len(successful_dfs),
                            "total_rows": len(combined_df)
                        }
                    )
            except Exception as e:
                logger.error(f"Error combining DataFrames: {str(e)}")
        
        return combined_df, results
    
    def connect_to_dms(self, vendor: str, 
                     credentials_path: Optional[str] = None) -> DMSConnector:
        """
        Create a connection to a Dealer Management System.
        
        Args:
            vendor: DMS vendor name
            credentials_path: Path to credentials file
            
        Returns:
            DMSConnector object
        """
        return DMSConnector(vendor, credentials_path)
    
    def ingest_from_dms(self, connector: DMSConnector, 
                       data_type: str,
                       dealer_id: Optional[str] = None,
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None) -> IngestionResult:
        """
        Ingest data directly from a DMS.
        
        Args:
            connector: DMSConnector object
            data_type: Type of data to fetch
            dealer_id: Dealer ID for schema profile selection
            start_date: Start date for data range
            end_date: End date for data range
            
        Returns:
            IngestionResult object
        """
        try:
            # 1. Fetch data from DMS
            df = connector.fetch_data(data_type, start_date, end_date)
            
            if df.empty:
                return IngestionResult(
                    success=False,
                    error_message=f"No data retrieved from {connector.vendor} DMS",
                    validation_summary={"dms_retrieval": "empty_result"}
                )
            
            # 2. Apply schema normalization
            normalized_df, processing_summary = self.schema_applier.apply_schema(df, dealer_id)
            
            # 3. Track lineage if enabled
            if self.lineage_tracking:
                self.lineage.track_file_ingestion(
                    file_id=f"dms:{connector.vendor}:{data_type}",
                    vendor=connector.vendor,
                    metadata={
                        "dealer_id": dealer_id,
                        "data_type": data_type,
                        "start_date": start_date,
                        "end_date": end_date,
                        "row_count": len(df)
                    }
                )
            
            # 4. Create result
            result = IngestionResult(
                success=True,
                dataframe=normalized_df,
                source_file=f"dms:{connector.vendor}:{data_type}",
                validation_summary={
                    "normalization_result": processing_summary
                },
                metadata={
                    "dealer_id": dealer_id,
                    "vendor": connector.vendor,
                    "data_type": data_type,
                    "start_date": start_date,
                    "end_date": end_date,
                    "original_row_count": len(df),
                    "normalized_row_count": len(normalized_df)
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error ingesting from DMS {connector.vendor}: {str(e)}", exc_info=True)
            return IngestionResult(
                success=False,
                error_message=f"Error during DMS ingestion: {str(e)}",
                metadata={
                    "vendor": connector.vendor,
                    "data_type": data_type
                }
            )
    
    def _track_lineage(self, file_path: str, df: pd.DataFrame, 
                     processing_summary: Dict[str, Any],
                     vendor: str) -> None:
        """
        Track data lineage for ingested file.
        
        Args:
            file_path: Path to the source file
            df: Processed DataFrame
            processing_summary: Processing summary
            vendor: Vendor name
        """
        if not self.lineage_tracking:
            return
        
        try:
            # Track file ingestion
            file_id = os.path.basename(file_path)
            self.lineage.track_file_ingestion(
                file_id=file_id,
                vendor=vendor,
                metadata={
                    "file_path": file_path,
                    "file_type": os.path.splitext(file_path)[1],
                    "row_count": len(df),
                    "column_count": len(df.columns)
                }
            )
            
            # Track column mappings
            if "column_mappings" in processing_summary:
                for mapping in processing_summary["column_mappings"]:
                    self.lineage.track_column_mapping(
                        source_column=mapping["source_column"],
                        target_column=mapping["target_column"],
                        confidence=mapping["confidence"],
                        vendor=vendor,
                        metadata={
                            "reason": mapping.get("reason", "Unknown"),
                            "file_id": file_id
                        }
                    )
        except Exception as e:
            logger.error(f"Error tracking lineage: {str(e)}")


def ingest_file(file_path: str, dealer_id: Optional[str] = None, 
              vendor: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Convenience function to ingest a file.
    
    Args:
        file_path: Path to the file
        dealer_id: Dealer ID for schema profile selection
        vendor: Vendor name for lineage tracking
        
    Returns:
        Tuple of (DataFrame, ingestion summary)
    """
    orchestrator = IngestionOrchestrator()
    result = orchestrator.ingest_file(file_path, dealer_id, vendor)
    
    if result.success and result.dataframe is not None:
        return result.dataframe, result.validation_summary
    else:
        error_msg = result.error_message or "Unknown error during ingestion"
        raise ValueError(f"Failed to ingest file: {error_msg}")