"""
Data ingestion pipeline for Nova Act.

This module provides the pipeline for normalizing, validating, 
and processing data collected from external systems.
"""

import os
import json
import logging
import asyncio
import pandas as pd
import numpy as np
import re
import traceback
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple, Set, Union
from pathlib import Path

from .logging_config import log_error, log_info, log_warning

logger = logging.getLogger(__name__)

# Define constants
DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data"
)

# Column mapping for different report types
COLUMN_MAPPINGS = {
    "sales": {
        # Common mappings across vendors
        "date": ["sale_date", "saledate", "date", "deal_date", "transaction_date"],
        "customer_name": ["customer", "customer_name", "buyer", "client"],
        "gross": ["gross_profit", "gross", "total_gross", "front_gross"],
        "vehicle": ["vehicle", "vehicle_desc", "car_description"],
        "vin": ["vin", "vin_number", "vehicle_id"],
        "salesperson": ["salesperson", "sales_person", "sales_rep", "associate"],
        "lead_source": ["lead_source", "leadsource", "source", "lead_type", "origin"],
        "sale_price": ["sale_price", "selling_price", "amount", "price"],
        "stock_num": ["stock", "stock_number", "stocknumber"],
        "vehicle_year": ["year", "vehicle_year", "modelyear"],
        "vehicle_make": ["make", "vehicle_make"],
        "vehicle_model": ["model", "vehicle_model"],
        
        # Vendor-specific mappings
        "dealersocket": {
            "date": ["deal_date_alternate_format", "sale_timestamp"],
            "gross": ["front_gross_alt", "total_before_pack"],
            "lead_source": ["lead_category", "customer_source"]
        },
        "vinsolutions": {
            "date": ["closing_date", "deal_created"],
            "gross": ["final_gross", "dealership_gross"],
            "lead_source": ["lead_gen_source", "customer_acquisition_source"]
        },
        "eleads": {
            "date": ["dt_sale", "sale_time"],
            "gross": ["profit", "gross_margin"],
            "lead_source": ["lead_origin", "traffic_source"]
        }
    },
    "inventory": {
        # Common mappings
        "stock_num": ["stock", "stock_number", "stocknumber", "stock_#", "inventory_id"],
        "vin": ["vin", "vin_number", "vehicle_id"],
        "days_in_stock": ["days_in_stock", "age", "days_on_lot", "inventory_age"],
        "vehicle_year": ["year", "vehicle_year", "modelyear"],
        "vehicle_make": ["make", "vehicle_make"],
        "vehicle_model": ["model", "vehicle_model"],
        "mileage": ["odometer", "mileage", "miles"],
        "list_price": ["list_price", "price", "asking_price", "msrp"],
        "cost": ["cost", "acquisition_cost", "dealer_cost"],
        "certified": ["certified", "is_certified", "certified_flag"],
        "ext_color": ["exterior_color", "ext_color", "color", "vehicle_color"],
        "int_color": ["interior_color", "int_color", "trim_color"],
        "stock_date": ["stock_date", "date_added", "in_stock_date"],
        
        # Vendor-specific mappings
        "dealersocket": {
            "days_in_stock": ["age_in_days", "inventory_days"],
            "certified": ["certified_pre_owned", "cpo_flag"]
        },
        "vinsolutions": {
            "days_in_stock": ["days_in_inventory", "lot_age"],
            "certified": ["is_cpo", "cpo_status"]
        },
        "eleads": {
            "days_in_stock": ["days_aged", "age_days"],
            "certified": ["cert_flag", "cpo"]
        }
    },
    "leads": {
        # Common mappings
        "date": ["lead_date", "created_date", "submission_date", "date_received"],
        "customer_name": ["customer", "customer_name", "prospect", "lead_name"],
        "email": ["email", "email_address", "customer_email"],
        "phone": ["phone", "phone_number", "contact_phone", "customer_phone"],
        "lead_source": ["lead_source", "leadsource", "source", "lead_type", "origin"],
        "vehicle_interest": ["vehicle_interest", "interested_in", "vehicle_of_interest"],
        "status": ["status", "lead_status", "state", "disposition"],
        "salesperson": ["salesperson", "sales_person", "sales_rep", "owner"],
        "vehicle_year": ["year", "vehicle_year", "modelyear"],
        "vehicle_make": ["make", "vehicle_make"],
        "vehicle_model": ["model", "vehicle_model"],
        
        # Vendor-specific mappings
        "dealersocket": {
            "lead_source": ["lead_provider", "traffic_source"],
            "status": ["lead_outcome", "disposition"]
        },
        "vinsolutions": {
            "lead_source": ["source_name", "origin_source"],
            "status": ["current_status", "lead_disposition"]
        },
        "eleads": {
            "lead_source": ["lead_campaign", "source_type"],
            "status": ["status_code", "current_state"]
        }
    }
}

# Type conversion mappings
TYPE_CONVERSIONS = {
    "date": "datetime64[ns]",
    "datetime": "datetime64[ns]",
    "timestamp": "datetime64[ns]",
    "price": "float64",
    "cost": "float64",
    "gross": "float64",
    "days": "int64",
    "mileage": "int64",
    "year": "int64"
}

# Fields that should be deduplicated
DEDUPE_FIELDS = {
    "sales": ["vin", "stock_num", "date", "customer_name"],
    "inventory": ["vin", "stock_num"],
    "leads": ["email", "phone", "date", "customer_name"]
}

# Define schemas for validated data
SCHEMAS = {
    "sales": {
        "required_columns": ["date", "gross", "vin", "lead_source", "salesperson"],
        "column_types": {
            "date": "datetime64[ns]",
            "gross": "float64",
            "vin": "str",
            "lead_source": "str",
            "salesperson": "str",
            "sale_price": "float64"
        }
    },
    "inventory": {
        "required_columns": ["vin", "days_in_stock", "list_price"],
        "column_types": {
            "vin": "str",
            "days_in_stock": "int64",
            "list_price": "float64",
            "vehicle_year": "int64",
            "vehicle_make": "str",
            "vehicle_model": "str"
        }
    },
    "leads": {
        "required_columns": ["date", "lead_source", "status"],
        "column_types": {
            "date": "datetime64[ns]",
            "lead_source": "str",
            "status": "str",
            "email": "str",
            "phone": "str"
        }
    }
}

async def normalize_and_validate(
    file_path: str,
    vendor_id: str,
    report_type: str,
    dealer_id: str
) -> Dict[str, Any]:
    """
    Run the full normalization and validation pipeline on a data file.
    
    Args:
        file_path: Path to the data file
        vendor_id: ID of the vendor (e.g., 'dealersocket')
        report_type: Type of report (e.g., 'sales', 'inventory', 'leads')
        dealer_id: ID of the dealer
        
    Returns:
        Dictionary with the result of the pipeline
    """
    try:
        # Start tracking pipeline steps
        pipeline_result = {
            "success": False,
            "file_path": file_path,
            "vendor_id": vendor_id,
            "report_type": report_type,
            "dealer_id": dealer_id,
            "steps": [],
            "start_time": datetime.now(timezone.utc).isoformat()
        }
        
        # Step 1: Load the file
        log_info(
            f"Loading data file {file_path} for dealer {dealer_id}, vendor {vendor_id}, report {report_type}",
            dealer_id,
            "load_data"
        )
        
        try:
            if file_path.lower().endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.lower().endswith(('.xls', '.xlsx')):
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            
            pipeline_result["steps"].append({
                "step": "load_data",
                "success": True,
                "message": f"Loaded {len(df)} rows with {len(df.columns)} columns",
                "time": datetime.now(timezone.utc).isoformat()
            })
        except Exception as e:
            error_msg = f"Failed to load data file: {str(e)}"
            log_error(
                e,
                dealer_id,
                "load_data"
            )
            pipeline_result["steps"].append({
                "step": "load_data",
                "success": False,
                "message": error_msg,
                "time": datetime.now(timezone.utc).isoformat()
            })
            pipeline_result["error"] = error_msg
            return pipeline_result
        
        # Step 2: Map columns
        try:
            df, mapping_result = map_columns(df, vendor_id, report_type)
            
            pipeline_result["steps"].append({
                "step": "map_columns",
                "success": True,
                "message": f"Mapped {len(mapping_result['mapped'])} columns, {len(mapping_result['unmapped'])} unmapped",
                "details": mapping_result,
                "time": datetime.now(timezone.utc).isoformat()
            })
        except Exception as e:
            error_msg = f"Failed to map columns: {str(e)}"
            log_error(
                e,
                dealer_id,
                "map_columns"
            )
            pipeline_result["steps"].append({
                "step": "map_columns",
                "success": False,
                "message": error_msg,
                "time": datetime.now(timezone.utc).isoformat()
            })
            pipeline_result["error"] = error_msg
            return pipeline_result
        
        # Step 3: Clean data
        try:
            df, cleaning_result = clean_data(df, report_type)
            
            pipeline_result["steps"].append({
                "step": "clean_data",
                "success": True,
                "message": f"Cleaned data with {len(cleaning_result['actions'])} actions",
                "details": cleaning_result,
                "time": datetime.now(timezone.utc).isoformat()
            })
        except Exception as e:
            error_msg = f"Failed to clean data: {str(e)}"
            log_error(
                e,
                dealer_id,
                "clean_data"
            )
            pipeline_result["steps"].append({
                "step": "clean_data",
                "success": False,
                "message": error_msg,
                "time": datetime.now(timezone.utc).isoformat()
            })
            # Continue with validation despite cleaning errors
        
        # Step 4: Validate against schema
        try:
            validation_result = validate_schema(df, report_type)
            
            pipeline_result["steps"].append({
                "step": "validate_schema",
                "success": validation_result["valid"],
                "message": validation_result["message"],
                "details": validation_result,
                "time": datetime.now(timezone.utc).isoformat()
            })
            
            if not validation_result["valid"]:
                pipeline_result["warning"] = f"Schema validation issues: {validation_result['message']}"
                # Continue with saving despite validation issues
        except Exception as e:
            error_msg = f"Failed to validate schema: {str(e)}"
            log_error(
                e,
                dealer_id,
                "validate_schema"
            )
            pipeline_result["steps"].append({
                "step": "validate_schema",
                "success": False,
                "message": error_msg,
                "time": datetime.now(timezone.utc).isoformat()
            })
            pipeline_result["warning"] = error_msg
            # Continue with saving despite validation errors
        
        # Step 5: Save processed data
        try:
            output_path = save_processed_data(df, vendor_id, report_type, dealer_id)
            
            pipeline_result["steps"].append({
                "step": "save_data",
                "success": True,
                "message": f"Saved processed data to {output_path}",
                "output_path": output_path,
                "time": datetime.now(timezone.utc).isoformat()
            })
            
            pipeline_result["output_path"] = output_path
        except Exception as e:
            error_msg = f"Failed to save processed data: {str(e)}"
            log_error(
                e,
                dealer_id,
                "save_data"
            )
            pipeline_result["steps"].append({
                "step": "save_data",
                "success": False,
                "message": error_msg,
                "time": datetime.now(timezone.utc).isoformat()
            })
            pipeline_result["error"] = error_msg
            return pipeline_result
        
        # Final step: Create a metadata file
        try:
            metadata_path = create_metadata_file(
                file_path,
                output_path,
                vendor_id,
                report_type,
                dealer_id,
                pipeline_result
            )
            
            pipeline_result["steps"].append({
                "step": "create_metadata",
                "success": True,
                "message": f"Created metadata file at {metadata_path}",
                "metadata_path": metadata_path,
                "time": datetime.now(timezone.utc).isoformat()
            })
            
            pipeline_result["metadata_path"] = metadata_path
        except Exception as e:
            error_msg = f"Failed to create metadata file: {str(e)}"
            log_error(
                e,
                dealer_id,
                "create_metadata"
            )
            pipeline_result["steps"].append({
                "step": "create_metadata",
                "success": False,
                "message": error_msg,
                "time": datetime.now(timezone.utc).isoformat()
            })
            # Not critical, continue
        
        # Mark as success if we got this far
        pipeline_result["success"] = True
        pipeline_result["end_time"] = datetime.now(timezone.utc).isoformat()
        
        # Calculate row counts for summary
        if "output_path" in pipeline_result:
            try:
                result_df = pd.read_csv(pipeline_result["output_path"])
                pipeline_result["row_count"] = len(result_df)
                pipeline_result["column_count"] = len(result_df.columns)
            except:
                pass
        
        log_info(
            f"Completed ingestion pipeline for dealer {dealer_id}, vendor {vendor_id}, report {report_type}",
            dealer_id,
            "ingestion_pipeline"
        )
        
        return pipeline_result
    
    except Exception as e:
        log_error(
            e,
            dealer_id,
            "ingestion_pipeline"
        )
        
        return {
            "success": False,
            "file_path": file_path,
            "vendor_id": vendor_id,
            "report_type": report_type,
            "dealer_id": dealer_id,
            "error": f"Unhandled error in ingestion pipeline: {str(e)}",
            "traceback": traceback.format_exc()
        }


def map_columns(df: pd.DataFrame, vendor_id: str, report_type: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Map columns from vendor-specific names to standard names.
    
    Args:
        df: Input DataFrame
        vendor_id: ID of the vendor
        report_type: Type of report
        
    Returns:
        Tuple of (mapped DataFrame, mapping results)
    """
    # Tracking results
    result = {
        "original_columns": list(df.columns),
        "mapped": {},
        "unmapped": [],
        "vendor_specific": {}
    }
    
    # Get mappings for this report type
    if report_type not in COLUMN_MAPPINGS:
        raise ValueError(f"Unknown report type: {report_type}")
    
    mappings = COLUMN_MAPPINGS[report_type]
    
    # Create a new DataFrame to build with mapped columns
    mapped_df = pd.DataFrame(index=df.index)
    
    # First pass: try to map using standard column mappings
    for target_col, source_options in mappings.items():
        # Skip vendor-specific mappings (they're dictionaries, not lists)
        if not isinstance(source_options, list):
            continue
        
        # Try each possible source column
        mapped = False
        for source_col in source_options:
            # Try case insensitive matching
            matching_cols = [col for col in df.columns if col.lower() == source_col.lower()]
            if matching_cols:
                # Use the first match
                match = matching_cols[0]
                mapped_df[target_col] = df[match]
                result["mapped"][target_col] = match
                mapped = True
                break
        
        if not mapped:
            # Second attempt: try partial matching (e.g., "Customer Name" matches "customer")
            for source_col in source_options:
                matching_cols = [
                    col for col in df.columns 
                    if source_col.lower() in col.lower() or col.lower() in source_col.lower()
                ]
                if matching_cols:
                    # Use the first match
                    match = matching_cols[0]
                    mapped_df[target_col] = df[match]
                    result["mapped"][target_col] = match
                    mapped = True
                    break
    
    # Second pass: try vendor-specific mappings if available
    vendor_mappings = mappings.get(vendor_id, {})
    if vendor_mappings:
        for target_col, source_options in vendor_mappings.items():
            # Skip if already mapped
            if target_col in result["mapped"]:
                continue
            
            # Try each possible source column
            mapped = False
            for source_col in source_options:
                # Try case insensitive matching
                matching_cols = [col for col in df.columns if col.lower() == source_col.lower()]
                if matching_cols:
                    # Use the first match
                    match = matching_cols[0]
                    mapped_df[target_col] = df[match]
                    result["mapped"][target_col] = match
                    result["vendor_specific"][target_col] = match
                    mapped = True
                    break
            
            if not mapped:
                # Second attempt: try partial matching
                for source_col in source_options:
                    matching_cols = [
                        col for col in df.columns 
                        if source_col.lower() in col.lower() or col.lower() in source_col.lower()
                    ]
                    if matching_cols:
                        # Use the first match
                        match = matching_cols[0]
                        mapped_df[target_col] = df[match]
                        result["mapped"][target_col] = match
                        result["vendor_specific"][target_col] = match
                        mapped = True
                        break
    
    # Track unmapped original columns
    mapped_sources = set(result["mapped"].values())
    result["unmapped"] = [col for col in df.columns if col not in mapped_sources]
    
    # Copy any unmapped columns with their original names
    for col in result["unmapped"]:
        mapped_df[col] = df[col]
    
    return mapped_df, result


def clean_data(df: pd.DataFrame, report_type: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Clean and normalize data.
    
    Args:
        df: Input DataFrame
        report_type: Type of report
        
    Returns:
        Tuple of (cleaned DataFrame, cleaning results)
    """
    # Make a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Tracking results
    result = {
        "actions": [],
        "row_count": {
            "before": len(df),
            "after": len(df)
        }
    }
    
    # 1. Convert data types based on column names
    for col in cleaned_df.columns:
        # Try to infer type from column name
        inferred_type = None
        
        # Check for date columns
        if any(date_term in col.lower() for date_term in ['date', 'time', 'created']):
            inferred_type = "datetime64[ns]"
        
        # Check for monetary columns
        elif any(price_term in col.lower() for price_term in ['price', 'cost', 'gross', 'msrp', 'amount']):
            inferred_type = "float64"
        
        # Check for numeric columns
        elif any(num_term in col.lower() for num_term in ['days', 'count', 'number', 'id', 'year', 'mileage']):
            inferred_type = "int64"
        
        # Apply type conversion if inferred
        if inferred_type:
            try:
                original_type = cleaned_df[col].dtype
                
                if inferred_type == "datetime64[ns]":
                    cleaned_df[col] = pd.to_datetime(cleaned_df[col], errors='coerce')
                elif inferred_type == "float64":
                    # Handle currency strings (e.g. "$1,234.56")
                    if cleaned_df[col].dtype == 'object':
                        cleaned_df[col] = cleaned_df[col].astype(str).str.replace('[$,]', '', regex=True)
                    cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
                elif inferred_type == "int64":
                    cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce').fillna(0).astype('int64')
                
                result["actions"].append({
                    "action": "convert_type",
                    "column": col,
                    "from": str(original_type),
                    "to": inferred_type
                })
            except Exception as e:
                # Log but continue
                logger.warning(f"Type conversion failed for {col}: {str(e)}")
    
    # 2. Handle missing values
    for col in cleaned_df.columns:
        missing_count = cleaned_df[col].isna().sum()
        if missing_count > 0:
            # Different handling based on column type
            if cleaned_df[col].dtype == 'datetime64[ns]':
                # For date columns, use the earliest date in the column
                if cleaned_df[col].notna().any():
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].min())
                    result["actions"].append({
                        "action": "fill_missing",
                        "column": col,
                        "count": int(missing_count),
                        "method": "min_date"
                    })
            elif pd.api.types.is_numeric_dtype(cleaned_df[col]):
                # For numeric columns, use zero
                cleaned_df[col] = cleaned_df[col].fillna(0)
                result["actions"].append({
                    "action": "fill_missing",
                    "column": col,
                    "count": int(missing_count),
                    "method": "zero"
                })
            else:
                # For string columns, use empty string
                cleaned_df[col] = cleaned_df[col].fillna("")
                result["actions"].append({
                    "action": "fill_missing",
                    "column": col,
                    "count": int(missing_count),
                    "method": "empty_string"
                })
    
    # 3. Normalize text fields
    for col in cleaned_df.columns:
        if cleaned_df[col].dtype == 'object':
            # Normalize strings: strip whitespace, handle case
            if any(text_term in col.lower() for text_term in ['source', 'name', 'model', 'make', 'salesperson']):
                # Convert to string, strip whitespace, standardize case
                cleaned_df[col] = cleaned_df[col].astype(str).str.strip()
                
                if 'lead_source' in col.lower():
                    # For lead sources, convert to title case
                    cleaned_df[col] = cleaned_df[col].str.title()
                    result["actions"].append({
                        "action": "normalize_text",
                        "column": col,
                        "method": "title_case"
                    })
                else:
                    # For other text fields, basic normalization
                    result["actions"].append({
                        "action": "normalize_text",
                        "column": col,
                        "method": "strip_whitespace"
                    })
    
    # 4. Deduplicate based on report type
    if report_type in DEDUPE_FIELDS:
        dedupe_cols = [col for col in DEDUPE_FIELDS[report_type] if col in cleaned_df.columns]
        
        if dedupe_cols:
            before_count = len(cleaned_df)
            cleaned_df = cleaned_df.drop_duplicates(subset=dedupe_cols)
            after_count = len(cleaned_df)
            
            if before_count > after_count:
                result["actions"].append({
                    "action": "deduplicate",
                    "columns": dedupe_cols,
                    "removed_rows": before_count - after_count
                })
                
                result["row_count"]["after"] = after_count
    
    # 5. Special handling for specific column types
    
    # VIN normalization
    if 'vin' in cleaned_df.columns:
        # Convert to uppercase and remove spaces
        cleaned_df['vin'] = cleaned_df['vin'].astype(str).str.upper().str.replace(' ', '')
        result["actions"].append({
            "action": "normalize_vin",
            "column": "vin"
        })
    
    # Date range validation
    date_cols = [col for col in cleaned_df.columns if cleaned_df[col].dtype == 'datetime64[ns]']
    for col in date_cols:
        # Check for future dates
        now = pd.Timestamp.now()
        future_dates = cleaned_df[col] > now
        
        if future_dates.any():
            # Replace future dates with current date
            cleaned_df.loc[future_dates, col] = now
            result["actions"].append({
                "action": "fix_future_dates",
                "column": col,
                "count": int(future_dates.sum())
            })
        
        # Check for very old dates (likely errors)
        old_threshold = pd.Timestamp('2000-01-01')
        very_old_dates = cleaned_df[col] < old_threshold
        
        if very_old_dates.any():
            # Use the median date for old dates
            if cleaned_df[col].notna().any():
                median_date = cleaned_df.loc[cleaned_df[col] >= old_threshold, col].median()
                if pd.notna(median_date):
                    cleaned_df.loc[very_old_dates, col] = median_date
                    result["actions"].append({
                        "action": "fix_old_dates",
                        "column": col,
                        "count": int(very_old_dates.sum())
                    })
    
    return cleaned_df, result


def validate_schema(df: pd.DataFrame, report_type: str) -> Dict[str, Any]:
    """
    Validate DataFrame against the expected schema.
    
    Args:
        df: DataFrame to validate
        report_type: Type of report
        
    Returns:
        Dictionary with validation results
    """
    result = {
        "valid": True,
        "missing_required": [],
        "type_mismatches": [],
        "message": "Validation passed"
    }
    
    # Check if schema exists for this report type
    if report_type not in SCHEMAS:
        return {
            "valid": False,
            "message": f"No schema defined for report type: {report_type}"
        }
    
    schema = SCHEMAS[report_type]
    
    # Check required columns
    for col in schema["required_columns"]:
        if col not in df.columns:
            result["missing_required"].append(col)
            result["valid"] = False
    
    # Check column types
    for col, expected_type in schema["column_types"].items():
        if col in df.columns:
            actual_type = df[col].dtype.name
            
            # For datetime types, need a different check
            if expected_type == "datetime64[ns]":
                if not pd.api.types.is_datetime64_dtype(df[col]):
                    result["type_mismatches"].append({
                        "column": col,
                        "expected": expected_type,
                        "actual": actual_type
                    })
                    result["valid"] = False
            # For numeric types, accept any numeric
            elif expected_type in ["float64", "int64"] and not pd.api.types.is_numeric_dtype(df[col]):
                result["type_mismatches"].append({
                    "column": col,
                    "expected": expected_type,
                    "actual": actual_type
                })
                result["valid"] = False
            # For string types
            elif expected_type == "str" and not pd.api.types.is_string_dtype(df[col]) and not pd.api.types.is_object_dtype(df[col]):
                result["type_mismatches"].append({
                    "column": col,
                    "expected": expected_type,
                    "actual": actual_type
                })
                result["valid"] = False
    
    # Update validation message
    if not result["valid"]:
        missing_msg = f"Missing columns: {', '.join(result['missing_required'])}" if result["missing_required"] else ""
        type_msg = f"Type mismatches: {len(result['type_mismatches'])}" if result["type_mismatches"] else ""
        result["message"] = "; ".join(filter(None, [missing_msg, type_msg]))
    
    return result


def save_processed_data(
    df: pd.DataFrame,
    vendor_id: str,
    report_type: str,
    dealer_id: str
) -> str:
    """
    Save processed data to the appropriate location.
    
    Args:
        df: Processed DataFrame
        vendor_id: ID of the vendor
        report_type: Type of report
        dealer_id: ID of the dealer
        
    Returns:
        Path to the saved file
    """
    # Determine target directory structure
    target_dir = os.path.join(
        DEFAULT_DATA_DIR,
        "processed",
        vendor_id,
        dealer_id,
        report_type
    )
    os.makedirs(target_dir, exist_ok=True)
    
    # Create a timestamped filename
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"{report_type}_processed_{timestamp}.csv"
    output_path = os.path.join(target_dir, filename)
    
    # Save the DataFrame
    df.to_csv(output_path, index=False)
    
    return output_path


def create_metadata_file(
    input_path: str,
    output_path: str,
    vendor_id: str,
    report_type: str,
    dealer_id: str,
    pipeline_result: Dict[str, Any]
) -> str:
    """
    Create a metadata file for the processed data.
    
    Args:
        input_path: Path to the input file
        output_path: Path to the output file
        vendor_id: ID of the vendor
        report_type: Type of report
        dealer_id: ID of the dealer
        pipeline_result: Results from the pipeline
        
    Returns:
        Path to the metadata file
    """
    # Create metadata directory
    metadata_dir = os.path.join(
        DEFAULT_DATA_DIR,
        "metadata",
        vendor_id,
        dealer_id,
        report_type
    )
    os.makedirs(metadata_dir, exist_ok=True)
    
    # Extract timestamp from output path
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if "timestamp" in pipeline_result:
        timestamp = pipeline_result["timestamp"]
    
    # Create metadata filename
    metadata_filename = f"{report_type}_metadata_{timestamp}.json"
    metadata_path = os.path.join(metadata_dir, metadata_filename)
    
    # Create metadata content
    metadata = {
        "input_file": input_path,
        "output_file": output_path,
        "vendor_id": vendor_id,
        "report_type": report_type,
        "dealer_id": dealer_id,
        "processing_time": datetime.now(timezone.utc).isoformat(),
        "pipeline_result": {
            key: value for key, value in pipeline_result.items()
            if key != "steps"  # Exclude detailed steps to keep metadata compact
        }
    }
    
    # Create a file stats section if output file exists
    if os.path.exists(output_path):
        try:
            df = pd.read_csv(output_path)
            metadata["file_stats"] = {
                "row_count": len(df),
                "column_count": len(df.columns),
                "columns": list(df.columns),
                "file_size_bytes": os.path.getsize(output_path)
            }
            
            # Add date range if there's a date column
            date_cols = [col for col in df.columns if any(date_term in col.lower() for date_term in ['date', 'time'])]
            if date_cols:
                try:
                    df[date_cols[0]] = pd.to_datetime(df[date_cols[0]], errors='coerce')
                    min_date = df[date_cols[0]].min()
                    max_date = df[date_cols[0]].max()
                    
                    if pd.notna(min_date) and pd.notna(max_date):
                        metadata["file_stats"]["date_range"] = {
                            "start": min_date.isoformat(),
                            "end": max_date.isoformat(),
                            "days": (max_date - min_date).days
                        }
                except:
                    pass  # Skip date range if conversion fails
        except:
            pass  # Skip file stats if reading fails
    
    # Write metadata to file
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata_path


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def test():
        # Define test file - replace with a real path
        test_file = os.path.join(DEFAULT_DATA_DIR, "example_sales.csv")
        
        if not os.path.exists(test_file):
            print(f"Test file not found: {test_file}")
            return
        
        # Run the pipeline
        result = await normalize_and_validate(
            test_file,
            "dealersocket",
            "sales",
            "test_dealer"
        )
        
        # Print result
        print(json.dumps(result, indent=2))
    
    asyncio.run(test())