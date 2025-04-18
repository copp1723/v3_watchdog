"""
Data I/O utilities for Watchdog AI.
Provides cached functions for loading and processing data with Redis support.
"""

import pandas as pd
import streamlit as st
from typing import Dict, Any, Optional, Tuple
import logging
import re
import os
import zipfile
import sentry_sdk
import hashlib
from pandas.errors import ParserError
from .data_normalization import normalize_columns, get_supported_aliases
from .errors import ValidationError
from .term_normalizer import normalize, normalizer
from .cache import get_cache

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    'LeadSource',
    'TotalGross',
    'VIN',
    'SaleDate',
    'SalePrice'
]

def validate_schema(df: pd.DataFrame, normalize_terms_data: bool = True) -> pd.DataFrame:
    """
    Validate and normalize the DataFrame schema and content.
    
    Args:
        df: Input DataFrame
        normalize_terms_data: Whether to normalize term variations in key columns
        
    Returns:
        Normalized DataFrame with canonical column names and standardized terms
        
    Raises:
        ValidationError if required columns are missing
    """
    # Normalize column names
    df = normalize_columns(df)
    
    # Check for required columns
    expected = set(REQUIRED_COLUMNS)
    missing = expected - set(df.columns)
    
    if missing:
        # Get the original column names for better error messages
        found_cols = list(df.columns)
        
        error_msg = (
            f"Missing required columns: {', '.join(sorted(missing))}\n"
            f"Found columns: {', '.join(found_cols)}"
        )
        
        raise ValidationError(error_msg)
    
    # Normalize term variations if requested
    if normalize_terms_data:
        # Identify columns that should be normalized
        normalizable_columns = [
            'LeadSource',
            'VehicleType' if 'VehicleType' in df.columns else None,
            'SalesRepName' if 'SalesRepName' in df.columns else None
        ]
        normalizable_columns = [col for col in normalizable_columns if col]
        if normalizable_columns:
            try:
                # Apply term normalization
                df = normalize(df, columns=normalizable_columns)
                logger.info(f"Applied term normalization to columns: {normalizable_columns}")
            except Exception as e:
                logger.warning(f"Term normalization failed: {str(e)}")
        
    return df

@st.cache_data
def load_data(uploaded_file) -> pd.DataFrame:
    """
    Load data from uploaded file with Redis caching.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        Loaded DataFrame
        
    Raises:
        ValueError if required columns are missing or file format is invalid
    """
    try:
        # Determine file type by extension or MIME
        file_name = uploaded_file.name
        file_ext = os.path.splitext(file_name)[1].lower()
        mime_type = getattr(uploaded_file, 'type', None)
        allowed_exts = {'.csv', '.xlsx', '.xls'}
        allowed_mimes = {
            'text/csv',
            'application/vnd.ms-excel',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        }
        if file_ext not in allowed_exts and (not mime_type or mime_type not in allowed_mimes):
            msg = (
                f"Unsupported file format: '{file_name}'. "
                "Please upload CSV (.csv) or Excel (.xlsx, .xls) files."
            )
            logger.error(f"Invalid file format: {file_name} (mime: {mime_type})")
            sentry_sdk.capture_message(f"Upload failed: {file_name} (unsupported format)", level='error')
            raise ValueError(msg)
        
        # Get file content as bytes for caching
        file_bytes = uploaded_file.getvalue()
        uploaded_file.seek(0)  # Reset file position after reading
        
        # Check cache using Redis before parsing
        # Create cache key based on file content hash and rules version
        redis_cache = get_cache()
        rules_version = getattr(normalizer, 'rules_version', None)
        cache_key = redis_cache.create_key(file_bytes, rules_version)
        
        # Try to get from cache first
        if redis_cache.is_available:
            sentry_sdk.set_tag("cache_check", "enabled")
            cached_df = redis_cache.get(cache_key)
            if cached_df is not None:
                logger.info(f"Cache hit for file '{file_name}' with {len(cached_df)} rows")
                sentry_sdk.set_tag("cache_result", "hit")
                sentry_sdk.capture_message(f"Cache hit for upload: {file_name}", level='info')
                return cached_df
            sentry_sdk.set_tag("cache_result", "miss")
        else:
            sentry_sdk.set_tag("cache_check", "disabled")
            
        # Encrypt and store the uploaded file
        try:
            from src.utils.encryption import encrypt_bytes
            
            # Ensure uploads directory exists
            from datetime import datetime
            
            # Create a structured directory for uploads
            base_uploads_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'uploads')
            # Create subdirectory by date for better organization
            date_str = datetime.now().strftime("%Y-%m-%d")
            uploads_dir = os.path.join(base_uploads_dir, date_str)
            os.makedirs(uploads_dir, exist_ok=True)
            
            # Create a unique filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_hash = hashlib.md5(file_bytes[:1024]).hexdigest()[:8]  # Use part of file hash for uniqueness
            encrypted_filename = f"{timestamp}_{file_hash}_{uploaded_file.name}.enc"
            encrypted_file_path = os.path.join(uploads_dir, encrypted_filename)
            
            # Encrypt file bytes and save
            encrypted_bytes = encrypt_bytes(file_bytes)
            with open(encrypted_file_path, 'wb') as f:
                f.write(encrypted_bytes)
            
            # Store metadata about the upload
            metadata_path = encrypted_file_path + '.meta'
            import json
            metadata = {
                'original_filename': uploaded_file.name,
                'upload_time': datetime.now().isoformat(),
                'encrypted_path': encrypted_file_path,
                'file_size': len(file_bytes),
                'content_hash': hashlib.sha256(file_bytes).hexdigest(),
                'rules_version': rules_version
            }
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"File '{uploaded_file.name}' encrypted and saved to {encrypted_file_path}")
            
            # Send success event to Sentry if available
            try:
                sentry_sdk.set_tag("file_type", os.path.splitext(uploaded_file.name)[1])
                sentry_sdk.set_tag("file_size_kb", len(file_bytes) // 1024)
                sentry_sdk.capture_message(f"Upload success: {uploaded_file.name} encrypted and stored.", level="info")
            except ImportError:
                pass  # Sentry not available
                
        except Exception as encrypt_error:
            logger.error(f"Error encrypting file '{uploaded_file.name}': {str(encrypt_error)}")
            try:
                sentry_sdk.capture_exception(encrypt_error)
            except ImportError:
                pass  # Sentry not available
        
        # Read file based on extension with robust error handling
        try:
            if file_ext == '.csv':
                # Try reading as UTF-8 CSV
                try:
                    df = pd.read_csv(uploaded_file, encoding='utf-8')
                except UnicodeDecodeError as e:
                    sentry_sdk.capture_exception(e)
                    uploaded_file.seek(0)
                    try:
                        df = pd.read_csv(uploaded_file, encoding='latin-1')
                    except Exception as e2:
                        msg = f"Error parsing CSV file '{file_name}': {str(e2)}"
                        logger.error(msg)
                        sentry_sdk.capture_exception(e2)
                        raise ValueError(msg)
                except ParserError as e:
                    msg = f"Error parsing CSV file '{file_name}': {str(e)}"
                    logger.error(msg)
                    sentry_sdk.capture_exception(e)
                    raise ValueError(msg)
                # Handle delimiter issues (e.g., only one column parsed)
                if len(df.columns) == 1 and ',' in df.columns[0]:
                    uploaded_file.seek(0)
                    try:
                        df = pd.read_csv(uploaded_file, sep=';')
                        logger.info(f"CSV file '{file_name}' was read using ';' as delimiter")
                    except ParserError as e:
                        msg = f"Error parsing CSV with ';' delimiter for file '{file_name}': {str(e)}"
                        logger.error(msg)
                        sentry_sdk.capture_exception(e)
                        raise ValueError(msg)
            else:
                # Read Excel file (.xlsx, .xls)
                try:
                    df = pd.read_excel(uploaded_file)
                except Exception as excel_error:
                    msg = (
                        f"Error reading Excel file '{file_name}'. "
                        f"The file may be corrupted or password-protected. Details: {str(excel_error)}"
                    )
                    logger.error(msg)
                    sentry_sdk.capture_exception(excel_error)
                    raise ValueError(msg)
        except Exception as read_error:
            msg = (
                f"Error reading file '{file_name}'. "
                f"Please check that the file is not corrupted and is in the correct format. "
                f"Error details: {str(read_error)}"
            )
            logger.error(msg)
            sentry_sdk.capture_exception(read_error)
            raise ValueError(msg)
        
        # Validate basic DataFrame properties
        if df.empty:
            error_msg = f"The uploaded file '{uploaded_file.name}' is empty. Please upload a file with data."
            logger.error(f"Empty DataFrame from file: {uploaded_file.name}")
            raise ValueError(error_msg)
        
        # Show the user what columns were found
        found_cols = list(df.columns)
        logger.info(f"Loaded file '{uploaded_file.name}' with columns: {found_cols}")
        
        try:
            # Normalize column names and validate schema
            df = validate_schema(df)
            
            # Normalize headers & values after schema validation
            sentry_sdk.set_tag("normalization_step", "post_schema")
            sentry_sdk.set_tag("normalization_rules_version", normalizer.rules_version)
            df = normalize(df)
            
            # Store in Redis cache after successful parsing and normalization
            if redis_cache.is_available:
                cache_result = redis_cache.set(cache_key, df)
                if cache_result:
                    logger.info(f"Cached DataFrame for file '{file_name}' with key {cache_key[:20]}...")
                else:
                    logger.warning(f"Failed to cache DataFrame for file '{file_name}'")
            
            # Log upload success to Sentry
            sentry_sdk.capture_message(f"Upload succeeded: {file_name}", level='info')
            return df
            
        except ValidationError as e:
            # Get supported aliases for missing columns
            aliases = get_supported_aliases()
            missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
            missing_details = []
            
            for col in missing_cols:
                if col in aliases:
                    supported = [f"'{alias}'" for alias in aliases[col]]
                    missing_details.append(f"{col} (accepts: {', '.join(supported)})")
                else:
                    missing_details.append(col)
            
            error_msg = (
                f"Your file is missing some required columns. Found these columns: {', '.join(found_cols)}\n\n"
                f"Missing required columns: {', '.join(missing_details)}\n\n"
                "Please ensure your file has these columns with any of the supported names."
            )
            logger.error(f"Schema validation error: {error_msg}")
            raise ValueError(error_msg)
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

@st.cache_data
def compute_lead_gross(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute lead source gross metrics with caching.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with lead source metrics
    """
    try:
        # Check if we can use Redis cache
        redis_cache = get_cache()
        if redis_cache.is_available:
            # Create a cache key based on DataFrame content hash
            df_json = df.to_json().encode('utf-8')
            cache_key = redis_cache.create_key(df_json, "compute_lead_gross")
            
            # Try to get from cache
            cached_result = redis_cache.get(cache_key)
            if cached_result is not None:
                sentry_sdk.set_tag("metrics_cache_result", "hit")
                logger.info("Cache hit for lead gross computation")
                return cached_result
            
            sentry_sdk.set_tag("metrics_cache_result", "miss")
        
        # Group by lead source and calculate metrics
        lead_gross = df.groupby('LeadSource').agg({
            'TotalGross': ['sum', 'mean', 'count']
        }).reset_index()
        
        # Flatten column names
        lead_gross.columns = ['LeadSource', 'TotalGross', 'AvgGross', 'Count']
        
        # Sort by total gross descending
        lead_gross = lead_gross.sort_values('TotalGross', ascending=False)
        
        # Cache the result if Redis is available
        if redis_cache.is_available:
            cache_result = redis_cache.set(cache_key, lead_gross)
            if cache_result:
                logger.info("Cached lead gross computation result")
        
        return lead_gross
        
    except Exception as e:
        logger.error(f"Error computing lead gross: {str(e)}")
        sentry_sdk.capture_exception(e)
        raise

def load_encrypted_file(file_path: str) -> pd.DataFrame:
    """
    Load an encrypted data file.
    
    Args:
        file_path: Path to the encrypted file
        
    Returns:
        DataFrame with the decrypted data
        
    Raises:
        FileNotFoundError if the file doesn't exist
        ValueError if the file can't be decrypted or parsed
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        # Import decrypt function
        from .encryption import decrypt_bytes
        
        # Read the encrypted file
        with open(file_path, 'rb') as f:
            encrypted_data = f.read()
        
        # Decrypt the data
        decrypted_data = decrypt_bytes(encrypted_data)
        
        # Determine file type from original filename
        if file_path.endswith('.enc'):
            # Try to get original filename from metadata
            meta_path = file_path + '.meta'
            if os.path.exists(meta_path):
                import json
                with open(meta_path, 'r') as f:
                    metadata = json.load(f)
                original_filename = metadata.get('original_filename', '')
            else:
                # Extract original filename from the encrypted filename
                parts = os.path.basename(file_path).split('_', 2)
                original_filename = parts[2][:-4] if len(parts) > 2 else ""
        else:
            original_filename = os.path.basename(file_path)
        
        # Parse based on file type
        import io
        if original_filename.lower().endswith('.csv'):
            # Try different encodings for CSV
            try:
                return pd.read_csv(io.BytesIO(decrypted_data), encoding='utf-8')
            except UnicodeDecodeError:
                return pd.read_csv(io.BytesIO(decrypted_data), encoding='latin-1')
        elif original_filename.lower().endswith(('.xlsx', '.xls')):
            return pd.read_excel(io.BytesIO(decrypted_data))
        else:
            # Attempt to determine type from content
            try:
                return pd.read_csv(io.BytesIO(decrypted_data))
            except:
                return pd.read_excel(io.BytesIO(decrypted_data))
    
    except Exception as e:
        logger.error(f"Error loading encrypted file {file_path}: {str(e)}")
        raise ValueError(f"Could not load encrypted file: {str(e)}")


@st.cache_data
def list_encrypted_files() -> pd.DataFrame:
    """
    List all encrypted files in the uploads directory.
    
    Returns:
        DataFrame with metadata about available encrypted files
    """
    import os
    import glob
    import json
    from datetime import datetime
    
    # Base directory for uploads
    base_uploads_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                  'data', 'uploads')
    
    # Get all encrypted files
    encrypted_files = []
    for root, _, _ in os.walk(base_uploads_dir):
        encrypted_files.extend(glob.glob(os.path.join(root, "*.enc")))
    
    if not encrypted_files:
        return pd.DataFrame(columns=['filename', 'upload_date', 'size_kb', 'path'])
    
    # Extract metadata
    file_data = []
    for file_path in encrypted_files:
        meta_path = file_path + '.meta'
        
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r') as f:
                    metadata = json.load(f)
                
                file_data.append({
                    'filename': metadata.get('original_filename', os.path.basename(file_path)),
                    'upload_date': datetime.fromisoformat(metadata.get('upload_time', '')),
                    'size_kb': metadata.get('file_size', 0) // 1024,
                    'path': file_path
                })
            except:
                # If metadata is missing or invalid, use file properties
                stat = os.stat(file_path)
                file_data.append({
                    'filename': os.path.basename(file_path).split('_', 2)[-1][:-4],
                    'upload_date': datetime.fromtimestamp(stat.st_mtime),
                    'size_kb': stat.st_size // 1024,
                    'path': file_path
                })
        else:
            # If no metadata, use file properties
            stat = os.stat(file_path)
            file_data.append({
                'filename': os.path.basename(file_path).split('_', 2)[-1][:-4],
                'upload_date': datetime.fromtimestamp(stat.st_mtime),
                'size_kb': stat.st_size // 1024,
                'path': file_path
            })
    
    # Create DataFrame and sort by upload date (newest first)
    df = pd.DataFrame(file_data)
    return df.sort_values('upload_date', ascending=False).reset_index(drop=True)


@st.cache_data
def validate_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Validate DataFrame schema and data quality with caching.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Tuple of (validated DataFrame, validation summary)
    """
    try:
        # Check if we can use Redis cache
        redis_cache = get_cache()
        cache_hit = False
        
        if redis_cache.is_available:
            # Create a cache key based on DataFrame content hash
            df_json = df.to_json().encode('utf-8')
            cache_key = redis_cache.create_key(df_json, "validate_data")
            
            # Try to get validation summary from cache
            cached_summary = redis_cache.get(cache_key)
            if cached_summary is not None and isinstance(cached_summary, pd.DataFrame) and len(cached_summary) == 1:
                # Extract validation summary from single-row DataFrame
                try:
                    validation_summary = cached_summary.iloc[0, 0]
                    sentry_sdk.set_tag("validation_cache_result", "hit")
                    logger.info("Cache hit for data validation")
                    cache_hit = True
                    return df, validation_summary
                except Exception as cache_error:
                    logger.warning(f"Error retrieving validation summary from cache: {str(cache_error)}")
                    
            sentry_sdk.set_tag("validation_cache_result", "miss")
            
        # Calculate validation metrics
        validation_summary = {
            'total_records': len(df),
            'missing_values': {},
            'invalid_values': {}
        }
        
        # Check for missing values
        for col in REQUIRED_COLUMNS:
            missing = df[col].isna().sum()
            if missing > 0:
                validation_summary['missing_values'][col] = missing
        
        # Check for invalid values
        validation_summary['invalid_values']['negative_gross'] = (
            (df['TotalGross'] < 0).sum()
        )
        
        validation_summary['invalid_values']['empty_lead_source'] = (
            ((df['LeadSource'].isna()) | (df['LeadSource'] == '')).sum()
        )
        
        validation_summary['invalid_values']['invalid_vin'] = (
            ((df['VIN'].str.len() != 17) | (df['VIN'].isna())).sum()
        )
        
        # Calculate overall quality score
        total_issues = (
            sum(validation_summary['missing_values'].values()) +
            sum(validation_summary['invalid_values'].values())
        )
        validation_summary['quality_score'] = max(
            0, 100 - (total_issues / len(df) * 100)
        )
        
        # Cache the validation summary if Redis is available
        if redis_cache.is_available and not cache_hit:
            # We need to store the dictionary in a DataFrame for the cache to work properly
            summary_df = pd.DataFrame([{'summary': validation_summary}])
            cache_result = redis_cache.set(cache_key, summary_df)
            if cache_result:
                logger.info("Cached data validation result")
        
        return df, validation_summary
        
    except Exception as e:
        logger.error(f"Error validating data: {str(e)}")
        sentry_sdk.capture_exception(e)
        raise