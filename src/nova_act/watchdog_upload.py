import requests
import streamlit as st # Import streamlit for session state access
import sys
import os
import pandas as pd
import time
import mimetypes
import threading
import logging
from typing import Optional, Dict, Tuple, List, Any, Union, Callable
import uuid
from datetime import datetime
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s [%(levelname)s] [WatchdogUpload] %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('watchdog_upload')

# Ensure the project root is in the path for sibling imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    # Attempt to import the processing function
    from src.app import process_uploaded_file
except ImportError:
    logger.error("Could not import process_uploaded_file from src.app.")
    # Define a dummy function if import fails to prevent crashing
    def process_uploaded_file(file_obj, apply_auto_cleaning=False):
        logger.warning("Using dummy process_uploaded_file. Real processing will not occur.")
        return None, {"status": "error", "message": "Processing function not available"}, None, None


def upload_to_watchdog(file_path: str, auto_cleaning: bool = True, timeout: int = 300) -> bool:
    """
    Simulates uploading a file to the Watchdog endpoint and triggers processing.

    NOTE: Streamlit apps typically don't expose HTTP endpoints like '/upload'.
          This function directly calls the 'process_uploaded_file' logic from
          src/app.py after simulating a successful "upload". A real-world
          scenario might involve a separate API endpoint or a different
          trigger mechanism.

    Args:
        file_path (str): The path to the file collected by Nova Act.
        auto_cleaning (bool): Whether to apply automatic data cleaning.
        timeout (int): Maximum seconds to wait for processing (default 300).

    Returns:
        bool: True if processing was triggered successfully, False otherwise.
    """
    logger.info(f"Preparing to process file: {file_path}")
    
    # Track progress
    progress_key = f"upload_progress_{os.path.basename(file_path)}"
    st.session_state[progress_key] = 0
    
    # Validate file existence
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        st.session_state.nova_act_error = f"Upload failed: File not found at {file_path}"
        return False
    
    # Validate file size
    file_size = os.path.getsize(file_path)
    if file_size == 0:
        logger.error(f"File is empty: {file_path}")
        st.session_state.nova_act_error = f"Upload failed: File is empty ({file_path})"
        return False
    if file_size > 50 * 1024 * 1024:  # 50MB limit
        logger.error(f"File is too large: {file_path} ({file_size/1024/1024:.2f} MB)")
        st.session_state.nova_act_error = f"Upload failed: File exceeds 50MB limit ({file_size/1024/1024:.2f} MB)"
        return False
    
    # Validate file type
    file_type = _get_file_type(file_path)
    logger.info(f"Detected file type: {file_type}")
    if file_type not in ['csv', 'excel', 'text']:
        logger.error(f"Unsupported file type: {file_type}")
        st.session_state.nova_act_error = f"Upload failed: Unsupported file type ({file_type})"
        return False
    
    st.session_state[progress_key] = 10  # 10% - File validation complete
        
    try:
        # Simulate the upload and trigger processing directly
        # We need to open the file and pass the file object to process_uploaded_file
        with open(file_path, 'rb') as f:
            # Mimic the structure of a Streamlit UploadedFile if necessary
            f.name = os.path.basename(file_path)
            
            st.session_state[progress_key] = 30  # 30% - File opened and ready for processing
            logger.info(f"Calling process_uploaded_file for: {f.name}")
            
            # Use a thread to track progress during processing
            processing_done = threading.Event()
            
            # Start progress tracking in background
            threading.Thread(target=_track_progress, args=(progress_key, processing_done, timeout)).start()
            
            # Assuming process_uploaded_file handles the file object 'f'
            try:
                df, summary, report, schema_info = process_uploaded_file(f, apply_auto_cleaning=auto_cleaning)
                processing_done.set()  # Signal that processing is complete
                
                # Ensure the progress is updated to at least 80% on completion
                st.session_state[progress_key] = max(80, st.session_state[progress_key])
                
                # Process results
                if summary.get("status") == "success" and df is not None:
                    logger.info(f"Processing successful. Updating session state.")
                    st.session_state[progress_key] = 90  # 90% - Processing complete
                    
                    # Validate the processed data
                    if not _validate_processed_data(df, summary):
                        logger.error("Processed data validation failed")
                        st.session_state.nova_act_error = "Upload succeeded but data validation failed"
                        return False
                    
                    # Update session state directly - requires Streamlit context
                    try:
                        st.session_state.validated_data = df
                        st.session_state.validation_summary = summary
                        st.session_state.validation_report = report
                        st.session_state.schema_info = schema_info
                        st.session_state.last_uploaded_file = f.name
                        
                        # Store upload metadata
                        st.session_state.nova_act_last_upload = {
                            'file': f.name,
                            'timestamp': time.time(),
                            'size': file_size,
                            'type': file_type,
                            'rows': len(df) if df is not None else 0,
                            'columns': len(df.columns) if df is not None else 0
                        }
                        
                        st.session_state[progress_key] = 100  # 100% - All complete
                        logger.info("Session state updated successfully.")
                        return True
                    except Exception as session_e:
                        logger.error(f"Failed to update Streamlit session state: {session_e}")
                        st.session_state.nova_act_error = "Processing complete but UI update failed."
                        return False  # Return False as the full workflow didn't complete
                else:
                    logger.error(f"Processing failed: {summary.get('message', 'Unknown processing error')}")
                    st.session_state.nova_act_error = f"Processing failed for {os.path.basename(file_path)}: {summary.get('message', 'Unknown error')}"
                    return False
            
            except Exception as proc_e:
                processing_done.set()  # Signal that processing is complete (with error)
                logger.error(f"Error during file processing: {proc_e}")
                st.session_state.nova_act_error = f"Error during file processing: {str(proc_e)}"
                return False

    except Exception as e:
        logger.error(f"Exception during upload/processing simulation: {e}")
        try:
            st.session_state.nova_act_error = f"Upload/Processing failed for {os.path.basename(file_path)}: {str(e)}"
        except Exception as session_e:
            logger.error(f"Failed to set error in Streamlit session state: {session_e}")
        return False


def _get_file_type(file_path: str) -> str:
    """
    Determine the file type based on extension and content.
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        str: File type identifier ('csv', 'excel', 'text', 'unknown')
    """
    # Check extension first
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    if ext == '.csv':
        return 'csv'
    elif ext in ['.xls', '.xlsx', '.xlsm']:
        return 'excel'
    elif ext in ['.txt', '.dat', '.json']:
        return 'text'
    
    # If extension is ambiguous, check content type
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type:
        if mime_type == 'text/csv':
            return 'csv'
        elif 'spreadsheet' in mime_type or 'excel' in mime_type:
            return 'excel'
        elif mime_type.startswith('text/'):
            return 'text'
    
    # Try to read the first few bytes to determine type
    try:
        with open(file_path, 'rb') as f:
            header = f.read(8)
            # Excel signature check
            if header.startswith(b'\x50\x4B\x03\x04') or header.startswith(b'\xD0\xCF\x11\xE0'):
                return 'excel'
    except Exception:
        pass
    
    # Default to unknown
    return 'unknown'


def _validate_processed_data(df: pd.DataFrame, summary: dict) -> bool:
    """
    Validate the processed dataframe to ensure it meets minimal requirements.
    
    Args:
        df (pd.DataFrame): The processed dataframe
        summary (dict): Processing summary
        
    Returns:
        bool: True if validation passes, False otherwise
    """
    # Skip validation if dataframe is None
    if df is None:
        return False
    
    # Check for empty dataframe
    if df.empty:
        logger.warning("Processed dataframe is empty")
        return False
    
    # Check for minimum rows
    if len(df) < 1:
        logger.warning(f"Processed dataframe contains only {len(df)} rows")
        return False
    
    # Check for minimum columns
    if len(df.columns) < 1:
        logger.warning(f"Processed dataframe contains only {len(df.columns)} columns")
        return False
    
    # Check for summary status
    if summary.get('status') != 'success':
        logger.warning(f"Processing summary indicates failure: {summary.get('message', 'Unknown error')}")
        return False
    
    # Additional validation can be added here
    
    return True


def _track_progress(progress_key: str, done_event: threading.Event, timeout: int = 300):
    """
    Track and update progress in session state during file processing.
    
    Args:
        progress_key (str): Session state key for progress tracking
        done_event (threading.Event): Event signaling when processing is complete
        timeout (int): Maximum seconds to wait
    """
    start_time = time.time()
    progress_points = [40, 50, 60, 70, 75, 80]  # Progress percentages
    
    # Calculate intervals for progress updates
    interval = timeout / len(progress_points)
    
    for progress in progress_points:
        # Wait for either the done event or the next progress interval
        elapsed = time.time() - start_time
        remaining = min(interval, max(0.1, interval - (elapsed % interval)))
        
        if done_event.wait(remaining):
            # Processing is done
            return
        
        # Update progress if we haven't reached the target yet
        try:
            if progress_key in st.session_state and st.session_state[progress_key] < progress:
                st.session_state[progress_key] = progress
                logger.debug(f"Updated progress to {progress}%")
        except Exception as e:
            logger.error(f"Error updating progress: {e}")
        
        # Check if we've exceeded timeout
        if time.time() - start_time > timeout:
            logger.warning(f"Progress tracking timed out after {timeout} seconds")
            return


def get_upload_status() -> Dict[str, Any]:
    """
    Get the status of the most recent upload.
    
    Returns:
        dict: Upload status information
    """
    if hasattr(st.session_state, 'nova_act_last_upload'):
        return st.session_state.nova_act_last_upload
    return {
        'file': None,
        'timestamp': None,
        'status': 'none',
        'message': 'No upload has been recorded'
    }


def get_active_progress() -> Dict[str, float]:
    """
    Get all active upload progress values.
    
    Returns:
        dict: Map of active upload keys to progress percentages
    """
    progress = {}
    for key in st.session_state:
        if key.startswith('upload_progress_'):
            file_name = key.replace('upload_progress_', '')
            progress[file_name] = st.session_state[key]
    return progress


# Example Usage (if run directly)
if __name__ == "__main__":
    print("Running WatchdogUpload Example...")
    
    # Create a dummy file to upload
    dummy_file = "./dummy_upload_test.csv"
    with open(dummy_file, 'w') as f:
        # Create a simple CSV file with some sample data
        f.write("Header1,Header2,Header3\n")
        for i in range(10):
            f.write(f"Value{i},Data{i*2},{i*10}\n")
    
    print(f"Attempting to upload and process dummy file: {dummy_file}")
    
    # Note: This direct call will likely fail to update session_state
    # if not run within an active Streamlit session context.
    success = upload_to_watchdog(dummy_file)
    print(f"Example upload successful: {success}")
    
    # Clean up dummy file
    if os.path.exists(dummy_file):
        os.remove(dummy_file)
        print(f"Cleaned up dummy file: {dummy_file}")


class WatchdogUploader:
    """
    Handles file uploads, validation, and processing for the Watchdog AI system.
    Manages both UI components for file upload and the backend processing logic.
    """
    
    ALLOWED_EXTENSIONS = {'.csv', '.xlsx', '.xls', '.json', '.xml'}
    MAX_FILE_SIZE_MB = 100  # 100MB file size limit
    
    def __init__(self, upload_dir: str = "uploads"):
        """
        Initialize the WatchdogUploader.
        
        Args:
            upload_dir: Directory to store uploaded files
        """
        self.upload_dir = upload_dir
        self.logger = logging.getLogger("WatchdogUploader")
        
        # Ensure upload directory exists
        os.makedirs(upload_dir, exist_ok=True)
        
        # Initialize validation handlers
        self._validation_handlers = []
        self._post_upload_processors = []
    
    def render_upload_ui(self, 
                       key: str = "watchdog_upload",
                       upload_label: str = "Upload data file",
                       help_text: str = "Supported formats: CSV, Excel, JSON, XML",
                       show_progress: bool = True) -> Dict:
        """
        Render the file upload UI component.
        
        Args:
            key: Unique key for the Streamlit widget
            upload_label: Label to display for the upload widget
            help_text: Help text to display below the upload widget
            show_progress: Whether to show a progress bar during processing
            
        Returns:
            Dict containing upload status and file info if successful
        """
        st.markdown(f"### {upload_label}")
        st.markdown(help_text)
        
        # Display file size limit
        st.caption(f"Maximum file size: {self.MAX_FILE_SIZE_MB}MB")
        
        # Create the file uploader
        uploaded_file = st.file_uploader(
            "Choose a file", 
            type=["csv", "xlsx", "xls", "json", "xml"],
            key=key
        )
        
        result = {
            "success": False,
            "file_info": None,
            "error": None,
            "data": None
        }
        
        # Process the uploaded file if one was provided
        if uploaded_file is not None:
            if show_progress:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    result = self._process_uploaded_file(uploaded_file)
            else:
                result = self._process_uploaded_file(uploaded_file)
            
            # Display success or error message
            if result["success"]:
                st.success(f"✅ File uploaded successfully: {uploaded_file.name}")
                
                # Store in session state for reference
                st.session_state[f"{key}_result"] = result
                
                # Display file info
                with st.expander("File Details"):
                    self._render_file_info(result["file_info"])
            else:
                st.error(f"❌ Upload failed: {result['error']}")
        
        return result
    
    def _render_file_info(self, file_info: Dict):
        """Render file information in the UI"""
        if not file_info:
            return
            
        # Create a two-column layout for file details
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**File Information**")
            st.text(f"Name: {file_info['filename']}")
            st.text(f"Type: {file_info['file_type']}")
            st.text(f"Size: {file_info['size_readable']}")
            st.text(f"Uploaded: {file_info['upload_time']}")
        
        with col2:
            st.markdown("**Data Preview**")
            if "preview" in file_info and file_info["preview"] is not None:
                if isinstance(file_info["preview"], pd.DataFrame):
                    st.dataframe(file_info["preview"].head(5), use_container_width=True)
                else:
                    st.code(str(file_info["preview"])[:500] + "...")
    
    def _process_uploaded_file(self, uploaded_file) -> Dict:
        """
        Process an uploaded file, run validations, and save it.
        
        Args:
            uploaded_file: The file object from st.file_uploader
            
        Returns:
            Dict with processing status and file information
        """
        result = {
            "success": False,
            "file_info": None,
            "error": None,
            "data": None
        }
        
        try:
            # Get file information
            file_info = self._get_file_info(uploaded_file)
            result["file_info"] = file_info
            
            # Check file size
            if file_info["size_mb"] > self.MAX_FILE_SIZE_MB:
                result["error"] = f"File size ({file_info['size_mb']:.1f}MB) exceeds maximum allowed ({self.MAX_FILE_SIZE_MB}MB)"
                return result
            
            # Check file extension
            if not any(file_info["filename"].lower().endswith(ext) for ext in self.ALLOWED_EXTENSIONS):
                result["error"] = f"File type not supported: {file_info['file_ext']}. Allowed types: {', '.join(self.ALLOWED_EXTENSIONS)}"
                return result
            
            # Read the file into memory
            data = self._read_file(uploaded_file)
            
            if data is None:
                result["error"] = "Could not read file contents"
                return result
            
            # Add data to result and file_info
            result["data"] = data
            if isinstance(data, pd.DataFrame):
                file_info["row_count"] = len(data)
                file_info["column_count"] = len(data.columns)
                file_info["columns"] = data.columns.tolist()
                file_info["preview"] = data
            
            # Run validation handlers
            validation_results = self._run_validators(data, file_info)
            
            if not validation_results["success"]:
                result["error"] = validation_results["error"]
                result["validation_results"] = validation_results
                return result
            
            # Save the file
            save_path = self._save_file(uploaded_file, file_info["unique_id"])
            file_info["save_path"] = save_path
            
            # Run post-upload processors
            self._run_post_processors(data, file_info)
            
            # Set success
            result["success"] = True
            
        except Exception as e:
            self.logger.exception(f"Error processing uploaded file: {str(e)}")
            result["error"] = f"Failed to process file: {str(e)}"
        
        return result
    
    def _get_file_info(self, uploaded_file) -> Dict:
        """Extract information about the uploaded file"""
        # Generate a unique ID for this upload
        unique_id = str(uuid.uuid4())
        
        # Get file size in bytes
        uploaded_file.seek(0, os.SEEK_END)
        size_bytes = uploaded_file.tell()
        uploaded_file.seek(0)
        
        # Convert to MB for readable size
        size_mb = size_bytes / (1024 * 1024)
        
        # Get file extension
        filename = uploaded_file.name
        file_ext = os.path.splitext(filename)[1].lower()
        
        # Get MIME type if available
        file_type = uploaded_file.type if hasattr(uploaded_file, 'type') else f"Unknown ({file_ext})"
        
        # Create file info dictionary
        file_info = {
            "unique_id": unique_id,
            "filename": filename,
            "file_ext": file_ext,
            "file_type": file_type,
            "size_bytes": size_bytes,
            "size_mb": size_mb,
            "size_readable": f"{size_mb:.2f} MB" if size_mb >= 1 else f"{size_bytes / 1024:.2f} KB",
            "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return file_info
    
    def _read_file(self, uploaded_file) -> Any:
        """
        Read the contents of the uploaded file based on its type.
        
        Returns:
            DataFrame for CSV/Excel, Dict for JSON, str for XML/text, or None on error
        """
        filename = uploaded_file.name.lower()
        
        try:
            # Handle CSV files
            if filename.endswith('.csv'):
                return pd.read_csv(uploaded_file)
            
            # Handle Excel files
            elif filename.endswith(('.xlsx', '.xls')):
                return pd.read_excel(uploaded_file)
            
            # Handle JSON files
            elif filename.endswith('.json'):
                # First try to parse as pandas DataFrame
                try:
                    return pd.read_json(uploaded_file)
                except:
                    # If that fails, return as dict
                    uploaded_file.seek(0)
                    import json
                    return json.load(uploaded_file)
            
            # Handle XML files
            elif filename.endswith('.xml'):
                uploaded_file.seek(0)
                content = uploaded_file.read().decode('utf-8')
                
                # Try to parse as DataFrame if possible
                try:
                    import xml.etree.ElementTree as ET
                    root = ET.fromstring(content)
                    
                    # This is a very simple XML to DataFrame conversion
                    # For complex XML, this would need customization
                    data = []
                    for child in root:
                        row = {subchild.tag: subchild.text for subchild in child}
                        data.append(row)
                    
                    if data:
                        return pd.DataFrame(data)
                    return content
                except:
                    return content
            
            # Unsupported file type
            else:
                self.logger.warning(f"Unsupported file type: {filename}")
                return None
                
        except Exception as e:
            self.logger.exception(f"Error reading file {filename}: {str(e)}")
            return None
    
    def _save_file(self, uploaded_file, unique_id: str) -> str:
        """
        Save the uploaded file to disk with a unique filename.
        
        Args:
            uploaded_file: The file to save
            unique_id: Unique identifier for the file
            
        Returns:
            Path to the saved file
        """
        # Create a unique filename
        original_filename = uploaded_file.name
        file_ext = os.path.splitext(original_filename)[1]
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{timestamp}_{unique_id}{file_ext}"
        
        # Create the save path
        save_path = os.path.join(self.upload_dir, unique_filename)
        
        # Save the file
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        self.logger.info(f"Saved uploaded file to {save_path}")
        return save_path
    
    def add_validation_handler(self, handler: Callable[[Any, Dict], Tuple[bool, Optional[str]]]):
        """
        Add a validation handler function that will be run on uploaded files.
        
        Args:
            handler: Function that takes (data, file_info) and returns (is_valid, error_message)
        """
        self._validation_handlers.append(handler)
    
    def add_post_upload_processor(self, processor: Callable[[Any, Dict], None]):
        """
        Add a post-upload processor function that will be run after successful upload.
        
        Args:
            processor: Function that takes (data, file_info) and returns None
        """
        self._post_upload_processors.append(processor)
    
    def _run_validators(self, data: Any, file_info: Dict) -> Dict:
        """
        Run all registered validation handlers on the uploaded data.
        
        Returns:
            Dict with validation results
        """
        validation_results = {
            "success": True,
            "error": None,
            "details": []
        }
        
        for validator in self._validation_handlers:
            try:
                is_valid, error_message = validator(data, file_info)
                
                validation_results["details"].append({
                    "validator": validator.__name__ if hasattr(validator, "__name__") else "unnamed_validator",
                    "success": is_valid,
                    "message": error_message
                })
                
                if not is_valid:
                    validation_results["success"] = False
                    validation_results["error"] = error_message
                    break
                    
            except Exception as e:
                self.logger.exception(f"Error running validator {validator}: {str(e)}")
                validation_results["success"] = False
                validation_results["error"] = f"Validation error: {str(e)}"
                validation_results["details"].append({
                    "validator": validator.__name__ if hasattr(validator, "__name__") else "unnamed_validator",
                    "success": False,
                    "message": str(e),
                    "exception": True
                })
                break
        
        return validation_results
    
    def _run_post_processors(self, data: Any, file_info: Dict):
        """Run all registered post-upload processors on the uploaded data."""
        for processor in self._post_upload_processors:
            try:
                processor(data, file_info)
            except Exception as e:
                self.logger.exception(f"Error running post-processor {processor}: {str(e)}")
    
    # Common validation handlers
    @staticmethod
    def validate_csv_columns(required_columns: List[str]) -> Callable:
        """
        Create a validator function that checks if a CSV file has required columns.
        
        Args:
            required_columns: List of column names that must be present
            
        Returns:
            Validator function that can be added with add_validation_handler
        """
        def validator(data, file_info):
            if not isinstance(data, pd.DataFrame):
                return False, "Data is not in tabular format"
                
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                return False, f"Missing required columns: {', '.join(missing_columns)}"
                
            return True, None
            
        return validator
    
    @staticmethod
    def validate_row_count(min_rows: int = 1, max_rows: Optional[int] = None) -> Callable:
        """
        Create a validator function that checks if a file has an acceptable number of rows.
        
        Args:
            min_rows: Minimum number of rows required
            max_rows: Maximum number of rows allowed (None for no limit)
            
        Returns:
            Validator function that can be added with add_validation_handler
        """
        def validator(data, file_info):
            if not isinstance(data, pd.DataFrame):
                return False, "Data is not in tabular format"
                
            row_count = len(data)
            
            if row_count < min_rows:
                return False, f"File contains {row_count} rows, but at least {min_rows} required"
                
            if max_rows is not None and row_count > max_rows:
                return False, f"File contains {row_count} rows, exceeding maximum of {max_rows}"
                
            return True, None
            
        return validator
    
    @staticmethod
    def validate_no_empty_values(columns: Optional[List[str]] = None) -> Callable:
        """
        Create a validator function that checks for empty values in specified columns.
        
        Args:
            columns: List of columns to check (None for all columns)
            
        Returns:
            Validator function that can be added with add_validation_handler
        """
        def validator(data, file_info):
            if not isinstance(data, pd.DataFrame):
                return False, "Data is not in tabular format"
                
            cols_to_check = columns if columns is not None else data.columns
            
            for col in cols_to_check:
                if col not in data.columns:
                    continue
                    
                empty_count = data[col].isna().sum()
                if empty_count > 0:
                    return False, f"Column '{col}' contains {empty_count} empty values"
            
            return True, None
            
        return validator 