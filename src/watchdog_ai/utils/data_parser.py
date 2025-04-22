"""
Data Parser Module for Watchdog AI.

This module provides functionality for parsing different file formats
(CSV, PDF) and converting them to pandas DataFrames.
"""

import os
import io
import csv
import chardet
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union, Set, BinaryIO
from datetime import datetime
import tempfile

# Import PDF extractor if available
try:
    from .pdf_extractor import PDFExtractor, extract_from_pdf
    PDF_EXTRACTION_AVAILABLE = True
except ImportError:
    PDF_EXTRACTION_AVAILABLE = False

logger = logging.getLogger(__name__)

class DataParserResult:
    """Results from parsing a data file."""
    
    def __init__(self, 
                dataframe: pd.DataFrame = None, 
                file_type: str = None,
                file_path: str = None):
        """
        Initialize the parser result.
        
        Args:
            dataframe: Parsed DataFrame
            file_type: Detected file type
            file_path: Path to the source file
        """
        self.dataframe = dataframe if dataframe is not None else pd.DataFrame()
        self.file_type = file_type
        self.file_path = file_path
        self.metadata = {
            "file_path": file_path,
            "file_type": file_type,
            "parsed_at": datetime.now().isoformat()
        }
        self.errors = []
        self.warnings = []
    
    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.errors.append(error)
        logger.error(f"Parser error: {error}")
    
    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)
        logger.warning(f"Parser warning: {warning}")
    
    def update_metadata(self, new_metadata: Dict[str, Any]) -> None:
        """Update metadata with new information."""
        self.metadata.update(new_metadata)
    
    def is_successful(self) -> bool:
        """Check if parsing was successful."""
        return (not self.errors and 
                self.dataframe is not None and 
                not self.dataframe.empty)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the parsing result."""
        return {
            "success": self.is_successful(),
            "file_type": self.file_type,
            "file_path": self.file_path,
            "errors": self.errors,
            "warnings": self.warnings,
            "row_count": len(self.dataframe) if self.dataframe is not None else 0,
            "column_count": len(self.dataframe.columns) if self.dataframe is not None else 0,
            "metadata": self.metadata
        }


class FileValidator:
    """Validates uploaded files for size, type, and basic structure."""
    
    def __init__(self):
        """Initialize the file validator."""
        self.max_file_size_mb = 100
        self.supported_extensions = ['.csv', '.pdf']
        self.supported_encodings = ['utf-8', 'latin1', 'iso-8859-1', 'windows-1252']
    
    def validate_file_size(self, file: BinaryIO) -> Tuple[bool, str]:
        """
        Check if file size is within acceptable limits.
        
        Args:
            file: File-like object
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            file.seek(0, 2)  # Seek to end
            size_mb = file.tell() / (1024 * 1024)
            file.seek(0)
            
            if size_mb > self.max_file_size_mb:
                return False, f"File size ({size_mb:.1f}MB) exceeds maximum allowed size ({self.max_file_size_mb}MB)"
            return True, ""
        except Exception as e:
            return False, f"Error checking file size: {str(e)}"
    
    def detect_file_type(self, file_path: str) -> str:
        """
        Detect file type from extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Detected file type
        """
        extension = os.path.splitext(file_path.lower())[1]
        
        if extension == '.csv':
            return 'csv'
        elif extension == '.pdf':
            return 'pdf'
        else:
            return 'unknown'
    
    def validate_file_type(self, file_path: str) -> Tuple[bool, str]:
        """
        Validate if file type is supported.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        file_type = self.detect_file_type(file_path)
        
        if file_type == 'unknown':
            return False, f"Unsupported file type. Supported types: {', '.join(self.supported_extensions)}"
        
        return True, ""
    
    def validate_csv_structure(self, content: bytes) -> Tuple[bool, str, str]:
        """
        Validate basic CSV structure and detect encoding.
        
        Args:
            content: CSV file content
            
        Returns:
            Tuple of (is_valid, error_message, detected_encoding)
        """
        # Detect encoding
        detection = chardet.detect(content)
        encoding = detection['encoding']
        confidence = detection['confidence']
        
        if encoding is None:
            return False, "Could not detect file encoding", None
        
        if encoding.lower() not in [enc.lower() for enc in self.supported_encodings]:
            # Try supported encodings in order of preference
            for fallback_encoding in self.supported_encodings:
                try:
                    decoded = content.decode(fallback_encoding)
                    reader = csv.reader(io.StringIO(decoded))
                    headers = next(reader)
                    if headers:
                        return True, "", fallback_encoding
                except:
                    continue
            
            return False, f"Unsupported file encoding: {encoding}", None
        
        # Try to decode and read headers
        try:
            decoded = content.decode(encoding)
            reader = csv.reader(io.StringIO(decoded))
            headers = next(reader)
            
            if not headers:
                return False, "File appears to be empty or has no headers", encoding
            
            return True, "", encoding
        except Exception as e:
            return False, f"Error validating CSV structure: {str(e)}", encoding
    
    def validate_pdf_structure(self, file_path: str) -> Tuple[bool, str]:
        """
        Validate basic PDF structure.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not PDF_EXTRACTION_AVAILABLE:
            return False, "PDF extraction libraries not available. Install tabula-py, PyPDF2, and pdfminer.six for PDF support."
        
        try:
            # Check if file is a valid PDF
            with open(file_path, 'rb') as f:
                header = f.read(5)
                if header != b'%PDF-':
                    return False, "File does not appear to be a valid PDF"
            
            # Try to get page count as basic validation
            extractor = PDFExtractor()
            page_count = extractor._get_page_count(file_path)
            
            if page_count == 0:
                return False, "PDF file appears to be empty or invalid"
            
            return True, ""
        except Exception as e:
            return False, f"Error validating PDF structure: {str(e)}"


class DataParser:
    """Parser for CSV and PDF files."""
    
    def __init__(self):
        """Initialize the data parser."""
        self.validator = FileValidator()
        
    def parse_file(self, file_path: str, **kwargs) -> DataParserResult:
        """
        Parse a file into a DataFrame.
        
        Args:
            file_path: Path to the file
            **kwargs: Additional arguments for specific file types
            
        Returns:
            DataParserResult object
        """
        # Check if file exists
        if not os.path.exists(file_path):
            result = DataParserResult(file_path=file_path)
            result.add_error(f"File not found: {file_path}")
            return result
        
        # Validate file type
        is_valid, error_msg = self.validator.validate_file_type(file_path)
        if not is_valid:
            result = DataParserResult(file_path=file_path)
            result.add_error(error_msg)
            return result
        
        # Detect file type
        file_type = self.validator.detect_file_type(file_path)
        
        # Parse based on file type
        if file_type == 'csv':
            return self.parse_csv(file_path, **kwargs)
        elif file_type == 'pdf':
            return self.parse_pdf(file_path, **kwargs)
        else:
            result = DataParserResult(file_path=file_path, file_type=file_type)
            result.add_error("Unsupported file type")
            return result
    
    def parse_csv(self, file_path: str, encoding: str = None, 
                 skiprows: int = None, header: int = 0, 
                 delimiter: str = None, **kwargs) -> DataParserResult:
        """
        Parse a CSV file into a DataFrame.
        
        Args:
            file_path: Path to the CSV file
            encoding: File encoding (auto-detected if None)
            skiprows: Number of rows to skip
            header: Row index to use as header
            delimiter: Column delimiter
            **kwargs: Additional arguments for pd.read_csv
            
        Returns:
            DataParserResult object
        """
        result = DataParserResult(file_path=file_path, file_type='csv')
        
        try:
            # Read file content for validation
            with open(file_path, 'rb') as f:
                content = f.read()
            
            # Validate CSV structure and detect encoding
            if encoding is None:
                is_valid, error_msg, detected_encoding = self.validator.validate_csv_structure(content)
                if not is_valid:
                    result.add_error(error_msg)
                    return result
                encoding = detected_encoding
            
            # Read the CSV file
            df = pd.read_csv(
                file_path,
                encoding=encoding,
                skiprows=skiprows,
                header=header,
                delimiter=delimiter,
                **kwargs
            )
            
            # Basic data cleanup
            for col in df.columns:
                # Remove unnamed columns
                if 'unnamed' in str(col).lower():
                    new_col = f"column_{df.columns.get_loc(col)}"
                    df.rename(columns={col: new_col}, inplace=True)
                    result.add_warning(f"Renamed unnamed column to {new_col}")
            
            # Update result
            result.dataframe = df
            result.update_metadata({
                "encoding": encoding,
                "delimiter": delimiter or ',',
                "skiprows": skiprows,
                "header": header,
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "dtypes": {col: str(df[col].dtype) for col in df.columns}
            })
            
            return result
            
        except Exception as e:
            result.add_error(f"Error parsing CSV file: {str(e)}")
            return result
    
    def parse_pdf(self, file_path: str, extract_tables: bool = True,
                extract_text: bool = True, **kwargs) -> DataParserResult:
        """
        Parse a PDF file into a DataFrame.
        
        Args:
            file_path: Path to the PDF file
            extract_tables: Whether to extract tables
            extract_text: Whether to extract text
            **kwargs: Additional arguments for pdf extraction
            
        Returns:
            DataParserResult object
        """
        result = DataParserResult(file_path=file_path, file_type='pdf')
        
        # Check if PDF extraction is available
        if not PDF_EXTRACTION_AVAILABLE:
            result.add_error("PDF extraction libraries not available")
            return result
        
        # Validate PDF structure
        is_valid, error_msg = self.validator.validate_pdf_structure(file_path)
        if not is_valid:
            result.add_error(error_msg)
            return result
        
        try:
            # Extract from PDF
            df, metadata = extract_from_pdf(file_path)
            
            # Update result
            result.dataframe = df
            result.update_metadata(metadata)
            
            # Add warning if DataFrame is empty or small
            if df.empty:
                result.add_warning("No tables found in PDF")
            elif len(df) < 5:
                result.add_warning(f"Only {len(df)} rows extracted from PDF")
            
            return result
            
        except Exception as e:
            result.add_error(f"Error parsing PDF file: {str(e)}")
            return result


def parse_file(file_path: str, **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Convenience function to parse a file into a DataFrame.
    
    Args:
        file_path: Path to the file
        **kwargs: Additional arguments for specific file types
        
    Returns:
        Tuple of (DataFrame, metadata)
    """
    parser = DataParser()
    result = parser.parse_file(file_path, **kwargs)
    
    if result.is_successful():
        return result.dataframe, result.metadata
    else:
        error_msg = result.errors[0] if result.errors else "Unknown error"
        raise ValueError(f"Failed to parse file: {error_msg}")


def detect_file_type(file_path: str) -> str:
    """
    Detect file type from extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Detected file type ('csv', 'pdf', or 'unknown')
    """
    validator = FileValidator()
    return validator.detect_file_type(file_path)