"""
PDF Extraction Utility for Watchdog AI.

This module provides functionality for extracting data from PDF files,
including both tables and text content.
"""

import os
import re
import io
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union, Set
from dataclasses import dataclass, field
from datetime import datetime
import tempfile

logger = logging.getLogger(__name__)

# Try to import PDF libraries, but make them optional
try:
    import tabula
    import PyPDF2
    from pdfminer.high_level import extract_text as pdfminer_extract_text
    from pdfminer.layout import LAParams
    PDF_EXTRACTION_AVAILABLE = True
except ImportError:
    logger.warning("PDF extraction libraries not available. Install tabula-py, PyPDF2, and pdfminer.six for PDF support.")
    PDF_EXTRACTION_AVAILABLE = False

# Define constants
DEFAULT_TABLE_EXTRACTION_METHODS = ["lattice", "stream"]
DEFAULT_TEXT_EXTRACTION_METHOD = "pdfminer"  # "pdfminer" or "pypdf2"
DEFAULT_MAX_PAGES = 50  # Limit for safety

@dataclass
class PDFExtractionResult:
    """Results from PDF extraction."""
    tables: List[pd.DataFrame] = field(default_factory=list)
    text: Dict[int, str] = field(default_factory=dict)  # Page number -> text content
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    combined_df: Optional[pd.DataFrame] = None  # If tables are combined
    selected_df: Optional[pd.DataFrame] = None  # Best table based on heuristics

class PDFExtractor:
    """PDF data extraction with multiple strategies."""
    
    def __init__(self):
        """Initialize the PDF extractor."""
        self.check_dependencies()
    
    def check_dependencies(self) -> bool:
        """Check if required dependencies are available."""
        return PDF_EXTRACTION_AVAILABLE
    
    def extract_tables(self, 
                      file_path: str, 
                      pages: Union[str, List[int]] = "all",
                      extraction_methods: List[str] = DEFAULT_TABLE_EXTRACTION_METHODS,
                      area: Optional[List[float]] = None,
                      guess: bool = True,
                      max_pages: int = DEFAULT_MAX_PAGES) -> List[pd.DataFrame]:
        """
        Extract tables from a PDF file using tabula-py.
        
        Args:
            file_path: Path to the PDF file
            pages: Pages to extract from, "all" or list of page numbers (1-based)
            extraction_methods: List of methods to try ("lattice", "stream")
            area: Rectangular area to analyze [top, left, bottom, right]
            guess: Whether to guess table structure
            max_pages: Maximum number of pages to process
            
        Returns:
            List of pandas DataFrames containing extracted tables
        """
        if not PDF_EXTRACTION_AVAILABLE:
            logger.error("PDF extraction libraries not available")
            return []
        
        # Validate file
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return []
        
        # Get total page count and validate pages parameter
        num_pages = self._get_page_count(file_path)
        if num_pages == 0:
            logger.error(f"Could not determine page count or file has no pages: {file_path}")
            return []
        
        # Limit pages to process
        if pages == "all":
            pages = list(range(1, min(num_pages + 1, max_pages + 1)))
        elif isinstance(pages, list):
            pages = [p for p in pages if 1 <= p <= num_pages]
            if len(pages) > max_pages:
                pages = pages[:max_pages]
                logger.warning(f"Limiting to {max_pages} pages for processing")
        
        # Try each extraction method
        all_tables = []
        for method in extraction_methods:
            try:
                logger.info(f"Extracting tables using method: {method}")
                tables = tabula.read_pdf(
                    file_path,
                    pages=pages,
                    multiple_tables=True,
                    guess=guess,
                    lattice=(method == "lattice"),
                    stream=(method == "stream"),
                    area=area,
                    silent=True
                )
                
                if tables:
                    logger.info(f"Extracted {len(tables)} tables using {method} method")
                    all_tables.extend(tables)
            except Exception as e:
                logger.error(f"Error extracting tables with {method} method: {str(e)}")
        
        # Filter out empty tables
        all_tables = [df for df in all_tables if not df.empty]
        
        return all_tables
    
    def extract_text(self, 
                    file_path: str, 
                    pages: Union[str, List[int]] = "all",
                    extraction_method: str = DEFAULT_TEXT_EXTRACTION_METHOD,
                    max_pages: int = DEFAULT_MAX_PAGES) -> Dict[int, str]:
        """
        Extract text from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            pages: Pages to extract from, "all" or list of page numbers (1-based)
            extraction_method: Method to use ("pdfminer" or "pypdf2")
            max_pages: Maximum number of pages to process
            
        Returns:
            Dictionary mapping page numbers to extracted text
        """
        if not PDF_EXTRACTION_AVAILABLE:
            logger.error("PDF extraction libraries not available")
            return {}
        
        # Validate file
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return {}
        
        # Get total page count and validate pages parameter
        num_pages = self._get_page_count(file_path)
        if num_pages == 0:
            logger.error(f"Could not determine page count or file has no pages: {file_path}")
            return {}
        
        # Limit pages to process
        if pages == "all":
            page_indices = list(range(min(num_pages, max_pages)))
        elif isinstance(pages, list):
            page_indices = [p-1 for p in pages if 1 <= p <= num_pages]
            if len(page_indices) > max_pages:
                page_indices = page_indices[:max_pages]
                logger.warning(f"Limiting to {max_pages} pages for processing")
        
        result = {}
        
        if extraction_method == "pdfminer":
            # Extract using pdfminer
            try:
                # Extract from whole document
                if pages == "all" or len(page_indices) == num_pages:
                    text = pdfminer_extract_text(
                        file_path, 
                        laparams=LAParams()
                    )
                    
                    # Split text by page markers (if any)
                    page_texts = self._split_text_into_pages(text)
                    
                    # If we couldn't split by pages, just treat as one page
                    if not page_texts:
                        result = {1: text}
                    else:
                        result = {i+1: page_texts[i] for i in range(min(len(page_texts), max_pages))}
                else:
                    # Extract page by page
                    for page_idx in page_indices:
                        try:
                            text = pdfminer_extract_text(
                                file_path, 
                                page_numbers=[page_idx],
                                laparams=LAParams()
                            )
                            result[page_idx + 1] = text
                        except Exception as e:
                            logger.error(f"Error extracting text from page {page_idx + 1}: {str(e)}")
            except Exception as e:
                logger.error(f"Error extracting text with pdfminer: {str(e)}")
        
        elif extraction_method == "pypdf2":
            # Extract using PyPDF2
            try:
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    for i in page_indices:
                        try:
                            if i < len(reader.pages):
                                text = reader.pages[i].extract_text()
                                result[i + 1] = text
                        except Exception as e:
                            logger.error(f"Error extracting text from page {i + 1}: {str(e)}")
            except Exception as e:
                logger.error(f"Error extracting text with PyPDF2: {str(e)}")
        
        return result
    
    def get_pdf_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dictionary of metadata
        """
        if not PDF_EXTRACTION_AVAILABLE:
            logger.error("PDF extraction libraries not available")
            return {}
        
        # Validate file
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return {}
        
        metadata = {
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "file_size": os.path.getsize(file_path),
            "extraction_time": datetime.now().isoformat()
        }
        
        try:
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                metadata["page_count"] = len(reader.pages)
                
                if reader.metadata:
                    info = reader.metadata
                    for key in info:
                        try:
                            value = info[key]
                            if value:
                                metadata[key.lower()] = str(value)
                        except:
                            pass
        except Exception as e:
            logger.error(f"Error extracting PDF metadata: {str(e)}")
            metadata["error"] = str(e)
        
        return metadata
    
    def extract_all(self, file_path: str, extract_tables: bool = True, 
                   extract_text: bool = True, pages: Union[str, List[int]] = "all",
                   combine_tables: bool = True,
                   select_best_table: bool = True) -> PDFExtractionResult:
        """
        Extract all data from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            extract_tables: Whether to extract tables
            extract_text: Whether to extract text
            pages: Pages to extract from, "all" or list of page numbers (1-based)
            combine_tables: Whether to combine extracted tables
            select_best_table: Whether to select the best table based on heuristics
            
        Returns:
            PDFExtractionResult object with extracted data
        """
        result = PDFExtractionResult()
        
        # Get metadata first
        result.metadata = self.get_pdf_metadata(file_path)
        
        # Extract tables if requested
        if extract_tables:
            try:
                result.tables = self.extract_tables(file_path, pages=pages)
                result.metadata["table_count"] = len(result.tables)
            except Exception as e:
                error_msg = f"Error extracting tables: {str(e)}"
                logger.error(error_msg)
                result.errors.append(error_msg)
        
        # Extract text if requested
        if extract_text:
            try:
                result.text = self.extract_text(file_path, pages=pages)
                result.metadata["text_page_count"] = len(result.text)
            except Exception as e:
                error_msg = f"Error extracting text: {str(e)}"
                logger.error(error_msg)
                result.errors.append(error_msg)
        
        # Combine tables if requested and tables are available
        if combine_tables and result.tables:
            try:
                result.combined_df = self._combine_tables(result.tables)
                result.metadata["combined_shape"] = result.combined_df.shape
            except Exception as e:
                error_msg = f"Error combining tables: {str(e)}"
                logger.error(error_msg)
                result.errors.append(error_msg)
        
        # Select best table if requested and tables are available
        if select_best_table and result.tables:
            try:
                result.selected_df = self._select_best_table(result.tables)
                result.metadata["selected_shape"] = result.selected_df.shape
            except Exception as e:
                error_msg = f"Error selecting best table: {str(e)}"
                logger.error(error_msg)
                result.errors.append(error_msg)
        
        return result
    
    def convert_text_to_dataframe(self, text: str, delimiter_pattern: str = r'\s{2,}') -> pd.DataFrame:
        """
        Convert text to DataFrame by detecting delimited columns.
        
        Args:
            text: Text to convert
            delimiter_pattern: Regex pattern for delimiter
            
        Returns:
            DataFrame created from text
        """
        if not text or not text.strip():
            return pd.DataFrame()
        
        lines = text.strip().split('\n')
        if not lines:
            return pd.DataFrame()
        
        # Detect column names from the first line
        header = re.split(delimiter_pattern, lines[0].strip())
        
        # Process data rows
        data = []
        for line in lines[1:]:
            if line.strip():
                row = re.split(delimiter_pattern, line.strip())
                # Pad or truncate to match header length
                if len(row) < len(header):
                    row.extend([''] * (len(header) - len(row)))
                elif len(row) > len(header):
                    row = row[:len(header)]
                data.append(row)
        
        # Create DataFrame
        if data:
            df = pd.DataFrame(data, columns=header)
            return df
        
        return pd.DataFrame()
    
    def extract_tables_from_text(self, text_dict: Dict[int, str]) -> List[pd.DataFrame]:
        """
        Attempt to extract tables from text by identifying delimited patterns.
        
        Args:
            text_dict: Dictionary of page numbers to text
            
        Returns:
            List of DataFrames extracted from text
        """
        tables = []
        
        for page_num, text in text_dict.items():
            if not text or not text.strip():
                continue
            
            # Split text into potential tables based on line breaks
            text_blocks = self._split_text_into_blocks(text)
            
            for block in text_blocks:
                if len(block.split('\n')) >= 3:  # At least header + 2 rows
                    df = self.convert_text_to_dataframe(block)
                    if not df.empty and len(df.columns) >= 2:  # At least 2 columns
                        df['_pdf_page'] = page_num
                        df['_text_block'] = True
                        tables.append(df)
        
        return tables
    
    def pdf_to_dataframe(self, file_path: str, prefer_combined: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Convert a PDF file to a DataFrame, using the best available extraction method.
        
        Args:
            file_path: Path to the PDF file
            prefer_combined: Whether to prefer the combined table over the best table
            
        Returns:
            Tuple of (DataFrame, metadata)
        """
        # Extract all data
        result = self.extract_all(file_path)
        
        # No tables found from tabula, try extracting from text
        if not result.tables and result.text:
            text_tables = self.extract_tables_from_text(result.text)
            result.tables.extend(text_tables)
            result.metadata["text_table_count"] = len(text_tables)
            
            if text_tables:
                result.combined_df = self._combine_tables(text_tables)
                result.selected_df = self._select_best_table(text_tables)
                result.metadata["combined_shape"] = result.combined_df.shape
                result.metadata["selected_shape"] = result.selected_df.shape
        
        # Determine best DataFrame to return
        if prefer_combined and result.combined_df is not None and not result.combined_df.empty:
            df = result.combined_df
            result.metadata["source"] = "combined_tables"
        elif result.selected_df is not None and not result.selected_df.empty:
            df = result.selected_df
            result.metadata["source"] = "best_table"
        elif result.tables:
            df = result.tables[0]  # Return first table
            result.metadata["source"] = "first_table"
        else:
            # No tables found, create a DataFrame with text content
            data = []
            columns = ["page", "text"]
            for page_num, text in result.text.items():
                data.append([page_num, text])
            df = pd.DataFrame(data, columns=columns)
            result.metadata["source"] = "text_content"
        
        # Add extraction info to metadata
        result.metadata["extraction_method"] = "tabula+pdfminer"
        result.metadata["preferred_method"] = "prefer_combined" if prefer_combined else "prefer_best"
        
        return df, result.metadata
    
    def _get_page_count(self, file_path: str) -> int:
        """Get the number of pages in a PDF file."""
        try:
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                return len(reader.pages)
        except Exception as e:
            logger.error(f"Error getting page count: {str(e)}")
            return 0
    
    def _split_text_into_pages(self, text: str) -> List[str]:
        """
        Split text into pages based on common page markers.
        
        This is a heuristic method - pdfminer sometimes includes page break indicators.
        """
        # Try common page break markers
        patterns = [
            r'\f',  # Form feed character
            r'-----+ ?Page \d+ ?-----+',  # "---- Page N ----" pattern
            r'\n\s*Page \d+\s*\n'  # "Page N" pattern
        ]
        
        for pattern in patterns:
            parts = re.split(pattern, text)
            if len(parts) > 1:
                return parts
        
        # No page breaks found
        return []
    
    def _split_text_into_blocks(self, text: str) -> List[str]:
        """Split text into potential table blocks based on blank lines."""
        blocks = []
        current_block = []
        
        lines = text.split('\n')
        for line in lines:
            if not line.strip():
                if current_block:
                    blocks.append('\n'.join(current_block))
                    current_block = []
            else:
                current_block.append(line)
        
        if current_block:
            blocks.append('\n'.join(current_block))
        
        return blocks
    
    def _combine_tables(self, tables: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Combine multiple tables into a single DataFrame.
        
        This handles cases where tables might have different columns.
        """
        if not tables:
            return pd.DataFrame()
        
        if len(tables) == 1:
            return tables[0].copy()
        
        # Check if tables have similar columns (at least 50% overlap)
        all_columns = set()
        for df in tables:
            all_columns.update(df.columns)
        
        compatible_tables = []
        for df in tables:
            overlap = len(set(df.columns).intersection(all_columns)) / len(all_columns)
            if overlap >= 0.5:
                compatible_tables.append(df)
        
        if compatible_tables:
            # Fill missing columns with NaN
            for i, df in enumerate(compatible_tables):
                if set(df.columns) != all_columns:
                    for col in all_columns:
                        if col not in df.columns:
                            compatible_tables[i][col] = np.nan
            
            # Concatenate tables
            return pd.concat(compatible_tables, ignore_index=True)
        
        # If no compatible tables, return the largest table
        largest_table = max(tables, key=lambda df: df.size)
        return largest_table.copy()
    
    def _select_best_table(self, tables: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Select the best table from a list based on heuristics.
        
        Currently uses size as the main criteria, but could be extended with more
        sophisticated heuristics like column count, data types, etc.
        """
        if not tables:
            return pd.DataFrame()
        
        if len(tables) == 1:
            return tables[0].copy()
        
        # Score each table
        scored_tables = []
        for i, df in enumerate(tables):
            # Calculate a quality score based on various factors
            
            # 1. Size factor (larger is better but not necessarily)
            size_score = min(df.size / 100, 10)  # Cap at 10
            
            # 2. Column count (more columns often means more structured data)
            col_score = min(len(df.columns), 10)
            
            # 3. Non-null ratio (higher is better)
            non_null_ratio = df.count().sum() / (df.size or 1)
            null_score = non_null_ratio * 10
            
            # 4. Numeric column ratio (typically good for analytical tables)
            numeric_cols = df.select_dtypes(include=['number']).columns
            numeric_ratio = len(numeric_cols) / (len(df.columns) or 1)
            numeric_score = numeric_ratio * 5
            
            # 5. Penalty for unnamed columns
            unnamed_cols = sum(1 for c in df.columns if 'unnamed' in str(c).lower())
            unnamed_penalty = -unnamed_cols
            
            # 6. Row count reasonable (not too few, not excessive)
            row_score = min(df.shape[0] / 10, 5)
            
            # Calculate total score
            total_score = size_score + col_score + null_score + numeric_score + unnamed_penalty + row_score
            
            scored_tables.append((i, total_score, df))
        
        # Return the table with the highest score
        best_table_idx = max(scored_tables, key=lambda x: x[1])[0]
        return tables[best_table_idx].copy()


def extract_from_pdf(file_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Convenience function to extract data from a PDF file.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Tuple of (DataFrame, metadata)
    """
    extractor = PDFExtractor()
    return extractor.pdf_to_dataframe(file_path)