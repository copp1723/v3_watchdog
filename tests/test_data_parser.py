"""
Tests for the data_parser module.

This module tests the parsing functionality that handles different file formats
(CSV, Excel, PDF) and converts them to pandas DataFrames.
"""

import pytest
import pandas as pd
import numpy as np
import os
import tempfile
from typing import Dict, List, Tuple, Any

# Import the modules to test
try:
    from v2_watchdog_ai.excel_utils import (
        get_excel_sheet_names,
        get_excel_metadata,
        read_excel_sheet,
        auto_detect_header,
        guess_data_types,
        read_all_excel_sheets,
        excel_to_dataframes
    )
except ImportError:
    # Define empty functions with the same signatures for testing
    def get_excel_sheet_names(excel_path): return []
    def get_excel_metadata(excel_path): return {}
    def read_excel_sheet(excel_path, sheet_name=0, header=0, skiprows=None, usecols=None, nrows=None): return None, {}
    def auto_detect_header(df): return 0, df
    def guess_data_types(df): return df
    def read_all_excel_sheets(excel_path, auto_detect_headers=True, guess_types=True): return {}, {}
    def excel_to_dataframes(excel_path, sheet_selection="all", auto_detect_headers=True, guess_types=True): return [], {}

try:
    from v2_watchdog_ai.pdf_utils import (
        extract_tables_from_pdf,
        extract_text_from_pdf,
        get_pdf_info,
        pdf_to_dataframe
    )
except ImportError:
    # Define empty functions with the same signatures for testing
    def extract_tables_from_pdf(pdf_path, pages="all", guess=True, lattice=True, stream=True, area=None, silent=False): return [], {}
    def extract_text_from_pdf(pdf_path, pages="all"): return {}, {}
    def get_pdf_info(pdf_path): return {}
    def pdf_to_dataframe(pdf_path, extract_tables=True, extract_text=True, pages="all", guess=True): return [], {}

try:
    from v2_watchdog_ai.data_parser import (
        parse_file,
        parse_csv,
        parse_excel,
        parse_pdf,
        detect_file_type,
        create_dummy_pdf,
        create_dummy_excel
    )
except ImportError:
    # Define empty functions with the same signatures for testing
    def parse_file(file_path, **kwargs): return [], {}
    def parse_csv(csv_path, **kwargs): return [], {}
    def parse_excel(excel_path, **kwargs): return [], {}
    def parse_pdf(pdf_path, **kwargs): return [], {}
    def detect_file_type(file_path): return "unknown"
    def create_dummy_pdf(output_path, num_tables=2): pass
    def create_dummy_excel(output_path, num_sheets=3): pass


class TestFileTypeDetection:
    """Tests for file type detection functionality."""
    
    def test_detect_csv_file(self, sample_clean_csv_path):
        """Test detection of CSV files."""
        # Act
        file_type = detect_file_type(sample_clean_csv_path)
        
        # Assert
        assert file_type == "csv"
    
    def test_detect_excel_file(self, sample_excel_path):
        """Test detection of Excel files."""
        # Act
        file_type = detect_file_type(sample_excel_path)
        
        # Assert
        assert file_type == "excel"
    
    def test_detect_unknown_file(self, assets_dir):
        """Test detection of unknown file types."""
        # Arrange
        unknown_file = os.path.join(assets_dir, "unknown_file.txt")
        with open(unknown_file, 'w') as f:
            f.write("This is not a supported format")
        
        # Act
        file_type = detect_file_type(unknown_file)
        
        # Assert
        assert file_type == "unknown"


class TestCSVParsing:
    """Tests for CSV parsing functionality."""
    
    def test_parse_clean_csv(self, sample_clean_csv_path):
        """Test parsing of a clean CSV file."""
        # Act
        dfs, metadata = parse_csv(sample_clean_csv_path)
        
        # Assert
        assert len(dfs) == 1
        assert isinstance(dfs[0], pd.DataFrame)
        assert not dfs[0].empty
        assert metadata.get('file_path') == sample_clean_csv_path
    
    def test_parse_messy_csv(self, sample_messy_csv_path):
        """Test parsing of a messy CSV file."""
        # Act
        dfs, metadata = parse_csv(sample_messy_csv_path)
        
        # Assert
        assert len(dfs) == 1
        assert isinstance(dfs[0], pd.DataFrame)
        assert not dfs[0].empty
        assert metadata.get('file_path') == sample_messy_csv_path
    
    def test_csv_with_custom_options(self, sample_clean_csv_path):
        """Test parsing CSV with custom options."""
        # Act
        dfs, metadata = parse_csv(
            sample_clean_csv_path,
            skiprows=1,
            header=None,
            names=['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9', 'col10']
        )
        
        # Assert
        assert len(dfs) == 1
        assert list(dfs[0].columns) == ['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9', 'col10']
        assert metadata.get('custom_options', {}).get('skiprows') == 1


@pytest.mark.skipif(not os.path.exists(os.path.join(os.path.dirname(__file__), 'assets/sample_excel.xlsx')), 
                   reason="Excel file not available")
class TestExcelParsing:
    """Tests for Excel parsing functionality."""
    
    def test_parse_excel_all_sheets(self, sample_excel_path):
        """Test parsing all sheets from an Excel file."""
        # Act
        dfs, metadata = parse_excel(sample_excel_path)
        
        # Assert
        assert len(dfs) > 0
        assert all(isinstance(df, pd.DataFrame) for df in dfs)
        assert metadata.get('file_path') == sample_excel_path
        assert 'sheet_count' in metadata
    
    def test_parse_excel_specific_sheet(self, sample_excel_path):
        """Test parsing a specific sheet from an Excel file."""
        # Act
        dfs, metadata = parse_excel(
            sample_excel_path,
            sheet_selection='Clean Data'
        )
        
        # Assert
        assert len(dfs) == 1
        assert isinstance(dfs[0], pd.DataFrame)
        assert not dfs[0].empty
        assert metadata.get('selected_sheets') == ['Clean Data']
    
    def test_excel_with_header_detection(self, sample_excel_path):
        """Test Excel parsing with header detection."""
        # Act
        dfs, metadata = parse_excel(
            sample_excel_path,
            sheet_selection='Header Issues',
            auto_detect_headers=True
        )
        
        # Assert
        assert len(dfs) == 1
        assert isinstance(dfs[0], pd.DataFrame)
        assert not dfs[0].empty
        assert 'auto_detect_headers' in metadata
        assert metadata.get('auto_detect_headers') is True


@pytest.mark.skipif(not os.path.exists(os.path.join(os.path.dirname(__file__), 'assets/test.pdf')), 
                   reason="PDF file not available")
class TestPDFParsing:
    """Tests for PDF parsing functionality."""
    
    @pytest.fixture
    def sample_pdf_path(self, assets_dir):
        """Fixture that provides a path to a sample PDF file."""
        pdf_path = os.path.join(assets_dir, "test.pdf")
        
        # Create a dummy PDF file if it doesn't exist
        if not os.path.exists(pdf_path):
            # Try to create a dummy PDF
            try:
                create_dummy_pdf(pdf_path)
            except:
                # If creation fails, create an empty file to skip tests
                with open(pdf_path, 'w') as f:
                    f.write("dummy pdf content")
        
        return pdf_path
    
    def test_parse_pdf_tables(self, sample_pdf_path):
        """Test parsing tables from a PDF file."""
        # Skip if package not available
        pytest.importorskip("tabula")
        
        # Act
        dfs, metadata = parse_pdf(
            sample_pdf_path,
            extract_text=False,
            extract_tables=True
        )
        
        # Assert
        assert isinstance(dfs, list)
        assert isinstance(metadata, dict)
        assert 'extraction_types' in metadata
        
        # Tables might not be found in dummy PDF
        if len(dfs) > 0:
            assert all(isinstance(df, pd.DataFrame) for df in dfs)
    
    def test_parse_pdf_text(self, sample_pdf_path):
        """Test parsing text from a PDF file."""
        # Skip if package not available
        pytest.importorskip("PyPDF2")
        
        # Act
        dfs, metadata = parse_pdf(
            sample_pdf_path,
            extract_text=True,
            extract_tables=False
        )
        
        # Assert
        assert isinstance(dfs, list)
        assert isinstance(metadata, dict)
        assert 'extraction_types' in metadata
        
        # Text might be extracted as DataFrames if structured
        # but we can't guarantee results with a dummy PDF


class TestMainParsingFunction:
    """Tests for the main parse_file function."""
    
    def test_parse_csv_through_main_function(self, sample_clean_csv_path):
        """Test parsing a CSV file through the main parse_file function."""
        # Act
        dfs, metadata = parse_file(sample_clean_csv_path)
        
        # Assert
        assert len(dfs) == 1
        assert isinstance(dfs[0], pd.DataFrame)
        assert not dfs[0].empty
        assert metadata.get('file_type') == 'csv'
    
    def test_parse_excel_through_main_function(self, sample_excel_path):
        """Test parsing an Excel file through the main parse_file function."""
        # Act
        dfs, metadata = parse_file(sample_excel_path)
        
        # Assert
        assert len(dfs) > 0
        assert all(isinstance(df, pd.DataFrame) for df in dfs)
        assert metadata.get('file_type') == 'excel'
    
    def test_unsupported_file_type(self, assets_dir):
        """Test behavior with unsupported file types."""
        # Arrange
        unsupported_file = os.path.join(assets_dir, "unsupported.txt")
        with open(unsupported_file, 'w') as f:
            f.write("This is not a supported format")
        
        # Act/Assert
        with pytest.raises(ValueError):
            parse_file(unsupported_file)
    
    def test_nonexistent_file(self):
        """Test behavior with nonexistent files."""
        # Act/Assert
        with pytest.raises(FileNotFoundError):
            parse_file("nonexistent_file.csv")
