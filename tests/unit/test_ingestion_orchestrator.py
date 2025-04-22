"""
Unit tests for the IngestionOrchestrator.
"""

import os
import pytest
import pandas as pd
import tempfile
import json
import shutil
from unittest.mock import patch, MagicMock, ANY

from src.watchdog_ai.utils.ingestion_orchestrator import (
    IngestionOrchestrator, 
    IngestionResult,
    DMSConnector,
    ingest_file
)

class TestIngestionOrchestrator:
    """Tests for the IngestionOrchestrator class."""
    
    @pytest.fixture
    def test_csv_file(self):
        """Create a temporary CSV file for testing."""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp:
            temp.write(b'col1,col2,col3\n1,a,2023-01-01\n2,b,2023-01-02\n3,c,2023-01-03\n')
            return temp.name
    
    @pytest.fixture
    def test_directory(self):
        """Create a temporary directory with test files."""
        temp_dir = tempfile.mkdtemp()
        
        # Create CSV files
        with open(os.path.join(temp_dir, 'file1.csv'), 'w') as f:
            f.write('col1,col2,col3\n1,a,2023-01-01\n2,b,2023-01-02\n')
            
        with open(os.path.join(temp_dir, 'file2.csv'), 'w') as f:
            f.write('col1,col2,col3\n3,c,2023-01-03\n4,d,2023-01-04\n')
            
        with open(os.path.join(temp_dir, 'readme.txt'), 'w') as f:
            f.write('This is a test file, not data.')
        
        yield temp_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_init(self):
        """Test initialization of IngestionOrchestrator."""
        orchestrator = IngestionOrchestrator()
        assert orchestrator.schema_profiles_dir == "config/schema_profiles"
        assert orchestrator.lineage_tracking == True
        assert os.path.exists(orchestrator.output_dir)
    
    def test_ingestion_result(self):
        """Test IngestionResult class."""
        df = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
        result = IngestionResult(
            success=True,
            dataframe=df,
            source_file="test.csv",
            validation_summary={"key": "value"}
        )
        
        assert result.success == True
        assert result.dataframe is df
        assert result.source_file == "test.csv"
        
        # Test to_dict method
        result_dict = result.to_dict()
        assert result_dict["success"] == True
        assert result_dict["source_file"] == "test.csv"
        assert result_dict["validation_summary"] == {"key": "value"}
        assert result_dict["data_shape"] == (2, 2)
        assert result_dict["columns"] == ['col1', 'col2']
    
    @patch('src.watchdog_ai.utils.ingestion_orchestrator.DataParser')
    @patch('src.watchdog_ai.utils.ingestion_orchestrator.DataSchemaApplier')
    def test_ingest_file_success(self, mock_schema_applier, mock_parser, test_csv_file):
        """Test successful file ingestion."""
        # Setup mocks
        mock_parse_result = MagicMock()
        mock_parse_result.is_successful.return_value = True
        mock_parse_result.dataframe = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
        mock_parse_result.file_type = 'csv'
        mock_parse_result.get_summary.return_value = {"success": True}
        
        mock_parser_instance = MagicMock()
        mock_parser_instance.parse_file.return_value = mock_parse_result
        mock_parser.return_value = mock_parser_instance
        
        mock_schema_applier_instance = MagicMock()
        mock_schema_applier_instance.apply_schema.return_value = (
            pd.DataFrame({'normalized_col1': [1, 2], 'normalized_col2': ['a', 'b']}),
            {"normalization": "success"}
        )
        mock_schema_applier.return_value = mock_schema_applier_instance
        
        # Run test
        orchestrator = IngestionOrchestrator(output_dir=tempfile.mkdtemp())
        result = orchestrator.ingest_file(test_csv_file, dealer_id="test_dealer", vendor="test_vendor")
        
        # Verify
        assert result.success == True
        assert isinstance(result.dataframe, pd.DataFrame)
        assert result.source_file == test_csv_file
        assert result.validation_summary["parsing_result"] == {"success": True}
        assert result.validation_summary["normalization_result"] == {"normalization": "success"}
        assert result.metadata["dealer_id"] == "test_dealer"
        assert result.metadata["vendor"] == "test_vendor"
    
    @patch('src.watchdog_ai.utils.ingestion_orchestrator.DataParser')
    def test_ingest_file_parsing_failure(self, mock_parser, test_csv_file):
        """Test file ingestion with parsing failure."""
        # Setup mocks
        mock_parse_result = MagicMock()
        mock_parse_result.is_successful.return_value = False
        mock_parse_result.errors = ["Parsing error"]
        mock_parse_result.get_summary.return_value = {"success": False, "error": "Parsing error"}
        
        mock_parser_instance = MagicMock()
        mock_parser_instance.parse_file.return_value = mock_parse_result
        mock_parser.return_value = mock_parser_instance
        
        # Run test
        orchestrator = IngestionOrchestrator()
        result = orchestrator.ingest_file(test_csv_file)
        
        # Verify
        assert result.success == False
        assert result.error_message == "Parsing error"
        assert result.validation_summary["parsing_result"] == {"success": False, "error": "Parsing error"}
    
    @patch('src.watchdog_ai.utils.ingestion_orchestrator.DataParser')
    @patch('src.watchdog_ai.utils.ingestion_orchestrator.DataSchemaApplier')
    def test_ingest_directory(self, mock_schema_applier, mock_parser, test_directory):
        """Test ingesting all files in a directory."""
        # Setup mocks
        mock_parse_result = MagicMock()
        mock_parse_result.is_successful.return_value = True
        mock_parse_result.file_type = 'csv'
        mock_parse_result.get_summary.return_value = {"success": True}
        
        # Return different DataFrames for different files
        def mock_parse_file(file_path, **kwargs):
            if 'file1.csv' in file_path:
                mock_parse_result.dataframe = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
            else:
                mock_parse_result.dataframe = pd.DataFrame({'col1': [3, 4], 'col2': ['c', 'd']})
            return mock_parse_result
        
        mock_parser_instance = MagicMock()
        mock_parser_instance.parse_file.side_effect = mock_parse_file
        mock_parser.return_value = mock_parser_instance
        
        mock_schema_applier_instance = MagicMock()
        mock_schema_applier_instance.apply_schema.return_value = (
            pd.DataFrame({'normalized_col1': [1, 2, 3, 4], 'normalized_col2': ['a', 'b', 'c', 'd']}),
            {"normalization": "success"}
        )
        mock_schema_applier.return_value = mock_schema_applier_instance
        
        # Run test
        orchestrator = IngestionOrchestrator()
        combined_df, results = orchestrator.ingest_directory(
            test_directory, 
            file_pattern="*.csv", 
            dealer_id="test_dealer", 
            vendor="test_vendor"
        )
        
        # Verify
        assert combined_df is not None
        assert len(results) == 2
        assert all(r.success for r in results)
        assert mock_parser_instance.parse_file.call_count == 2
    
    def test_dms_connector(self):
        """Test DMSConnector."""
        connector = DMSConnector(vendor="test_vendor")
        assert connector.vendor == "test_vendor"
        assert connector.test_connection() == True  # Placeholder always returns True
        
        # Fetch data (placeholder returns empty DataFrame)
        df = connector.fetch_data("inventory", "2023-01-01", "2023-01-31")
        assert isinstance(df, pd.DataFrame)
        assert df.empty
    
    @patch('src.watchdog_ai.utils.ingestion_orchestrator.IngestionOrchestrator')
    def test_ingest_file_convenience_function(self, mock_orchestrator_class, test_csv_file):
        """Test the convenience function for ingesting a file."""
        # Setup mock
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.dataframe = pd.DataFrame({'col1': [1, 2]})
        mock_result.validation_summary = {"test": "summary"}
        
        mock_orchestrator = MagicMock()
        mock_orchestrator.ingest_file.return_value = mock_result
        mock_orchestrator_class.return_value = mock_orchestrator
        
        # Run test
        df, summary = ingest_file(test_csv_file, dealer_id="test_dealer", vendor="test_vendor")
        
        # Verify
        assert isinstance(df, pd.DataFrame)
        assert summary == {"test": "summary"}
        mock_orchestrator.ingest_file.assert_called_once_with(
            test_csv_file, "test_dealer", "test_vendor"
        )
    
    @patch('src.watchdog_ai.utils.ingestion_orchestrator.IngestionOrchestrator')
    def test_ingest_file_convenience_function_failure(self, mock_orchestrator_class, test_csv_file):
        """Test the convenience function with a failure."""
        # Setup mock
        mock_result = MagicMock()
        mock_result.success = False
        mock_result.error_message = "Test error"
        
        mock_orchestrator = MagicMock()
        mock_orchestrator.ingest_file.return_value = mock_result
        mock_orchestrator_class.return_value = mock_orchestrator
        
        # Run test and verify exception
        with pytest.raises(ValueError, match="Failed to ingest file: Test error"):
            df, summary = ingest_file(test_csv_file)