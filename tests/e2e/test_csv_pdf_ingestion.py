"""
End-to-end tests for CSV and PDF ingestion pipeline.
"""

import os
import pytest
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

from src.watchdog_ai.utils.pdf_extractor import PDFExtractor
from src.watchdog_ai.utils.data_parser import DataParser
from src.watchdog_ai.utils.data_normalization import DataSchemaApplier
from src.watchdog_ai.utils.ingestion_orchestrator import IngestionOrchestrator

# Sample data for test files
SAMPLE_CSV_DATA = """vehicle_id,make,model,year,sale_price,sale_date,customer_id
1001,Toyota,Camry,2020,25000,2023-01-15,C001
1002,Honda,Accord,2021,27500,2023-01-20,C002
1003,Ford,F-150,2019,32000,2023-01-25,C003
1004,Chevrolet,Malibu,2022,26000,2023-01-30,C004
1005,Nissan,Altima,2020,23000,2023-02-05,C005
"""

class TestCSVPDFIngestion:
    """End-to-end tests for the ingestion pipeline."""
    
    @pytest.fixture
    def test_data_dir(self):
        """Create a temporary directory with test data files."""
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        
        # Create a CSV file
        csv_path = os.path.join(temp_dir, "sample_vehicles.csv")
        with open(csv_path, "w") as f:
            f.write(SAMPLE_CSV_DATA)
        
        # Create a PDF file (test-only, just a renamed CSV)
        # In a real test, this would be an actual PDF
        pdf_path = os.path.join(temp_dir, "sample_invoice.pdf")
        with open(pdf_path, "w") as f:
            f.write(SAMPLE_CSV_DATA)
        
        yield temp_dir
        
        # Clean up
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def schema_profiles_dir(self, test_data_dir):
        """Create temporary schema profiles for testing."""
        profiles_dir = os.path.join(test_data_dir, "schema_profiles")
        os.makedirs(profiles_dir, exist_ok=True)
        
        # Create a simple default profile
        default_profile = {
            "id": "default",
            "name": "Default Schema Profile",
            "description": "Default schema profile for testing",
            "role": "general_manager",
            "columns": [
                {
                    "name": "vehicle_id",
                    "display_name": "Vehicle ID",
                    "description": "Unique identifier for the vehicle",
                    "data_type": "string",
                    "aliases": ["id", "vin", "inventory_id"]
                },
                {
                    "name": "make",
                    "display_name": "Make",
                    "description": "Vehicle manufacturer",
                    "data_type": "string",
                    "aliases": ["manufacturer", "brand"]
                },
                {
                    "name": "model",
                    "display_name": "Model",
                    "description": "Vehicle model",
                    "data_type": "string",
                    "aliases": ["vehicle_model"]
                },
                {
                    "name": "year",
                    "display_name": "Year",
                    "description": "Model year",
                    "data_type": "integer",
                    "aliases": ["model_year"]
                },
                {
                    "name": "sale_price",
                    "display_name": "Sale Price",
                    "description": "Sale price of the vehicle",
                    "data_type": "float",
                    "aliases": ["price", "amount", "cost"]
                },
                {
                    "name": "sale_date",
                    "display_name": "Sale Date",
                    "description": "Date of the sale",
                    "data_type": "date",
                    "aliases": ["date", "transaction_date"]
                },
                {
                    "name": "customer_id",
                    "display_name": "Customer ID",
                    "description": "Unique identifier for the customer",
                    "data_type": "string",
                    "aliases": ["cust_id", "buyer_id"]
                }
            ]
        }
        
        # Write profile to file
        with open(os.path.join(profiles_dir, "default.json"), "w") as f:
            import json
            json.dump(default_profile, f, indent=2)
        
        return profiles_dir
    
    def test_pdf_extractor(self, test_data_dir):
        """Test the PDF extraction functionality."""
        # Note: This test would normally use a real PDF
        # For this implementation test, we'll skip actual extraction
        
        extractor = PDFExtractor()
        assert extractor.check_dependencies() == True, "PDF dependencies should be available"
    
    def test_data_parser_csv(self, test_data_dir):
        """Test parsing a CSV file."""
        csv_path = os.path.join(test_data_dir, "sample_vehicles.csv")
        
        parser = DataParser()
        result = parser.parse_file(csv_path)
        
        # Verify results
        assert result.is_successful()
        assert result.file_type == "csv"
        assert len(result.dataframe) == 5
        assert list(result.dataframe.columns) == [
            "vehicle_id", "make", "model", "year", "sale_price", "sale_date", "customer_id"
        ]
    
    def test_data_normalization(self, test_data_dir, schema_profiles_dir):
        """Test data normalization with schema application."""
        csv_path = os.path.join(test_data_dir, "sample_vehicles.csv")
        
        # Parse the file first
        parser = DataParser()
        parse_result = parser.parse_file(csv_path)
        
        # Apply schema
        applier = DataSchemaApplier(schema_profiles_dir)
        df, summary = applier.apply_schema(parse_result.dataframe)
        
        # Verify normalization results
        assert df is not None
        assert not df.empty
        assert len(df) == 5
        assert summary["profile_id"] == "default"
        assert summary["success"] == True
        
        # Check data types
        assert pd.api.types.is_numeric_dtype(df["year"])
        assert pd.api.types.is_numeric_dtype(df["sale_price"])
    
    def test_ingestion_orchestrator_csv(self, test_data_dir, schema_profiles_dir):
        """Test the full ingestion pipeline with a CSV file."""
        csv_path = os.path.join(test_data_dir, "sample_vehicles.csv")
        
        # Create orchestrator with test dirs
        orchestrator = IngestionOrchestrator(
            schema_profiles_dir=schema_profiles_dir,
            output_dir=test_data_dir,
            lineage_tracking=False  # Disable for testing
        )
        
        # Ingest the file
        result = orchestrator.ingest_file(
            csv_path,
            dealer_id=None,
            vendor="test"
        )
        
        # Verify results
        assert result.success == True
        assert result.dataframe is not None
        assert len(result.dataframe) == 5
        assert result.source_file == csv_path
        assert result.validation_summary["parsing_result"]["success"] == True
        
        # Check that a summary file was created
        summary_files = [f for f in os.listdir(test_data_dir) if f.endswith("_ingestion_summary_") and f.endswith(".json")]
        assert len(summary_files) > 0
    
    def test_batch_ingestion(self, test_data_dir, schema_profiles_dir):
        """Test ingesting a directory of files."""
        # Create orchestrator with test dirs
        orchestrator = IngestionOrchestrator(
            schema_profiles_dir=schema_profiles_dir,
            output_dir=test_data_dir,
            lineage_tracking=False  # Disable for testing
        )
        
        # Ingest the directory
        combined_df, results = orchestrator.ingest_directory(
            test_data_dir,
            file_pattern="*.csv",
            dealer_id=None,
            vendor="test"
        )
        
        # Verify results
        assert combined_df is not None
        assert len(combined_df) == 5  # Just one CSV file in the test
        assert len(results) == 1
        assert results[0].success == True