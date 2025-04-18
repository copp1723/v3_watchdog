"""
End-to-end tests for Nova Act data fetching.
"""

import pytest
import pandas as pd
from datetime import date, datetime
import os
import asyncio
from unittest.mock import patch

from src.utils.datasource.novact import fetch_sales_data, fetch_inventory_data
from src.utils.data_io import load_data
from src.nova_act.core import NovaActClient

class MockStreamlitUploadedFile:
    """Mock Streamlit uploaded file."""
    def __init__(self, content):
        self.content = content
        self._position = 0
    
    def read(self):
        return self.content
    
    def getvalue(self):
        return self.content
    
    def seek(self, position):
        self._position = position

@pytest.fixture
def mock_environment():
    """Set up test environment variables."""
    os.environ["USE_NOVACT"] = "true"
    os.environ["NOVACT_API_KEY"] = "test_key"
    yield
    del os.environ["USE_NOVACT"]
    del os.environ["NOVACT_API_KEY"]

@pytest.fixture
def mock_nova_client():
    """Create mock Nova Act client."""
    async def mock_collect_report(vendor, credentials, report_config):
        # Create mock data based on report type
        if report_config["report_type"] == "sales_performance":
            df = pd.DataFrame({
                'SaleDate': pd.date_range('2023-01-01', periods=5),
                'TotalGross': range(1000, 6000, 1000),
                'SalesRepName': ['Alice', 'Bob', 'Charlie', 'Alice', 'Bob'],
                'LeadSource': ['Web', 'Phone', 'Walk-in', 'Web', 'Phone'],
                'VIN': [f'VIN00{i}' for i in range(1, 6)],
                'SalePrice': range(20000, 45000, 5000)
            })
        else:  # inventory_report
            df = pd.DataFrame({
                'StockDate': pd.date_range('2023-01-01', periods=5),
                'DaysInStock': [15, 30, 45, 60, 75],
                'VIN': [f'VIN00{i}' for i in range(1, 6)],
                'Make': ['Toyota', 'Honda', 'Ford', 'Chevrolet', 'Toyota'],
                'Model': ['Camry', 'Civic', 'F-150', 'Silverado', 'RAV4'],
                'Year': [2022, 2023, 2022, 2023, 2022],
                'ListPrice': range(25000, 50000, 5000)
            })
        
        # Save to temp file
        temp_file = f"temp_{report_config['report_type']}.csv"
        df.to_csv(temp_file, index=False)
        
        return {
            "success": True,
            "file_path": temp_file,
            "timestamp": datetime.now().isoformat()
        }
    
    with patch.object(NovaActClient, 'collect_report', new=mock_collect_report):
        yield NovaActClient()

@pytest.mark.asyncio
async def test_full_data_pipeline(mock_environment, mock_nova_client):
    """Test complete data fetching and processing pipeline."""
    try:
        # Fetch sales data
        sales_df = await fetch_sales_data(
            dealership_id="test123",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 1, 5),
            client=mock_nova_client
        )
        
        # Verify sales data
        assert isinstance(sales_df, pd.DataFrame)
        assert len(sales_df) == 5
        assert all(col in sales_df.columns for col in [
            'SaleDate', 'TotalGross', 'SalesRepName', 'LeadSource', 'VIN', 'SalePrice'
        ])
        
        # Fetch inventory data
        inventory_df = await fetch_inventory_data(
            dealership_id="test123",
            client=mock_nova_client
        )
        
        # Verify inventory data
        assert isinstance(inventory_df, pd.DataFrame)
        assert len(inventory_df) == 5
        assert all(col in inventory_df.columns for col in [
            'StockDate', 'DaysInStock', 'VIN', 'Make', 'Model', 'Year', 'ListPrice'
        ])
        
        # Test data loading through Streamlit interface
        mock_file = MockStreamlitUploadedFile(sales_df.to_csv().encode())
        loaded_df = load_data(mock_file)
        
        # Verify loaded data
        assert isinstance(loaded_df, pd.DataFrame)
        assert len(loaded_df) == len(sales_df)
        assert all(col in loaded_df.columns for col in sales_df.columns)
        
    finally:
        # Clean up temp files
        for file in ['temp_sales_performance.csv', 'temp_inventory_report.csv']:
            if os.path.exists(file):
                os.remove(file)

@pytest.mark.asyncio
async def test_data_normalization(mock_environment, mock_nova_client):
    """Test data normalization and cleaning."""
    # Fetch sales data
    sales_df = await fetch_sales_data(
        dealership_id="test123",
        start_date=date(2023, 1, 1),
        end_date=date(2023, 1, 5),
        client=mock_nova_client
    )
    
    # Check data types
    assert pd.api.types.is_datetime64_any_dtype(sales_df['SaleDate'])
    assert sales_df['TotalGross'].dtype in ['int64', 'float64']
    assert sales_df['SalePrice'].dtype in ['int64', 'float64']
    
    # Check for missing values
    assert not sales_df['VIN'].isna().any()
    assert not sales_df['SalesRepName'].isna().any()
    assert not sales_df['LeadSource'].isna().any()

@pytest.mark.asyncio
async def test_error_handling(mock_environment):
    """Test error handling in data fetching."""
    # Create client that always fails
    async def mock_failed_collect_report(*args, **kwargs):
        return {
            "success": False,
            "error": "API error"
        }
    
    with patch.object(NovaActClient, 'collect_report', new=mock_failed_collect_report):
        client = NovaActClient()
        
        # Test sales data error handling
        with pytest.raises(Exception) as exc_info:
            await fetch_sales_data(
                dealership_id="test123",
                start_date=date(2023, 1, 1),
                end_date=date(2023, 1, 5),
                client=client
            )
        assert "Failed to fetch sales data" in str(exc_info.value)
        
        # Test inventory data error handling
        with pytest.raises(Exception) as exc_info:
            await fetch_inventory_data(
                dealership_id="test123",
                client=client
            )
        assert "Failed to fetch inventory data" in str(exc_info.value)