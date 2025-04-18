"""
Unit tests for Nova Act data connector.
"""

import pytest
import pandas as pd
from datetime import date, datetime
import json

from src.utils.datasource.novact import fetch_sales_data, fetch_inventory_data
from src.nova_act.core import NovaActClient
from src.utils.errors import ProcessingError

class MockNovaActClient:
    """Mock Nova Act client for testing."""
    
    def __init__(self, mock_responses=None):
        self.mock_responses = mock_responses or {}
    
    async def collect_report(self, vendor, credentials, report_config):
        """Mock report collection."""
        report_type = report_config["report_type"]
        
        if report_type in self.mock_responses:
            return self.mock_responses[report_type]
            
        # Default mock responses
        if report_type == "sales_performance":
            return {
                "success": True,
                "file_path": "tests/fixtures/mock_sales.csv",
                "timestamp": datetime.now().isoformat()
            }
        elif report_type == "inventory_report":
            return {
                "success": True,
                "file_path": "tests/fixtures/mock_inventory.csv",
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "success": False,
                "error": f"Unknown report type: {report_type}"
            }

@pytest.fixture
def mock_sales_data():
    """Create mock sales data CSV."""
    df = pd.DataFrame({
        'sale_date': pd.date_range('2023-01-01', periods=5),
        'total_gross': ['$1,000', '$2,000', '$1,500', '$3,000', '$2,500'],
        'sales_rep': ['Alice', 'Bob', 'Charlie', 'Alice', 'Bob'],
        'lead_source': ['Web', 'Phone', 'Walk-in', 'Web', 'Phone'],
        'vin': ['VIN001', 'VIN002', 'VIN003', 'VIN004', 'VIN005'],
        'sale_price': ['$20,000', '$25,000', '$22,000', '$30,000', '$28,000']
    })
    df.to_csv('tests/fixtures/mock_sales.csv', index=False)
    return df

@pytest.fixture
def mock_inventory_data():
    """Create mock inventory data CSV."""
    df = pd.DataFrame({
        'stock_date': pd.date_range('2023-01-01', periods=5),
        'days_in_stock': [15, 30, 45, 60, 75],
        'vin': ['VIN001', 'VIN002', 'VIN003', 'VIN004', 'VIN005'],
        'make': ['Toyota', 'Honda', 'Ford', 'Chevrolet', 'Toyota'],
        'model': ['Camry', 'Civic', 'F-150', 'Silverado', 'RAV4'],
        'year': [2022, 2023, 2022, 2023, 2022],
        'list_price': ['$25,000', '$22,000', '$45,000', '$48,000', '$32,000']
    })
    df.to_csv('tests/fixtures/mock_inventory.csv', index=False)
    return df

@pytest.mark.asyncio
async def test_fetch_sales_data(mock_sales_data):
    """Test fetching sales data."""
    client = MockNovaActClient()
    
    df = await fetch_sales_data(
        dealership_id="test123",
        start_date=date(2023, 1, 1),
        end_date=date(2023, 1, 5),
        client=client
    )
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5
    assert all(col in df.columns for col in [
        'SaleDate', 'TotalGross', 'SalesRepName', 'LeadSource', 'VIN', 'SalePrice'
    ])
    assert df['TotalGross'].dtype in ['int64', 'float64']
    assert df['SalePrice'].dtype in ['int64', 'float64']
    assert pd.api.types.is_datetime64_any_dtype(df['SaleDate'])

@pytest.mark.asyncio
async def test_fetch_inventory_data(mock_inventory_data):
    """Test fetching inventory data."""
    client = MockNovaActClient()
    
    df = await fetch_inventory_data(
        dealership_id="test123",
        client=client
    )
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5
    assert all(col in df.columns for col in [
        'StockDate', 'DaysInStock', 'VIN', 'Make', 'Model', 'Year', 'ListPrice'
    ])
    assert df['ListPrice'].dtype in ['int64', 'float64']
    assert df['DaysInStock'].dtype in ['int64', 'float64']
    assert pd.api.types.is_datetime64_any_dtype(df['StockDate'])

@pytest.mark.asyncio
async def test_fetch_sales_data_error():
    """Test error handling in sales data fetch."""
    client = MockNovaActClient({
        "sales_performance": {
            "success": False,
            "error": "API error"
        }
    })
    
    with pytest.raises(ProcessingError) as exc_info:
        await fetch_sales_data(
            dealership_id="test123",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 1, 5),
            client=client
        )
    
    assert "Failed to fetch sales data" in str(exc_info.value)

@pytest.mark.asyncio
async def test_fetch_inventory_data_error():
    """Test error handling in inventory data fetch."""
    client = MockNovaActClient({
        "inventory_report": {
            "success": False,
            "error": "API error"
        }
    })
    
    with pytest.raises(ProcessingError) as exc_info:
        await fetch_inventory_data(
            dealership_id="test123",
            client=client
        )
    
    assert "Failed to fetch inventory data" in str(exc_info.value)

@pytest.mark.asyncio
async def test_column_normalization(mock_sales_data):
    """Test column name normalization."""
    # Modify column names in mock data
    mock_sales_data.columns = [
        'Sale Date', 'Total Gross', 'Sales Rep',
        'Lead Source', 'VIN Number', 'Sale Price'
    ]
    mock_sales_data.to_csv('tests/fixtures/mock_sales.csv', index=False)
    
    client = MockNovaActClient()
    df = await fetch_sales_data(
        dealership_id="test123",
        start_date=date(2023, 1, 1),
        end_date=date(2023, 1, 5),
        client=client
    )
    
    assert all(col in df.columns for col in [
        'SaleDate', 'TotalGross', 'SalesRepName', 'LeadSource', 'VIN', 'SalePrice'
    ])