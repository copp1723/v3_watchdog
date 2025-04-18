"""
Unit tests for the report generators.
"""

import pandas as pd
import pytest
from datetime import datetime

from src.scheduler.reports import ReportGenerator
from src.scheduler.reports.sales_report import SalesReportGenerator
from src.scheduler.reports.inventory_report import InventoryReportGenerator


class TestSalesReportGenerator:
    """Tests for the SalesReportGenerator class."""
    
    @pytest.fixture
    def generator(self):
        """Fixture to create a generator instance."""
        return SalesReportGenerator()
    
    @pytest.fixture
    def sample_data(self):
        """Fixture to create sample sales data."""
        return pd.DataFrame({
            'Sale_Date': pd.date_range(start='2023-01-01', end='2023-01-10'),
            'VIN': [f'VIN{i:06d}' for i in range(10)],
            'Gross_Profit': [1000, 1500, 2000, 1200, 1800, 2200, 1300, 1700, 1900, 2100],
            'LeadSource': ['Website', 'Walk-in', 'Referral', 'Website', 'Third-party', 
                           'Website', 'Walk-in', 'Referral', 'Third-party', 'Unknown'],
            'VehicleMake': ['Toyota', 'Honda', 'Ford', 'Chevrolet', 'BMW', 
                            'Mercedes', 'Toyota', 'Honda', 'Ford', 'Chevrolet']
        })
    
    def test_generate_with_charts_and_tables(self, generator, sample_data):
        """Test generating a report with charts and tables."""
        parameters = {
            "title": "Test Sales Report",
            "include_charts": True,
            "include_tables": True,
            "metadata": {"test": True}
        }
        
        result = generator.generate(sample_data, parameters)
        
        assert result["title"] == "Test Sales Report"
        assert "generated_at" in result
        assert "Sales Summary" in result["summary"]
        assert len(result["charts"]) > 0
        assert len(result["tables"]) > 0
        assert result["metadata"] == {"test": True}
    
    def test_generate_without_charts(self, generator, sample_data):
        """Test generating a report without charts."""
        parameters = {
            "title": "Test Sales Report",
            "include_charts": False,
            "include_tables": True
        }
        
        result = generator.generate(sample_data, parameters)
        
        assert result["title"] == "Test Sales Report"
        assert "generated_at" in result
        assert "Sales Summary" in result["summary"]
        assert len(result["charts"]) == 0
        assert len(result["tables"]) > 0
    
    def test_generate_without_tables(self, generator, sample_data):
        """Test generating a report without tables."""
        parameters = {
            "title": "Test Sales Report",
            "include_charts": True,
            "include_tables": False
        }
        
        result = generator.generate(sample_data, parameters)
        
        assert result["title"] == "Test Sales Report"
        assert "generated_at" in result
        assert "Sales Summary" in result["summary"]
        assert len(result["charts"]) > 0
        assert len(result["tables"]) == 0
    
    def test_generate_with_empty_data(self, generator):
        """Test generating a report with empty data."""
        empty_data = pd.DataFrame()
        
        result = generator.generate(empty_data)
        
        assert "No data available" in result["summary"]
        assert len(result["charts"]) == 0
        assert len(result["tables"]) == 0


class TestInventoryReportGenerator:
    """Tests for the InventoryReportGenerator class."""
    
    @pytest.fixture
    def generator(self):
        """Fixture to create a generator instance."""
        return InventoryReportGenerator()
    
    @pytest.fixture
    def sample_data(self):
        """Fixture to create sample inventory data."""
        return pd.DataFrame({
            'VIN': [f'INV{i:06d}' for i in range(10)],
            'DaysInInventory': [10, 25, 45, 60, 75, 90, 105, 120, 15, 30],
            'VehicleMake': ['Toyota', 'Honda', 'Ford', 'Chevrolet', 'BMW', 
                            'Mercedes', 'Toyota', 'Honda', 'Ford', 'Chevrolet'],
            'VehicleModel': ['Sedan', 'SUV', 'Truck', 'Sedan', 'SUV',
                            'Sedan', 'SUV', 'Truck', 'Sedan', 'SUV']
        })
    
    def test_generate_with_charts_and_tables(self, generator, sample_data):
        """Test generating a report with charts and tables."""
        parameters = {
            "title": "Test Inventory Report",
            "include_charts": True,
            "include_tables": True,
            "metadata": {"test": True}
        }
        
        result = generator.generate(sample_data, parameters)
        
        assert result["title"] == "Test Inventory Report"
        assert "generated_at" in result
        assert "Inventory Health" in result["summary"]
        assert len(result["charts"]) > 0
        assert len(result["tables"]) > 0
        assert result["metadata"] == {"test": True}
    
    def test_generate_without_charts(self, generator, sample_data):
        """Test generating a report without charts."""
        parameters = {
            "title": "Test Inventory Report",
            "include_charts": False,
            "include_tables": True
        }
        
        result = generator.generate(sample_data, parameters)
        
        assert result["title"] == "Test Inventory Report"
        assert "generated_at" in result
        assert "Inventory Health" in result["summary"]
        assert len(result["charts"]) == 0
        assert len(result["tables"]) > 0
    
    def test_generate_without_tables(self, generator, sample_data):
        """Test generating a report without tables."""
        parameters = {
            "title": "Test Inventory Report",
            "include_charts": True,
            "include_tables": False
        }
        
        result = generator.generate(sample_data, parameters)
        
        assert result["title"] == "Test Inventory Report"
        assert "generated_at" in result
        assert "Inventory Health" in result["summary"]
        assert len(result["charts"]) > 0
        assert len(result["tables"]) == 0
    
    def test_generate_with_empty_data(self, generator):
        """Test generating a report with empty data."""
        empty_data = pd.DataFrame()
        
        result = generator.generate(empty_data)
        
        assert "No data available" in result["summary"]
        assert len(result["charts"]) == 0
        assert len(result["tables"]) == 0