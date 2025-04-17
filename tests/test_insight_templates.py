"""
Tests for insight templates module.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytest

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.insight_templates import TemplateManager, InsightTemplate

# Sample data for testing
@pytest.fixture
def sample_sales_data():
    """Create sample sales data for testing."""
    # Generate dates for the last year
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Create 500 random sales
    np.random.seed(42)  # For reproducibility
    n_sales = 500
    sales_indices = np.random.choice(len(dates), size=n_sales, replace=True)
    
    data = {
        'Sale_Date': dates[sales_indices],
        'VIN': [f'VIN{i:06d}' for i in range(n_sales)],
        'Gross_Profit': np.random.normal(2000, 800, n_sales),
        'LeadSource': np.random.choice(['Website', 'Walk-in', 'Referral', 'Third-party', 'Unknown'], n_sales),
        'VehicleMake': np.random.choice(['Toyota', 'Honda', 'Ford', 'Chevrolet', 'BMW', 'Mercedes'], n_sales),
        'VehicleModel': np.random.choice(['Sedan', 'SUV', 'Truck', 'Coupe'], n_sales)
    }
    
    # Create some pattern - more sales in certain months
    df = pd.DataFrame(data)
    
    # Add summer boost (more sales in summer months)
    summer_dates = dates[(dates.month >= 6) & (dates.month <= 8)]
    summer_sales_indices = np.random.choice(len(summer_dates), size=100, replace=True)
    
    summer_data = {
        'Sale_Date': summer_dates[summer_sales_indices],
        'VIN': [f'VIN_SUMMER{i:06d}' for i in range(100)],
        'Gross_Profit': np.random.normal(2200, 700, 100),  # Slightly higher gross in summer
        'LeadSource': np.random.choice(['Website', 'Walk-in', 'Referral', 'Third-party', 'Unknown'], 100),
        'VehicleMake': np.random.choice(['Toyota', 'Honda', 'Ford', 'Chevrolet', 'BMW', 'Mercedes'], 100),
        'VehicleModel': np.random.choice(['Sedan', 'SUV', 'Truck', 'Coupe'], 100)
    }
    
    # Combine datasets
    df = pd.concat([df, pd.DataFrame(summer_data)], ignore_index=True)
    
    # Sort by date
    return df.sort_values('Sale_Date').reset_index(drop=True)

@pytest.fixture
def sample_inventory_data():
    """Create sample inventory data for testing."""
    np.random.seed(42)  # For reproducibility
    
    data = {
        'VIN': [f'INV{i:06d}' for i in range(200)],
        'DaysInInventory': np.random.exponential(scale=45, size=200),
        'VehicleMake': np.random.choice(['Toyota', 'Honda', 'Ford', 'Chevrolet', 'BMW', 'Mercedes'], 200),
        'VehicleModel': np.random.choice(['Sedan', 'SUV', 'Truck', 'Coupe'], 200)
    }
    
    return pd.DataFrame(data)

def test_template_manager_initialization():
    """Test that TemplateManager initializes correctly."""
    manager = TemplateManager()
    
    # Should have loaded templates from the templates directory
    assert len(manager.templates) > 0
    
    # Should have the sales trend template
    assert 'sales_trend_analysis' in manager.templates
    
    # Check template structure
    template = manager.templates['sales_trend_analysis']
    assert template.name == 'Sales Trend Analysis'
    assert 'Sale_Date' in template.required_columns
    assert 'VIN' in template.required_columns
    assert 'chart_data' in template.expected_response_format

def test_template_compatibility(sample_sales_data, sample_inventory_data):
    """Test template compatibility checking."""
    manager = TemplateManager()
    
    # Sales templates should be compatible with sales data
    sales_compatible = manager.get_applicable_templates(sample_sales_data)
    assert len(sales_compatible) > 0
    
    # The sales trend template should be first (most compatible)
    assert sales_compatible[0][0].template_id == 'sales_trend_analysis'
    
    # Inventory template should be compatible with inventory data
    inventory_compatible = manager.get_applicable_templates(sample_inventory_data)
    assert len(inventory_compatible) > 0
    
    # Find the inventory health template
    inventory_template = next((t for t, s in inventory_compatible if t.template_id == 'inventory_health'), None)
    assert inventory_template is not None

def test_apply_sales_template(sample_sales_data):
    """Test applying the sales trend template."""
    manager = TemplateManager()
    
    # Apply the sales trend template
    prompt = manager.apply_template('sales_trend_analysis', sample_sales_data)
    
    # Check that the prompt includes key elements
    assert 'Sales Trend Analysis' in prompt
    assert 'Total Sales' in prompt
    assert 'based on the following dealership data' in prompt
    
    # Should include chart data
    assert 'Chart Data' in prompt
    
    # The prompt should be substantial
    assert len(prompt) > 500

def test_apply_inventory_template(sample_inventory_data):
    """Test applying the inventory health template."""
    manager = TemplateManager()
    
    # Apply the inventory health template
    prompt = manager.apply_template('inventory_health', sample_inventory_data)
    
    # Check that the prompt includes key elements
    assert 'Inventory Health Analysis' in prompt
    assert 'Average Days in Inventory' in prompt
    assert 'Inventory Age Distribution' in prompt
    
    # Should include chart data
    assert 'Chart Data' in prompt
    
    # The prompt should be substantial
    assert len(prompt) > 500

if __name__ == '__main__':
    pytest.main(['-xvs', __file__])