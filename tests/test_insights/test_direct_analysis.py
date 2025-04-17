"""
Tests for direct data analysis functions.
"""

import pytest
import pandas as pd
import numpy as np
from src.insights.direct_analysis import (
    analyze_negative_profits,
    analyze_by_lead_source,
    find_metric_column,
    find_category_column
)

@pytest.fixture
def sample_df():
    """Create sample DataFrame for testing."""
    return pd.DataFrame({
        'Gross_Profit': [1000, -500, 2000, -750, 1500],
        'Lead_Source': ['Web', 'Phone', 'Web', 'Walk-in', 'Phone'],
        'Sale_Price': [25000, 22000, 30000, 20000, 28000]
    })

def test_analyze_negative_profits(sample_df):
    """Test negative profit analysis."""
    metrics, viz_data = analyze_negative_profits(sample_df, 'Gross_Profit')
    
    assert metrics['count'] == 2  # Two negative profits
    assert metrics['percentage'] == 40.0  # 2 out of 5 = 40%
    assert metrics['total_loss'] == -1250  # Sum of -500 and -750
    assert metrics['avg_loss'] == -625  # Average of -500 and -750
    
    assert len(viz_data) == 2
    assert viz_data['Count'].sum() == len(sample_df)

def test_analyze_by_lead_source(sample_df):
    """Test lead source analysis."""
    results = analyze_by_lead_source(sample_df, 'Gross_Profit', 'Lead_Source')
    
    assert len(results) == 3  # Three unique lead sources
    web_result = next(r for r in results if r['Lead_Source'] == 'Web')
    assert web_result['Gross_Profit_count'] == 2
    assert web_result['Gross_Profit_sum'] == 3000

def test_find_metric_column():
    """Test metric column finding."""
    columns = ['Total_Gross_Profit', 'Sale_Price', 'Revenue']
    
    assert find_metric_column(columns, 'gross') == 'Total_Gross_Profit'
    assert find_metric_column(columns, 'price') == 'Sale_Price'
    assert find_metric_column(columns, 'revenue') == 'Revenue'
    assert find_metric_column(columns, 'nonexistent') is None

def test_find_category_column():
    """Test category column finding."""
    columns = ['Lead_Source', 'SalesRep_Name', 'Vehicle_Make']
    
    assert find_category_column(columns, 'source') == 'Lead_Source'
    assert find_category_column(columns, 'rep') == 'SalesRep_Name'
    assert find_category_column(columns, 'vehicle') == 'Vehicle_Make'
    assert find_category_column(columns, 'nonexistent') is None