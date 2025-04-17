"""
Tests for simple insight generation module.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from src.insights.simple_insight import (
    query_insight,
    analyze_lead_source,
    analyze_all_lead_sources,
    analyze_top_sales_reps,
    analyze_negative_gross,
    analyze_time_period
)

@pytest.fixture
def sample_sales_df():
    """Create a sample sales DataFrame."""
    return pd.DataFrame({
        'date': pd.date_range(start='2024-01-01', periods=100),
        'gross': [1000 + i * 100 for i in range(100)],
        'lead_source': ['CarGurus', 'AutoTrader', 'Website'] * 33 + ['CarGurus'],
        'sales_rep': ['Alice', 'Bob', 'Charlie', 'Diana'] * 25,
        'vin': [f'VIN{i}' for i in range(100)]
    })

@pytest.fixture
def sample_df_dict(sample_sales_df):
    """Create a sample DataFrame dictionary."""
    return {
        'sales': sample_sales_df,
        'inventory': pd.DataFrame({
            'vin': [f'VIN{i}' for i in range(50)],
            'days_in_stock': [i for i in range(50)],
            'price': [20000 + i * 1000 for i in range(50)]
        })
    }

def test_query_insight_lead_source(sample_df_dict):
    """Test lead source analysis query."""
    result = query_insight(sample_df_dict, "how many sales from cargurus")
    assert "CarGurus" in result.title
    assert "sales" in result.summary.lower()
    assert len(result.metrics) > 0
    assert result.chart_data is not None

def test_query_insight_sales_rep(sample_df_dict):
    """Test sales rep analysis query."""
    result = query_insight(sample_df_dict, "who are the top sales reps")
    assert "Sales Representatives" in result.title
    assert len(result.metrics) > 0
    assert result.chart_data is not None

def test_query_insight_negative_gross(sample_df_dict):
    """Test negative gross analysis query."""
    # Add some negative gross deals
    sample_df_dict['sales'].loc[0:4, 'gross'] = -1000
    result = query_insight(sample_df_dict, "show me negative gross deals")
    assert "Negative" in result.title
    assert "deals" in result.summary.lower()
    assert len(result.metrics) > 0
    assert result.chart_data is not None

def test_query_insight_time_period(sample_df_dict):
    """Test time period analysis query."""
    result = query_insight(sample_df_dict, "how many sales this month")
    assert "Month" in result.title
    assert "deals" in result.summary.lower()
    assert len(result.metrics) > 0
    assert result.chart_data is not None

def test_query_insight_unknown_question(sample_df_dict):
    """Test handling of unknown questions."""
    result = query_insight(sample_df_dict, "what is the meaning of life")
    assert "Not Understood" in result.title
    assert len(result.recommendations) > 0

def test_analyze_lead_source(sample_sales_df):
    """Test specific lead source analysis."""
    result = analyze_lead_source(sample_sales_df, "cargurus")
    assert "CarGurus" in result.title
    assert isinstance(result.metrics, list)
    assert len(result.metrics) > 0
    assert result.chart_data is not None
    assert len(result.recommendations) > 0

def test_analyze_all_lead_sources(sample_sales_df):
    """Test analysis of all lead sources."""
    result = analyze_all_lead_sources(sample_sales_df)
    assert "Lead Source" in result.title
    assert isinstance(result.metrics, list)
    assert len(result.metrics) > 0
    assert result.chart_data is not None
    assert len(result.recommendations) > 0

def test_analyze_top_sales_reps(sample_sales_df):
    """Test top sales reps analysis."""
    result = analyze_top_sales_reps(sample_sales_df)
    assert "Top Sales" in result.title
    assert isinstance(result.metrics, list)
    assert len(result.metrics) == 5  # Top 5 reps
    assert result.chart_data is not None
    assert len(result.recommendations) > 0

def test_analyze_negative_gross(sample_sales_df):
    """Test negative gross analysis."""
    # Add some negative gross deals
    sample_sales_df.loc[0:4, 'gross'] = -1000
    result = analyze_negative_gross(sample_sales_df)
    assert "Negative" in result.title
    assert isinstance(result.metrics, list)
    assert len(result.metrics) > 0
    assert result.chart_data is not None
    assert len(result.recommendations) > 0

def test_analyze_negative_gross_no_negatives(sample_sales_df):
    """Test negative gross analysis with no negative deals."""
    result = analyze_negative_gross(sample_sales_df)
    assert "No Negative" in result.title
    assert len(result.metrics) == 0
    assert result.chart_data is None
    assert len(result.recommendations) > 0

def test_analyze_time_period_this_month(sample_sales_df):
    """Test time period analysis for current month."""
    result = analyze_time_period(sample_sales_df, "how many sales this month")
    assert "Month" in result.title
    assert isinstance(result.metrics, list)
    assert len(result.metrics) > 0
    assert result.chart_data is not None
    assert len(result.recommendations) > 0

def test_analyze_time_period_last_week(sample_sales_df):
    """Test time period analysis for last week."""
    result = analyze_time_period(sample_sales_df, "show me last week's sales")
    assert "Week" in result.title
    assert isinstance(result.metrics, list)
    assert len(result.metrics) > 0
    assert result.chart_data is not None
    assert len(result.recommendations) > 0

def test_analyze_time_period_invalid(sample_sales_df):
    """Test time period analysis with invalid period."""
    result = analyze_time_period(sample_sales_df, "show me sales from next year")
    assert "Not Understood" in result.title
    assert len(result.metrics) == 0
    assert result.chart_data is None
    assert len(result.recommendations) > 0

def test_analyze_time_period_no_data(sample_sales_df):
    """Test time period analysis with no data in period."""
    # Set all dates to last year
    sample_sales_df['date'] = pd.date_range(start='2023-01-01', periods=100)
    result = analyze_time_period(sample_sales_df, "show me sales today")
    assert "No Sales" in result.title
    assert len(result.metrics) == 0
    assert result.chart_data is None
    assert len(result.recommendations) > 0

def test_query_insight_missing_sales_data():
    """Test query with missing sales data."""
    df_dict = {'inventory': pd.DataFrame()}
    result = query_insight(df_dict, "show me top sales reps")
    assert "Missing Data" in result.title
    assert len(result.metrics) == 0
    assert result.chart_data is None
    assert len(result.recommendations) > 0

def test_query_insight_empty_sales_data(sample_df_dict):
    """Test query with empty sales DataFrame."""
    sample_df_dict['sales'] = pd.DataFrame(columns=['date', 'gross', 'lead_source', 'sales_rep', 'vin'])
    result = query_insight(sample_df_dict, "show me top sales reps")
    assert len(result.metrics) == 0
    assert result.chart_data is not None  # Empty chart data is still valid
    assert len(result.recommendations) > 0