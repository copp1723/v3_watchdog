"""
Tests for data quality tracking and multi-metric support.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from src.utils.metrics_logger import MetricsLogger
from src.watchdog_ai.insights.models import InsightErrorType, InsightResponse

@pytest.fixture
def sample_data():
    """Create sample DataFrame with known quality issues."""
    return pd.DataFrame({
        'sales': [100, 200, np.nan, 400, 500],
        'profit': [10, np.nan, 30, np.nan, 50],
        'date': pd.date_range(start='2023-01-01', periods=5)
    })

@pytest.fixture
def metrics_logger(tmp_path):
    """Create MetricsLogger instance with temporary directory."""
    return MetricsLogger(log_dir=str(tmp_path / "metrics"))

def test_nan_percentage_tracking(metrics_logger, sample_data):
    """Test tracking of NaN percentages."""
    # Calculate NaN percentage for profit column
    nan_pct = (sample_data['profit'].isna().sum() / len(sample_data)) * 100
    
    metrics = metrics_logger.log_query(
        query_id="test_query",
        query="Test query",
        result={},
        start_time=datetime.now().timestamp(),
        llm_tokens=0,
        cache_hit=False,
        nan_percentage=nan_pct,
        excluded_rows=2
    )
    
    assert metrics.nan_percentage == 40.0  # 2 out of 5 rows are NaN
    assert metrics.excluded_rows == 2

def test_data_quality_thresholds(sample_data):
    """Test data quality threshold checks."""
    from src.watchdog_ai.insights.models import InsightResponse
    
    # Calculate quality metrics
    quality_metrics = {
        'nan_percentage': 40.0,
        'sample_size': len(sample_data),
        'warning_level': 'medium'
    }
    
    response = InsightResponse(
        summary="Test insight",
        metrics={},
        recommendations=[],
        confidence="medium",
        data_quality=quality_metrics
    )
    
    assert response.data_quality['warning_level'] == 'medium'
    assert response.data_quality['nan_percentage'] == 40.0

def test_error_type_handling():
    """Test error type handling."""
    # Test no valid data error
    response = InsightResponse(
        summary="No valid data available",
        metrics={},
        recommendations=[],
        confidence="low",
        error_type=InsightErrorType.NO_VALID_DATA
    )
    
    assert response.error_type == InsightErrorType.NO_VALID_DATA
    
    # Test insufficient data error
    response = InsightResponse(
        summary="Insufficient data for analysis",
        metrics={},
        recommendations=[],
        confidence="low",
        error_type=InsightErrorType.INSUFFICIENT_DATA
    )
    
    assert response.error_type == InsightErrorType.INSUFFICIENT_DATA

def test_multi_metric_analysis(sample_data):
    """Test analysis with multiple metrics."""
    metrics = {}
    
    # Calculate metrics for non-NaN values
    for column in ['sales', 'profit']:
        valid_data = sample_data[column].dropna()
        metrics[column] = {
            'mean': valid_data.mean(),
            'count': len(valid_data),
            'nan_pct': (sample_data[column].isna().sum() / len(sample_data)) * 100
        }
    
    # Verify sales metrics
    assert metrics['sales']['nan_pct'] == 20.0  # 1 out of 5 is NaN
    assert metrics['sales']['count'] == 4
    
    # Verify profit metrics
    assert metrics['profit']['nan_pct'] == 40.0  # 2 out of 5 are NaN
    assert metrics['profit']['count'] == 3

def test_data_quality_trends(metrics_logger):
    """Test data quality trend analysis."""
    # Log multiple entries with decreasing quality
    for i in range(3):
        metrics_logger.log_query(
            query_id=f"test_query_{i}",
            query="Test query",
            result={},
            start_time=datetime.now().timestamp(),
            llm_tokens=0,
            cache_hit=False,
            nan_percentage=12.0 - i,  # Decreasing values: 12.0, 11.0, 10.0
            excluded_rows=i
        )
    
    # Get trends
    trends = metrics_logger.get_data_quality_trends()
    
    assert len(trends['nan_percentage']) == 3
    assert len(trends['excluded_rows']) == 3
    # First entry should have higher nan_percentage than second entry
    assert trends['nan_percentage'][0]['value'] == 12.0
    assert trends['nan_percentage'][1]['value'] == 11.0
    assert trends['nan_percentage'][2]['value'] == 10.0
    # Excluded rows should increase
    assert trends['excluded_rows'][0]['value'] == 0
    assert trends['excluded_rows'][1]['value'] == 1
    assert trends['excluded_rows'][2]['value'] == 2