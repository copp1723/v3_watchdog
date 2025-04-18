"""
Unit tests for predictive forecasting.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.insights.forecast import (
    ForecastConfig,
    ForecastResult,
    SalesPerformanceForecaster,
    InventoryTurnoverForecaster,
    MarginTrendForecaster
)

@pytest.fixture
def sample_sales_data():
    """Create sample sales data."""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    
    # Create synthetic data with trend and seasonality
    trend = np.linspace(1000, 2000, len(dates))
    seasonality = np.sin(np.arange(len(dates)) * 2 * np.pi / 7) * 200
    noise = np.random.normal(0, 100, len(dates))
    
    sales = trend + seasonality + noise
    
    return pd.DataFrame({
        'date': dates,
        'gross': sales,
        'price': sales * 2
    })

@pytest.fixture
def sample_inventory_data():
    """Create sample inventory data."""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    
    # Create synthetic aging data
    base_aging = np.ones(len(dates)) * 30
    trend = np.linspace(0, 10, len(dates))
    noise = np.random.normal(0, 5, len(dates))
    
    aging = base_aging + trend + noise
    
    return pd.DataFrame({
        'date': dates,
        'days_on_lot': aging
    })

@pytest.fixture
def sample_margin_data():
    """Create sample margin data."""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    
    # Create synthetic margin data
    base_margin = np.ones(len(dates)) * 0.2
    trend = np.linspace(0, 0.05, len(dates))
    seasonality = np.sin(np.arange(len(dates)) * 2 * np.pi / 30) * 0.02
    noise = np.random.normal(0, 0.01, len(dates))
    
    margins = base_margin + trend + seasonality + noise
    
    return pd.DataFrame({
        'date': dates,
        'gross': margins * 1000,
        'price': 1000
    })

def test_sales_forecaster_initialization():
    """Test initialization of sales forecaster."""
    config = ForecastConfig(periods=30)
    forecaster = SalesPerformanceForecaster(config)
    
    assert forecaster.config.periods == 30
    assert forecaster.config.confidence_level == 0.95

def test_sales_forecast_generation(sample_sales_data):
    """Test sales forecast generation."""
    forecaster = SalesPerformanceForecaster()
    result = forecaster.generate(sample_sales_data)
    
    assert isinstance(result, ForecastResult)
    assert len(result.forecast) == forecaster.config.periods
    assert isinstance(result.confidence_intervals, pd.DataFrame)
    assert "mean" in result.metrics
    assert "seasonality" in result.model_info

def test_inventory_forecast_generation(sample_inventory_data):
    """Test inventory forecast generation."""
    forecaster = InventoryTurnoverForecaster()
    result = forecaster.generate(sample_inventory_data)
    
    assert isinstance(result, ForecastResult)
    assert len(result.forecast) == forecaster.config.periods
    assert isinstance(result.confidence_intervals, pd.DataFrame)
    assert result.metrics["mean"] > 0

def test_margin_forecast_generation(sample_margin_data):
    """Test margin forecast generation."""
    forecaster = MarginTrendForecaster()
    result = forecaster.generate(sample_margin_data)
    
    assert isinstance(result, ForecastResult)
    assert len(result.forecast) == forecaster.config.periods
    assert isinstance(result.confidence_intervals, pd.DataFrame)
    assert 0 <= result.metrics["mean"] <= 1

def test_forecast_with_insufficient_data():
    """Test handling of insufficient historical data."""
    # Create small dataset
    dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
    data = pd.DataFrame({
        'date': dates,
        'gross': np.random.normal(1000, 100, len(dates))
    })
    
    forecaster = SalesPerformanceForecaster()
    
    with pytest.raises(ValueError) as excinfo:
        forecaster.generate(data)
    
    assert "Insufficient history" in str(excinfo.value)

def test_seasonality_detection(sample_sales_data):
    """Test seasonality detection in time series."""
    forecaster = SalesPerformanceForecaster()
    prepared_data = forecaster._prepare_data(sample_sales_data)
    seasonality_info = forecaster._analyze_seasonality(prepared_data)
    
    assert isinstance(seasonality_info, dict)
    assert "seasonal_strength" in seasonality_info
    assert "has_seasonality" in seasonality_info
    assert seasonality_info["period"] == 7  # Weekly seasonality

def test_forecast_metrics(sample_sales_data):
    """Test forecast accuracy metrics calculation."""
    forecaster = SalesPerformanceForecaster()
    result = forecaster.generate(sample_sales_data)
    
    assert "mean" in result.metrics
    assert "std" in result.metrics
    assert "forecast_mean" in result.metrics
    assert "forecast_std" in result.metrics
    
    # Forecast mean should be reasonably close to historical mean
    historical_mean = sample_sales_data['gross'].mean()
    forecast_mean = result.metrics["forecast_mean"]
    assert abs(historical_mean - forecast_mean) / historical_mean < 0.5

def test_confidence_intervals(sample_sales_data):
    """Test confidence interval generation."""
    forecaster = SalesPerformanceForecaster(
        ForecastConfig(confidence_level=0.95)
    )
    result = forecaster.generate(sample_sales_data)
    
    # Check interval bounds
    assert (result.confidence_intervals.iloc[:, 1] >= 
            result.confidence_intervals.iloc[:, 0]).all()
    
    # Check forecast falls within intervals
    assert (result.forecast >= result.confidence_intervals.iloc[:, 0]).all()
    assert (result.forecast <= result.confidence_intervals.iloc[:, 1]).all()

def test_error_handling():
    """Test error handling with invalid data."""
    # Create invalid dataset
    data = pd.DataFrame({
        'date': ['invalid_date'] * 100,
        'gross': ['invalid_number'] * 100
    })
    
    forecaster = SalesPerformanceForecaster()
    
    with pytest.raises(Exception):
        forecaster.generate(data)

def test_forecast_result_serialization(sample_sales_data):
    """Test ForecastResult serialization."""
    forecaster = SalesPerformanceForecaster()
    result = forecaster.generate(sample_sales_data)
    
    # Convert to dict
    result_dict = result.to_dict()
    
    assert isinstance(result_dict, dict)
    assert "forecast" in result_dict
    assert "confidence_intervals" in result_dict
    assert "metrics" in result_dict
    assert "model_info" in result_dict
    assert "generated_at" in result_dict

def test_different_forecast_horizons(sample_sales_data):
    """Test forecasting with different time horizons."""
    # Test short horizon
    short_config = ForecastConfig(periods=7)
    forecaster = SalesPerformanceForecaster(short_config)
    short_result = forecaster.generate(sample_sales_data)
    
    assert len(short_result.forecast) == 7
    
    # Test long horizon
    long_config = ForecastConfig(periods=90)
    forecaster = SalesPerformanceForecaster(long_config)
    long_result = forecaster.generate(sample_sales_data)
    
    assert len(long_result.forecast) == 90
    
    # Longer horizon should have wider confidence intervals
    short_interval_width = (
        short_result.confidence_intervals.iloc[:, 1] - 
        short_result.confidence_intervals.iloc[:, 0]
    ).mean()
    
    long_interval_width = (
        long_result.confidence_intervals.iloc[:, 1] - 
        long_result.confidence_intervals.iloc[:, 0]
    ).mean()
    
    assert long_interval_width > short_interval_width