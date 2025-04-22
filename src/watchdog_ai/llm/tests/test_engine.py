"""
Tests for the Watchdog AI LLM Engine.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from watchdog_ai.llm import LLMEngine, EngineSettings

# Test data setup
@pytest.fixture
def sample_data():
    """Create sample DataFrame for testing."""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    
    return pd.DataFrame({
        'date': dates,
        'sales': np.random.normal(1000, 100, len(dates)) + np.linspace(0, 500, len(dates)),  # Trend
        'inventory': np.random.normal(500, 50, len(dates)),
        'profit': np.random.normal(200, 20, len(dates)) + 50 * np.sin(np.linspace(0, 4*np.pi, len(dates)))  # Seasonal
    })

@pytest.fixture
def engine():
    """Create LLM engine instance for testing."""
    settings = EngineSettings()
    return LLMEngine(settings=settings)

def test_engine_initialization(engine):
    """Test engine initializes correctly."""
    assert engine is not None
    assert engine.settings is not None
    assert engine.api_config is not None
    assert engine.system_prompts is not None

def test_pattern_analysis(engine, sample_data):
    """Test pattern analysis functionality."""
    enhanced_prompt = engine._enhance_prompt_with_analysis(
        "Analyze sales trends",
        sample_data
    )
    
    # Check that analysis results are included
    assert "Based on data analysis:" in enhanced_prompt
    assert "Patterns detected:" in enhanced_prompt
    assert "Metrics analyzed:" in enhanced_prompt

def test_metric_analysis(engine, sample_data):
    """Test metric analysis functionality."""
    metrics = engine._analyze_metrics(sample_data, "analyze sales and profit trends")
    
    assert metrics is not None
    assert len(metrics) > 0
    assert any('sales' in metric.lower() for metric in metrics.keys())
    assert any('profit' in metric.lower() for metric in metrics.keys())

def test_error_handling(engine):
    """Test error response generation."""
    error_msg = "Test error"
    response = engine._generate_error_response(error_msg)
    
    assert response["summary"] == f"Error: {error_msg}"
    assert response["confidence"] == "low"
    assert "timestamp" in response
    assert len(response["value_insights"]) > 0
    assert len(response["actionable_flags"]) > 0

def test_insight_generation_with_context(engine, sample_data):
    """Test insight generation with data context."""
    response = engine.generate_insight(
        "Analyze sales trends",
        context={'data': sample_data}
    )
    
    assert response is not None
    assert "summary" in response
    assert "value_insights" in response
    assert "actionable_flags" in response
    assert "confidence" in response

def test_insight_generation_without_context(engine):
    """Test insight generation without data context."""
    response = engine.generate_insight(
        "What are best practices for inventory management?"
    )
    
    assert response is not None
    assert "summary" in response
    assert "value_insights" in response
    assert "actionable_flags" in response
    assert "confidence" in response

def test_settings_validation():
    """Test engine settings validation."""
    settings = EngineSettings()
    
    # Test analysis settings
    assert 0 < settings.analysis.pattern_confidence_threshold < 1
    assert settings.analysis.min_data_points > 0
    assert settings.analysis.max_anomaly_percentage > 0
    
    # Test validation settings
    assert settings.validation.min_summary_length > 0
    assert settings.validation.max_summary_length > settings.validation.min_summary_length
    assert settings.validation.min_insights > 0
    assert settings.validation.max_insights > settings.validation.min_insights

def test_prompt_enhancement(engine, sample_data):
    """Test prompt enhancement with analysis results."""
    # Test with time series data
    prompt = "Analyze sales performance"
    enhanced = engine._enhance_prompt_with_analysis(prompt, sample_data)
    
    assert prompt in enhanced
    assert "Based on data analysis:" in enhanced
    assert any(pattern in enhanced.lower() for pattern in ["trend", "seasonal", "correlation"])
    
    # Test with invalid data
    empty_df = pd.DataFrame()
    enhanced = engine._enhance_prompt_with_analysis(prompt, empty_df)
    assert prompt in enhanced  # Original prompt should still be there
    assert enhanced == prompt  # Should return original prompt on error

def test_api_configuration():
    """Test API configuration handling."""
    settings = EngineSettings()
    config = settings.api_config.get_client_config()
    
    assert "api_key" in config
    assert "timeout" in config
    assert "max_retries" in config
    assert config["timeout"] > 0
    assert config["max_retries"] > 0

@pytest.mark.parametrize("prompt,expected_terms", [
    ("Analyze sales trends", ["sales", "trends"]),
    ("How is inventory performing?", ["inventory", "performing"]),
    ("Calculate profit margins", ["profit", "margins"])
])
def test_query_term_extraction(engine, prompt, expected_terms):
    """Test query term extraction for metric analysis."""
    metrics = engine._analyze_metrics(pd.DataFrame(), prompt)
    assert isinstance(metrics, dict)

