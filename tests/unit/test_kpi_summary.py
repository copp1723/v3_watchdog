"""
Unit tests for KPI dashboard summary generation.
"""

import pytest
from datetime import datetime
import pandas as pd
from src.insights.summarizer import Summarizer

@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    class MockLLMClient:
        def generate(self, prompt):
            return "Sales rep Alice's gross margin increased by 15% month-over-month, outperforming the group average."
    return MockLLMClient()

@pytest.fixture
def sample_metrics():
    """Create sample metrics data."""
    return pd.DataFrame({
        "Metric": ["Total Sales", "Avg Gross", "Aged Inventory"],
        "Value": [100, 5000, 15],
        "Change": ["+10%", "+15%", "-5%"]
    })

def test_summary_generation(mock_llm_client, sample_metrics):
    """Test basic summary generation."""
    summarizer = Summarizer(mock_llm_client)
    
    summary = summarizer.summarize(
        "kpi_summary.tpl",
        entity_name="Dealership",
        date_range="2023-01-01 to 2023-01-31",
        metrics_table=sample_metrics.to_markdown()
    )
    
    assert isinstance(summary, str)
    assert len(summary) > 0
    assert "gross margin" in summary.lower()
    assert "%" in summary

def test_summary_validation(mock_llm_client, sample_metrics):
    """Test summary validation rules."""
    summarizer = Summarizer(mock_llm_client)
    
    summary = summarizer.summarize(
        "kpi_summary.tpl",
        entity_name="Dealership",
        date_range="2023-01-01 to 2023-01-31",
        metrics_table=sample_metrics.to_markdown()
    )
    
    # Check for required elements
    assert "Dealership" in summary
    assert any(term in summary.lower() for term in ['increase', 'decrease', 'change'])
    assert any(term in summary.lower() for term in ['%', 'percent'])

def test_summary_with_feedback(mock_llm_client, sample_metrics):
    """Test summary generation with feedback context."""
    summarizer = Summarizer(mock_llm_client)
    
    summary = summarizer.summarize(
        "kpi_summary.tpl",
        entity_name="Dealership",
        date_range="2023-01-01 to 2023-01-31",
        metrics_table=sample_metrics.to_markdown(),
        feedback_context="Previous feedback: 10 ratings, 80% helpful"
    )
    
    assert isinstance(summary, str)
    assert len(summary) > 0

def test_summary_with_threshold_context(mock_llm_client, sample_metrics):
    """Test summary generation with threshold context."""
    summarizer = Summarizer(mock_llm_client)
    
    summary = summarizer.summarize(
        "kpi_summary.tpl",
        entity_name="Dealership",
        date_range="2023-01-01 to 2023-01-31",
        metrics_table=sample_metrics.to_markdown(),
        threshold_context="Using learned threshold of 0.85"
    )
    
    assert isinstance(summary, str)
    assert len(summary) > 0

def test_empty_metrics_handling(mock_llm_client):
    """Test handling of empty metrics."""
    summarizer = Summarizer(mock_llm_client)
    empty_metrics = pd.DataFrame(columns=["Metric", "Value", "Change"])
    
    with pytest.raises(Exception):
        summarizer.summarize(
            "kpi_summary.tpl",
            entity_name="Dealership",
            date_range="2023-01-01 to 2023-01-31",
            metrics_table=empty_metrics.to_markdown()
        )

def test_invalid_date_range(mock_llm_client, sample_metrics):
    """Test handling of invalid date range."""
    summarizer = Summarizer(mock_llm_client)
    
    with pytest.raises(Exception):
        summarizer.summarize(
            "kpi_summary.tpl",
            entity_name="Dealership",
            date_range="invalid_date_range",
            metrics_table=sample_metrics.to_markdown()
        )

def test_summary_caching(mock_llm_client, sample_metrics):
    """Test summary caching behavior."""
    summarizer = Summarizer(mock_llm_client)
    
    # Generate summary twice with same inputs
    summary1 = summarizer.summarize(
        "kpi_summary.tpl",
        entity_name="Dealership",
        date_range="2023-01-01 to 2023-01-31",
        metrics_table=sample_metrics.to_markdown()
    )
    
    summary2 = summarizer.summarize(
        "kpi_summary.tpl",
        entity_name="Dealership",
        date_range="2023-01-01 to 2023-01-31",
        metrics_table=sample_metrics.to_markdown()
    )
    
    # Should return cached result
    assert summary1 == summary2