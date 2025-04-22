"""
Unit tests for LLM response parsing and validation.
"""

import pytest
import json
from datetime import datetime
from watchdog_ai.models import InsightResponse, BreakdownItem
from watchdog_ai.llm.llm_engine import LLMEngine, sanitize_query, build_prompt
from watchdog_ai.utils import parse_llm_response

def test_valid_insight_response():
    """Test parsing a valid LLM response."""
    valid_json = {
        "summary": "Test summary",
        "metrics": {"total": 100, "average": 50.5},
        "breakdown": [
            {"label": "A", "value": 30.0},
            {"label": "B", "value": 70.0}
        ],
        "recommendations": ["Do X", "Try Y"],
        "confidence": "high"
    }
    
    insight = InsightResponse(**valid_json)
    assert insight.summary == "Test summary"
    assert insight.metrics["total"] == 100
    assert len(insight.breakdown) == 2
    assert insight.confidence == "high"
    assert not insight.is_error
    assert not insight.is_mock

def test_missing_required_fields():
    """Test handling missing required fields."""
    invalid_json = {
        "summary": "Test summary",
        "metrics": {"total": 100}
        # Missing required fields
    }
    
    with pytest.raises(ValueError):
        InsightResponse(**invalid_json)

def test_invalid_confidence_level():
    """Test handling invalid confidence level."""
    invalid_json = {
        "summary": "Test summary",
        "metrics": {"total": 100},
        "breakdown": [],
        "recommendations": [],
        "confidence": "invalid"  # Must be low/medium/high
    }
    
    with pytest.raises(ValueError):
        InsightResponse(**invalid_json)

def test_mock_insight():
    """Test generating a mock insight."""
    mock = InsightResponse.mock_insight()
    assert mock.is_mock
    assert mock.confidence == "low"
    assert len(mock.recommendations) > 0

def test_error_response():
    """Test generating an error response."""
    error = "Test error"
    response = InsightResponse.error_response(error)
    assert response.is_error
    assert response.error == error
    assert response.confidence == "low"

def test_llm_engine_mock_mode():
    """Test LLM engine in mock mode."""
    engine = LLMEngine(use_mock=True)
    response = engine.generate_insight("test query", {})
    assert response.is_mock
    assert not response.is_error

def test_llm_engine_sanitize_query():
    """Test query sanitization."""
    engine = LLMEngine(use_mock=True)
    dangerous_query = '{{dangerous}} {% injection %} <script>alert("xss")</script>'
    safe_query = engine._sanitize_query(dangerous_query)
    assert "{{" not in safe_query
    assert "{%" not in safe_query
    assert "<script>" not in safe_query

def test_breakdown_item_validation():
    """Test BreakdownItem validation."""
    valid_item = BreakdownItem(label="Test", value=42.0)
    assert valid_item.label == "Test"
    assert valid_item.value == 42.0
    
    with pytest.raises(ValueError):
        BreakdownItem(label="Test", value="not a number")

def test_prompt_and_parse_end_to_end(tmp_path):
    """Test the prompt building and parsing flow end-to-end."""
    # Test sanitize
    assert sanitize_query("What {{hack}}?") == "What hack?"
    
    # Test build prompt (smoke test)
    prompt = build_prompt(["colA", "colB"], "How many?")
    assert "Dataset Information:" in prompt
    
    # Test parse valid response
    good = '{"summary":"foo","metrics":{},"breakdown":[],"recommendations":[],"confidence":"low"}'
    insight = parse_llm_response(good)
    assert insight.summary == "foo"
    
    # Test parsing invalid response
    bad = '{"summary": "incomplete'
    fallback = parse_llm_response(bad)
    assert fallback.is_mock == True