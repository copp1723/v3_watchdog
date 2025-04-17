"""
Unit tests for the LLM engine module.
"""

import pytest
import streamlit as st
import pandas as pd
import json
from ..src.llm_engine import LLMEngine

@pytest.fixture
def llm_engine():
    """Fixture providing an LLMEngine instance with mock responses."""
    return LLMEngine(use_mock=True)

@pytest.fixture
def sample_prompt():
    """Fixture providing a sample prompt."""
    return "How do sales compare to last month?"

@pytest.fixture
def sample_json_prompt():
    """Fixture providing a sample JSON formatted prompt."""
    return json.dumps({
        "query": "How do sales compare to last month?",
        "context": {"entity": "sales", "timeframe": "last month"},
        "previous_insights": []
    })

@pytest.fixture
def valid_json_response():
    """Fixture providing a valid JSON response string."""
    return json.dumps({
        "summary": "Sales increased by 15% compared to last month.",
        "chart_data": {
            "type": "bar",
            "data": {
                "Period": ["Last Month", "This Month"],
                "Value": [100, 115]
            }
        },
        "recommendation": "Increase marketing budget to sustain growth.",
        "risk_flag": False
    })

@pytest.fixture
def invalid_json_response():
    """Fixture providing an invalid JSON response string."""
    return "This is not JSON at all."

@pytest.fixture
def json_with_extra_keys():
    """Fixture providing a JSON response with unexpected keys."""
    return json.dumps({
        "summary": "Sales increased by 15% compared to last month.",
        "chart_data": {
            "type": "bar",
            "data": {
                "Period": ["Last Month", "This Month"],
                "Value": [100, 115]
            }
        },
        "recommendation": "Increase marketing budget to sustain growth.",
        "risk_flag": False,
        "extra_field": "This should not be here",
        "another_extra": 123
    })

def test_llm_engine_init(llm_engine):
    """Test initialization of LLMEngine."""
    assert llm_engine.use_mock is True
    assert llm_engine.client is None
    assert hasattr(llm_engine, 'response_schema')
    assert 'summary' in llm_engine.response_schema
    assert 'system_prompt' in dir(llm_engine)

def test_parse_llm_response_valid_json(llm_engine, valid_json_response):
    """Test parsing a valid JSON response from the LLM."""
    parsed = llm_engine.parse_llm_response(valid_json_response)
    
    assert isinstance(parsed, dict)
    assert parsed['summary'] == "Sales increased by 15% compared to last month."
    assert isinstance(parsed['chart_data'], dict)
    assert parsed['chart_data']['type'] == "bar"
    assert isinstance(parsed['chart_data']['data'], pd.DataFrame)
    assert parsed['recommendation'] == "Increase marketing budget to sustain growth."
    assert parsed['risk_flag'] is False

def test_parse_llm_response_invalid_json(llm_engine, invalid_json_response):
    """Test parsing an invalid JSON response from the LLM."""
    parsed = llm_engine.parse_llm_response(invalid_json_response)
    
    assert isinstance(parsed, dict)
    assert "Error: Unable to parse insight." in parsed['summary']
    assert parsed['chart_data'] == {}
    assert "parsing error" in parsed['recommendation']
    assert parsed['risk_flag'] is False

def test_parse_llm_response_extra_keys(llm_engine, json_with_extra_keys):
    """Test parsing a JSON response with unexpected keys."""
    parsed = llm_engine.parse_llm_response(json_with_extra_keys)
    
    assert isinstance(parsed, dict)
    assert parsed['summary'] == "Sales increased by 15% compared to last month."
    assert 'extra_field' not in parsed
    assert 'another_extra' not in parsed

def test_get_mock_response_simple_prompt(llm_engine, sample_prompt):
    """Test mock response generation for a simple prompt."""
    response = llm_engine.get_mock_response(sample_prompt)
    
    assert isinstance(response, dict)
    assert 'summary' in response
    assert 'chart_data' in response
    assert 'recommendation' in response
    assert 'risk_flag' in response
    assert 'timestamp' in response
    assert sample_prompt in response['summary']
    assert isinstance(response['chart_data'], dict)

def test_get_mock_response_json_prompt(llm_engine, sample_json_prompt):
    """Test mock response generation for a JSON formatted prompt."""
    response = llm_engine.get_mock_response(sample_json_prompt)
    
    assert isinstance(response, dict)
    assert 'summary' in response
    assert 'How do sales compare to last month?' in response['summary']

def test_generate_insight(llm_engine, sample_prompt):
    """Test generating an insight with a mock response."""
    response = llm_engine.generate_insight(sample_prompt)
    
    assert isinstance(response, dict)
    assert 'summary' in response
    assert 'chart_data' in response
    assert 'recommendation' in response
    assert 'risk_flag' in response
    assert 'timestamp' in response 