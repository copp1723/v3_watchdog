"""
Unit tests for the insight flow module.
"""

import pytest
import pandas as pd
from ..src.insight_flow import (
    PromptGenerator,
    FollowUpPrompt,
    generate_llm_prompt
)

@pytest.fixture
def sample_schema():
    """Fixture providing a sample schema for testing."""
    return {
        "sales": "Sales performance metrics",
        "inventory": "Inventory levels and turnover",
        "customers": "Customer satisfaction and retention",
        "revenue": "Revenue and profitability metrics"
    }

@pytest.fixture
def sample_summary():
    """Fixture providing a sample insight summary for testing."""
    return """
    Sales increased by $1,234,567 (15.5%) compared to last month. 
    Customer satisfaction improved by 8% in Q2, while inventory turnover 
    decreased by 12%. Revenue growth was strong at 22% year-over-year.
    """

@pytest.fixture
def sample_previous_insights():
    """Fixture providing sample previous insights for testing."""
    return [
        {
            "summary": "Sales were flat in Q1 with no significant growth.",
            "timestamp": "2023-04-01T10:00:00Z"
        },
        {
            "summary": "Customer satisfaction dropped by 5% in March.",
            "timestamp": "2023-04-15T14:30:00Z"
        }
    ]

def test_extract_entities(sample_schema, sample_summary):
    """Test entity extraction from summary text."""
    generator = PromptGenerator(sample_schema)
    entities = generator.extract_entities(sample_summary)
    
    # Check that all mentioned entities are extracted
    assert "sales" in entities
    assert "customers" in entities
    assert "inventory" in entities
    assert "revenue" in entities
    assert len(entities) == 4

def test_extract_metrics(sample_summary):
    """Test metric extraction from summary text."""
    generator = PromptGenerator({})
    metrics = generator.extract_metrics(sample_summary)
    
    # Check that all metrics are extracted
    assert "$1,234,567" in metrics
    assert "15.5%" in metrics
    assert "8%" in metrics
    assert "12%" in metrics
    assert "22%" in metrics
    assert len(metrics) == 5

def test_extract_timeframes(sample_summary):
    """Test timeframe extraction from summary text."""
    generator = PromptGenerator({})
    timeframes = generator.extract_timeframes(sample_summary)
    
    # Check that timeframes are extracted (case-insensitive)
    assert "last month" in timeframes
    assert "q2" in timeframes  # Check for lowercase version
    assert "year-over-year" in timeframes

def test_generate_comparison_prompts(sample_schema):
    """Test generation of comparison prompts."""
    generator = PromptGenerator(sample_schema)
    entities = ["sales", "inventory"]
    timeframes = ["last month", "Q2"]
    
    prompts = generator.generate_comparison_prompts(entities, timeframes)
    
    # Check that prompts are generated correctly
    assert len(prompts) > 0
    assert all(isinstance(prompt, FollowUpPrompt) for prompt in prompts)
    assert all(prompt.category == "comparison" for prompt in prompts)
    
    # Check that context variables are set
    for prompt in prompts:
        assert "entity" in prompt.context_vars
        assert "timeframe" in prompt.context_vars
        if "dimension" in prompt.context_vars:
            assert prompt.context_vars["dimension"] == "category"

def test_generate_dig_deeper_prompts(sample_schema):
    """Test generation of dig deeper prompts."""
    generator = PromptGenerator(sample_schema)
    entities = ["sales", "inventory"]
    metrics = ["$1,234,567", "15.5%"]
    
    prompts = generator.generate_dig_deeper_prompts(entities, metrics)
    
    # Check that prompts are generated correctly
    assert len(prompts) > 0
    assert all(isinstance(prompt, FollowUpPrompt) for prompt in prompts)
    
    # Check that we have both analysis and recommendation prompts
    categories = [prompt.category for prompt in prompts]
    assert "analysis" in categories
    assert "recommendation" in categories
    
    # Check that context variables are set
    for prompt in prompts:
        assert "entity" in prompt.context_vars

def test_generate_context_prompts(sample_schema):
    """Test generation of context prompts."""
    generator = PromptGenerator(sample_schema)
    entities = ["sales", "inventory"]
    
    prompts = generator.generate_context_prompts(entities)
    
    # Check that prompts are generated correctly
    assert len(prompts) > 0
    assert all(isinstance(prompt, FollowUpPrompt) for prompt in prompts)
    assert all(prompt.category == "context" for prompt in prompts)
    
    # Check that context variables are set
    for prompt in prompts:
        assert "entity" in prompt.context_vars
        assert "related_entity" in prompt.context_vars
        assert prompt.context_vars["entity"] != prompt.context_vars["related_entity"]

def test_generate_follow_up_prompts(sample_schema, sample_summary):
    """Test generation of all follow-up prompts."""
    generator = PromptGenerator(sample_schema)
    prompts = generator.generate_follow_up_prompts(sample_summary)
    
    # Check that prompts are generated correctly
    assert len(prompts) > 0
    assert all(isinstance(prompt, FollowUpPrompt) for prompt in prompts)
    
    # Check that prompts are sorted by priority
    priorities = [prompt.priority for prompt in prompts]
    assert priorities == sorted(priorities, reverse=True)

def test_generate_llm_prompt(sample_previous_insights):
    """Test generation of LLM prompt."""
    selected_prompt = "How does sales compare to last month?"
    context_vars = {"entity": "sales", "timeframe": "last month"}
    
    prompt_json = generate_llm_prompt(
        selected_prompt, 
        context_vars, 
        sample_previous_insights
    )
    
    # Check that the prompt is valid JSON
    import json
    prompt = json.loads(prompt_json)
    
    # Check that the prompt has the expected structure
    assert "query" in prompt
    assert prompt["query"] == selected_prompt
    
    assert "context" in prompt
    assert prompt["context"] == context_vars
    
    assert "previous_insights" in prompt
    assert len(prompt["previous_insights"]) == 2
    
    # Check that previous insights are included correctly
    for i, insight in enumerate(prompt["previous_insights"]):
        assert "summary" in insight
        assert insight["summary"] == sample_previous_insights[i]["summary"]
        assert "timestamp" in insight
        assert insight["timestamp"] == sample_previous_insights[i]["timestamp"] 