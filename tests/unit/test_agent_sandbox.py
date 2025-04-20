"""
Tests for the agent sandbox module.
"""

import pytest
import pandas as pd
import numpy as np
import json
import time
from typing import Dict, Any, List, Optional

from src.utils.agent_sandbox import (
    code_execution_in_sandbox,
    SandboxConfig,
    SandboxExecutionError,
    SchemaValidationError,
    validate_output_against_schema,
    DEFAULT_OUTPUT_SCHEMA
)

# Sample dataframe for testing
@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=10),
        'sales': [100, 150, 200, 120, 80, 250, 300, 180, 220, 190],
        'gross': [25, 40, 50, 30, 15, 60, 75, 45, 55, 47],
        'lead_source': ['Web', 'Referral', 'Web', 'Walk-in', 'Web', 
                        'Referral', 'Web', 'Walk-in', 'Referral', 'Web']
    })

# Sample schema for testing
@pytest.fixture
def sample_schema():
    """Create a sample schema for testing."""
    return {
        "type": "object",
        "required": ["answer", "data", "chart_type"],
        "properties": {
            "answer": {
                "type": "string"
            },
            "data": {
                "type": ["object", "array"]
            },
            "chart_type": {
                "type": "string",
                "enum": ["table", "bar", "line", "pie", "scatter", "none"]
            }
        }
    }

# Mock LLM service for retries
def mock_llm_service(prompt: str) -> str:
    """Mock LLM service that returns fixed code."""
    return """
# Fixed code that works correctly
result = {
    "answer": "The average sales is 179.0",
    "data": {"average_sales": 179.0},
    "chart_type": "none",
    "confidence": 0.9
}
"""

def test_successful_code_execution(sample_df):
    """Test that valid code executes successfully."""
    code = """
# Calculate summary statistics
avg_sales = df['sales'].mean()
max_sales = df['sales'].max()
min_sales = df['sales'].min()

# Create result
result = {
    "answer": f"The average sales is {avg_sales:.1f}, with a maximum of {max_sales} and minimum of {min_sales}",
    "data": {
        "average_sales": float(avg_sales),
        "max_sales": float(max_sales),
        "min_sales": float(min_sales)
    },
    "chart_type": "bar",
    "confidence": 0.95
}
"""
    
    config = SandboxConfig(execution_timeout_seconds=5)
    result = code_execution_in_sandbox(code, sample_df, config=config)
    
    assert result["answer"] == "The average sales is 179.0, with a maximum of 300 and minimum of 80"
    assert result["data"]["average_sales"] == 179.0
    assert result["data"]["max_sales"] == 300.0
    assert result["data"]["min_sales"] == 80.0
    assert result["chart_type"] == "bar"
    assert result["confidence"] == 0.95
    assert "metadata" in result
    assert "execution_id" in result["metadata"]
    assert "execution_time" in result["metadata"]

def test_syntax_error_handling(sample_df):
    """Test that syntax errors are caught and properly reported."""
    code = """
# This code has a syntax error
if df['sales'].mean() > 100
    result = {"answer": "High sales", "data": {}, "chart_type": "none", "confidence": 0.8}
else:
    result = {"answer": "Low sales", "data": {}, "chart_type": "none", "confidence": 0.8}
"""
    
    with pytest.raises(SandboxExecutionError) as excinfo:
        code_execution_in_sandbox(code, sample_df)
    
    assert "syntax error" in str(excinfo.value).lower() or "SyntaxError" in str(excinfo.value)

def test_runtime_error_handling(sample_df):
    """Test that runtime errors are caught and properly reported."""
    code = """
# This code has a runtime error (division by zero)
x = 10 / 0
result = {"answer": "This won't execute", "data": {}, "chart_type": "none", "confidence": 0.8}
"""
    
    with pytest.raises(SandboxExecutionError) as excinfo:
        code_execution_in_sandbox(code, sample_df)
    
    assert "division by zero" in str(excinfo.value).lower() or "ZeroDivisionError" in str(excinfo.value)

def test_timeout_handling(sample_df):
    """Test that code execution timeouts are handled properly."""
    code = """
# This code will timeout
import time
time.sleep(10)  # Sleep for longer than the timeout
result = {"answer": "This won't execute", "data": {}, "chart_type": "none", "confidence": 0.8}
"""
    
    config = SandboxConfig(execution_timeout_seconds=1)
    
    with pytest.raises(SandboxExecutionError) as excinfo:
        code_execution_in_sandbox(code, sample_df, config=config)
    
    # Different systems might report timeouts differently
    assert any(err in str(excinfo.value).lower() for err in ["timeout", "timed out", "killed", "terminated"])

def test_schema_validation(sample_df, sample_schema):
    """Test that output is properly validated against a schema."""
    # Code missing required field
    code = """
# This code produces output missing a required field
result = {
    "answer": "The average sales is 179.0",
    # Missing 'data' field
    "chart_type": "bar"
}
"""
    
    with pytest.raises(SchemaValidationError) as excinfo:
        code_execution_in_sandbox(code, sample_df, schema=sample_schema)
    
    assert "schema" in str(excinfo.value).lower()
    assert "data" in str(excinfo.value)  # Should mention the missing field

def test_retry_mechanism(sample_df):
    """Test that the retry mechanism works with a corrected prompt."""
    # Initial code with an error
    bad_code = """
# This code has a problem (accessing non-existent column)
avg_revenue = df['revenue'].mean()  # 'revenue' column doesn't exist
result = {"answer": f"Average revenue: {avg_revenue}", "data": {}, "chart_type": "none", "confidence": 0.8}
"""
    
    # Test with retry enabled
    result = code_execution_in_sandbox(
        bad_code, 
        sample_df,
        llm_service_func=mock_llm_service,
        original_prompt="What's the average revenue?",
        enable_retry=True
    )
    
    # The mock LLM service returns code that produces a different result
    assert result["answer"] == "The average sales is 179.0"
    assert "metadata" in result
    assert "retry_attempt" in result["metadata"]
    assert result["metadata"]["retry_attempt"] >= 1

def test_automatic_result_wrapping(sample_df):
    """Test that code without explicit return structures gets properly wrapped."""
    code = """
# Just calculate a value without explicit return structure
answer = f"The average sales is {df['sales'].mean():.1f}"
"""
    
    result = code_execution_in_sandbox(code, sample_df)
    
    assert result["answer"] == "The average sales is 179.0"
    assert "chart_type" in result
    assert "confidence" in result

def test_validate_output_against_schema():
    """Test the schema validation function directly."""
    # Valid output
    valid_output = {
        "answer": "The average sales is 179.0",
        "data": {"average": 179.0},
        "chart_type": "bar",
        "confidence": 0.9
    }
    
    # Invalid output (missing required field)
    invalid_output = {
        "answer": "The average sales is 179.0",
        "data": {"average": 179.0},
        # Missing chart_type
        "confidence": 0.9
    }
    
    # Test valid output
    is_valid, errors = validate_output_against_schema(valid_output, DEFAULT_OUTPUT_SCHEMA)
    assert is_valid is True
    assert errors is None
    
    # Test invalid output
    is_valid, errors = validate_output_against_schema(invalid_output, DEFAULT_OUTPUT_SCHEMA)
    assert is_valid is False
    assert errors is not None
    assert "chart_type" in str(errors)

def test_dataframe_modification(sample_df):
    """Test that code can modify and analyze the DataFrame."""
    code = """
# Create a new column and perform analysis
df['profit_margin'] = df['gross'] / df['sales'] * 100
avg_margin = df['profit_margin'].mean()
max_margin_source = df.loc[df['profit_margin'].idxmax(), 'lead_source']

result = {
    "answer": f"The average profit margin is {avg_margin:.1f}%. The highest margin came from {max_margin_source}.",
    "data": {
        "margins_by_source": df.groupby('lead_source')['profit_margin'].mean().to_dict()
    },
    "chart_type": "bar",
    "confidence": 0.9
}
"""
    
    result = code_execution_in_sandbox(code, sample_df)
    
    assert "The average profit margin is" in result["answer"]
    assert "highest margin came from" in result["answer"]
    assert "margins_by_source" in result["data"]
    assert isinstance(result["data"]["margins_by_source"], dict)
    assert "Web" in result["data"]["margins_by_source"]
    assert "Referral" in result["data"]["margins_by_source"]

def test_different_return_styles(sample_df):
    """Test different ways code might return results."""
    # Test with creating a new DataFrame
    code1 = """
# Create a summary DataFrame
summary_df = pd.DataFrame({
    'metric': ['Average Sales', 'Average Gross', 'Count'],
    'value': [df['sales'].mean(), df['gross'].mean(), len(df)]
})
"""
    
    result1 = code_execution_in_sandbox(code1, sample_df)
    assert "chart_type" in result1
    assert result1["chart_type"] == "table"  # Should detect the DataFrame and use table
    
    # Test with just returning a value
    code2 = """
# Just return a simple answer
answer = f"Found {len(df)} records with average sales of {df['sales'].mean():.1f}"
"""
    
    result2 = code_execution_in_sandbox(code2, sample_df)
    assert result2["answer"] == "Found 10 records with average sales of 179.0"
    assert result2["chart_type"] == "none"

def test_security_restrictions(sample_df):
    """Test that security restrictions prevent dangerous operations."""
    # Try to import a potentially dangerous module
    code = """
# Try to import subprocess
import subprocess
result = {"answer": "Imported subprocess", "data": {}, "chart_type": "none", "confidence": 0.8}
"""
    
    with pytest.raises(SandboxExecutionError) as excinfo:
        code_execution_in_sandbox(code, sample_df)
    
    assert "subprocess" in str(excinfo.value).lower() or "ModuleNotFoundError" in str(excinfo.value)
    
    # Try to execute a shell command
    code = """
# Try to execute a shell command
import os
os.system("echo 'Hello World'")
result = {"answer": "Executed command", "data": {}, "chart_type": "none", "confidence": 0.8}
"""
    
    with pytest.raises(SandboxExecutionError) as excinfo:
        code_execution_in_sandbox(code, sample_df)
    
    assert "os" in str(excinfo.value).lower() or "ModuleNotFoundError" in str(excinfo.value)