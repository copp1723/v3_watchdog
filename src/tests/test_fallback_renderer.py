"""
Tests for the fallback renderer module.
"""

import pytest
from datetime import datetime
from fallback_renderer import FallbackRenderer, ErrorCode, ErrorContext

@pytest.fixture
def fallback_renderer():
    """Create a FallbackRenderer instance for testing."""
    return FallbackRenderer()

@pytest.fixture
def error_context():
    """Create a sample ErrorContext for testing."""
    return ErrorContext(
        error_code=ErrorCode.COLUMN_MISSING,
        error_message="Column 'sales' not found in dataset",
        details={"column_name": "sales"},
        timestamp=datetime.now().isoformat(),
        user_query="Show me sales trends",
        affected_columns=["sales"],
        stack_trace="Traceback (most recent call last)..."
    )

def test_render_fallback_column_missing(fallback_renderer, error_context):
    """Test rendering a column missing error."""
    result = fallback_renderer.render_fallback(error_context)
    
    assert result["type"] == "error"
    assert result["title"] == "Required Data Not Found"
    assert "sales" in result["message"]
    assert "check your data file" in result["action"].lower()
    assert result["error_code"] == "column_missing"
    assert result["timestamp"] == error_context.timestamp

def test_render_fallback_invalid_data_type(fallback_renderer):
    """Test rendering an invalid data type error."""
    context = ErrorContext(
        error_code=ErrorCode.INVALID_DATA_TYPE,
        error_message="Invalid data type in column",
        details={
            "column_name": "price",
            "expected_type": "numeric",
            "actual_type": "string"
        },
        timestamp=datetime.now().isoformat()
    )
    
    result = fallback_renderer.render_fallback(context)
    
    assert result["type"] == "error"
    assert result["title"] == "Data Type Mismatch"
    assert "price" in result["message"]
    assert "numeric" in result["action"].lower()
    assert result["error_code"] == "invalid_data_type"

def test_render_fallback_code_execution_error(fallback_renderer):
    """Test rendering a code execution error."""
    context = ErrorContext(
        error_code=ErrorCode.CODE_EXECUTION_ERROR,
        error_message="Failed to execute analysis code",
        details={"error_details": "Division by zero"},
        timestamp=datetime.now().isoformat()
    )
    
    result = fallback_renderer.render_fallback(context)
    
    assert result["type"] == "error"
    assert result["title"] == "Analysis Execution Error"
    assert "error while running" in result["message"].lower()
    assert "try again" in result["action"].lower()
    assert result["error_code"] == "code_execution_error"

def test_render_fallback_unknown_error_code(fallback_renderer):
    """Test rendering with an unknown error code."""
    context = ErrorContext(
        error_code="UNKNOWN_ERROR",  # type: ignore
        error_message="Unknown error occurred",
        details={},
        timestamp=datetime.now().isoformat()
    )
    
    result = fallback_renderer.render_fallback(context)
    
    assert result["type"] == "error"
    assert result["title"] == "System Error"
    assert "unexpected system error" in result["message"].lower()
    assert result["error_code"] == "UNKNOWN_ERROR"

def test_render_fallback_missing_template_variables(fallback_renderer):
    """Test rendering with missing template variables."""
    context = ErrorContext(
        error_code=ErrorCode.COLUMN_MISSING,
        error_message="Column missing",
        details={},  # Missing column_name
        timestamp=datetime.now().isoformat()
    )
    
    result = fallback_renderer.render_fallback(context)
    
    assert result["type"] == "error"
    assert result["title"] == "Required Data Not Found"
    assert "check your data file" in result["action"].lower()
    assert result["error_code"] == "column_missing"

def test_render_fallback_with_stack_trace(fallback_renderer):
    """Test rendering with stack trace information."""
    context = ErrorContext(
        error_code=ErrorCode.CODE_EXECUTION_ERROR,
        error_message="Execution failed",
        details={"error_details": "Runtime error"},
        timestamp=datetime.now().isoformat(),
        stack_trace="Traceback (most recent call last)..."
    )
    
    result = fallback_renderer.render_fallback(context)
    
    assert result["type"] == "error"
    assert result["title"] == "Analysis Execution Error"
    assert "error while running" in result["message"].lower()
    assert result["error_code"] == "code_execution_error"

def test_render_fallback_with_user_query(fallback_renderer):
    """Test rendering with user query information."""
    context = ErrorContext(
        error_code=ErrorCode.NO_MATCHING_DATA,
        error_message="No data found",
        details={"criteria": "sales > 1000"},
        timestamp=datetime.now().isoformat(),
        user_query="Show me high sales"
    )
    
    result = fallback_renderer.render_fallback(context)
    
    assert result["type"] == "error"
    assert result["title"] == "No Matching Data Found"
    assert "couldn't find any data" in result["message"].lower()
    assert "adjust your query" in result["action"].lower()
    assert result["error_code"] == "no_matching_data" 