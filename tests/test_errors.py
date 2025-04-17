"""
Tests for the error handling system.
"""

import pytest
from datetime import datetime
from src.utils.errors import (
    WatchdogError,
    ValidationError,
    ProcessingError,
    ConfigurationError,
    APIError,
    handle_error,
    format_error_for_ui
)

def test_watchdog_error_basic():
    """Test basic WatchdogError functionality."""
    error = WatchdogError("Test error")
    assert error.message == "Test error"
    assert error.error_code == "UNKNOWN_ERROR"
    assert isinstance(error.timestamp, str)
    assert isinstance(error.details, dict)

def test_watchdog_error_with_details():
    """Test WatchdogError with custom details."""
    details = {"key": "value"}
    error = WatchdogError("Test error", "CUSTOM_ERROR", details)
    assert error.message == "Test error"
    assert error.error_code == "CUSTOM_ERROR"
    assert error.details == details

def test_validation_error():
    """Test ValidationError specifics."""
    error = ValidationError("Invalid data")
    assert error.error_code == "VALIDATION_ERROR"
    assert str(error) == "Invalid data"

def test_processing_error():
    """Test ProcessingError specifics."""
    error = ProcessingError("Processing failed")
    assert error.error_code == "PROCESSING_ERROR"
    assert str(error) == "Processing failed"

def test_configuration_error():
    """Test ConfigurationError specifics."""
    error = ConfigurationError("Bad config")
    assert error.error_code == "CONFIG_ERROR"
    assert str(error) == "Bad config"

def test_api_error():
    """Test APIError specifics."""
    error = APIError("API failed")
    assert error.error_code == "API_ERROR"
    assert str(error) == "API failed"

def test_handle_error_basic():
    """Test basic error handling."""
    error = ValueError("Test error")
    response = handle_error(error)
    assert response["status"] == "error"
    assert response["error_type"] == "ValueError"
    assert "timestamp" in response

def test_handle_error_with_user_message():
    """Test error handling with custom user message."""
    error = ValueError("Internal error")
    user_message = "Something went wrong"
    response = handle_error(error, user_message)
    assert response["message"] == user_message

def test_handle_error_with_watchdog_error():
    """Test handling of WatchdogError with details."""
    details = {"source": "test"}
    error = WatchdogError("Test error", "TEST_ERROR", details)
    response = handle_error(error)
    assert response["error_code"] == "TEST_ERROR"
    assert response["details"] == details

def test_format_error_for_ui():
    """Test UI formatting of errors."""
    error_response = {
        "message": "Test error",
        "timestamp": datetime.now().isoformat()
    }
    ui_format = format_error_for_ui(error_response)
    assert ui_format["type"] == "error"
    assert "style" in ui_format["content"]
    assert ui_format["content"]["icon"] == "⚠️"

def test_error_inheritance():
    """Test error class inheritance."""
    assert issubclass(ValidationError, WatchdogError)
    assert issubclass(ProcessingError, WatchdogError)
    assert issubclass(ConfigurationError, WatchdogError)
    assert issubclass(APIError, WatchdogError)

def test_error_recovery_info():
    """Test error recovery information."""
    error = ValidationError(
        "Invalid data",
        details={
            "recoverable": True,
            "retry_after": 5,
            "suggested_action": "Check input format"
        }
    )
    response = handle_error(error)
    assert "details" in response
    assert response["details"]["recoverable"] is True

def test_error_chain():
    """Test error chaining."""
    try:
        try:
            raise ValueError("Original error")
        except ValueError as e:
            raise ProcessingError("Processing failed") from e
    except ProcessingError as e:
        response = handle_error(e)
        assert response["error_type"] == "ProcessingError"
        assert response["status"] == "error"

def test_multiple_errors():
    """Test handling multiple errors."""
    errors = [
        ValidationError("Error 1"),
        ProcessingError("Error 2"),
        ConfigurationError("Error 3")
    ]
    responses = [handle_error(e) for e in errors]
    assert len(responses) == 3
    assert all(r["status"] == "error" for r in responses)
    assert len(set(r["error_type"] for r in responses)) == 3