"""
Tests for the global error handling decorator.
"""

import pytest
import asyncio
import logging
from unittest.mock import patch, MagicMock

from src.watchdog_ai.utils.global_error_handler import (
    with_global_error_handling,
    global_try_except,
    handle_api_errors
)
from src.utils.errors import (
    WatchdogError,
    ValidationError,
    ProcessingError,
    ConfigurationError,
    APIError
)

# Test functions to be decorated
def sync_function_that_raises():
    """Test function that raises an exception."""
    raise ValueError("Test error")

def sync_function_with_args(a, b, c=3):
    """Test function with arguments that raises an exception."""
    raise ValueError(f"Error with args: {a}, {b}, {c}")

async def async_function_that_raises():
    """Async test function that raises an exception."""
    await asyncio.sleep(0.01)  # Simulate async operation
    raise ValueError("Async test error")

async def async_function_with_args(a, b, c=3):
    """Async test function with arguments that raises an exception."""
    await asyncio.sleep(0.01)  # Simulate async operation
    raise ValueError(f"Async error with args: {a}, {b}, {c}")

# Test functions with specific error types
def function_with_validation_error():
    """Function that raises a ValidationError."""
    raise ValidationError("Invalid data")

def function_with_processing_error():
    """Function that raises a ProcessingError."""
    raise ProcessingError("Processing failed")

def function_with_config_error():
    """Function that raises a ConfigurationError."""
    raise ConfigurationError("Bad configuration")

def function_with_api_error():
    """Function that raises an APIError."""
    raise APIError("API request failed")

# Test classes for the tests that need setup/teardown
class TestGlobalErrorHandler:
    """Test class for global error handler."""
    
    @pytest.fixture
    def mock_logger(self):
        """Mock the logger to test logging levels."""
        with patch('src.watchdog_ai.utils.global_error_handler.logger') as mock:
            yield mock
    
    def test_basic_sync_error_handling(self):
        """Test basic synchronous error handling."""
        decorated = with_global_error_handling()(sync_function_that_raises)
        
        # Function should not raise but return error response
        response = decorated()
        
        assert response is not None
        assert response["status"] == "error"
        assert response["error_type"] == "ValueError"
        assert "message" in response
        assert "timestamp" in response
    
    def test_sync_error_with_custom_message(self):
        """Test synchronous error handling with custom message."""
        friendly_msg = "Something went wrong"
        decorated = with_global_error_handling(friendly_message=friendly_msg)(sync_function_that_raises)
        
        response = decorated()
        
        assert response["message"] == friendly_msg
    
    def test_sync_error_with_args(self):
        """Test synchronous error handling with function arguments."""
        decorated = with_global_error_handling()(sync_function_with_args)
        
        response = decorated(1, 2, c=4)
        
        assert response["status"] == "error"
        assert "1, 2, 4" in str(response)
    
    @pytest.mark.asyncio
    async def test_basic_async_error_handling(self):
        """Test basic asynchronous error handling."""
        decorated = with_global_error_handling()(async_function_that_raises)
        
        # Function should not raise but return error response
        response = await decorated()
        
        assert response is not None
        assert response["status"] == "error"
        assert response["error_type"] == "ValueError"
        assert "message" in response
        assert "timestamp" in response
    
    @pytest.mark.asyncio
    async def test_async_error_with_custom_message(self):
        """Test asynchronous error handling with custom message."""
        friendly_msg = "Something went wrong async"
        decorated = with_global_error_handling(friendly_message=friendly_msg)(async_function_that_raises)
        
        response = await decorated()
        
        assert response["message"] == friendly_msg
    
    @pytest.mark.asyncio
    async def test_async_error_with_args(self):
        """Test asynchronous error handling with function arguments."""
        decorated = with_global_error_handling()(async_function_with_args)
        
        response = await decorated(1, 2, c=4)
        
        assert response["status"] == "error"
        assert "1, 2, 4" in str(response)
    
    def test_different_error_types(self):
        """Test handling of different error types."""
        # ValidationError
        decorated_validation = with_global_error_handling()(function_with_validation_error)
        response_validation = decorated_validation()
        assert response_validation["error_type"] == "ValidationError"
        
        # ProcessingError
        decorated_processing = with_global_error_handling()(function_with_processing_error)
        response_processing = decorated_processing()
        assert response_processing["error_type"] == "ProcessingError"
        
        # ConfigurationError
        decorated_config = with_global_error_handling()(function_with_config_error)
        response_config = decorated_config()
        assert response_config["error_type"] == "ConfigurationError"
        
        # APIError
        decorated_api = with_global_error_handling()(function_with_api_error)
        response_api = decorated_api()
        assert response_api["error_type"] == "APIError"
    
    def test_log_levels(self, mock_logger):
        """Test different logging levels."""
        # Test error level (default)
        decorated_error = with_global_error_handling()(sync_function_that_raises)
        decorated_error()
        mock_logger.error.assert_called()
        
        # Test debug level
        mock_logger.reset_mock()
        decorated_debug = with_global_error_handling(log_level="debug")(sync_function_that_raises)
        decorated_debug()
        mock_logger.debug.assert_called()
        
        # Test warning level
        mock_logger.reset_mock()
        decorated_warning = with_global_error_handling(log_level="warning")(sync_function_that_raises)
        decorated_warning()
        mock_logger.warning.assert_called()
        
        # Test info level
        mock_logger.reset_mock()
        decorated_info = with_global_error_handling(log_level="info")(sync_function_that_raises)
        decorated_info()
        mock_logger.info.assert_called()
    
    def test_ui_feedback_integration(self):
        """Test UI feedback integration."""
        # With UI feedback (default)
        decorated_with_ui = with_global_error_handling()(sync_function_that_raises)
        response_with_ui = decorated_with_ui()
        assert "ui_feedback" in response_with_ui
        assert response_with_ui["ui_feedback"]["type"] == "error"
        
        # Without UI feedback
        decorated_without_ui = with_global_error_handling(include_ui_feedback=False)(sync_function_that_raises)
        response_without_ui = decorated_without_ui()
        assert "ui_feedback" not in response_without_ui
    
    def test_reraise_option(self):
        """Test the reraise option."""
        # Without reraise (default)
        decorated_no_reraise = with_global_error_handling()(sync_function_that_raises)
        response = decorated_no_reraise()  # Should not raise
        assert response["status"] == "error"
        
        # With reraise
        decorated_reraise = with_global_error_handling(reraise=True)(sync_function_that_raises)
        with pytest.raises(ValueError):
            decorated_reraise()  # Should raise
    
    def test_global_try_except_no_args(self):
        """Test global_try_except decorator without arguments."""
        @global_try_except
        def test_func():
            raise ValueError("Test error")
        
        response = test_func()
        assert response["status"] == "error"
        assert response["error_type"] == "ValueError"
    
    def test_global_try_except_with_args(self):
        """Test global_try_except decorator with arguments."""
        @global_try_except(friendly_message="Custom message", log_level="warning")
        def test_func():
            raise ValueError("Test error")
        
        response = test_func()
        assert response["message"] == "Custom message"
    
    def test_api_error_handler(self):
        """Test API error handler decorator."""
        decorated = handle_api_errors()(function_with_api_error)
        
        response = decorated()
        
        assert response["status"] == "error"
        assert response["error_code"] == "API_ERROR"
        assert "message" in response
        assert "details" in response
    
    def test_api_error_handler_with_custom_message(self):
        """Test API error handler with custom message."""
        friendly_msg = "API error occurred"
        decorated = handle_api_errors(friendly_message=friendly_msg)(function_with_api_error)
        
        response = decorated()
        
        assert response["message"] == friendly_msg
    
    @pytest.mark.asyncio
    async def test_api_error_handler_async(self):
        """Test API error handler with async function."""
        @handle_api_errors()
        async def async_api_function():
            await asyncio.sleep(0.01)
            raise APIError("Async API error")
        
        response = await async_api_function()
        
        assert response["status"] == "error"
        assert response["error_code"] == "API_ERROR"
    
    def test_error_response_structure(self):
        """Test the structure of error responses."""
        decorated = with_global_error_handling()(sync_function_that_raises)
        
        response = decorated()
        
        # Check main response fields
        assert "status" in response
        assert "error_type" in response
        assert "message" in response
        assert "timestamp" in response
        
        # Check UI feedback structure
        assert "ui_feedback" in response
        ui_feedback = response["ui_feedback"]
        assert ui_feedback["type"] == "error"
        assert "content" in ui_feedback
        assert "title" in ui_feedback["content"]
        assert "message" in ui_feedback["content"]
        assert "style" in ui_feedback["content"]
    
    def test_error_response_with_watchdog_error(self):
        """Test error response for WatchdogError with details."""
        @with_global_error_handling()
        def function_with_details():
            raise WatchdogError(
                "Error with details",
                details={"key1": "value1", "key2": "value2"}
            )
        
        response = function_with_details()
        
        assert "details" in response
        assert response["details"]["key1"] == "value1"
        assert response["details"]["key2"] == "value2"
    
    def test_function_metadata_preservation(self):
        """Test that function metadata is preserved by the decorator."""
        @with_global_error_handling()
        def function_with_docstring():
            """This is a test docstring."""
            raise ValueError("Test")
        
        assert function_with_docstring.__doc__ == "This is a test docstring."
        assert function_with_docstring.__name__ == "function_with_docstring"

