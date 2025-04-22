"""
Centralized error handling system for Watchdog AI.
Provides custom exceptions and error handling utilities.
"""

from typing import Optional, Dict, Any, List
import logging
import traceback
from datetime import datetime
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

class WatchdogError(Exception):
    """Base exception class for Watchdog AI."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.timestamp = datetime.now().isoformat()
        
        # Log the error
        logger.error(f"{self.__class__.__name__}: {self.message}")
        if self.details:
            logger.error(f"Error details: {self.details}")

class ValidationError(WatchdogError):
    """Raised when data validation fails."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None,
                 missing_columns: Optional[List[str]] = None,
                 type_errors: Optional[Dict[str, List[str]]] = None,
                 validation_errors: Optional[Dict[str, List[str]]] = None):
        super().__init__(message, details)
        self.missing_columns = missing_columns or []
        self.type_errors = type_errors or {}
        self.validation_errors = validation_errors or {}
    
    def format_error_details(self) -> str:
        """Format error details for display."""
        details = []
        if self.missing_columns:
            details.append(f"Missing required columns: {', '.join(self.missing_columns)}")
        
        if self.type_errors:
            type_error_msgs = []
            for col, errors in self.type_errors.items():
                type_error_msgs.append(f"Column '{col}': {'; '.join(errors)}")
            details.append("Type errors:\n" + "\n".join(type_error_msgs))
        
        if self.validation_errors:
            validation_error_msgs = []
            for col, errors in self.validation_errors.items():
                validation_error_msgs.append(f"Column '{col}': {'; '.join(errors)}")
            details.append("Validation errors:\n" + "\n".join(validation_error_msgs))
        
        return "\n\n".join(details)

class ProcessingError(WatchdogError):
    """Raised when data processing fails."""
    pass

class ConfigurationError(WatchdogError):
    """Raised when there's a configuration issue."""
    pass

class APIError(WatchdogError):
    """Raised when an API call fails."""
    pass

class InsightGenerationError(WatchdogError):
    """Raised when insight generation fails."""
    pass

def handle_error(error: Exception, user_message: Optional[str] = None) -> Dict[str, Any]:
    """
    Centralized error handler that logs errors and returns user-friendly messages.
    
    Args:
        error: The exception to handle
        user_message: Optional custom message to display to the user
        
    Returns:
        Dictionary with error details formatted for display
    """
    # Log the error
    logger.error(f"Error type: {type(error).__name__}")
    logger.error(f"Error message: {str(error)}")
    logger.error(f"Stack trace:\n{traceback.format_exc()}")
    
    # Base error response
    error_response = {
        "status": "error",
        "error_type": type(error).__name__,
        "message": user_message or str(error),
        "timestamp": datetime.now().isoformat()
    }
    
    # Add specific handling for ValidationError
    if isinstance(error, ValidationError):
        error_response.update({
            "details": {
                "missing_columns": error.missing_columns,
                "type_errors": error.type_errors,
                "validation_errors": error.validation_errors
            },
            "formatted_details": error.format_error_details()
        })
    
    # Add any additional error details
    if hasattr(error, 'details'):
        error_response["details"] = error_response.get("details", {})
        error_response["details"].update(getattr(error, 'details'))
    
    return error_response

def format_error_for_ui(error_response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format error message for UI display with appropriate styling.
    
    Args:
        error_response: Error response from handle_error
        
    Returns:
        Dict with UI-friendly error formatting
    """
    severity_styles = {
        "ValidationError": {
            "color": "orange",
            "icon": "⚠️",
            "border": "1px solid orange"
        },
        "ProcessingError": {
            "color": "red",
            "icon": "❌",
            "border": "1px solid red"
        },
        "ConfigurationError": {
            "color": "purple",
            "icon": "⚙️",
            "border": "1px solid purple"
        },
        "APIError": {
            "color": "red",
            "icon": "🌐",
            "border": "1px solid red"
        },
        "InsightGenerationError": {
            "color": "orange",
            "icon": "💡",
            "border": "1px solid orange"
        }
    }
    
    error_type = error_response.get("error_type", "UnknownError")
    style = severity_styles.get(error_type, {
        "color": "red",
        "icon": "⚠️",
        "border": "1px solid red"
    })
    
    return {
        "type": "error",
        "content": {
            "title": f"Error: {error_type}",
            "message": error_response.get("message", "An unknown error occurred"),
            "details": error_response.get("formatted_details", ""),
            "timestamp": error_response.get("timestamp", datetime.now().isoformat()),
            "style": style
        }
    }