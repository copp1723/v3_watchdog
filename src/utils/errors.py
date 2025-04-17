"""
Centralized error handling system for Watchdog AI.
Provides custom exceptions and error handling utilities.
"""

from typing import Optional, Dict, Any
import logging
import traceback
from datetime import datetime

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
    pass

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

def handle_error(error: Exception) -> str:
    """
    Centralized error handler that logs errors and returns user-friendly messages.
    
    Args:
        error: The exception to handle
        
    Returns:
        User-friendly error message
    """
    # Log the error
    logger.error(f"Error type: {type(error).__name__}")
    logger.error(f"Error message: {str(error)}")
    logger.error(f"Stack trace:\n{traceback.format_exc()}")
    
    # Return user-friendly message
    if isinstance(error, ValidationError):
        return f"Validation error: {str(error)}"
    elif isinstance(error, ProcessingError):
        return f"Processing error: {str(error)}"
    elif isinstance(error, InsightGenerationError):
        return f"Insight generation error: {str(error)}"
    else:
        return f"An unexpected error occurred: {str(error)}"

def format_error_for_ui(error_message: str) -> Dict[str, Any]:
    """
    Format error message for UI display with appropriate styling.
    
    Args:
        error_message: Error message to format
        
    Returns:
        Dict with UI-friendly error formatting
    """
    return {
        "type": "error",
        "content": {
            "title": "Error",
            "message": error_message,
            "timestamp": datetime.now().isoformat(),
            "style": {
                "color": "red",
                "icon": "⚠️",
                "border": "1px solid red",
                "padding": "10px"
            }
        }
    }