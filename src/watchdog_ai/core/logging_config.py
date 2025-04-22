"""
Centralized logging configuration for Watchdog AI.

This module provides a standardized logging setup for the entire application,
supporting different log levels based on environment, logging to both file
and console, and including correlation IDs for request tracing.
"""

import os
import sys
import logging
import logging.handlers
import time
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, Union

from .config import settings, get_project_root

# Constants
DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | [%(correlation_id)s] | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_DIR = get_project_root() / "logs"

# Create logs directory if it doesn't exist
os.makedirs(LOG_DIR, exist_ok=True)

class CorrelationIDFilter(logging.Filter):
    """Filter that adds correlation ID to log records."""
    
    _correlation_id = ""
    
    def filter(self, record):
        """Add correlation_id to the record."""
        if not hasattr(record, 'correlation_id'):
            record.correlation_id = self._correlation_id or 'no_correlation_id'
        return True
    
    @classmethod
    def get_correlation_id(cls) -> str:
        """Get the current correlation ID."""
        return cls._correlation_id
    
    @classmethod
    def set_correlation_id(cls, correlation_id: Optional[str] = None) -> str:
        """Set the correlation ID for the current context."""
        cls._correlation_id = correlation_id or str(uuid.uuid4())
        return cls._correlation_id
    
    @classmethod
    def reset_correlation_id(cls) -> None:
        """Reset the correlation ID."""
        cls._correlation_id = ""

def get_log_level() -> int:
    """Get the log level from configuration."""
    log_level_name = settings.application.log_level.value
    return getattr(logging, log_level_name)

def configure_logging(level: Optional[Union[int, str]] = None, 
                     correlation_id: Optional[str] = None) -> None:
    """
    Configure logging for the application.
    
    Args:
        level: Override the log level from settings if provided
        correlation_id: Set a correlation ID for this logging session
    """
    # Set correlation ID
    if correlation_id:
        CorrelationIDFilter.set_correlation_id(correlation_id)
    
    # Determine log level
    if level is None:
        level = get_log_level()
    elif isinstance(level, str):
        level = getattr(logging, level.upper())
    
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(DEFAULT_LOG_FORMAT, DEFAULT_DATE_FORMAT)
    
    # Add correlation ID filter to root logger
    correlation_filter = CorrelationIDFilter()
    root_logger.addFilter(correlation_filter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    root_logger.addHandler(console_handler)
    
    # File handler - daily rotating
    log_file = LOG_DIR / f"watchdog_ai_{time.strftime('%Y%m%d')}.log"
    file_handler = logging.handlers.TimedRotatingFileHandler(
        log_file, when='midnight', backupCount=14
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    root_logger.addHandler(file_handler)
    
    # Set specific levels for third-party libraries
    if level <= logging.INFO:
        # Decrease verbosity of some noisy libraries
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("streamlit").setLevel(logging.INFO)
    
    # Log initial message
    logging.info(
        f"Logging configured with level={logging.getLevelName(level)}, "
        f"correlation_id={CorrelationIDFilter.get_correlation_id()}"
    )

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    This is the preferred way to get a logger in the application.
    """
    return logging.getLogger(name)

# Configure logging when the module is imported
if not logging.getLogger().handlers:
    configure_logging()

