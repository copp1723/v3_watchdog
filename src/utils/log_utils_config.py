
"""
Logging utility configuration for Watchdog AI.

This module provides logging configuration and utility functions
to ensure consistent logging across the application.

DEPRECATED: This module is deprecated and will be removed in v4.0.0.
            Please use 'watchdog_ai.core.config.logging' instead.
"""

import os
import sys
import logging
import warnings
from datetime import datetime
from typing import Optional, Dict, Any

warnings.warn(
    "The 'utils.log_utils_config' module is deprecated and will be removed in v4.0.0. "
    "Please use 'watchdog_ai.core.config.logging' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import from new location to maintain backward compatibility
try:
    from watchdog_ai.core.config.logging import (
        get_logger, 
        configure_logger, 
        LOG_LEVELS, 
        DEFAULT_LOG_FORMAT, 
        DEFAULT_LOG_LEVEL,
        setup_logging
    )
except ImportError:
    # Fall back to minimal implementation if imports fail (during migration)
    # These are just stubs in case the import fails during the transition
    DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    DEFAULT_LOG_LEVEL = logging.INFO
    
    LOG_LEVELS = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL
    }
    
    def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
        logger = logging.getLogger(name)
        if level and level.lower() in LOG_LEVELS:
            logger.setLevel(LOG_LEVELS[level.lower()])
        return logger
    
    def configure_logger(logger_name: str, log_level: str = "info", 
                         log_file: Optional[str] = None, 
                         log_format: Optional[str] = None) -> logging.Logger:
        return logging.getLogger(logger_name)
    
    def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None,
                     max_bytes: int = 10_485_760, backup_count: int = 5) -> None:
        pass

# For backward compatibility, we keep all exports
__all__ = [
    "get_logger", 
    "configure_logger", 
    "setup_logging",
    "LOG_LEVELS", 
    "DEFAULT_LOG_FORMAT", 
    "DEFAULT_LOG_LEVEL"
]

"""
Logging utility configuration for Watchdog AI.

This module provides logging configuration and utility functions
to ensure consistent logging across the application.
"""

import os
import logging
import logging.handlers
from typing import Optional, Dict, Any
import sys
from datetime import datetime

# Default log format
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_LOG_LEVEL = logging.INFO

# Log directory
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")

# Ensure log directory exists
os.makedirs(LOG_DIR, exist_ok=True)

# Configure default logging
logging.basicConfig(
    level=DEFAULT_LOG_LEVEL,
    format=DEFAULT_LOG_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.handlers.RotatingFileHandler(
            os.path.join(LOG_DIR, f"watchdog_{datetime.now().strftime('%Y%m%d')}.log"),
            maxBytes=10485760,  # 10MB
            backupCount=10
        )
    ]
)

# Create a mapping of log level strings to their corresponding constants
LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL
}

def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Get a logger with the specified name and level.
    
    Args:
        name: Name of the logger
        level: Optional log level (debug, info, warning, error, critical)
              If None, uses the default log level
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    if level and level.lower() in LOG_LEVELS:
        logger.setLevel(LOG_LEVELS[level.lower()])
    
    return logger

def configure_logger(
    logger_name: str, 
    log_level: str = "info", 
    log_file: Optional[str] = None, 
    log_format: Optional[str] = None
) -> logging.Logger:
    """
    Configure a logger with custom settings.
    
    Args:
        logger_name: Name of the logger to configure
        log_level: Log level (debug, info, warning, error, critical)
        log_file: Optional log file path
        log_format: Optional log format string
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(logger_name)
    
    # Set log level
    if log_level.lower() in LOG_LEVELS:
        logger.setLevel(LOG_LEVELS[log_level.lower()])
    else:
        logger.setLevel(DEFAULT_LOG_LEVEL)
        logger.warning(f"Unknown log level '{log_level}', using default")
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(log_format or DEFAULT_LOG_FORMAT)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if specified
    if log_file:
        try:
            # Ensure directory exists
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
                
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=10485760,  # 10MB
                backupCount=5
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"Failed to create log file handler: {str(e)}")
    
    return logger

# Export primary functions and constants
__all__ = ["get_logger", "configure_logger", "LOG_LEVELS", "DEFAULT_LOG_FORMAT", "DEFAULT_LOG_LEVEL"]

