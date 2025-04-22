"""
Consolidated logging configuration for Watchdog AI.

This module provides a unified logging system that consolidates functionality from:
- src/utils/log_utils_config.py
- src/utils/logging_config.py
- src/nova_act/logging_config.py
"""

import os
import logging
import logging.handlers
import sys
from typing import Optional, Dict, Any
from datetime import datetime
from pathlib import Path

# Default log format
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DETAILED_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s"
SIMPLE_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_LOG_LEVEL = logging.INFO

# Log directory
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "logs")

# Ensure log directory exists
os.makedirs(LOG_DIR, exist_ok=True)

# Create a mapping of log level strings to their corresponding constants
LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL
}

def get_log_level(level_name: str) -> int:
    """
    Convert a log level name to its corresponding integer value.
    
    Args:
        level_name: String representation of log level
        
    Returns:
        Integer log level or DEFAULT_LOG_LEVEL if not found
    """
    if isinstance(level_name, int):
        return level_name
        
    level_name = str(level_name).lower()
    return LOG_LEVELS.get(level_name, DEFAULT_LOG_LEVEL)

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
    
    # Ensure logger has at least one handler to avoid "No handlers could be found" warnings
    if not logger.handlers and not logger.parent.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT))
        logger.addHandler(console_handler)
    
    return logger

def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    max_bytes: int = 10_485_760,  # 10MB
    backup_count: int = 5,
    log_format: str = DETAILED_LOG_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT
) -> None:
    """
    Configure logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup files to keep
        log_format: Format string for log messages
        date_format: Format string for timestamps
    """
    # Default log file if none provided
    if log_file is None:
        log_file = os.path.join(LOG_DIR, f"watchdog_{datetime.now().strftime('%Y%m%d')}.log")
    
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Get the integer log level
    level = get_log_level(log_level)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(log_format, datefmt=date_format)
    simple_formatter = logging.Formatter(SIMPLE_LOG_FORMAT, datefmt=date_format)
    
    # Console handler (with simple format)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (with detailed format)
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Set specific levels for some loggers to reduce noise
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('streamlit').setLevel(logging.WARNING)
    
    # Log the configuration
    root_logger.info(f"Logging initialized: level={log_level}, file={log_file}")

def configure_logger(
    logger_name: str, 
    log_level: str = "info", 
    log_file: Optional[str] = None, 
    log_format: str = DETAILED_LOG_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
    max_bytes: int = 10_485_760,  # 10MB
    backup_count: int = 5,
    console_output: bool = True
) -> logging.Logger:
    """
    Configure a specific logger with custom settings.
    
    Args:
        logger_name: Name of the logger to configure
        log_level: Logging level (debug, info, warning, error, critical)
        log_file: Optional path to log file
        log_format: Format string for log messages
        date_format: Format string for timestamps
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup files to keep
        console_output: Whether to output logs to console
        
    Returns:
        Configured logger instance
    """
    # Get or create logger
    logger = logging.getLogger(logger_name)
    
    # Set log level
    level = get_log_level(log_level)
    logger.setLevel(level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatters
    detailed_formatter = logging.Formatter(log_format, datefmt=date_format)
    simple_formatter = logging.Formatter(SIMPLE_LOG_FORMAT, datefmt=date_format)
    
    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)
    
    # Add file handler if log file specified
    if log_file:
        # Create log directory if needed
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Create rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger to avoid duplicate logs
    logger.propagate = False
    
    # Log configuration
    logger.debug(
        f"Logger configured: name={logger_name}, level={log_level}, "
        f"file={log_file if log_file else 'None'}"
    )
    
    return logger
