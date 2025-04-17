"""
Logging configuration for Nova Act integration.
"""

import logging
import logging.handlers
import os
from typing import Optional
from .constants import LOGGING

class NovaActLogger:
    """Configures and manages logging for Nova Act operations."""
    
    _instance: Optional['NovaActLogger'] = None
    
    def __new__(cls):
        """Ensure singleton pattern for logger configuration."""
        if cls._instance is None:
            cls._instance = super(NovaActLogger, cls).__new__(cls)
            cls._instance._configure_logging()
        return cls._instance
    
    def _configure_logging(self):
        """Configure the logging system."""
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(LOGGING["file_path"])
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Configure root logger
        logger = logging.getLogger('nova_act')
        logger.setLevel(LOGGING["level"])
        
        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            LOGGING["file_path"],
            maxBytes=LOGGING["max_size"],
            backupCount=LOGGING["backup_count"]
        )
        file_handler.setFormatter(logging.Formatter(
            fmt=LOGGING["format"],
            datefmt=LOGGING["date_format"]
        ))
        logger.addHandler(file_handler)
        
        # Console handler for development
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(
            fmt=LOGGING["format"],
            datefmt=LOGGING["date_format"]
        ))
        logger.addHandler(console_handler)
        
        self.logger = logger
    
    @classmethod
    def get_logger(cls) -> logging.Logger:
        """Get the configured logger instance."""
        if cls._instance is None:
            cls()
        return cls._instance.logger

def log_error(error: Exception, vendor: str, operation: str):
    """
    Log an error with context.
    
    Args:
        error: The exception that occurred
        vendor: The vendor system where the error occurred
        operation: The operation being performed
    """
    logger = NovaActLogger.get_logger()
    logger.error(
        f"Error during {operation} for {vendor}: {str(error)}",
        exc_info=True,
        extra={
            'vendor': vendor,
            'operation': operation
        }
    )

def log_warning(message: str, vendor: str, operation: str):
    """
    Log a warning with context.
    
    Args:
        message: The warning message
        vendor: The vendor system
        operation: The operation being performed
    """
    logger = NovaActLogger.get_logger()
    logger.warning(
        f"{message}",
        extra={
            'vendor': vendor,
            'operation': operation
        }
    )

def log_info(message: str, vendor: str, operation: str):
    """
    Log an info message with context.
    
    Args:
        message: The info message
        vendor: The vendor system
        operation: The operation being performed
    """
    logger = NovaActLogger.get_logger()
    logger.info(
        f"{message}",
        extra={
            'vendor': vendor,
            'operation': operation
        }
    )