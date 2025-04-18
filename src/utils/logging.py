"""
Logging configuration for Watchdog AI.
"""

import logging
import os
from typing import Optional

def get_logger(name: str) -> logging.Logger:
    """Get a configured logger instance."""
    logger = logging.getLogger(name)
    
    # Only configure if no handlers exist
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Create console handler
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console.setFormatter(formatter)
        
        # Add handler
        logger.addHandler(console)
    
    return logger

# Create audit logger
audit_logger = get_logger('audit')