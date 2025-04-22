"""
Watchdog AI Configuration Interface.

This module provides a unified configuration system for Watchdog AI,
centralizing environment variables, secrets, and logging configuration.
"""

import warnings

# Import secure config components
from .secure import (
    SecureConfig, 
    config, 
    ConfigurationError,
    MIN_CONFIDENCE_TO_AUTOMAP,
    DROP_UNMAPPED_COLUMNS
)

# Import logging components
from .logging import (
    get_logger,
    setup_logging,
    configure_logger,
    LOG_LEVELS,
    DEFAULT_LOG_FORMAT,
    DETAILED_LOG_FORMAT,
    SIMPLE_LOG_FORMAT
)

# Session state keys (from the old simple config)
class SessionKeys:
    """Session state keys."""
    CONVERSATION_HISTORY = "conversation_history"
    VALIDATED_DATA = "validated_data"
    CONVERSATION_MANAGER = "conversation_manager"
    VALIDATION_SUMMARY = "validation_summary"

# Public API
__all__ = [
    # Classes
    'SecureConfig',
    'ConfigurationError',
    'SessionKeys',
    
    # Instances
    'config',
    
    # Logging functions
    'get_logger',
    'setup_logging',
    'configure_logger',
    
    # Constants
    'LOG_LEVELS',
    'DEFAULT_LOG_FORMAT',
    'DETAILED_LOG_FORMAT',
    'SIMPLE_LOG_FORMAT',
    'MIN_CONFIDENCE_TO_AUTOMAP',
    'DROP_UNMAPPED_COLUMNS'
]

