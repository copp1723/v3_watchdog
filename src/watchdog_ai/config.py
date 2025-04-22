
"""
Configuration module for Watchdog AI.

DEPRECATED: This module is deprecated and will be removed in v4.0.0.
            Please use 'watchdog_ai.core.config' instead.
"""

import os
import warnings

warnings.warn(
    "The 'watchdog_ai.config' module is deprecated and will be removed in v4.0.0. "
    "Please use 'watchdog_ai.core.config' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import from new location to maintain backward compatibility
try:
    from watchdog_ai.core.config import SessionKeys
    from watchdog_ai.core.config.secure import config
    # Get OPENAI_API_KEY from config for backward compatibility
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
except ImportError:
    # Fall back to original implementation if imports fail (during migration)
    class SessionKeys:
        """Session state keys."""
        CONVERSATION_HISTORY = "conversation_history"
        VALIDATED_DATA = "validated_data"
        CONVERSATION_MANAGER = "conversation_manager"
        VALIDATION_SUMMARY = "validation_summary"

    # OpenAI API key from environment variable
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# For backward compatibility, we keep all exports
__all__ = [
    'SessionKeys',
    'OPENAI_API_KEY'
]

"""
Configuration module for Watchdog AI.
"""

import os

class SessionKeys:
    """Session state keys."""
    CONVERSATION_HISTORY = "conversation_history"
    VALIDATED_DATA = "validated_data"
    CONVERSATION_MANAGER = "conversation_manager"
    VALIDATION_SUMMARY = "validation_summary"

# OpenAI API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")