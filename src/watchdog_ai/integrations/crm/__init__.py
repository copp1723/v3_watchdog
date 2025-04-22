"""
CRM Integration Package

This package provides adapters for integrating with various CRM systems.
"""

__version__ = '0.1.0'
__author__ = 'Watchdog AI Team'

# Export base interface and exceptions
from .base import (
    BaseCRMAdapter,
    AuthenticationError,
    DataFetchError,
    DataPushError
)

# Export implementations
from .nova_act import NovaActAdapter

# Public API
__all__ = [
    'BaseCRMAdapter',
    'NovaActAdapter',
    'AuthenticationError',
    'DataFetchError',
    'DataPushError',
]
