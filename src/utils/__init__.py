"""
Utils Package for Watchdog AI.

This package contains utility functions and helpers used throughout the application.
"""

from .openai_client import get_openai_client, generate_completion

__all__ = [
    'get_openai_client',
    'generate_completion'
]