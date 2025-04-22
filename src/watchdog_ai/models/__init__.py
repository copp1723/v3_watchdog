"""
Watchdog AI Models Package.

This package contains all the data models and type definitions.
"""

from .insight import InsightResponse, InsightErrorType, BreakdownItem
from .intent import IntentSchema

__all__ = [
    'InsightResponse',
    'InsightErrorType',
    'IntentSchema',
    'BreakdownItem'
] 