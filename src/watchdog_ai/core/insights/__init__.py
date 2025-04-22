"""
Core insight processing components for Watchdog AI.

This package provides the foundational classes and interfaces for:
- Insight generation and processing
- Conversation management
- Insight metadata handling
- Standardized insight structures
"""

from .metadata import InsightMetadata
from .base import InsightBase, InsightFormatter
from .card import InsightCard
from .conversation import ConversationManager

__all__ = [
    'InsightMetadata',
    'InsightBase',
    'InsightFormatter',
    'InsightCard',
    'ConversationManager'
]

