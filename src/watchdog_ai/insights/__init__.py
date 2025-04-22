"""
Insight generation and processing modules.
"""

from .intent_manager import IntentManager
from .intents import Intent, TopMetricIntent, BottomMetricIntent, AverageMetricIntent
from .engine import InsightEngine

__all__ = [
    'IntentManager',
    'Intent',
    'TopMetricIntent',
    'BottomMetricIntent',
    'AverageMetricIntent',
    'InsightEngine'
]