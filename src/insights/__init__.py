"""
Intent-based insight generation system.
"""

from .intents import (
    TopMetricIntent,
    BottomMetricIntent,
    AverageMetricIntent,
    CountMetricIntent,
    HighestCountIntent,
    NegativeProfitIntent
)
from .models import InsightResult

__all__ = [
    'TopMetricIntent',
    'BottomMetricIntent',
    'AverageMetricIntent',
    'CountMetricIntent',
    'HighestCountIntent',
    'NegativeProfitIntent',
    'InsightResult'
]