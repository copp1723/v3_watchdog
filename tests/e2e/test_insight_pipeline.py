"""
End-to-end tests for insight pipeline.
"""

import pytest
from src.watchdog_ai.insights.engine import InsightEngine

def test_insight_engine():
    """Test insight engine initialization."""
    engine = InsightEngine()
    assert engine is not None