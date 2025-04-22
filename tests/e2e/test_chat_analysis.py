"""
End-to-end tests for chat analysis.
"""

import pytest
from src.watchdog_ai.insights.intent_manager import IntentManager

def test_intent_manager():
    """Test intent manager initialization."""
    manager = IntentManager()
    assert manager is not None