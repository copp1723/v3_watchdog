"""
Insight card component for rendering analysis results.

DEPRECATED: This module is deprecated and will be removed in v4.0.0.
Please use 'watchdog_ai.core.insights.card' instead.
"""

import warnings
import streamlit as st
from typing import Dict, Any, Union

warnings.warn(
    "The 'watchdog_ai.insight_card' module is deprecated and will be removed in v4.0.0. "
    "Please use 'watchdog_ai.core.insights.card' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import from new location for backward compatibility
from watchdog_ai.core.insights.card import InsightCard

def render_insight_card(insight: Union[Dict[str, Any], object]) -> None:
    """
    Render an insight card with analysis results.
    
    DEPRECATED: Please use watchdog_ai.core.insights.card.InsightCard.render() instead.
    
    Args:
        insight: Dictionary or InsightResponse object containing insight data
    """
    warnings.warn(
        "render_insight_card() is deprecated. "
        "Please use watchdog_ai.core.insights.card.InsightCard.render() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Delegate to new implementation
    InsightCard.render(insight)
