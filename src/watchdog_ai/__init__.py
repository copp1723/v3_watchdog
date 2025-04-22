"""
Watchdog AI package.

This package provides AI-powered dealership analytics.
"""

from .app import (
    initialize_session_state,
    process_uploaded_file,
    render_data_validation,
    render_insight_generation
)

from .insights import (
    IntentManager,
    Intent,
    TopMetricIntent,
    BottomMetricIntent,
    AverageMetricIntent,
    InsightEngine
)

from .ui import (
    ChatInterface,
    render_data_uploader,
    render_chat_tab
)

__all__ = [
    'initialize_session_state',
    'process_uploaded_file',
    'render_data_validation',
    'render_insight_generation',
    'IntentManager',
    'Intent',
    'TopMetricIntent',
    'BottomMetricIntent',
    'AverageMetricIntent',
    'InsightEngine',
    'ChatInterface',
    'render_data_uploader',
    'render_chat_tab'
]