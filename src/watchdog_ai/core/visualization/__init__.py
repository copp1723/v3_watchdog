"""
Visualization components for Watchdog AI.

This module provides a unified interface for chart generation and rendering,
consolidating functionality from various chart utilities throughout the codebase.
"""

from .chart_base import ChartBase, ChartConfig, ChartType
from .chart_renderer import render_chart, extract_chart_data, build_chart
from .chart_utils import build_chart_data, extract_chart_data_from_llm_response

__all__ = [
    'ChartBase',
    'ChartConfig',
    'ChartType',
    'render_chart',
    'extract_chart_data',
    'build_chart',
    'build_chart_data',
    'extract_chart_data_from_llm_response'
]

