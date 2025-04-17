"""
UI Components for Watchdog AI.
"""

from .chat_interface import ChatInterface
from .insight_generator import InsightGenerator
from .system_connect import render_system_connect
from .data_upload import render_data_upload, render_validation_summary
from .flag_panel import render_flag_summary, render_flag_metrics
from .dashboard import (
    render_dashboard_from_insight,
    render_kpi_metrics,
    render_sales_dashboard,
    render_inventory_dashboard,
    render_interactive_chart
)

__all__ = [
    'ChatInterface',
    'InsightGenerator',
    'render_system_connect',
    'render_data_upload',
    'render_validation_summary',
    'render_flag_summary',
    'render_flag_metrics',
    'render_dashboard_from_insight',
    'render_kpi_metrics',
    'render_sales_dashboard',
    'render_inventory_dashboard',
    'render_interactive_chart'
]
