"""
UI package for Watchdog AI.

This package contains UI components and utilities for building
interactive dashboards and visualizations with Watchdog AI.
"""

# Import UI components for easy access
from .components import (
    # Classes
    ChatInterface,
    InsightGenerator,
    
    # Functions
    render_system_connect,
    render_data_upload,
    render_validation_summary,
    render_flag_summary,
    render_flag_metrics,
    render_dashboard_from_insight,
    render_kpi_metrics,
    render_sales_dashboard,
    render_inventory_dashboard,
    render_interactive_chart
)

# Import pages
from .pages import modern_analyst_ui

# Define __all__ to control wildcard imports
__all__ = [
    # Classes
    'ChatInterface',
    'InsightGenerator',
    
    # Functions
    'render_system_connect',
    'render_data_upload',
    'render_validation_summary',
    'render_flag_summary',
    'render_flag_metrics',
    'render_dashboard_from_insight',
    'render_kpi_metrics',
    'render_sales_dashboard',
    'render_inventory_dashboard',
    'render_interactive_chart',
    
    # Pages
    'modern_analyst_ui'
]
