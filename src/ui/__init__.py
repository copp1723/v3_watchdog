"""
UI package for Watchdog AI.

This package contains UI components and utilities for building
interactive dashboards and visualizations with Watchdog AI.
"""

# Import UI components for easy access
from .components.flag_panel import (
    render_flag_summary,
    render_flag_metrics,
    highlight_flagged_rows
)

# Import the new data upload component
from .components.data_upload import (
    render_file_upload,
    render_sample_data_option
)
