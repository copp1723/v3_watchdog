"""
UI Components module for Watchdog AI.

This module contains reusable UI components for building
Watchdog AI dashboards and visualizations.
"""

# Import components for easy access
from .flag_panel import (
    render_flag_summary,
    render_flag_metrics,
    highlight_flagged_rows
)

from .data_upload import (
    render_file_upload,
    render_sample_data_option
)
