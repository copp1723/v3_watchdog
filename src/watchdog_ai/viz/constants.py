"""
Constants for the visualization layer.

This module defines constants related to chart dimensions, colors, and other
visualization parameters to ensure consistency across all chart builders.
"""

from typing import Dict, Any, List

# ==== Chart Dimensions ====
# Standard desktop chart dimensions
DEFAULT_CHART_WIDTH: int = 700
DEFAULT_CHART_HEIGHT: int = 400

# Mobile dimensions and breakpoints
MOBILE_WIDTH_THRESHOLD: int = 768  # Bootstrap md breakpoint
DEFAULT_MOBILE_CHART_HEIGHT: int = 300
DEFAULT_MOBILE_CHART_WIDTH: int = 340

# Tablet dimensions
TABLET_WIDTH_THRESHOLD: int = 992  # Bootstrap lg breakpoint
TABLET_CHART_HEIGHT: int = 350

# Aspect ratios
LANDSCAPE_RATIO: float = 16/9  # Widescreen
PORTRAIT_RATIO: float = 3/4    # Vertical displays
SQUARE_RATIO: float = 1        # Equal width/height
GOLDEN_RATIO: float = 1.618    # Aesthetically pleasing ratio

# ==== Font Settings ====
# Font sizes for desktop
TITLE_FONT_SIZE: int = 16
AXIS_LABEL_FONT_SIZE: int = 14
TICK_LABEL_FONT_SIZE: int = 12
LEGEND_FONT_SIZE: int = 12

# Font sizes for mobile (slightly smaller)
MOBILE_TITLE_FONT_SIZE: int = 14
MOBILE_AXIS_LABEL_FONT_SIZE: int = 12
MOBILE_TICK_LABEL_FONT_SIZE: int = 10
MOBILE_LEGEND_FONT_SIZE: int = 10

# ==== Color Schemes ====
# Default color schemes based on Vega/Altair schemes
DEFAULT_COLOR_SCHEME: str = "tableau10"  # Good general-purpose scheme
SEQUENTIAL_COLOR_SCHEME: str = "blues"   # For quantitative data
DIVERGING_COLOR_SCHEME: str = "redblue"  # For data with midpoint
QUALITATIVE_COLOR_SCHEME: str = "category10"  # For categorical data

# Color scheme recommendations by chart type
COLOR_SCHEMES_BY_CHART: Dict[str, str] = {
    "bar": "tableau10",
    "line": "category10",
    "pie": "tableau10",
    "scatter": "tableau10",
    "heatmap": "viridis",
    "map": "blues"
}

# ==== Animation and Interaction ====
# Animation durations in milliseconds
DEFAULT_ANIMATION_DURATION: int = 500
FAST_ANIMATION_DURATION: int = 300
SLOW_ANIMATION_DURATION: int = 800

# Opacity settings
DEFAULT_OPACITY: float = 0.8
HIGHLIGHT_OPACITY: float = 1.0
INACTIVE_OPACITY: float = 0.3
BACKGROUND_OPACITY: float = 0.1

# ==== Chart Margins and Padding ====
# Standard chart margins
DEFAULT_CHART_PADDING: Dict[str, int] = {
    "top": 10,
    "right": 10,
    "bottom": 40,
    "left": 40
}

# Mobile chart margins (reduced for small screens)
MOBILE_CHART_PADDING: Dict[str, int] = {
    "top": 5,
    "right": 5,
    "bottom": 30,
    "left": 30
}

# ==== Category and Data Limits ====
# Max number of categories to display before grouping as "Other"
MAX_CATEGORIES: int = 10
MAX_PIE_CATEGORIES: int = 7  # Pie charts need fewer slices for readability
MAX_LINE_SERIES: int = 5     # Too many lines becomes unreadable

# ==== Performance Thresholds ====
# Maximum data points to display before sampling/aggregation
MAX_POINTS_MOBILE: int = 500   # Limit for mobile devices
MAX_POINTS_TABLET: int = 1000  # Limit for tablets
MAX_POINTS_DESKTOP: int = 2000 # Limit for desktop

# ==== Other Settings ====
# Tooltip settings
DEFAULT_TOOLTIP_FORMAT: Dict[str, str] = {
    "number": ",",        # Thousands separator
    "date": "%b %d, %Y",  # Jan 01, 2025
    "percent": ".1%"      # One decimal, with percent sign
}

