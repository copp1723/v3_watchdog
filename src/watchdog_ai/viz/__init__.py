"""
Watchdog AI Visualization Layer.

This package provides encapsulated chart builders with mobile-responsive sizing
for Streamlit applications. Each chart type is available as a standalone function 
with a consistent API.

Example:
    ```python
    import streamlit as st
    from watchdog_ai.viz import sales_trend_plot, category_distribution_plot
    
    # Get container width for responsive charts
    container_width = st.container().width
    
    # Create sales trend chart with responsive sizing
    sales_df = get_sales_data()  # Your data loading function
    sales_chart = sales_trend_plot(sales_df, container_width=container_width)
    st.altair_chart(sales_chart)
    ```
"""

__version__ = "0.1.0"
__author__ = "Watchdog AI Team"

from typing import List, Dict, Optional, Any, Union, Tuple

# Import utilities
from .utils import (
    get_responsive_dimensions,
    make_chart_responsive,
    get_mobile_optimized_chart_config,
    configure_responsive_chart
)

# Import key constants for public use
from .constants import (
    # Chart dimensions
    DEFAULT_CHART_WIDTH,
    DEFAULT_CHART_HEIGHT,
    MOBILE_WIDTH_THRESHOLD,
    
    # Color schemes
    DEFAULT_COLOR_SCHEME,
    SEQUENTIAL_COLOR_SCHEME,
    DIVERGING_COLOR_SCHEME,
    
    # Aspect ratios
    LANDSCAPE_RATIO,
    PORTRAIT_RATIO,
    SQUARE_RATIO,
    
    # Max categories
    MAX_CATEGORIES
)

# Chart builders
from .builders.sales_trend_plot import sales_trend_plot
from .builders.category_distribution_plot import category_distribution_plot

# TODO: Add future chart builders as they are implemented
# from .builders.correlation_matrix import correlation_matrix
# from .builders.geographic_map import geographic_map
# from .builders.time_comparison import time_comparison

# Export everything needed in the public API
__all__ = [
    # Version and metadata
    '__version__',
    '__author__',
    
    # Utilities
    'get_responsive_dimensions',
    'make_chart_responsive',
    'get_mobile_optimized_chart_config',
    'configure_responsive_chart',
    
    # Constants
    'DEFAULT_CHART_WIDTH',
    'DEFAULT_CHART_HEIGHT',
    'MOBILE_WIDTH_THRESHOLD',
    'DEFAULT_COLOR_SCHEME',
    'SEQUENTIAL_COLOR_SCHEME',
    'DIVERGING_COLOR_SCHEME',
    'LANDSCAPE_RATIO',
    'PORTRAIT_RATIO',
    'SQUARE_RATIO',
    'MAX_CATEGORIES',
    
    # Chart builders
    'sales_trend_plot',
    'category_distribution_plot',
    
    # TODO: Add future chart builders to __all__ as they are implemented
    # 'correlation_matrix',
    # 'geographic_map',
    # 'time_comparison',
]

# Type aliases for public use
ChartDimensions = Dict[str, Union[int, str]]
ColorScheme = str
ResponsiveConfig = Dict[str, Any]

