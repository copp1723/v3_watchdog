"""
Category distribution visualization chart builder.

This module provides functions to create category distribution charts (bar, pie),
supporting responsive sizing for both desktop and mobile views.
"""

import pandas as pd
import altair as alt
import logging
from typing import Dict, Optional, Any, Union, List, Literal

from watchdog_ai.viz.utils import get_responsive_dimensions
from watchdog_ai.viz.constants import (
    DEFAULT_COLOR_SCHEME,
    MAX_CATEGORIES,
    DEFAULT_OPACITY
)

logger = logging.getLogger(__name__)

def category_distribution_plot(
    df: pd.DataFrame,
    category_column: Optional[str] = None,
    value_column: Optional[str] = None,
    chart_type: Literal['bar', 'pie'] = 'bar',
    title: Optional[str] = "Category Distribution",
    sort_by: Literal['value', 'category', 'none'] = 'value',
    sort_order: Literal['ascending', 'descending'] = 'descending',
    color_scheme: str = DEFAULT_COLOR_SCHEME,
    container_width: Optional[int] = None,
    height: Optional[int] = None,
    max_categories: int = MAX_CATEGORIES,
    tooltip: Union[bool, List[str]] = True,
) -> alt.Chart:
    """
    Create a category distribution chart (bar or pie).
    
    Args:
        df: DataFrame containing category data
        category_column: Column name with categories (auto-detected if None)
        value_column: Column name with values (auto-detected if None)
        chart_type: 'bar' or 'pie' chart type
        title: Chart title
        sort_by: How to sort the categories ('value', 'category', or 'none')
        sort_order: Sort direction ('ascending' or 'descending')
        color_scheme: Color scheme for categories
        container_width: Width of the container for responsive sizing
        height: Explicit chart height
        max_categories: Maximum number of categories to display (excess grouped into 'Other')
        tooltip: True for default tooltip, list of column names for custom tooltip
        
    Returns:
        Altair Chart object representing the category distribution
        
    Raises:
        ValueError: If the DataFrame is empty or required columns cannot be determined
    """
    try:
        # Validate input data
        if df.empty:
            raise ValueError("DataFrame is empty")
            
        # Make a copy to avoid modifying the original
        plot_df = df.copy()
        
        # Determine category column if not specified
        cat_col = category_column or _find_category_column(plot_df)
        if not cat_col:
            raise ValueError("Could not determine category column. Please specify category_column.")
            
        # Determine value column if not specified
        val_col = value_column or _find_numeric_column(plot_df)
        if not val_col:
            raise ValueError("Could not determine value column. Please specify value_column.")
            
        # Group and aggregate data
        if cat_col and val_col:
            # Group by category and sum values
            agg_df = plot_df.groupby(cat_col)[val_col].sum().reset_index()
            
            # Sort the data
            if sort_by == 'value':
                agg_df = agg_df.sort_values(
                    by=val_col, 
                    ascending=(sort_order == 'ascending')
                )
            elif sort_by == 'category':
                agg_df = agg_df.sort_values(
                    by=cat_col, 
                    ascending=(sort_order == 'ascending')
                )
            
            # Limit to max_categories
            if len(agg_df) > max_categories:
                # Keep top categories and group others
                top_df = agg_df.iloc[:max_categories-1]
                other_value = agg_df.iloc[max_categories-1:][val_col].sum()
                other_row = pd.DataFrame({cat_col: ['Other'], val_col: [other_value]})
                agg_df = pd.concat([top_df, other_row], ignore_index=True)
        else:
            raise ValueError("Both category and value columns are required")
            
        # Create base chart
        base = alt.Chart(agg_df)
        
        # Create tooltip encoding
        tooltip_encoding = _get_tooltip_encoding(agg_df, cat_col, val_col, tooltip)
        
        # Generate the appropriate chart type
        if chart_type == 'pie':
            # Create pie chart
            chart = base.mark_arc(opacity=DEFAULT_OPACITY).encode(
                theta=alt.Theta(f"{val_col}:Q"),
                color=alt.Color(
                    f"{cat_col}:N", 
                    scale=alt.Scale(scheme=color_scheme),
                    legend=alt.Legend(title=cat_col)
                ),
                tooltip=tooltip_encoding
            )
        else:
            # Create bar chart (default)
            chart = base.mark_bar(opacity=DEFAULT_OPACITY).encode(
                x=alt.X(f"{cat_col}:N", title=cat_col, sort=None),  # Use pre-sorted data
                y=alt.Y(f"{val_col}:Q", title=val_col),
                color=alt.Color(
                    f"{cat_col}:N",
                    scale=alt.Scale(scheme=color_scheme),
                    legend=alt.Legend(title=cat_col)
                ),
                tooltip=tooltip_encoding
            )
        
        # Set chart title
        chart = chart.properties(title=title)
        
        # Add responsive sizing
        responsive_props = get_responsive_dimensions(
            container_width=container_width,
            height=height
        )
        if responsive_props:
            chart = chart.properties(**responsive_props)
            
        return chart
        
    except Exception as e:
        logger.error(f"Error creating category distribution chart: {str(e)}")
        return _create_error_chart(str(e))

def _find_category_column(df: pd.DataFrame) -> Optional[str]:
    """Find appropriate column for categories."""
    # Prefer categorical/string columns for categories
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_cols:
        # Look for columns with category-related names
        category_indicators = ['category', 'type', 'group', 'segment', 'class', 'name', 'product']
        cat_cols = [col for col in categorical_cols if any(indicator in col.lower() for indicator in category_indicators)]
        if cat_cols:
            return cat_cols[0]
        return categorical_cols[0]
    
    # If no categorical columns, use the first column
    return df.columns[0] if len(df.columns) > 0 else None

def _find_numeric_column(df: pd.DataFrame) -> Optional[str]:
    """Find appropriate numeric column for values."""
    # Look for value-related column names first
    value_indicators = ['value', 'count', 'amount', 'quantity', 'total', 'sum', 'sales']
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Find numeric columns with value-related names
    value_cols = [col for col in numeric_cols if any(indicator in col.lower() for indicator in value_indicators)]
    if value_cols:
        return value_cols[0]
        
    # If no value-related columns found, return the first numeric column
    if numeric_cols:
        return numeric_cols[0]
        
    # If no numeric columns at all, return None
    return None

def _get_tooltip_encoding(
    df: pd.DataFrame, 
    category_col: str, 
    value_col: str, 
    tooltip: Union[bool, List[str]] = True
) -> Union[List[alt.Tooltip], alt.Tooltip]:
    """
    Create tooltip encoding configuration for the chart.
    
    Args:
        df: DataFrame with the data
        category_col: Category column name
        value_col: Value column name
        tooltip: True for default tooltips, or list of column names for custom tooltips
        
    Returns:
        Altair tooltip encoding configuration
    """
    # If tooltip is False, return minimal tooltip
    if tooltip is False:
        return [alt.Tooltip(category_col), alt.Tooltip(value_col)]
        
    # If tooltip is a list of columns, use those
    if isinstance(tooltip, list):
        return [
            alt.Tooltip(c) for c in tooltip 
            if c in df.columns
        ]
    
    # Default tooltip (when tooltip is True)
    return [alt.Tooltip(category_col), alt.Tooltip(value_col, format=',')]

def _create_error_chart

