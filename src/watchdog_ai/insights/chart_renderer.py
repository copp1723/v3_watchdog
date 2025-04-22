"""
Robust chart rendering utilities for insights with comprehensive error handling.
"""

import altair as alt
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
from datetime import datetime
import traceback

logger = logging.getLogger(__name__)

class ChartError(Exception):
    """Custom exception for chart rendering operations."""
    pass

class ChartRenderer:
    """Handles chart creation for insight visualization with comprehensive validation."""
    
    # Chart configuration constants
    MAX_CATEGORIES = 50
    MIN_DATA_POINTS = 2
    MAX_STRING_LENGTH = 30
    CHART_TYPES = ["bar", "line", "scatter", "area", "pie"]
    COLOR_SCHEMES = {
        "categorical": "tableau10",
        "sequential": "viridis",
        "diverging": "spectral"
    }
    
    @staticmethod
    def create_chart(
        breakdown: List[Dict[str, Any]], 
        df: pd.DataFrame,
        chart_type: str = "bar",
        title: Optional[str] = None,
        color_scheme: Optional[str] = None,
        interactive: bool = True,
        width: Optional[int] = None,
        height: Optional[int] = None
    ) -> Tuple[Optional[alt.Chart], Dict[str, Any]]:
        """
        Create an appropriate chart with comprehensive validation and error handling.
        
        Args:
            breakdown: List of dictionaries containing breakdown data
            df: Source DataFrame used for additional context
            chart_type: Type of chart to create
            title: Optional chart title
            color_scheme: Optional color scheme
            interactive: Whether to make the chart interactive
            width: Optional chart width
            height: Optional chart height
            
        Returns:
            Tuple[Optional[alt.Chart], Dict[str, Any]]: (chart, metadata)
        """
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "chart_type": chart_type,
            "error": None
        }
        
        try:
            # Validate inputs
            if not isinstance(breakdown, list):
                raise ChartError(f"Breakdown must be a list, got {type(breakdown)}")
                
            if not isinstance(df, pd.DataFrame):
                raise ChartError(f"df must be a DataFrame, got {type(df)}")
                
            if not breakdown:
                raise ChartError("Breakdown data is empty")
                
            # Convert breakdown to DataFrame with validation
            try:
                breakdown_df = pd.DataFrame(breakdown)
                metadata["original_columns"] = breakdown_df.columns.tolist()
                
                # Validate minimum required columns
                required_cols = ["value"]
                category_cols = ["category", "group", "dimension"]
                has_category = any(col in breakdown_df.columns for col in category_cols)
                
                if not has_category:
                    raise ChartError(f"Missing category column. Need one of: {category_cols}")
                    
                if "value" not in breakdown_df.columns:
                    raise ChartError("Missing 'value' column")
                    
                # Identify key columns
                category_col = next((col for col in category_cols if col in breakdown_df.columns), None)
                
                # Clean and prepare data
                # Convert value column to numeric
                breakdown_df["value"] = pd.to_numeric(breakdown_df["value"], errors='coerce')
                breakdown_df = breakdown_df.dropna(subset=["value"])
                
                if breakdown_df.empty:
                    raise ChartError("No valid numeric values found")
                    
                # Truncate long category names
                if category_col:
                    breakdown_df[category_col] = breakdown_df[category_col].astype(str).apply(
                        lambda x: x[:ChartRenderer.MAX_STRING_LENGTH] + '...' if len(x) > ChartRenderer.MAX_STRING_LENGTH else x
                    )
                
                # Limit number of categories
                if len(breakdown_df) > ChartRenderer.MAX_CATEGORIES:
                    logger.warning(f"Too many categories ({len(breakdown_df)}), limiting to {ChartRenderer.MAX_CATEGORIES}")
                    breakdown_df = breakdown_df.nlargest(ChartRenderer.MAX_CATEGORIES, "value")
                
                # Update metadata
                metadata.update({
                    "row_count": len(breakdown_df),
                    "value_range": {
                        "min": float(breakdown_df["value"].min()),
                        "max": float(breakdown_df["value"].max()),
                        "mean": float(breakdown_df["value"].mean())
                    }
                })
                
                # Validate chart type
                chart_type = chart_type.lower()
                if chart_type not in ChartRenderer.CHART_TYPES:
                    logger.warning(f"Invalid chart type: {chart_type}, defaulting to bar")
                    chart_type = "bar"
                
                # Set up chart configuration
                config = {
                    "width": width or "container",
                    "height": height or 400,
                    "title": title or f"{category_col.replace('_', ' ').title()} Distribution",
                    "color_scheme": color_scheme or ChartRenderer.COLOR_SCHEMES["categorical"]
                }
                
                # Create base chart
                base = alt.Chart(breakdown_df).properties(
                    width=config["width"],
                    height=config["height"],
                    title=config["title"]
                )
                
                # Create chart based on type
                if chart_type == "bar":
                    chart = base.mark_bar().encode(
                        x=alt.X("value:Q", title="Value"),
                        y=alt.Y(f"{category_col}:N", 
                               title=category_col.replace('_', ' ').title(),
                               sort="-x"),
                        tooltip=[
                            alt.Tooltip(f"{category_col}:N", title=category_col.replace('_', ' ').title()),
                            alt.Tooltip("value:Q", format=".2f")
                        ]
                    )
                elif chart_type == "pie":
                    chart = base.mark_arc().encode(
                        theta="value:Q",
                        color=alt.Color(f"{category_col}:N", scale=alt.Scale(scheme=config["color_scheme"])),
                        tooltip=[
                            alt.Tooltip(f"{category_col}:N", title=category_col.replace('_', ' ').title()),
                            alt.Tooltip("value:Q", format=".2f")
                        ]
                    )
                else:
                    # Add more chart types as needed
                    chart = base.mark_bar()  # Default to bar
                
                # Add interactivity if requested
                if interactive:
                    selection = alt.selection_single(
                        on="mouseover",
                        nearest=True,
                        empty="none"
                    )
                    chart = chart.add_selection(selection)
                    
                # Update metadata
                metadata.update({
                    "success": True,
                    "chart_type": chart_type,
                    "config": config
                })
                
                return chart, metadata
                
            except Exception as e:
                raise ChartError(f"Error creating chart: {str(e)}")
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Chart creation error: {traceback.format_exc()}")
            metadata["error"] = error_msg
            return None, metadata
            
    @staticmethod
    def validate_chart_data(data: Union[pd.DataFrame, List[Dict[str, Any]]]) -> Tuple[bool, List[str]]:
        """
        Validate data for chart creation.
        
        Args:
            data: Data to validate
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_issues)
        """
        issues = []
        
        try:
            # Convert list to DataFrame if needed
            if isinstance(data, list):
                if not data:
                    issues.append("Empty data list")
                    return False, issues
                data = pd.DataFrame(data)
            
            # Validate DataFrame
            if not isinstance(data, pd.DataFrame):
                issues.append(f"Invalid data type: {type(data)}")
                return False, issues
                
            if data.empty:
                issues.append("Empty DataFrame")
                return False, issues
                
            # Check for required columns
            if "value" not in data.columns:
                issues.append("Missing 'value' column")
                
            category_cols = ["category", "group", "dimension"]
            if not any(col in data.columns for col in category_cols):
                issues.append(f"Missing category column (need one of {category_cols})")
                
            # Validate value column
            if "value" in data.columns:
                if not pd.to_numeric(data["value"], errors='coerce').notna().any():
                    issues.append("No valid numeric values in 'value' column")
                    
            # Check for too many categories
            if len(data) > ChartRenderer.MAX_CATEGORIES:
                issues.append(f"Too many categories ({len(data)})")
                
            return len(issues) == 0, issues
            
        except Exception as e:
            issues.append(f"Validation error: {str(e)}")
            return False, issues 