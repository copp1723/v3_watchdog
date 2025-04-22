"""
Period change analysis module for LLM engine.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def analyze_period_changes(df: pd.DataFrame,
                         metric_column: str,
                         date_column: Optional[str] = None,
                         period: str = 'month') -> Dict[str, Any]:
    """
    Analyze changes in metrics over different time periods.
    
    Args:
        df: DataFrame to analyze
        metric_column: Column containing the metric values
        date_column: Column containing dates (optional)
        period: Analysis period ('day', 'week', 'month', 'quarter', 'year')
        
    Returns:
        Dictionary containing period change analysis
    """
    try:
        # Ensure we have the metric column
        if metric_column not in df.columns:
            return {
                "has_changes": False,
                "message": f"Metric column '{metric_column}' not found"
            }
            
        # Handle date column
        if date_column:
            if date_column not in df.columns:
                return {
                    "has_changes": False,
                    "message": f"Date column '{date_column}' not found"
                }
            try:
                df = df.copy()
                df[date_column] = pd.to_datetime(df[date_column])
            except Exception as e:
                return {
                    "has_changes": False,
                    "message": f"Error converting date column: {str(e)}"
                }
        else:
            # Try to find a date column
            date_columns = [col for col in df.columns 
                          if any(term in col.lower() 
                                for term in ['date', 'time', 'day', 'month', 'year'])]
            if date_columns:
                date_column = date_columns[0]
                try:
                    df = df.copy()
                    df[date_column] = pd.to_datetime(df[date_column])
                except:
                    date_column = None
        
        # Calculate period changes
        if date_column:
            # Time-based analysis
            changes = _analyze_time_based_changes(df, metric_column, date_column, period)
        else:
            # Sequential analysis
            changes = _analyze_sequential_changes(df[metric_column])
            
        return changes
        
    except Exception as e:
        logger.error(f"Error analyzing period changes: {str(e)}")
        return {
            "has_changes": False,
            "error": str(e),
            "message": "Error during period change analysis"
        }

def _analyze_time_based_changes(df: pd.DataFrame,
                              metric_column: str,
                              date_column: str,
                              period: str) -> Dict[str, Any]:
    """Analyze changes based on time periods."""
    # Sort by date
    df = df.sort_values(date_column)
    
    # Group by period
    if period == 'day':
        grouped = df.groupby(df[date_column].dt.date)[metric_column].sum()
    elif period == 'week':
        grouped = df.groupby(pd.Grouper(key=date_column, freq='W'))[metric_column].sum()
    elif period == 'month':
        grouped = df.groupby(pd.Grouper(key=date_column, freq='M'))[metric_column].sum()
    elif period == 'quarter':
        grouped = df.groupby(pd.Grouper(key=date_column, freq='Q'))[metric_column].sum()
    else:  # year
        grouped = df.groupby(pd.Grouper(key=date_column, freq='Y'))[metric_column].sum()
    
    # Calculate changes
    changes = grouped.pct_change() * 100
    
    # Calculate period statistics
    period_stats = {
        "total_periods": len(grouped),
        "increases": sum(changes > 0),
        "decreases": sum(changes < 0),
        "no_change": sum(changes == 0),
        "average_change": float(changes.mean()),
        "max_increase": float(changes.max()) if not pd.isna(changes.max()) else 0.0,
        "max_decrease": float(changes.min()) if not pd.isna(changes.min()) else 0.0
    }
    
    # Identify trends
    trend_direction = "increasing" if period_stats["average_change"] > 0 else "decreasing"
    trend_strength = abs(period_stats["average_change"])
    if trend_strength > 10:
        trend_magnitude = "strong"
    elif trend_strength > 5:
        trend_magnitude = "moderate"
    else:
        trend_magnitude = "weak"
    
    return {
        "has_changes": True,
        "period_type": period,
        "total_periods": period_stats["total_periods"],
        "changes": {
            "first_value": float(grouped.iloc[0]),
            "last_value": float(grouped.iloc[-1]),
            "total_change": float((grouped.iloc[-1] - grouped.iloc[0]) / grouped.iloc[0] * 100)
            if grouped.iloc[0] != 0 else 0.0
        },
        "period_stats": period_stats,
        "trend": {
            "direction": trend_direction,
            "magnitude": trend_magnitude,
            "strength": float(trend_strength)
        }
    }

def _analyze_sequential_changes(series: pd.Series) -> Dict[str, Any]:
    """Analyze changes in sequential order (no dates)."""
    changes = series.pct_change() * 100
    
    sequential_stats = {
        "total_points": len(series),
        "increases": int(sum(changes > 0)),
        "decreases": int(sum(changes < 0)),
        "no_change": int(sum(changes == 0)),
        "average_change": float(changes.mean()),
        "max_increase": float(changes.max()) if not pd.isna(changes.max()) else 0.0,
        "max_decrease": float(changes.min()) if not pd.isna(changes.min()) else 0.0
    }
    
    return {
        "has_changes": True,
        "analysis_type": "sequential",
        "total_points": sequential_stats["total_points"],
        "changes": {
            "first_value": float(series.iloc[0]),
            "last_value": float(series.iloc[-1]),
            "total_change": float((series.iloc[-1] - series.iloc[0]) / series.iloc[0] * 100)
            if series.iloc[0] != 0 else 0.0
        },
        "sequential_stats": sequential_stats,
        "volatility": {
            "std_dev": float(changes.std()),
            "range": float(changes.max() - changes.min())
            if not pd.isna(changes.max()) and not pd.isna(changes.min()) else 0.0
        }
    }

