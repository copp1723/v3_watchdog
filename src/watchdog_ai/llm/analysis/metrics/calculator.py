"""
Metrics calculation module for LLM engine.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

def calculate_metrics(df: pd.DataFrame, query_terms: List[str]) -> Dict[str, Any]:
    """
    Calculate relevant metrics based on query terms.
    
    Args:
        df: DataFrame to analyze
        query_terms: List of terms from the query
        
    Returns:
        Dictionary of calculated metrics
    """
    metric_results = {}
    
    try:
        # Define metric patterns to look for
        metric_patterns = {
            'sales': ['sales', 'revenue', 'volume'],
            'profit': ['profit', 'gross', 'margin'],
            'inventory': ['inventory', 'stock', 'vehicles'],
            'leads': ['leads', 'prospects', 'opportunities']
        }
        
        # Find relevant columns based on query terms
        query_set = set(term.lower() for term in query_terms)
        
        for metric_type, patterns in metric_patterns.items():
            if any(term in query_set for term in patterns):
                # Find matching columns
                matching_cols = [
                    col for col in df.columns 
                    if any(pattern in col.lower() for pattern in patterns)
                ]
                
                if matching_cols:
                    metric_results[metric_type] = _analyze_metric_columns(
                        df, matching_cols, metric_type
                    )
        
        return {
            "has_metrics": bool(metric_results),
            "metrics": metric_results,
            "query_coverage": len(metric_results) / len(metric_patterns) * 100
        }
        
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        return {
            "has_metrics": False,
            "error": str(e),
            "message": "Error during metrics calculation"
        }

def _analyze_metric_columns(df: pd.DataFrame,
                          columns: List[str],
                          metric_type: str) -> Dict[str, Any]:
    """
    Analyze specific metric columns with appropriate statistics.
    """
    results = {}
    
    for col in columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            stats = df[col].describe()
            
            result = {
                "current": float(stats['mean']),
                "min": float(stats['min']),
                "max": float(stats['max']),
                "std_dev": float(stats['std']),
                "sample_size": int(stats['count'])
            }
            
            # Add metric-specific analysis
            if metric_type == 'sales' or metric_type == 'profit':
                result.update(_analyze_financial_metric(df[col]))
            elif metric_type == 'inventory':
                result.update(_analyze_inventory_metric(df[col]))
            elif metric_type == 'leads':
                result.update(_analyze_lead_metric(df[col]))
                
            results[col] = result
    
    return results

def _analyze_financial_metric(series: pd.Series) -> Dict[str, Any]:
    """Add financial-specific analysis."""
    return {
        "total": float(series.sum()),
        "average": float(series.mean()),
        "growth_rate": float((series.iloc[-1] - series.iloc[0]) / series.iloc[0] * 100)
        if len(series) > 1 and series.iloc[0] != 0 else 0.0
    }

def _analyze_inventory_metric(series: pd.Series) -> Dict[str, Any]:
    """Add inventory-specific analysis."""
    return {
        "current_level": float(series.iloc[-1]) if len(series) > 0 else 0.0,
        "average_level": float(series.mean()),
        "turnover_rate": float(series.std() / series.mean() * 100)
        if series.mean() != 0 else 0.0
    }

def _analyze_lead_metric(series: pd.Series) -> Dict[str, Any]:
    """Add lead-specific analysis."""
    return {
        "total_leads": int(series.sum()),
        "daily_average": float(series.mean()),
        "conversion_potential": float(series.sum() * 0.2)  # Assumed 20% conversion
    }

