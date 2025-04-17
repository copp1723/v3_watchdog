"""
Direct data analysis functions for common insight queries.
"""

import pandas as pd
from typing import Dict, Any, Tuple
import numpy as np

def analyze_negative_profits(df: pd.DataFrame, gross_col: str) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Analyze negative profit transactions.
    
    Args:
        df: DataFrame to analyze
        gross_col: Name of the gross profit column
        
    Returns:
        Tuple of (metrics dict, visualization DataFrame)
    """
    # Clean and convert gross profit values
    df[gross_col] = pd.to_numeric(df[gross_col].astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce')
    
    # Find negative profits
    negative_profits = df[df[gross_col] < 0]
    count = len(negative_profits)
    total_loss = negative_profits[gross_col].sum()
    avg_loss = total_loss / count if count > 0 else 0
    
    # Calculate percentage
    total_deals = len(df)
    percentage = (count / total_deals) * 100 if total_deals > 0 else 0
    
    # Prepare metrics
    metrics = {
        "count": count,
        "total_loss": total_loss,
        "avg_loss": avg_loss,
        "percentage": percentage,
        "total_deals": total_deals
    }
    
    # Prepare visualization data
    viz_data = pd.DataFrame({
        'Category': ['Negative Profit Deals', 'Other Deals'],
        'Count': [count, len(df) - count]
    })
    
    return metrics, viz_data

def analyze_by_lead_source(df: pd.DataFrame, gross_col: str, source_col: str) -> Dict[str, Any]:
    """
    Analyze performance by lead source.
    
    Args:
        df: DataFrame to analyze
        gross_col: Name of the gross profit column
        source_col: Name of the lead source column
        
    Returns:
        Dictionary of analysis results
    """
    # Clean gross profit values
    df[gross_col] = pd.to_numeric(df[gross_col].astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce')
    
    # Group by lead source
    grouped = df.groupby(source_col).agg({
        gross_col: ['count', 'sum', 'mean'],
    }).round(2)
    
    # Flatten column names
    grouped.columns = [f"{col[0]}_{col[1]}" for col in grouped.columns]
    grouped = grouped.reset_index()
    
    return grouped.to_dict('records')

def find_metric_column(columns: list[str], metric_type: str) -> Optional[str]:
    """Find the most likely column for a given metric type."""
    metric_keywords = {
        'gross': ['gross', 'profit', 'margin'],
        'price': ['price', 'cost', 'amount'],
        'revenue': ['revenue', 'sales', 'income']
    }
    
    keywords = metric_keywords.get(metric_type, [metric_type])
    for col in columns:
        if any(keyword in col.lower() for keyword in keywords):
            return col
    return None

def find_category_column(columns: list[str], category_type: str) -> Optional[str]:
    """Find the most likely column for a given category type."""
    category_keywords = {
        'source': ['source', 'lead', 'channel'],
        'rep': ['rep', 'sales', 'person'],
        'vehicle': ['make', 'model', 'vehicle']
    }
    
    keywords = category_keywords.get(category_type, [category_type])
    for col in columns:
        if any(keyword in col.lower() for keyword in keywords):
            return col
    return None