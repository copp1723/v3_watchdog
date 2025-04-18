"""
Statistical baseline calculations for Watchdog AI insights.

This module provides core statistical functions for analyzing inventory aging
and sales performance metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
import logging

# Configure logger
logger = logging.getLogger(__name__)

def calculate_inventory_aging_stats(
    df: pd.DataFrame,
    days_in_stock_col: str = 'days_in_stock',
    model_col: Optional[str] = 'model',
    trim_col: Optional[str] = 'trim'
) -> Dict[str, Any]:
    """
    Calculate inventory aging statistics with optional model/trim segmentation.
    
    Args:
        df: DataFrame containing inventory data
        days_in_stock_col: Column name for days in stock
        model_col: Optional column name for vehicle model
        trim_col: Optional column name for vehicle trim
        
    Returns:
        Dictionary containing aging statistics
    """
    try:
        # Ensure days_in_stock is numeric
        df[days_in_stock_col] = pd.to_numeric(df[days_in_stock_col], errors='coerce')
        
        # Calculate overall statistics
        overall_stats = {
            'mean_days': float(df[days_in_stock_col].mean()),
            'median_days': float(df[days_in_stock_col].median()),
            'std_days': float(df[days_in_stock_col].std()),
            'total_vehicles': len(df),
            'aged_inventory_count': len(df[df[days_in_stock_col] > 90]),  # Standard aging threshold
            'percentiles': {
                '25th': float(df[days_in_stock_col].quantile(0.25)),
                '50th': float(df[days_in_stock_col].quantile(0.50)),
                '75th': float(df[days_in_stock_col].quantile(0.75)),
                '90th': float(df[days_in_stock_col].quantile(0.90))
            }
        }
        
        # Calculate aging by model if column exists
        model_stats = {}
        if model_col and model_col in df.columns:
            for model in df[model_col].unique():
                model_df = df[df[model_col] == model]
                if len(model_df) > 0:
                    model_stats[model] = {
                        'mean_days': float(model_df[days_in_stock_col].mean()),
                        'median_days': float(model_df[days_in_stock_col].median()),
                        'std_days': float(model_df[days_in_stock_col].std()),
                        'total_vehicles': len(model_df),
                        'aged_inventory_count': len(model_df[model_df[days_in_stock_col] > 90])
                    }
        
        # Calculate aging by model/trim combination if both columns exist
        model_trim_stats = {}
        if model_col and trim_col and model_col in df.columns and trim_col in df.columns:
            for model in df[model_col].unique():
                model_trim_stats[model] = {}
                model_df = df[df[model_col] == model]
                
                for trim in model_df[trim_col].unique():
                    trim_df = model_df[model_df[trim_col] == trim]
                    if len(trim_df) > 0:
                        model_trim_stats[model][trim] = {
                            'mean_days': float(trim_df[days_in_stock_col].mean()),
                            'median_days': float(trim_df[days_in_stock_col].median()),
                            'std_days': float(trim_df[days_in_stock_col].std()),
                            'total_vehicles': len(trim_df),
                            'aged_inventory_count': len(trim_df[trim_df[days_in_stock_col] > 90])
                        }
        
        # Identify outliers using z-score
        z_scores = np.abs(stats.zscore(df[days_in_stock_col]))
        outliers = df[z_scores > 2]  # More than 2 standard deviations
        
        outlier_stats = {
            'count': len(outliers),
            'percentage': (len(outliers) / len(df)) * 100,
            'mean_days': float(outliers[days_in_stock_col].mean()) if len(outliers) > 0 else 0
        }
        
        return {
            'overall_stats': overall_stats,
            'model_stats': model_stats,
            'model_trim_stats': model_trim_stats,
            'outlier_stats': outlier_stats
        }
        
    except Exception as e:
        logger.error(f"Error calculating inventory aging stats: {str(e)}")
        return {
            'error': str(e),
            'overall_stats': {},
            'model_stats': {},
            'model_trim_stats': {},
            'outlier_stats': {}
        }

def calculate_sales_performance_stats(
    df: pd.DataFrame,
    gross_col: str = 'gross',
    rep_col: str = 'sales_rep',
    date_col: Optional[str] = 'date'
) -> Dict[str, Any]:
    """
    Calculate sales performance statistics with benchmarking.
    
    Args:
        df: DataFrame containing sales data
        gross_col: Column name for gross profit
        rep_col: Column name for sales representative
        date_col: Optional column name for sale date
        
    Returns:
        Dictionary containing performance statistics
    """
    try:
        # Ensure gross is numeric
        df[gross_col] = pd.to_numeric(df[gross_col].astype(str).str.replace(r'[\$,]', '', regex=True), 
                                    errors='coerce')
        
        # Calculate overall statistics
        overall_stats = {
            'total_gross': float(df[gross_col].sum()),
            'mean_gross': float(df[gross_col].mean()),
            'median_gross': float(df[gross_col].median()),
            'std_gross': float(df[gross_col].std()),
            'total_deals': len(df),
            'negative_gross_count': len(df[df[gross_col] < 0])
        }
        
        # Calculate per-rep statistics
        rep_stats = {}
        for rep in df[rep_col].unique():
            rep_df = df[df[rep_col] == rep]
            if len(rep_df) > 0:
                rep_stats[rep] = {
                    'total_gross': float(rep_df[gross_col].sum()),
                    'mean_gross': float(rep_df[gross_col].mean()),
                    'median_gross': float(rep_df[gross_col].median()),
                    'std_gross': float(rep_df[gross_col].std()),
                    'total_deals': len(rep_df),
                    'negative_gross_count': len(rep_df[rep_df[gross_col] < 0])
                }
        
        # Calculate time-based trends if date column exists
        trend_stats = {}
        if date_col and date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])
            monthly_stats = df.groupby(df[date_col].dt.to_period('M')).agg({
                gross_col: ['sum', 'mean', 'count']
            }).reset_index()
            
            monthly_stats.columns = ['month', 'total_gross', 'mean_gross', 'deal_count']
            trend_stats = monthly_stats.to_dict('records')
        
        # Calculate benchmarks
        benchmarks = {
            'top_quartile_gross': float(df[gross_col].quantile(0.75)),
            'bottom_quartile_gross': float(df[gross_col].quantile(0.25)),
            'top_10_percent_gross': float(df[gross_col].quantile(0.90)),
            'deals_per_rep_mean': float(df.groupby(rep_col).size().mean()),
            'deals_per_rep_median': float(df.groupby(rep_col).size().median())
        }
        
        return {
            'overall_stats': overall_stats,
            'rep_stats': rep_stats,
            'trend_stats': trend_stats,
            'benchmarks': benchmarks
        }
        
    except Exception as e:
        logger.error(f"Error calculating sales performance stats: {str(e)}")
        return {
            'error': str(e),
            'overall_stats': {},
            'rep_stats': {},
            'trend_stats': {},
            'benchmarks': {}
        }

def detect_inventory_anomalies(
    df: pd.DataFrame,
    days_in_stock_col: str = 'days_in_stock',
    model_col: Optional[str] = 'model',
    price_col: Optional[str] = 'price'
) -> List[Dict[str, Any]]:
    """
    Detect anomalies in inventory aging patterns.
    
    Args:
        df: DataFrame containing inventory data
        days_in_stock_col: Column name for days in stock
        model_col: Optional column name for vehicle model
        price_col: Optional column name for vehicle price
        
    Returns:
        List of dictionaries containing anomaly information
    """
    anomalies = []
    
    try:
        # Ensure numeric columns
        df[days_in_stock_col] = pd.to_numeric(df[days_in_stock_col], errors='coerce')
        if price_col and price_col in df.columns:
            df[price_col] = pd.to_numeric(df[price_col].astype(str).str.replace(r'[\$,]', '', regex=True), 
                                        errors='coerce')
        
        # Calculate overall statistics
        overall_mean = df[days_in_stock_col].mean()
        overall_std = df[days_in_stock_col].std()
        
        # Function to check if a value is an outlier
        def is_outlier(value, mean, std):
            z_score = (value - mean) / std if std > 0 else 0
            return abs(z_score) > 2  # More than 2 standard deviations
        
        # Check for overall outliers
        outlier_mask = df[days_in_stock_col].apply(lambda x: is_outlier(x, overall_mean, overall_std))
        outliers = df[outlier_mask]
        
        for _, row in outliers.iterrows():
            anomaly = {
                'type': 'aging_outlier',
                'days_in_stock': float(row[days_in_stock_col]),
                'z_score': float((row[days_in_stock_col] - overall_mean) / overall_std),
                'severity': 'high' if abs((row[days_in_stock_col] - overall_mean) / overall_std) > 3 else 'medium'
            }
            
            # Add model information if available
            if model_col and model_col in df.columns:
                anomaly['model'] = row[model_col]
            
            # Add price information if available
            if price_col and price_col in df.columns:
                anomaly['price'] = float(row[price_col])
            
            anomalies.append(anomaly)
        
        # Check for model-specific anomalies if model column exists
        if model_col and model_col in df.columns:
            for model in df[model_col].unique():
                model_df = df[df[model_col] == model]
                if len(model_df) < 2:  # Skip if not enough data
                    continue
                
                model_mean = model_df[days_in_stock_col].mean()
                model_std = model_df[days_in_stock_col].std()
                
                # Find outliers within this model
                model_outliers = model_df[
                    model_df[days_in_stock_col].apply(lambda x: is_outlier(x, model_mean, model_std))
                ]
                
                for _, row in model_outliers.iterrows():
                    anomaly = {
                        'type': 'model_specific_outlier',
                        'model': model,
                        'days_in_stock': float(row[days_in_stock_col]),
                        'z_score': float((row[days_in_stock_col] - model_mean) / model_std),
                        'severity': 'high' if abs((row[days_in_stock_col] - model_mean) / model_std) > 3 else 'medium'
                    }
                    
                    if price_col and price_col in df.columns:
                        anomaly['price'] = float(row[price_col])
                    
                    anomalies.append(anomaly)
        
        return anomalies
        
    except Exception as e:
        logger.error(f"Error detecting inventory anomalies: {str(e)}")
        return [{'error': str(e)}]