"""
Anomaly detection module for LLM engine pattern analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

def detect_anomalies(series: pd.Series,
                    method: str = "iqr",
                    threshold: float = 1.5) -> Dict[str, Any]:
    """
    Detect anomalies in a data series using specified method.
    
    Args:
        series: Data series to analyze
        method: Detection method ('iqr' or 'zscore')
        threshold: Threshold for anomaly detection
        
    Returns:
        Dictionary containing anomaly detection results
    """
    try:
        if len(series) < 4:  # Minimum required for quartile calculation
            return {
                "has_anomalies": False,
                "message": "Not enough data points for anomaly detection"
            }
            
        clean_series = series.dropna()
        
        if method == "iqr":
            anomalies = _detect_iqr_anomalies(clean_series, threshold)
        else:  # zscore
            anomalies = _detect_zscore_anomalies(clean_series, threshold)
            
        if not anomalies["indices"]:
            return {
                "has_anomalies": False,
                "message": "No anomalies detected",
                "stats": anomalies["stats"]
            }
            
        return {
            "has_anomalies": True,
            "indices": anomalies["indices"],
            "values": anomalies["values"],
            "stats": anomalies["stats"],
            "percentage": len(anomalies["indices"]) / len(clean_series) * 100
        }
        
    except Exception as e:
        logger.error(f"Error in anomaly detection: {str(e)}")
        return {
            "has_anomalies": False,
            "error": str(e),
            "message": "Error during anomaly detection"
        }

def _detect_iqr_anomalies(series: pd.Series, 
                         threshold: float) -> Dict[str, Any]:
    """
    Detect anomalies using the Interquartile Range method.
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    anomaly_mask = (series < lower_bound) | (series > upper_bound)
    anomaly_indices = series[anomaly_mask].index.tolist()
    anomaly_values = series[anomaly_mask].tolist()
    
    return {
        "indices": anomaly_indices,
        "values": anomaly_values,
        "stats": {
            "Q1": float(Q1),
            "Q3": float(Q3),
            "IQR": float(IQR),
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound)
        }
    }

def _detect_zscore_anomalies(series: pd.Series,
                           threshold: float) -> Dict[str, Any]:
    """
    Detect anomalies using the Z-score method.
    """
    mean = series.mean()
    std = series.std()
    
    z_scores = np.abs((series - mean) / std)
    anomaly_mask = z_scores > threshold
    anomaly_indices = series[anomaly_mask].index.tolist()
    anomaly_values = series[anomaly_mask].tolist()
    
    return {
        "indices": anomaly_indices,
        "values": anomaly_values,
        "stats": {
            "mean": float(mean),
            "std": float(std),
            "threshold": float(threshold)
        }
    }

def analyze_anomaly_patterns(series: pd.Series,
                           window_size: int = 30) -> Dict[str, Any]:
    """
    Analyze patterns in anomaly occurrence.
    
    Args:
        series: Data series to analyze
        window_size: Size of rolling window for pattern detection
        
    Returns:
        Dictionary containing anomaly pattern analysis
    """
    try:
        anomalies = detect_anomalies(series)
        if not anomalies["has_anomalies"]:
            return {
                "has_patterns": False,
                "message": "No anomalies detected for pattern analysis"
            }
            
        # Convert to datetime index if not already
        if not isinstance(series.index, pd.DatetimeIndex):
            try:
                series.index = pd.to_datetime(series.index)
            except:
                return {
                    "has_patterns": False,
                    "message": "Cannot analyze temporal patterns without datetime index"
                }
                
        # Analyze temporal distribution
        anomaly_series = pd.Series(1, index=series.index[anomalies["indices"]])
        daily_counts = anomaly_series.resample('D').sum().fillna(0)
        
        # Detect patterns in anomaly occurrence
        patterns = []
        
        # Check for weekly patterns
        weekly_pattern = daily_counts.groupby(daily_counts.index.dayofweek).mean()
        if weekly_pattern.std() > weekly_pattern.mean() * 0.2:  # Significant variation
            patterns.append({
                "type": "weekly",
                "description": "Anomalies show weekly pattern",
                "values": weekly_pattern.to_dict()
            })
            
        # Check for monthly patterns
        monthly_pattern = daily_counts.groupby(daily_counts.index.month).mean()
        if monthly_pattern.std() > monthly_pattern.mean() * 0.2:
            patterns.append({
                "type": "monthly",
                "description": "Anomalies show monthly pattern",
                "values": monthly_pattern.to_dict()
            })
            
        return {
            "has_patterns": bool(patterns),
            "patterns": patterns,
            "temporal_stats": {
                "total_anomalies": int(anomaly_series.sum()),
                "mean_daily_anomalies": float(daily_counts.mean()),
                "max_daily_anomalies": int(daily_counts.max())
            }
        }
        
    except Exception as e:
        logger.error(f"Error in anomaly pattern analysis: {str(e)}")
        return {
            "has_patterns": False,
            "error": str(e),
            "message": "Error during anomaly pattern analysis"
        }

