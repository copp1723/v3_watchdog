"""
Trend detection module for LLM engine pattern analysis.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

def detect_trends(series: pd.Series, confidence_threshold: float = 0.05) -> Dict[str, Any]:
    """
    Detect significant trends in a time series.
    
    Args:
        series: Time series data to analyze
        confidence_threshold: P-value threshold for significance
        
    Returns:
        Dictionary containing trend analysis results
    """
    try:
        if len(series) < 2:
            return {
                "has_trend": False,
                "message": "Not enough data points for trend analysis"
            }
            
        # Create index array for regression
        x = np.arange(len(series))
        y = series.values
        
        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Determine trend strength
        strength = abs(r_value)
        if strength > 0.7:
            trend_strength = "strong"
        elif strength > 0.4:
            trend_strength = "moderate"
        else:
            trend_strength = "weak"
            
        # Check significance
        is_significant = p_value < confidence_threshold
        
        return {
            "has_trend": is_significant,
            "direction": "increasing" if slope > 0 else "decreasing",
            "strength": trend_strength,
            "slope": float(slope),
            "r_squared": float(r_value ** 2),
            "p_value": float(p_value),
            "confidence": 1 - p_value,
            "std_error": float(std_err)
        }
        
    except Exception as e:
        logger.error(f"Error in trend detection: {str(e)}")
        return {
            "has_trend": False,
            "error": str(e),
            "message": "Error during trend analysis"
        }

def detect_trend_changes(series: pd.Series, 
                        window_size: int = 10,
                        threshold: float = 0.1) -> List[Dict[str, Any]]:
    """
    Detect points where trend direction changes significantly.
    
    Args:
        series: Time series data to analyze
        window_size: Size of rolling window for trend detection
        threshold: Minimum change in slope to consider significant
        
    Returns:
        List of detected trend changes with metadata
    """
    try:
        changes = []
        if len(series) < window_size * 2:
            return changes
            
        # Calculate rolling trends
        for i in range(window_size, len(series) - window_size):
            window1 = series.iloc[i-window_size:i]
            window2 = series.iloc[i:i+window_size]
            
            trend1 = detect_trends(window1)
            trend2 = detect_trends(window2)
            
            if trend1["has_trend"] and trend2["has_trend"]:
                slope_change = abs(trend2["slope"] - trend1["slope"])
                if slope_change > threshold:
                    changes.append({
                        "index": i,
                        "timestamp": series.index[i],
                        "before_trend": trend1["direction"],
                        "after_trend": trend2["direction"],
                        "change_magnitude": float(slope_change)
                    })
                    
        return changes
        
    except Exception as e:
        logger.error(f"Error detecting trend changes: {str(e)}")
        return []

def extrapolate_trend(series: pd.Series, 
                     periods: int = 5,
                     confidence_level: float = 0.95) -> Dict[str, Any]:
    """
    Extrapolate detected trend for future periods.
    
    Args:
        series: Time series data to analyze
        periods: Number of periods to forecast
        confidence_level: Confidence level for prediction intervals
        
    Returns:
        Dictionary containing extrapolation results and confidence intervals
    """
    try:
        trend = detect_trends(series)
        if not trend["has_trend"]:
            return {
                "can_extrapolate": False,
                "message": "No significant trend detected for extrapolation"
            }
            
        # Create prediction points
        x = np.arange(len(series))
        X = np.vstack([x, np.ones(len(x))]).T
        y = series.values
        
        # Fit model
        model = np.linalg.lstsq(X, y, rcond=None)[0]
        
        # Generate future points
        future_x = np.arange(len(series), len(series) + periods)
        future_X = np.vstack([future_x, np.ones(len(future_x))]).T
        
        # Calculate predictions
        predictions = np.dot(future_X, model)
        
        # Calculate prediction intervals
        mse = np.sum((y - np.dot(X, model))**2) / (len(x) - 2)
        var_pred = mse * (1 + np.diag(np.dot(future_X, np.dot(np.linalg.pinv(np.dot(X.T, X)), future_X.T))))
        
        # Get confidence intervals
        from scipy.stats import t
        t_value = t.ppf((1 + confidence_level) / 2, len(x) - 2)
        ci = t_value * np.sqrt(var_pred)
        
        return {
            "can_extrapolate": True,
            "predictions": predictions.tolist(),
            "confidence_intervals": {
                "lower": (predictions - ci).tolist(),
                "upper": (predictions + ci).tolist()
            },
            "model_quality": {
                "r_squared": trend["r_squared"],
                "confidence_level": confidence_level,
                "mse": float(mse)
            }
        }
        
    except Exception as e:
        logger.error(f"Error in trend extrapolation: {str(e)}")
        return {
            "can_extrapolate": False,
            "error": str(e),
            "message": "Error during trend extrapolation"
        }

