"""
Statistical baseline calculations for Watchdog AI.
"""

import numpy as np
from scipy import stats
import pandas as pd
from typing import Dict, Any, Tuple, Optional

def calculate_baseline_stats(data: pd.Series) -> Dict[str, float]:
    """Calculate baseline statistics for a numeric series."""
    stats_dict = {
        "mean": float(data.mean()),
        "median": float(data.median()),
        "std": float(data.std()),
        "min": float(data.min()),
        "max": float(data.max())
    }
    
    # Add percentiles
    for p in [25, 75, 90, 95]:
        stats_dict[f"p{p}"] = float(np.percentile(data, p))
    
    return stats_dict

def detect_anomalies(data: pd.Series, threshold: float = 2.0) -> Tuple[pd.Series, Dict[str, Any]]:
    """Detect anomalies using z-score method."""
    z_scores = stats.zscore(data)
    anomalies = data[abs(z_scores) > threshold]
    
    summary = {
        "total_anomalies": len(anomalies),
        "anomaly_percentage": (len(anomalies) / len(data)) * 100,
        "threshold": threshold
    }
    
    return anomalies, summary

def calculate_trend(data: pd.Series) -> Dict[str, float]:
    """Calculate trend statistics."""
    x = np.arange(len(data))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, data)
    
    return {
        "slope": float(slope),
        "r_squared": float(r_value ** 2),
        "p_value": float(p_value),
        "std_err": float(std_err)
    }