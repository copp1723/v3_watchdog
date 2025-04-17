"""
Data models for insight generation.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import pandas as pd

@dataclass
class InsightResult:
    """Result from insight analysis."""
    title: str
    summary: str
    recommendations: list[str]
    chart_data: Optional[pd.DataFrame] = None
    chart_encoding: Optional[Dict[str, Any]] = None
    supporting_data: Optional[pd.DataFrame] = None
    confidence: str = "high"
    error: Optional[str] = None