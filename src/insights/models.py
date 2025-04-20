"""
Data models for insights.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
from pydantic import BaseModel
from enum import Enum

class ConfidenceLevel(str, Enum):
    """Confidence level for insights."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class ChartType(str, Enum):
    """Chart types for visualizations."""
    BAR = "bar"
    LINE = "line"
    PIE = "pie"
    SCATTER = "scatter"
    TABLE = "table"
    NONE = "none"

class InsightResult:
    """Result of an insight analysis."""
    
    def __init__(
        self,
        title: str,
        summary: str,
        recommendations: List[str],
        confidence: str = ConfidenceLevel.MEDIUM,
        chart_data: Optional[pd.DataFrame] = None,
        chart_encoding: Optional[Dict[str, Any]] = None,
        supporting_data: Optional[pd.DataFrame] = None,
        error: Optional[str] = None
    ):
        """Initialize an insight result."""
        self.title = title
        self.summary = summary
        self.recommendations = recommendations
        self.confidence = confidence
        self.chart_data = chart_data
        self.chart_encoding = chart_encoding or {}
        self.supporting_data = supporting_data
        self.error = error