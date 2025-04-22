"""
Data models for insights.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime
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

@dataclass
class IntentSchema:
    """Schema for intent detection results."""
    intent_type: str
    confidence: float
    parameters: Dict[str, Any]
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IntentSchema':
        return cls(
            intent_type=data.get('intent_type', ''),
            confidence=float(data.get('confidence', 0.0)),
            parameters=data.get('parameters', {})
        )

@dataclass
class InsightResponse:
    """Schema for insight response data."""
    summary: str
    metrics: Dict[str, float]
    breakdown: List[Dict[str, Any]]
    recommendations: List[str]
    confidence: str
    timestamp: datetime = datetime.now()
    chart_data: Optional[pd.DataFrame] = None
    chart_encoding: Optional[Dict[str, Any]] = None
    supporting_data: Optional[pd.DataFrame] = None
    error: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InsightResponse':
        return cls(
            summary=data.get('summary', ''),
            metrics=data.get('metrics', {}),
            breakdown=data.get('breakdown', []),
            recommendations=data.get('recommendations', []),
            confidence=data.get('confidence', 'low'),
            chart_data=data.get('chart_data'),
            chart_encoding=data.get('chart_encoding'),
            supporting_data=data.get('supporting_data'),
            error=data.get('error')
        )

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
        metrics: Optional[Dict[str, float]] = None,
        breakdown: Optional[List[Dict[str, Any]]] = None,
        error: Optional[str] = None
    ):
        self.title = title
        self.summary = summary
        self.recommendations = recommendations
        self.confidence = confidence
        self.chart_data = chart_data
        self.chart_encoding = chart_encoding
        self.metrics = metrics or {}
        self.breakdown = breakdown or []
        self.error = error
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert insight result to dictionary format."""
        return {
            'title': self.title,
            'summary': self.summary,
            'recommendations': self.recommendations,
            'confidence': self.confidence,
            'chart_data': self.chart_data.to_dict() if isinstance(self.chart_data, pd.DataFrame) else None,
            'chart_encoding': self.chart_encoding,
            'metrics': self.metrics,
            'breakdown': self.breakdown,
            'error': self.error,
            'timestamp': self.timestamp.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InsightResult':
        """Create an insight result from dictionary data."""
        chart_data = pd.DataFrame(data['chart_data']) if data.get('chart_data') else None
        return cls(
            title=data.get('title', ''),
            summary=data['summary'],
            recommendations=data.get('recommendations', []),
            confidence=data.get('confidence', ConfidenceLevel.MEDIUM),
            chart_data=chart_data,
            chart_encoding=data.get('chart_encoding'),
            metrics=data.get('metrics'),
            breakdown=data.get('breakdown'),
            error=data.get('error')
        )

    def __str__(self) -> str:
        """String representation of the insight result."""
        return f"{self.title}: {self.summary}"

@dataclass
class FeedbackEntry:
    """Schema for user feedback on insights."""
    insight_id: str
    rating: int
    comment: Optional[str] = None
    timestamp: datetime = datetime.now()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeedbackEntry':
        return cls(
            insight_id=data['insight_id'],
            rating=int(data['rating']),
            comment=data.get('comment'),
            timestamp=datetime.fromisoformat(data['timestamp']) if 'timestamp' in data else datetime.now()
        )
