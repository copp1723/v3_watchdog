"""
Data models for Watchdog AI.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Literal, Optional
from datetime import datetime

class BreakdownItem(BaseModel):
    """Single item in an insight breakdown."""
    label: str
    value: float

class InsightResponse(BaseModel):
    """Structured response from LLM insight generation."""
    summary: str
    metrics: Dict[str, float] = Field(default_factory=dict)
    breakdown: List[BreakdownItem] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"]
    timestamp: datetime = Field(default_factory=datetime.now)
    is_error: bool = False
    error: Optional[str] = None
    is_mock: bool = False
    chart_data: Optional[Dict] = None

    @classmethod
    def mock_insight(cls) -> "InsightResponse":
        """Generate a mock insight for testing or fallback."""
        return cls(
            summary="This is a mock insight response. LLM integration coming soon!",
            metrics={
                "Total Records": 0,
                "Total Columns": 0
            },
            recommendations=[
                "Upload real data to get actual insights",
                "Implement LLM integration for real analysis"
            ],
            confidence="low",
            is_mock=True
        )

    @classmethod
    def error_response(cls, error: str) -> "InsightResponse":
        """Generate an error response."""
        return cls(
            summary="Failed to generate insight",
            error=error,
            confidence="low",
            is_error=True
        )