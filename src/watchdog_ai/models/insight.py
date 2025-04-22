"""
Models for insight generation and responses.
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
from datetime import datetime

class InsightErrorType(str, Enum):
    """Enumeration of possible insight error types."""
    INVALID_COLUMN = "INVALID_COLUMN"
    ANALYSIS_ERROR = "ANALYSIS_ERROR"
    DATA_QUALITY = "DATA_QUALITY"
    PROCESSING_ERROR = "PROCESSING_ERROR"
    AMBIGUOUS_INTENT = "AMBIGUOUS_INTENT"
    NO_VALID_DATA = "NO_VALID_DATA"

class BreakdownItem(BaseModel):
    """Single item in an insight breakdown."""
    label: str
    value: float

class InsightResponse(BaseModel):
    """Model for insight response data."""
    success: bool
    message: str
    metrics: Optional[Dict[str, Any]] = {}
    breakdown: Optional[List[BreakdownItem]] = []
    recommendations: Optional[List[str]] = []
    confidence: Optional[float] = None
    error_type: Optional[InsightErrorType] = None
    timestamp: datetime = datetime.now()

    @classmethod
    def mock_insight(cls) -> "InsightResponse":
        """Generate a mock insight for testing or fallback."""
        return cls(
            success=True,
            message="This is a mock insight response. LLM integration coming soon!",
            metrics={
                "Total Records": 0,
                "Average Value": 0.0
            },
            breakdown=[
                BreakdownItem(label="Category A", value=0.0),
                BreakdownItem(label="Category B", value=0.0)
            ],
            recommendations=[
                "Upload data to analyze",
                "Implement LLM integration for real analysis"
            ],
            confidence=0.5
        )

    @classmethod
    def error_response(cls, error: str, error_type: Optional[InsightErrorType] = None) -> "InsightResponse":
        """Generate an error response."""
        return cls(
            success=False,
            message=f"Failed to generate insight: {error}",
            error_type=error_type or InsightErrorType.ANALYSIS_ERROR,
            confidence=0.0
        ) 