"""
Models for insight generation and error handling.
"""

from enum import Enum
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

class InsightErrorType(Enum):
    """Types of errors that can occur during insight generation."""
    NO_VALID_DATA = "no_valid_data"
    INSUFFICIENT_DATA = "insufficient_data"
    DATA_ERROR = "data_error"
    MISSING_COLUMNS = "missing_columns"
    INVALID_QUERY = "invalid_query"
    PROCESSING_ERROR = "processing_error"
    UNKNOWN_ERROR = "unknown_error"

class BreakdownItem(BaseModel):
    """Single item in a breakdown analysis."""
    label: str
    value: float
    percentage: Optional[float] = None

class InsightResponse(BaseModel):
    """Structured insight response."""
    summary: str
    metrics: Dict[str, Any] = {}
    breakdown: List[BreakdownItem] = []
    recommendations: List[str] = []
    confidence: str
    error_type: Optional[InsightErrorType] = None
    data_quality: Optional[Dict[str, Any]] = None  # Added data quality field
    warning_level: Optional[str] = None  # Added warning level field

def mock_insight() -> Dict[str, Any]:
    """Generate a mock insight for testing."""
    return {
        "summary": "This is a mock insight for testing purposes.",
        "metrics": {"total": 100, "average": 50.0},
        "breakdown": [
            {"label": "Category A", "value": 60.0, "percentage": 60.0},
            {"label": "Category B", "value": 40.0, "percentage": 40.0}
        ],
        "recommendations": [
            "First mock recommendation",
            "Second mock recommendation"
        ],
        "confidence": "medium",
        "is_mock": True
    }

def error_response(error_type: InsightErrorType, message: str) -> Dict[str, Any]:
    """Generate an error response."""
    return {
        "summary": message,
        "metrics": {},
        "breakdown": [],
        "recommendations": [],
        "confidence": "low",
        "error_type": error_type,
        "is_error": True
    }