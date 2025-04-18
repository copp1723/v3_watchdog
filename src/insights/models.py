"""
Data models for insight generation.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from datetime import datetime
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

@dataclass
class FeedbackEntry:
    """User feedback for an insight."""
    insight_id: str
    user_id: str
    rating: int  # 1-5 scale
    comment: Optional[str] = None
    tags: List[str] = None
    created_at: datetime = None
    context: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Initialize default values."""
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.tags is None:
            self.tags = []
        if self.context is None:
            self.context = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert feedback entry to dictionary format."""
        return {
            "insight_id": self.insight_id,
            "user_id": self.user_id,
            "rating": self.rating,
            "comment": self.comment,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "context": self.context
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeedbackEntry':
        """Create feedback entry from dictionary format."""
        if "created_at" in data and isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)