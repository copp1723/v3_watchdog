"""
Models for intent parsing and validation.
"""

from typing import Optional, Literal, List, Dict, Any
from pydantic import BaseModel, Field

class IntentSchema(BaseModel):
    """Schema for insight analysis intents."""
    intent: Literal["groupby_summary", "total_summary", "performance_analysis", "fallback"]
    metric: Optional[str] = None
    category: Optional[str] = None
    aggregation: Optional[Literal["sum", "count", "mean", "max", "min"]] = "sum"
    dimensions: Optional[List[str]] = Field(default_factory=list)
    filters: Optional[Dict[str, Any]] = Field(default_factory=dict)
    sort_by: Optional[str] = None
    sort_order: Optional[Literal["asc", "desc"]] = "desc"
    limit: Optional[int] = None
    
    class Config:
        use_enum_values = True 