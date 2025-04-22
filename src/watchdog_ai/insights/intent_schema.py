"""
Schema definitions for insight intents.
"""

from typing import Optional, Literal
from pydantic import BaseModel, Field

class IntentSchema(BaseModel):
    """Schema for insight analysis intents."""
    intent: Literal["groupby_summary", "total_summary", "fallback"]
    metric: Optional[str] = None
    category: Optional[str] = None
    aggregation: Optional[Literal["sum", "count", "mean", "max", "min"]] = "sum"
    
    class Config:
        use_enum_values = True 