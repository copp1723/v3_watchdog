"""
Shared type definitions for Watchdog AI.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
import pandas as pd

@dataclass
class InsightResponse:
    """Standard insight response format."""
    title: str
    summary: str
    value_insights: List[str]
    actionable_flags: List[str]
    chart_data: Optional[Dict[str, Any]] = None
    chart_encoding: Optional[Dict[str, str]] = None
    supporting_data: Optional[Dict[str, Any]] = None
    confidence: str = "medium"
    error: Optional[str] = None
    is_error: bool = False
    is_direct_calculation: bool = True
    timestamp: str = ""