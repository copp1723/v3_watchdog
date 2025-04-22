"""
Query models for the Watchdog AI system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Optional, Dict, Any, List, Set, Union, Tuple
from decimal import Decimal
from numbers import Real

# Use these types instead of float
NumericType = Union[int, float, Decimal, Real]
from numbers import Real as Number  # Use this instead of float for type hints
from ..insights.context import InsightExecutionContext

@dataclass
class TimeRange:
    """Time range for query analysis."""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    period: Optional[str] = None  # e.g. "this_month", "last_quarter", "ytd"
    
class TimeGranularity(Enum):
    """Time granularity for analysis grouping."""
    DAILY = auto()
    WEEKLY = auto()
    MONTHLY = auto()
    QUARTERLY = auto()
    YEARLY = auto()

@dataclass
class IntentSchema:
    """
    Schema for query intent.
    
    Attributes:
        intent: The primary intent type (e.g. performance_analysis, groupby_summary)
        metric: The primary metric being analyzed (e.g. profit, revenue)
        category: The category to group by (e.g. sales_representative, product)
        aggregation: How to aggregate the metric (e.g. sum, avg)
        sort_order: The order to sort results (asc or desc)
        limit: Number of results to return
        
        # Time-based analysis parameters
        time_range: Time period for the analysis
        time_granularity: Granularity for time-based grouping
        
        # Comparative analysis flags
        compare_previous_period: Whether to compare with previous period
        benchmark_metric: Optional metric to use as benchmark
        benchmark_value: Optional fixed value to use as benchmark
        
        # Confidence scoring
        intent_confidence: Confidence score for the detected intent
        metric_confidence: Confidence score for the detected metric
        
        # Multi-metric support
        additional_metrics: Additional metrics to include in analysis
        metric_relationships: How metrics relate (e.g. ratio, difference)
        
        # Validation fields
        validated_entities: Entities that have been validated against data source
        data_availability: Whether required data is available
        allowed_operations: Operations that are valid for this intent/metric combo
    """
    intent: str  # performance_analysis, groupby_summary, total_summary
    metric: Optional[str] = None  # e.g. profit, revenue
    category: Optional[str] = None  # e.g. sales_representative, product
    aggregation: Optional[str] = None  # e.g. sum, avg
    sort_order: Optional[str] = None  # asc or desc
    limit: Optional[int] = None  # number of results to return
    
    # Time-based analysis parameters
    time_range: Optional[TimeRange] = None
    time_granularity: Optional[TimeGranularity] = None
    
    # Comparative analysis flags
    compare_previous_period: bool = False
    benchmark_metric: Optional[str] = None
    benchmark_value: Optional[float] = None
    
    # Confidence scoring
    intent_confidence: float = 1.0
    metric_confidence: float = 1.0
    
    # Multi-metric support
    additional_metrics: List[str] = field(default_factory=list)
    metric_relationships: Dict[str, str] = field(default_factory=dict)
    
    # Validation fields
    validated_entities: Dict[str, List[str]] = field(default_factory=dict)
    data_availability: bool = True
    allowed_operations: Set[str] = field(default_factory=set)

@dataclass
class QueryContext:
    """Context for query processing."""
    query: str
    insight_context: InsightExecutionContext
    intent: Optional[IntentSchema] = None
    parameters: Optional[Dict[str, Any]] = None

@dataclass
class QueryResult:
    """Result of query processing."""
    query: str
    success: bool
    message: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    entities: Dict[str, List[str]] = field(default_factory=dict)
    intent: Optional[IntentSchema] = None
    
    # Enhanced result fields
    confidence_score: float = 1.0
    related_insights: List[Dict[str, Any]] = field(default_factory=list)
    drill_down_options: List[Dict[str, Any]] = field(default_factory=list)
    historical_context: Dict[str, Any] = field(default_factory=dict)
    benchmarks: Dict[str, Any] = field(default_factory=dict)
    statistical_significance: Dict[str, float] = field(default_factory=dict)
