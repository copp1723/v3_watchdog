"""
Insight Contract System for Watchdog AI.
Enforces schema validation and business rules for insight generation.
"""

from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import pandas as pd
from pydantic import BaseModel, Field, validator
import logging

logger = logging.getLogger(__name__)

class InsightMetrics(BaseModel):
    """Base metrics for insights."""
    total_records: int = Field(..., description="Total number of records analyzed")
    time_period: str = Field(..., description="Time period of analysis")
    confidence_score: float = Field(..., ge=0, le=1, description="Confidence in the insight")
    data_quality_score: float = Field(..., ge=0, le=1, description="Quality of input data")

class ResponseTimeMetrics(BaseModel):
    """Metrics for lead response time analysis."""
    avg_response_time: timedelta = Field(..., description="Average time to first response")
    within_1hour: float = Field(..., ge=0, le=100, description="Percentage responded within 1 hour")
    within_24hours: float = Field(..., ge=0, le=100, description="Percentage responded within 24 hours")
    longest_wait: timedelta = Field(..., description="Longest wait time")
    response_rate: float = Field(..., ge=0, le=100, description="Overall response rate")

class InventoryAgeMetrics(BaseModel):
    """Metrics for inventory age and profit analysis."""
    avg_days_on_lot: float = Field(..., ge=0, description="Average days on lot")
    avg_profit_by_age: Dict[str, float] = Field(..., description="Average profit by age group")
    best_performing_age: str = Field(..., description="Age group with highest profit")
    total_aged_inventory: int = Field(..., ge=0, description="Total units over age threshold")
    profit_correlation: float = Field(..., ge=-1, le=1, description="Correlation between age and profit")

class ChartSpec(BaseModel):
    """Base specification for insight visualization."""
    chart_type: str = Field(..., description="Type of chart to render")
    x_axis: str = Field(..., description="X-axis field")
    y_axis: str = Field(..., description="Y-axis field")
    aggregation: Optional[str] = Field(None, description="Aggregation method")
    color_by: Optional[str] = Field(None, description="Field to use for color encoding")

class TimeSeriesChartSpec(ChartSpec):
    """Time series specific chart configuration."""
    time_unit: str = Field(..., description="Time unit for aggregation (hour, day, week)")
    cumulative: bool = Field(default=False, description="Whether to show cumulative values")

class ScatterChartSpec(ChartSpec):
    """Scatter plot specific chart configuration."""
    trend_line: bool = Field(default=True, description="Whether to show trend line")
    size_by: Optional[str] = Field(None, description="Field to use for point sizes")

class DataRequirements(BaseModel):
    """Input data requirements."""
    required_columns: List[str] = Field(..., description="Required columns in input data")
    min_rows: int = Field(default=10, description="Minimum number of rows required")
    data_types: Dict[str, str] = Field(..., description="Expected data types for columns")
    time_window: Optional[timedelta] = Field(None, description="Required time window for analysis")

class InsightOutput(BaseModel):
    """Output requirements for insights."""
    metrics: InsightMetrics
    response_time_metrics: Optional[ResponseTimeMetrics] = None
    inventory_age_metrics: Optional[InventoryAgeMetrics] = None
    summary: str = Field(..., min_length=10, max_length=1000)
    key_findings: List[str] = Field(..., min_items=1)
    recommendations: List[str] = Field(..., min_items=1)
    visualization: Optional[Union[ChartSpec, TimeSeriesChartSpec, ScatterChartSpec]] = None

    @validator('recommendations')
    def validate_recommendations(cls, v):
        """Ensure recommendations are properly formatted."""
        if not all(isinstance(r, str) for r in v):
            raise ValueError("All recommendations must be strings")
        if not all(r.strip() for r in v):
            raise ValueError("Recommendations cannot be empty strings")
        return v

    @validator('key_findings')
    def validate_findings(cls, v):
        """Ensure key findings are properly formatted."""
        if not all(isinstance(f, str) for f in v):
            raise ValueError("All findings must be strings")
        if not all(f.strip() for f in v):
            raise ValueError("Findings cannot be empty strings")
        return v

class InsightContract(BaseModel):
    """Contract defining expected insight input and output."""
    # Metadata
    insight_id: str = Field(..., description="Unique identifier for the insight")
    insight_type: str = Field(..., description="Type of insight being generated")
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str = Field(..., description="Version of the insight contract")
    
    # Requirements
    data_requirements: DataRequirements = Field(..., description="Data requirements for analysis")
    output: InsightOutput = Field(..., description="Output requirements and data")

    @validator('insight_type')
    def validate_insight_type(cls, v):
        """Validate insight type is supported."""
        valid_types = {'response_time', 'inventory_age_profit', 'general'}
        if v not in valid_types:
            raise ValueError(f"Insight type must be one of: {', '.join(valid_types)}")
        return v

    @validator('output')
    def validate_metrics(cls, v, values):
        """Validate that the appropriate metrics are present based on insight type."""
        insight_type = values.get('insight_type')
        if insight_type == 'response_time' and not v.response_time_metrics:
            raise ValueError("Response time metrics required for response_time insight type")
        if insight_type == 'inventory_age_profit' and not v.inventory_age_metrics:
            raise ValueError("Inventory age metrics required for inventory_age_profit insight type")
        return v

def create_default_contract(insight_type: str) -> InsightContract:
    """Create a default contract for an insight type."""
    # Define base requirements
    required_columns = ["SaleDate", "TotalGross", "LeadSource"]
    data_types = {
        "SaleDate": "datetime64[ns]",
        "TotalGross": "float64",
        "LeadSource": "object"
    }
    
    # Add type-specific requirements
    if insight_type == "response_time":
        required_columns.extend(["LeadDate", "FirstResponseDate"])
        data_types.update({
            "LeadDate": "datetime64[ns]",
            "FirstResponseDate": "datetime64[ns]"
        })
    elif insight_type == "inventory_age_profit":
        required_columns.extend(["DaysOnLot", "GrossProfit"])
        data_types.update({
            "DaysOnLot": "float64",
            "GrossProfit": "float64"
        })
    
    return InsightContract(
        insight_id=f"{insight_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        insight_type=insight_type,
        version="1.0.0",
        data_requirements=DataRequirements(
            required_columns=required_columns,
            data_types=data_types,
            min_rows=10
        ),
        output=InsightOutput(
            metrics=InsightMetrics(
                total_records=0,
                time_period="",
                confidence_score=0.0,
                data_quality_score=0.0
            ),
            summary="",
            key_findings=[],
            recommendations=[]
        )
    )
