"""
Insight Contract System for Watchdog AI.
Enforces schema validation and business rules for insight generation.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
from pydantic import BaseModel, Field, validator
import logging

logger = logging.getLogger(__name__)

@dataclass
class ValidationRule:
    """Represents a business logic validation rule."""
    name: str
    description: str
    severity: str  # 'error' or 'warning'
    validation_fn: callable
    error_message: str

class InsightMetrics(BaseModel):
    """Metrics included in insight output."""
    total_records: int = Field(..., description="Total number of records analyzed")
    time_period: str = Field(..., description="Time period of analysis")
    confidence_score: float = Field(..., ge=0, le=1, description="Confidence in the insight")
    data_quality_score: float = Field(..., ge=0, le=1, description="Quality of input data")

class ChartSpec(BaseModel):
    """Specification for insight visualization."""
    chart_type: str = Field(..., description="Type of chart to render")
    x_axis: str = Field(..., description="X-axis field")
    y_axis: str = Field(..., description="Y-axis field")
    aggregation: Optional[str] = Field(None, description="Aggregation method")
    color_by: Optional[str] = Field(None, description="Field to use for color encoding")

class InsightContract(BaseModel):
    """Contract defining expected insight input and output."""
    # Metadata
    insight_id: str = Field(..., description="Unique identifier for the insight")
    insight_type: str = Field(..., description="Type of insight being generated")
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str = Field(..., description="Version of the insight contract")
    
    # Input Requirements
    required_columns: List[str] = Field(..., description="Required columns in input data")
    min_rows: int = Field(default=10, description="Minimum number of rows required")
    data_types: Dict[str, str] = Field(..., description="Expected data types for columns")
    
    # Output Requirements
    metrics: InsightMetrics
    summary: str = Field(..., min_length=10, max_length=1000)
    key_findings: List[str] = Field(..., min_items=1)
    recommendations: List[str] = Field(..., min_items=1)
    visualization: Optional[ChartSpec] = None
    
    # Execution Context
    execution_time_ms: float = Field(..., description="Time taken to generate insight")
    cache_hit: bool = Field(default=False, description="Whether result was cached")
    error: Optional[str] = None
    
    @validator('key_findings')
    def validate_findings_length(cls, v):
        """Validate that key findings are meaningful."""
        if any(len(finding) < 10 for finding in v):
            raise ValueError("Key findings must be at least 10 characters long")
        return v
    
    @validator('recommendations')
    def validate_recommendations(cls, v):
        """Validate that recommendations are actionable."""
        if any(not recommendation.startswith(('Consider', 'Review', 'Implement', 'Analyze', 'Monitor'))
               for recommendation in v):
            raise ValueError("Recommendations must start with an action verb")
        return v

class InsightContractEnforcer:
    """Enforces insight contracts during execution."""
    
    def __init__(self):
        """Initialize the contract enforcer."""
        self.validation_rules: List[ValidationRule] = []
    
    def add_rule(self, rule: ValidationRule) -> None:
        """Add a validation rule."""
        self.validation_rules.append(rule)
    
    def validate_input(self, df: pd.DataFrame, contract: InsightContract) -> Dict[str, Any]:
        """
        Validate input data against contract requirements.
        
        Args:
            df: Input DataFrame
            contract: Insight contract to validate against
            
        Returns:
            Dictionary with validation results
        """
        results = {
            "is_valid": True,
            "errors": [],
            "warnings": []
        }
        
        try:
            # Check required columns
            missing_cols = [col for col in contract.required_columns if col not in df.columns]
            if missing_cols:
                results["is_valid"] = False
                results["errors"].append(f"Missing required columns: {', '.join(missing_cols)}")
            
            # Check minimum rows
            if len(df) < contract.min_rows:
                results["is_valid"] = False
                results["errors"].append(
                    f"Insufficient data: {len(df)} rows, minimum required: {contract.min_rows}"
                )
            
            # Check data types
            for col, expected_type in contract.data_types.items():
                if col in df.columns:
                    actual_type = str(df[col].dtype)
                    if actual_type != expected_type:
                        results["warnings"].append(
                            f"Column '{col}' has type '{actual_type}', expected '{expected_type}'"
                        )
            
            # Run custom validation rules
            for rule in self.validation_rules:
                try:
                    is_valid = rule.validation_fn(df)
                    if not is_valid:
                        if rule.severity == 'error':
                            results["is_valid"] = False
                            results["errors"].append(rule.error_message)
                        else:
                            results["warnings"].append(rule.error_message)
                except Exception as e:
                    logger.error(f"Error running validation rule '{rule.name}': {e}")
                    results["warnings"].append(f"Validation rule '{rule.name}' failed: {str(e)}")
            
        except Exception as e:
            results["is_valid"] = False
            results["errors"].append(f"Validation error: {str(e)}")
        
        return results
    
    def validate_output(self, output: Dict[str, Any], contract: InsightContract) -> Dict[str, Any]:
        """
        Validate insight output against contract requirements.
        
        Args:
            output: Generated insight output
            contract: Insight contract to validate against
            
        Returns:
            Dictionary with validation results
        """
        results = {
            "is_valid": True,
            "errors": [],
            "warnings": []
        }
        
        try:
            # Validate against Pydantic model
            InsightContract(**output)
            
            # Additional business logic validation
            if output.get('confidence_score', 0) < 0.7:
                results["warnings"].append(
                    f"Low confidence score: {output.get('confidence_score')}"
                )
            
            if len(output.get('recommendations', [])) < 2:
                results["warnings"].append(
                    "Consider providing more recommendations"
                )
            
            # Validate metrics
            metrics = output.get('metrics', {})
            if metrics.get('data_quality_score', 0) < 0.8:
                results["warnings"].append(
                    f"Low data quality score: {metrics.get('data_quality_score')}"
                )
            
        except Exception as e:
            results["is_valid"] = False
            results["errors"].append(f"Output validation error: {str(e)}")
        
        return results

def create_default_contract(insight_type: str) -> InsightContract:
    """
    Create a default contract for an insight type.
    
    Args:
        insight_type: Type of insight to create contract for
        
    Returns:
        InsightContract instance
    """
    return InsightContract(
        insight_id=f"{insight_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        insight_type=insight_type,
        version="1.0.0",
        required_columns=["SaleDate", "TotalGross", "LeadSource"],
        data_types={
            "SaleDate": "datetime64[ns]",
            "TotalGross": "float64",
            "LeadSource": "object"
        },
        metrics=InsightMetrics(
            total_records=0,
            time_period="",
            confidence_score=0.0,
            data_quality_score=0.0
        ),
        summary="",
        key_findings=[],
        recommendations=[],
        execution_time_ms=0.0
    )