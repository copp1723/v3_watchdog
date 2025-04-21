"""
Context classes for Watchdog AI insight execution.

This module provides structured context objects for carrying data
throughout the insight generation pipeline.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import pandas as pd
import uuid
from datetime import datetime

from ..utils.adaptive_schema import SchemaProfile


@dataclass
class InsightExecutionContext:
    """
    Execution context for insight generation.
    
    This class encapsulates all necessary data for generating insights,
    including the DataFrame, schema, query, and additional context variables.
    It enables parameter-based architecture without session state dependencies.
    """
    df: pd.DataFrame
    query: str
    user_role: str
    schema: Optional[SchemaProfile] = None
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    context_vars: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def __post_init__(self):
        """Validate and initialize the context."""
        # Ensure DataFrame is valid
        if self.df is None or self.df.empty:
            raise ValueError("DataFrame cannot be None or empty")
        
        # Ensure query is non-empty
        if not self.query or not isinstance(self.query, str):
            raise ValueError("Query must be a non-empty string")
        
        # Add basic metadata to context
        self.context_vars.setdefault("record_count", len(self.df))
        self.context_vars.setdefault("column_count", len(self.df.columns))
        self.context_vars.setdefault("column_list", self.df.columns.tolist())
        self.context_vars.setdefault("data_types", {col: str(self.df[col].dtype) for col in self.df.columns})
        
        # Add execution metadata
        self.context_vars.setdefault("trace_id", self.trace_id)
        self.context_vars.setdefault("user_role", self.user_role)
        self.context_vars.setdefault("timestamp", self.timestamp)
    
    def with_additional_context(self, **kwargs) -> 'InsightExecutionContext':
        """
        Create a new context with additional variables.
        
        Args:
            **kwargs: Additional context variables to add
            
        Returns:
            New InsightExecutionContext with updated context_vars
        """
        # Create a new context with original values plus new kwargs
        new_context_vars = self.context_vars.copy()
        new_context_vars.update(kwargs)
        
        return InsightExecutionContext(
            df=self.df,
            query=self.query,
            user_role=self.user_role,
            schema=self.schema,
            trace_id=self.trace_id,
            context_vars=new_context_vars,
            timestamp=self.timestamp
        )
    
    def get_schema_context(self) -> Dict[str, Any]:
        """
        Get schema-related context for LLM prompts.
        
        Returns:
            Dictionary with schema information
        """
        schema_context = {}
        
        # Include schema profile information if available
        if self.schema:
            schema_context["schema_id"] = self.schema.id
            schema_context["schema_role"] = str(self.schema.role)
            schema_context["schema_metrics"] = self.schema.default_metrics
            schema_context["schema_dimensions"] = self.schema.default_dimensions
            
            # Get visible columns based on role
            visible_columns = self.schema.get_visible_columns(self.user_role)
            schema_context["visible_columns"] = [col.name for col in visible_columns]
            
            # Include column metadata
            column_info = {}
            for col in visible_columns:
                column_info[col.name] = {
                    "display_name": col.display_name,
                    "description": col.description,
                    "data_type": col.data_type,
                    "aliases": col.aliases,
                    "metric_type": col.metric_type,
                    "aggregations": col.aggregations,
                    "primary_groupings": col.primary_groupings,
                }
            schema_context["column_info"] = column_info
        
        return schema_context


@dataclass
class InsightPipelineStage:
    """
    Base class for pipeline stages in insight generation.
    
    Each stage in the insight pipeline performs a specific operation
    and passes its result to the next stage.
    """
    name: str
    description: str
    
    def process(self, context: InsightExecutionContext) -> InsightExecutionContext:
        """
        Process the context and return updated context.
        
        Args:
            context: InsightExecutionContext with input data
            
        Returns:
            Updated InsightExecutionContext with processing results
        """
        raise NotImplementedError("Subclasses must implement process()")


@dataclass
class InsightPipeline:
    """
    Pipeline for insight generation.
    
    This class orchestrates the execution of multiple pipeline stages
    to generate insights from data.
    """
    stages: List[InsightPipelineStage] = field(default_factory=list)
    
    def add_stage(self, stage: InsightPipelineStage) -> None:
        """Add a stage to the pipeline."""
        self.stages.append(stage)
    
    def execute(self, context: InsightExecutionContext) -> Dict[str, Any]:
        """
        Execute the full pipeline.
        
        Args:
            context: InsightExecutionContext with input data
            
        Returns:
            Dictionary with final insight result
        """
        current_context = context
        result_log = []
        
        try:
            # Process each stage in sequence
            for stage in self.stages:
                stage_start_time = datetime.now()
                
                # Process the stage
                try:
                    current_context = stage.process(current_context)
                    status = "success"
                    error = None
                except Exception as e:
                    status = "error"
                    error = str(e)
                
                # Log stage execution
                stage_duration = (datetime.now() - stage_start_time).total_seconds()
                stage_log = {
                    "stage": stage.name,
                    "status": status,
                    "duration_seconds": stage_duration,
                    "error": error
                }
                result_log.append(stage_log)
                
                # Stop processing if stage failed
                if status == "error":
                    break
            
            # Get final result from context
            if "result" in current_context.context_vars:
                final_result = current_context.context_vars["result"]
            else:
                final_result = {
                    "summary": "No result produced by pipeline",
                    "metrics": {},
                    "breakdown": [],
                    "recommendations": [],
                    "confidence": "low",
                    "error_type": "PIPELINE_ERROR"
                }
            
            # Add execution log to result
            final_result["_execution_log"] = result_log
            
            return final_result
            
        except Exception as e:
            # Handle pipeline execution errors
            return {
                "summary": f"Error executing insight pipeline: {str(e)}",
                "metrics": {},
                "breakdown": [],
                "recommendations": ["Please try again or contact support"],
                "confidence": "low",
                "error_type": "PIPELINE_ERROR",
                "_execution_log": result_log
            }