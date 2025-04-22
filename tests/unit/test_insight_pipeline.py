"""
Tests for the insight generation pipeline components.
"""

import pytest
import pandas as pd
from datetime import datetime
from typing import Dict, Any

from src.watchdog_ai.insights.context import (
    InsightExecutionContext,
    InsightPipelineStage,
    InsightPipeline
)

class ValidationStage(InsightPipelineStage):
    """Stage for data validation."""
    def process(self, context: InsightExecutionContext) -> InsightExecutionContext:
        # Validate data quality
        if context.df.empty:
            raise ValueError("Empty DataFrame")
        return context.with_additional_context(validation_passed=True)

class IntentDetectionStage(InsightPipelineStage):
    """Stage for intent detection."""
    def process(self, context: InsightExecutionContext) -> InsightExecutionContext:
        # Mock intent detection
        intent = {
            "type": "metric_query",
            "metric": "sales",
            "aggregation": "sum"
        }
        return context.with_additional_context(detected_intent=intent)

class ProcessingStage(InsightPipelineStage):
    """Stage for data processing."""
    def process(self, context: InsightExecutionContext) -> InsightExecutionContext:
        # Process based on intent
        intent = context.context_vars.get("detected_intent", {})
        if intent["type"] == "metric_query":
            result = {
                "summary": f"Analyzed {intent['metric']}",
                "metrics": {intent['metric']: 100},
                "breakdown": [],
                "recommendations": ["Sample recommendation"],
                "confidence": "high"
            }
            return context.with_additional_context(result=result)
        return context

@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'sales': [100, 200, 300],
        'profit': [10, 20, 30],
        'date': ['2023-01-01', '2023-01-02', '2023-01-03']
    })

@pytest.fixture
def sample_context(sample_df):
    """Create a sample execution context."""
    return InsightExecutionContext(
        df=sample_df,
        query="What are the total sales?",
        user_role="analyst"
    )

def test_validation_stage():
    """Test the validation stage."""
    # Test with valid data
    valid_df = pd.DataFrame({'a': [1, 2, 3]})
    context = InsightExecutionContext(
        df=valid_df,
        query="test",
        user_role="analyst"
    )
    
    stage = ValidationStage("validation", "Validate data")
    result = stage.process(context)
    
    assert result.context_vars["validation_passed"] is True
    
    # Test with invalid data
    with pytest.raises(ValueError):
        invalid_context = InsightExecutionContext(
            df=pd.DataFrame(),
            query="test",
            user_role="analyst"
        )
        stage.process(invalid_context)

def test_intent_detection_stage(sample_context):
    """Test the intent detection stage."""
    stage = IntentDetectionStage("intent", "Detect intent")
    result = stage.process(sample_context)
    
    assert "detected_intent" in result.context_vars
    assert result.context_vars["detected_intent"]["type"] == "metric_query"
    assert result.context_vars["detected_intent"]["metric"] == "sales"

def test_processing_stage(sample_context):
    """Test the processing stage."""
    # Add intent to context
    context_with_intent = sample_context.with_additional_context(
        detected_intent={
            "type": "metric_query",
            "metric": "sales",
            "aggregation": "sum"
        }
    )
    
    stage = ProcessingStage("processing", "Process data")
    result = stage.process(context_with_intent)
    
    assert "result" in result.context_vars
    assert result.context_vars["result"]["summary"] == "Analyzed sales"
    assert "metrics" in result.context_vars["result"]

def test_full_pipeline_execution(sample_context):
    """Test execution of full pipeline."""
    pipeline = InsightPipeline()
    
    # Add stages
    pipeline.add_stage(ValidationStage("validation", "Validate data"))
    pipeline.add_stage(IntentDetectionStage("intent", "Detect intent"))
    pipeline.add_stage(ProcessingStage("processing", "Process data"))
    
    result = pipeline.execute(sample_context)
    
    assert isinstance(result, dict)
    assert "_execution_log" in result
    assert len(result["_execution_log"]) == 3
    assert all(log["status"] == "success" for log in result["_execution_log"])

def test_pipeline_error_handling():
    """Test pipeline error handling."""
    class ErrorStage(InsightPipelineStage):
        def process(self, context):
            raise ValueError("Test error")
    
    pipeline = InsightPipeline()
    pipeline.add_stage(ErrorStage("error", "Error stage"))
    
    context = InsightExecutionContext(
        df=pd.DataFrame({'a': [1]}),
        query="test",
        user_role="analyst"
    )
    
    result = pipeline.execute(context)
    
    assert result["error_type"] == "PIPELINE_ERROR"
    assert len(result["_execution_log"]) == 1
    assert result["_execution_log"][0]["status"] == "error"

def test_pipeline_stage_timing():
    """Test stage execution timing."""
    class SlowStage(InsightPipelineStage):
        def process(self, context):
            import time
            time.sleep(0.1)
            return context
    
    pipeline = InsightPipeline()
    pipeline.add_stage(SlowStage("slow", "Slow stage"))
    
    context = InsightExecutionContext(
        df=pd.DataFrame({'a': [1]}),
        query="test",
        user_role="analyst"
    )
    
    result = pipeline.execute(context)
    
    assert result["_execution_log"][0]["duration_seconds"] >= 0.1

def test_pipeline_context_preservation():
    """Test that context is preserved through pipeline stages."""
    class ContextAddingStage(InsightPipelineStage):
        def process(self, context):
            return context.with_additional_context(
                stage_name=self.name
            )
    
    pipeline = InsightPipeline()
    pipeline.add_stage(ContextAddingStage("stage1", "First stage"))
    pipeline.add_stage(ContextAddingStage("stage2", "Second stage"))
    
    context = InsightExecutionContext(
        df=pd.DataFrame({'a': [1]}),
        query="test",
        user_role="analyst"
    )
    
    result = pipeline.execute(context)
    
    assert len(result["_execution_log"]) == 2
    assert all(log["status"] == "success" for log in result["_execution_log"])

def test_empty_pipeline():
    """Test execution of empty pipeline."""
    pipeline = InsightPipeline()
    
    context = InsightExecutionContext(
        df=pd.DataFrame({'a': [1]}),
        query="test",
        user_role="analyst"
    )
    
    result = pipeline.execute(context)
    
    assert isinstance(result, dict)
    assert len(result["_execution_log"]) == 0
    assert result["error_type"] == "PIPELINE_ERROR"