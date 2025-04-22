"""
Tests for the InsightExecutionContext class.
"""

import pytest
import pandas as pd
from datetime import datetime
from src.watchdog_ai.insights.context import InsightExecutionContext, InsightPipelineStage, InsightPipeline

def test_context_creation_with_required_fields():
    """Test context creation with required fields."""
    df = pd.DataFrame({'a': [1, 2, 3]})
    context = InsightExecutionContext(
        df=df,
        query="test query",
        user_role="analyst"
    )
    
    assert context.df is not None
    assert context.query == "test query"
    assert context.user_role == "analyst"
    assert context.trace_id is not None
    assert isinstance(context.context_vars, dict)
    assert isinstance(context.timestamp, str)

def test_context_validation():
    """Test context validation."""
    with pytest.raises(ValueError):
        InsightExecutionContext(
            df=pd.DataFrame(),  # Empty DataFrame
            query="test",
            user_role="analyst"
        )
    
    with pytest.raises(ValueError):
        InsightExecutionContext(
            df=pd.DataFrame({'a': [1]}),
            query="",  # Empty query
            user_role="analyst"
        )

def test_post_init_metadata_creation():
    """Test automatic metadata creation in post_init."""
    df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
    context = InsightExecutionContext(
        df=df,
        query="test query",
        user_role="analyst"
    )
    
    assert context.context_vars["record_count"] == 3
    assert context.context_vars["column_count"] == 2
    assert set(context.context_vars["column_list"]) == {'a', 'b'}
    assert isinstance(context.context_vars["data_types"], dict)

def test_with_additional_context():
    """Test adding additional context variables."""
    df = pd.DataFrame({'a': [1, 2, 3]})
    context = InsightExecutionContext(
        df=df,
        query="test query",
        user_role="analyst"
    )
    
    new_context = context.with_additional_context(
        test_var="test value",
        another_var=123
    )
    
    assert new_context.context_vars["test_var"] == "test value"
    assert new_context.context_vars["another_var"] == 123
    assert new_context.df is context.df
    assert new_context.query == context.query
    assert new_context.trace_id == context.trace_id

def test_get_schema_context():
    """Test getting schema context."""
    df = pd.DataFrame({'a': [1, 2, 3]})
    context = InsightExecutionContext(
        df=df,
        query="test query",
        user_role="analyst"
    )
    
    schema_context = context.get_schema_context()
    assert isinstance(schema_context, dict)

class MockPipelineStage(InsightPipelineStage):
    """Mock pipeline stage for testing."""
    def process(self, context: InsightExecutionContext) -> InsightExecutionContext:
        return context.with_additional_context(processed_by=self.name)

def test_pipeline_execution_success():
    """Test successful pipeline execution."""
    df = pd.DataFrame({'a': [1, 2, 3]})
    context = InsightExecutionContext(
        df=df,
        query="test query",
        user_role="analyst"
    )
    
    pipeline = InsightPipeline()
    pipeline.add_stage(MockPipelineStage("stage1", "Test stage 1"))
    pipeline.add_stage(MockPipelineStage("stage2", "Test stage 2"))
    
    result = pipeline.execute(context)
    
    assert isinstance(result, dict)
    assert "_execution_log" in result
    assert len(result["_execution_log"]) == 2

def test_pipeline_execution_error():
    """Test pipeline execution with error."""
    class ErrorStage(InsightPipelineStage):
        def process(self, context):
            raise ValueError("Test error")
    
    df = pd.DataFrame({'a': [1, 2, 3]})
    context = InsightExecutionContext(
        df=df,
        query="test query",
        user_role="analyst"
    )
    
    pipeline = InsightPipeline()
    pipeline.add_stage(ErrorStage("error_stage", "Error stage"))
    
    result = pipeline.execute(context)
    
    assert "error" in result["_execution_log"][0]
    assert result["error_type"] == "PIPELINE_ERROR"

def test_pipeline_exception_handling():
    """Test pipeline exception handling."""
    df = pd.DataFrame({'a': [1, 2, 3]})
    context = InsightExecutionContext(
        df=df,
        query="test query",
        user_role="analyst"
    )
    
    pipeline = InsightPipeline()
    
    # Test with no stages
    result = pipeline.execute(context)
    assert isinstance(result, dict)
    assert result["error_type"] == "PIPELINE_ERROR"

def test_stage_failure():
    """Test handling of stage failure."""
    class FailingStage(InsightPipelineStage):
        def process(self, context):
            context = context.with_additional_context(stage_failed=True)
            raise ValueError("Stage failure")
            return context
    
    df = pd.DataFrame({'a': [1, 2, 3]})
    context = InsightExecutionContext(
        df=df,
        query="test query",
        user_role="analyst"
    )
    
    pipeline = InsightPipeline()
    pipeline.add_stage(FailingStage("failing_stage", "Failing stage"))
    pipeline.add_stage(MockPipelineStage("next_stage", "Next stage"))
    
    result = pipeline.execute(context)
    
    assert len(result["_execution_log"]) == 1
    assert result["_execution_log"][0]["status"] == "error"
    assert "Stage failure" in result["_execution_log"][0]["error"]

def test_context_immutability():
    """Test that context modifications create new instances."""
    df = pd.DataFrame({'a': [1, 2, 3]})
    original_context = InsightExecutionContext(
        df=df,
        query="test query",
        user_role="analyst"
    )
    
    modified_context = original_context.with_additional_context(new_var=42)
    
    assert id(original_context) != id(modified_context)
    assert "new_var" not in original_context.context_vars
    assert modified_context.context_vars["new_var"] == 42

def test_execution_log_timing():
    """Test that execution log includes timing information."""
    df = pd.DataFrame({'a': [1, 2, 3]})
    context = InsightExecutionContext(
        df=df,
        query="test query",
        user_role="analyst"
    )
    
    pipeline = InsightPipeline()
    pipeline.add_stage(MockPipelineStage("timed_stage", "Timed stage"))
    
    result = pipeline.execute(context)
    
    assert "duration_seconds" in result["_execution_log"][0]
    assert isinstance(result["_execution_log"][0]["duration_seconds"], float)
    assert result["_execution_log"][0]["duration_seconds"] >= 0