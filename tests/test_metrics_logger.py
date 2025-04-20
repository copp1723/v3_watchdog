"""
Tests for the metrics logging system.
"""

import os
import json
import time
import pytest
from datetime import datetime
from src.utils.metrics_logger import MetricsLogger, QueryMetrics

@pytest.fixture
def temp_log_dir(tmpdir):
    """Fixture providing temporary log directory."""
    return str(tmpdir.mkdir("metrics"))

@pytest.fixture
def metrics_logger(temp_log_dir):
    """Fixture providing MetricsLogger instance."""
    return MetricsLogger(log_dir=temp_log_dir)

def test_log_query_success(metrics_logger, temp_log_dir):
    """Test logging successful query execution."""
    query_id = "test-query-1"
    query = "test query"
    result = {"status": "success"}
    start_time = time.time() - 1  # 1 second ago
    
    metrics = metrics_logger.log_query(
        query_id=query_id,
        query=query,
        result=result,
        start_time=start_time,
        llm_tokens=100,
        cache_hit=False
    )
    
    # Verify metrics object
    assert isinstance(metrics, QueryMetrics)
    assert metrics.query_id == query_id
    assert metrics.status == "success"
    assert metrics.execution_time_ms >= 1000  # Should be ~1000ms
    assert metrics.memory_mb > 0
    assert metrics.llm_tokens_used == 100
    assert not metrics.cache_hit
    
    # Verify log file
    log_file = os.path.join(temp_log_dir, "query_metrics.json")
    assert os.path.exists(log_file)
    
    with open(log_file) as f:
        log_entry = json.loads(f.read())
        assert log_entry["query"] == query
        assert log_entry["result"] == result
        assert log_entry["metrics"]["query_id"] == query_id

def test_log_query_error(metrics_logger):
    """Test logging failed query execution."""
    error = ValueError("test error")
    
    metrics = metrics_logger.log_query(
        query_id="test-query-2",
        query="test query",
        result={},
        start_time=time.time(),
        llm_tokens=0,
        cache_hit=False,
        error=error
    )
    
    assert metrics.status == "error"
    assert metrics.error_code == "ValueError"
    assert metrics.error_message == "test error"

def test_get_metrics_by_trace(metrics_logger):
    """Test retrieving metrics by trace ID."""
    trace_id = "test-trace-1"
    
    # Log some queries with the trace ID
    metrics_logger.log_query(
        query_id="query-1",
        query="test query 1",
        result={},
        start_time=time.time(),
        llm_tokens=100,
        cache_hit=False,
        trace_id=trace_id
    )
    
    metrics_logger.log_query(
        query_id="query-2",
        query="test query 2",
        result={},
        start_time=time.time(),
        llm_tokens=200,
        cache_hit=False,
        trace_id=trace_id
    )
    
    # Get metrics for trace
    metrics = metrics_logger.get_metrics_by_trace(trace_id)
    
    assert len(metrics) == 2
    assert all(m["metrics"]["trace_id"] == trace_id for m in metrics)

def test_get_recent_metrics(metrics_logger):
    """Test retrieving recent metrics."""
    # Log some queries
    for i in range(5):
        metrics_logger.log_query(
            query_id=f"query-{i}",
            query=f"test query {i}",
            result={},
            start_time=time.time(),
            llm_tokens=100,
            cache_hit=False
        )
    
    # Get recent metrics
    metrics = metrics_logger.get_recent_metrics(limit=3)
    
    assert len(metrics) == 3
    # Verify they're in reverse chronological order
    timestamps = [m["metrics"]["timestamp"] for m in metrics]
    assert timestamps == sorted(timestamps, reverse=True)

def test_log_rotation(metrics_logger, temp_log_dir):
    """Test log file rotation."""
    log_file = os.path.join(temp_log_dir, "query_metrics.json")
    
    # Generate enough log entries to trigger rotation
    large_result = {"data": "x" * 1000000}  # 1MB of data
    for i in range(15):  # Should generate >10MB of logs
        metrics_logger.log_query(
            query_id=f"query-{i}",
            query="test query",
            result=large_result,
            start_time=time.time(),
            llm_tokens=100,
            cache_hit=False
        )
    
    # Verify rotation occurred
    assert os.path.exists(log_file)
    assert os.path.exists(log_file + ".1")  # At least one backup file should exist