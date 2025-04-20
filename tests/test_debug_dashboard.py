"""
Tests for the Debug Dashboard component.

This module contains tests for the DebugDashboard class, including tests for
metrics calculation, visualization generation, and dashboard functionality.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import streamlit as st
from unittest.mock import MagicMock, patch
from src.insights.debug_dashboard import DebugDashboard

# Test Categories
pytestmark = [
    pytest.mark.dashboard,
    pytest.mark.integration
]

@pytest.fixture(scope="function")
def mock_metrics_data():
    """Fixture providing sample metrics data."""
    return [
        {
            "metrics": {
                "execution_time_ms": 150,
                "memory_mb": 256,
                "llm_tokens_used": 1000,
                "cache_hit": True,
                "status": "success",
                "timestamp": "2024-03-15T10:30:00"
            },
            "query": "Test query 1",
            "result": {"status": "success"}
        },
        {
            "metrics": {
                "execution_time_ms": 200,
                "memory_mb": 512,
                "llm_tokens_used": 2000,
                "cache_hit": False,
                "status": "error",
                "error_code": "invalid_input",
                "timestamp": "2024-03-15T10:31:00"
            },
            "query": "Test query 2",
            "result": {"status": "error"}
        }
    ]

@pytest.fixture
def dashboard():
    """Fixture providing a DebugDashboard instance."""
    return DebugDashboard()

@pytest.mark.performance
def test_large_dataset_performance():
    """Test dashboard performance with large datasets."""
    # Generate large mock dataset
    num_entries = 1000
    large_metrics_data = []
    
    for i in range(num_entries):
        large_metrics_data.append({
            "metrics": {
                "execution_time_ms": np.random.randint(100, 500),
                "memory_mb": np.random.randint(200, 1000),
                "llm_tokens_used": np.random.randint(500, 5000),
                "cache_hit": np.random.choice([True, False]),
                "status": np.random.choice(["success", "error"]),
                "timestamp": (datetime.now() + pd.Timedelta(minutes=i)).isoformat()
            },
            "query": f"Test query {i}",
            "result": {"status": "success"}
        })
    
    # Mock metrics logger
    with patch('src.utils.metrics_logger.metrics_logger') as mock_logger:
        mock_logger.get_recent_metrics.return_value = large_metrics_data
        
        # Create dashboard and measure performance
        import time
        
        # Initialize dashboard
        start_time = time.time()
        dashboard = DebugDashboard()
        init_time = time.time() - start_time
        assert init_time < 1.0, f"Dashboard initialization took {init_time:.2f}s, exceeding 1s limit"
        
        # Test execution metrics rendering
        start_time = time.time()
        with patch('streamlit.plotly_chart'):  # Mock streamlit rendering
            dashboard._render_execution_metrics()
        metrics_time = time.time() - start_time
        assert metrics_time < 3.0, f"Execution metrics rendering took {metrics_time:.2f}s, exceeding 3s limit"
        
        # Test cache statistics rendering
        start_time = time.time()
        with patch('streamlit.plotly_chart'):  # Mock streamlit rendering
            dashboard._render_cache_statistics()
        cache_time = time.time() - start_time
        assert cache_time < 3.0, f"Cache statistics rendering took {cache_time:.2f}s, exceeding 3s limit"
        
        # Verify memory usage
        import psutil
        import os
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / 1024 / 1024  # Convert to MB
        assert memory_usage < 500, f"Memory usage {memory_usage:.2f}MB exceeds 500MB limit"
        
        # Log performance metrics
        print("\nPerformance Metrics:")
        print(f"Initialization Time: {init_time:.2f}s")
        print(f"Execution Metrics Time: {metrics_time:.2f}s")
        print(f"Cache Statistics Time: {cache_time:.2f}s")
        print(f"Memory Usage: {memory_usage:.2f}MB")

@pytest.mark.error_handling
def test_invalid_data_handling(dashboard):
    """Test dashboard handling of corrupted or invalid metrics data."""
    # Test with missing required fields
    invalid_data_1 = [{
        "metrics": {}  # Empty metrics
    }]
    
    # Test with invalid data types
    invalid_data_2 = [{
        "metrics": {
            "execution_time_ms": "invalid",
            "memory_mb": None,
            "llm_tokens_used": "1000",
            "cache_hit": "yes",
            "status": 123,
            "timestamp": "invalid-date"
        }
    }]
    
    # Test with malformed structure
    invalid_data_3 = "not a list"
    
    # Test dashboard handling of invalid data
    with patch('src.utils.metrics_logger.metrics_logger') as mock_logger:
        for i, invalid_data in enumerate([invalid_data_1, invalid_data_2, invalid_data_3], 1):
            mock_logger.get_recent_metrics.return_value = invalid_data
            
            try:
                # Test execution metrics rendering
                with patch('streamlit.plotly_chart'):  # Mock streamlit rendering
                    dashboard._render_execution_metrics()
                
                # Test cache statistics rendering
                with patch('streamlit.plotly_chart'):  # Mock streamlit rendering
                    dashboard._render_cache_statistics()
                
            except Exception as e:
                pytest.fail(f"Dashboard failed to handle invalid data {i}: {str(e)}")
    
    # Test trace lookup with invalid trace ID
    with patch('src.utils.metrics_logger.metrics_logger') as mock_logger:
        mock_logger.get_metrics_by_trace.return_value = None
        
        with patch('streamlit.warning') as mock_warning:
            dashboard._render_trace_analysis()
            mock_warning.assert_called_once() 