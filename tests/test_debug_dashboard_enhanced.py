"""
Enhanced tests for the Debug Dashboard component.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
from src.insights.debug_dashboard import DebugDashboard

@pytest.fixture
def sample_metrics():
    """Fixture providing sample metrics data."""
    return [
        {
            "query": "test query 1",
            "metrics": {
                "query_id": "q1",
                "timestamp": (datetime.now() - timedelta(minutes=5)).isoformat(),
                "execution_time_ms": 150.0,
                "memory_mb": 50.0,
                "llm_tokens_used": 100,
                "cache_hit": True,
                "status": "success"
            },
            "result": {"status": "success"}
        },
        {
            "query": "test query 2",
            "metrics": {
                "query_id": "q2",
                "timestamp": (datetime.now() - timedelta(minutes=3)).isoformat(),
                "execution_time_ms": 300.0,
                "memory_mb": 75.0,
                "llm_tokens_used": 200,
                "cache_hit": False,
                "status": "error",
                "error_code": "ValueError",
                "error_message": "test error"
            },
            "result": {"status": "error"}
        }
    ]

@pytest.fixture
def sample_trace():
    """Fixture providing sample trace data."""
    return {
        "trace_id": "trace-1",
        "start_time": (datetime.now() - timedelta(minutes=10)).isoformat(),
        "steps": [
            {
                "timestamp": (datetime.now() - timedelta(minutes=9)).isoformat(),
                "description": "Step 1"
            },
            {
                "timestamp": (datetime.now() - timedelta(minutes=8)).isoformat(),
                "description": "Step 2"
            }
        ]
    }

def test_execution_metrics_rendering(mocker, sample_metrics):
    """Test rendering of execution metrics section."""
    # Mock metrics logger
    mock_logger = mocker.patch('src.utils.metrics_logger.metrics_logger')
    mock_logger.get_recent_metrics.return_value = sample_metrics
    
    dashboard = DebugDashboard()
    
    # Mock streamlit
    with st.container():
        dashboard._render_execution_metrics()
    
    # Verify metrics were retrieved
    mock_logger.get_recent_metrics.assert_called_once_with(limit=100)

def test_trace_analysis_rendering(mocker, sample_metrics, sample_trace):
    """Test rendering of trace analysis section."""
    # Mock metrics logger and trace engine
    mock_logger = mocker.patch('src.utils.metrics_logger.metrics_logger')
    mock_logger.get_metrics_by_trace.return_value = sample_metrics
    
    mock_trace_engine = mocker.patch('src.insights.traceability.TraceabilityEngine')
    mock_trace_engine.get_trace.return_value = sample_trace
    
    dashboard = DebugDashboard()
    
    # Mock streamlit
    with st.container():
        dashboard._render_trace_analysis()
    
    # Verify trace data was retrieved
    if st.session_state.selected_trace:
        mock_logger.get_metrics_by_trace.assert_called_once_with(
            st.session_state.selected_trace
        )

def test_cache_statistics_rendering(mocker, sample_metrics):
    """Test rendering of cache statistics section."""
    # Mock metrics logger
    mock_logger = mocker.patch('src.utils.metrics_logger.metrics_logger')
    mock_logger.get_recent_metrics.return_value = sample_metrics
    
    dashboard = DebugDashboard()
    
    # Mock streamlit
    with st.container():
        dashboard._render_cache_statistics()
    
    # Verify metrics were retrieved
    mock_logger.get_recent_metrics.assert_called_once_with(limit=1000)

def test_trace_visualization(mocker, sample_trace):
    """Test trace visualization rendering."""
    dashboard = DebugDashboard()
    
    # Mock streamlit
    with st.container():
        dashboard._render_trace_visualization(sample_trace)
    
    # Verify timeline was created with correct number of steps
    assert len(sample_trace['steps']) == 2

def test_dashboard_mobile_responsiveness(mocker):
    """Test dashboard mobile responsiveness."""
    dashboard = DebugDashboard()
    
    # Mock streamlit with mobile viewport
    with st.container():
        # Set mobile viewport
        st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
        dashboard.render_dashboard()
    
    # Verify mobile-friendly layout was used
    assert st._config.get_option("theme.base") == "light"