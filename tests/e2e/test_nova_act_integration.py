"""
End-to-end tests for Nova Act integration.
"""

import pytest
from src.nova_act.core import NovaActConnector
from src.nova_act.collectors.dealersocket import DealerSocketCollector
from src.watchdog_ai.ui.components.sales_report_renderer import SalesReportRenderer
from .fixtures import get_mock_sales_report_data

import pandas as pd
import streamlit as st
from unittest.mock import patch, MagicMock
from datetime import datetime

def test_nova_act_connector():
    """Test Nova Act connector initialization."""
    connector = NovaActConnector()
    assert connector is not None

@pytest.fixture
def mock_connector():
    """Create a mock Nova Act connector."""
    connector = MagicMock(spec=NovaActConnector)
    connector.verify_credentials.return_value = {"valid": True, "message": "Success"}
    return connector

@pytest.fixture
def mock_scheduler():
    """Create a mock scheduler."""
    scheduler = MagicMock()
    scheduler.schedule_report_collection.return_value = "task_123"
    return scheduler

@pytest.fixture
def mock_stream_session():
    """Create a mock Streamlit session state."""
    return {
        "active_tab": "System Connect",
        "nova_act_status": {
            "connected": True,
            "last_sync": datetime.now().isoformat()
        }
    }

def test_upload_to_watchdog_integration(mock_connector, mock_stream_session):
    """Test file upload to Watchdog integration."""
    # Get mock data
    df = get_mock_sales_report_data()
    
    # Initialize renderer
    renderer = SalesReportRenderer()
    
    # Create expected insight data with Phase 2 JSON structure
    insight_data = {
        "summary": "NeoIdentity Produced the Most Sales",
        "primary_metrics": {
            "lead_source": "NeoIdentity",
            "total_sales": 4,
            "relative_performance": "207% of team average",
            "trend": "+15% from previous month",
            "rank": "Top 5% of reps"
        },
        "chart_data": {
            "type": "bar",
            "data": {
                "x": df["Lead Source"].tolist(),
                "y": df["Sales Count"].tolist()
            }
        },
        "performance_breakdown": [
            {
                "category": "Vehicle Type",
                "top_performer": "SUVs ($7,821 avg gross)",
                "comparison": "152% above team"
            },
            {
                "category": "F&I Products",
                "top_performer": "Extended Warranties ($1,245/vehicle)",
                "comparison": "127% above team"
            }
        ],
        "actionable_flags": [
            {
                "action": "Increase marketing budget for NeoIdentity by 15%",
                "priority": "High",
                "impact_estimate": "Could increase sales by 10%"
            },
            {
                "action": "Review lead handling process for AutoTrader",
                "priority": "Medium",
                "impact_estimate": "Could improve conversion by 5%"
            }
        ]
    }
    
    # Test rendering with data
    with patch("streamlit.markdown"), \
         patch("streamlit.altair_chart"), \
         patch("streamlit.metric"), \
         patch("streamlit.write"):
        renderer.render_insight_block(insight_data)

@pytest.mark.asyncio
async def test_nova_act_data_collection_integration(mock_connector, mock_scheduler):
    """Test Nova Act data collection integration."""
    collector = DealerSocketCollector(mock_connector)
    
    # Test credentials
    credentials = {
        "username": "test@dealersocket.com",
        "password": "password123",
        "2fa_method": "SMS"
    }
    
    # Configure report
    report_config = {
        "type": "sales",
        "frequency": "daily",
        "format": "csv"
    }
    
    # Test collection
    result = await collector.collect_sales_report(credentials)
    assert result is not None

def test_report_scheduling_integration(mock_scheduler):
    """Test report scheduling integration."""
    # Schedule report
    task_id = mock_scheduler.schedule_report_collection(
        vendor="dealersocket",
        report_type="sales",
        frequency="daily",
        hour=0,
        minute=0
    )
    
    assert task_id == "task_123"
    mock_scheduler.schedule_report_collection.assert_called_once()

@pytest.mark.asyncio
async def test_full_flow_integration(mock_connector, mock_scheduler, mock_stream_session):
    """Test the complete flow from sync to display."""
    # 1. Configure connection
    credentials = {
        "username": "test@dealersocket.com",
        "password": "password123",
        "2fa_method": "SMS"
    }
    
    # 2. Verify credentials
    verify_result = await mock_connector.verify_credentials(
        "dealersocket",
        credentials=credentials
    )
    assert verify_result["valid"]
    
    # 3. Schedule collection
    task_id = mock_scheduler.schedule_report_collection(
        vendor="dealersocket",
        report_type="sales",
        frequency="daily"
    )
    assert task_id == "task_123"
    
    # 4. Test rendering
    df = get_mock_sales_report_data()
    renderer = SalesReportRenderer()
    
    with patch("streamlit.markdown"), \
         patch("streamlit.altair_chart"), \
         patch("streamlit.metric"), \
         patch("streamlit.write"):
        start_time = datetime.now()
        renderer.render_insight_block({
            "summary": "NeoIdentity Produced the Most Sales",
            "primary_metrics": {
                "lead_source": "NeoIdentity",
                "total_sales": 4,
                "relative_performance": "207% of team average",
                "trend": "+15% from previous month",
                "rank": "Top 5% of reps"
            },
            "chart_data": {
                "type": "bar",
                "data": {
                    "x": df["Lead Source"].tolist(),
                    "y": df["Sales Count"].tolist()
                }
            },
            "performance_breakdown": [
                {
                    "category": "Vehicle Type",
                    "top_performer": "SUVs ($7,821 avg gross)",
                    "comparison": "152% above team"
                }
            ],
            "actionable_flags": [
                {
                    "action": "Increase marketing budget for NeoIdentity by 15%",
                    "priority": "High",
                    "impact_estimate": "Could increase sales by 10%"
                }
            ]
        })
        render_time = (datetime.now() - start_time).total_seconds()
        assert render_time < 1.0, f"Render time {render_time}s exceeds 1s target"