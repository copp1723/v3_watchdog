"""
Unit tests for DealerSocket collector implementation.
"""

import pytest
from unittest.mock import MagicMock, patch
from src.nova_act.collectors.dealersocket import DealerSocketCollector
from src.nova_act.core import NovaActManager

@pytest.fixture
def mock_nova_manager():
    """Provide a mock Nova Act manager."""
    with patch('src.nova_act.core.NovaActManager') as mock:
        instance = mock.return_value
        instance.collect_data = MagicMock()
        yield instance

@pytest.fixture
def collector(mock_nova_manager):
    """Provide a configured DealerSocket collector."""
    return DealerSocketCollector(mock_nova_manager)

def test_collector_initialization(collector):
    """Test collector initialization and configuration."""
    assert collector.nova is not None
    assert isinstance(collector.selectors, dict)
    assert "login" in collector.selectors
    assert "reports" in collector.selectors
    assert "download" in collector.selectors
    
    # Verify DealerSocket-specific selectors
    assert collector.selectors["login"]["username"] == "#txtUsername"
    assert collector.selectors["reports"]["deals"] == "#dealReports"

@pytest.mark.asyncio
async def test_collect_deal_summary(collector, test_data):
    """Test deal summary report collection."""
    # Configure mock response
    collector.nova.collect_data.return_value = {
        "success": True,
        "file_path": "deal_summary.csv",
        "metadata": {"type": "deals"}
    }
    
    # Test collection
    result = await collector.collect_deal_summary(test_data["credentials"])
    
    # Verify correct configuration was passed
    collector.nova.collect_data.assert_called_once()
    args = collector.nova.collect_data.call_args[1]
    assert args["vendor"] == "dealersocket"
    assert args["credentials"] == test_data["credentials"]
    assert args["report_config"]["type"] == "deals"
    
    # Verify result
    assert result["success"] is True
    assert result["file_path"] == "deal_summary.csv"
    assert result["metadata"]["type"] == "deals"

@pytest.mark.asyncio
async def test_collect_lead_activity(collector, test_data):
    """Test lead activity report collection."""
    # Configure mock response
    collector.nova.collect_data.return_value = {
        "success": True,
        "file_path": "lead_activity.csv",
        "metadata": {"type": "leads"}
    }
    
    # Test collection
    result = await collector.collect_lead_activity(test_data["credentials"])
    
    # Verify correct configuration was passed
    collector.nova.collect_data.assert_called_once()
    args = collector.nova.collect_data.call_args[1]
    assert args["vendor"] == "dealersocket"
    assert args["credentials"] == test_data["credentials"]
    assert args["report_config"]["type"] == "leads"
    
    # Verify result
    assert result["success"] is True
    assert result["file_path"] == "lead_activity.csv"
    assert result["metadata"]["type"] == "leads"

@pytest.mark.asyncio
async def test_collect_sales_performance(collector, test_data):
    """Test sales performance report collection."""
    # Configure mock response
    collector.nova.collect_data.return_value = {
        "success": True,
        "file_path": "sales_performance.csv",
        "metadata": {"type": "sales"}
    }
    
    # Test collection
    result = await collector.collect_sales_performance(test_data["credentials"])
    
    # Verify correct configuration was passed
    collector.nova.collect_data.assert_called_once()
    args = collector.nova.collect_data.call_args[1]
    assert args["vendor"] == "dealersocket"
    assert args["credentials"] == test_data["credentials"]
    assert args["report_config"]["type"] == "sales"
    
    # Verify result
    assert result["success"] is True
    assert result["file_path"] == "sales_performance.csv"
    assert result["metadata"]["type"] == "sales"

@pytest.mark.asyncio
async def test_collect_inventory_report(collector, test_data):
    """Test inventory report collection."""
    # Configure mock response
    collector.nova.collect_data.return_value = {
        "success": True,
        "file_path": "inventory_report.csv",
        "metadata": {"type": "inventory"}
    }
    
    # Test collection
    result = await collector.collect_inventory_report(test_data["credentials"])
    
    # Verify correct configuration was passed
    collector.nova.collect_data.assert_called_once()
    args = collector.nova.collect_data.call_args[1]
    assert args["vendor"] == "dealersocket"
    assert args["credentials"] == test_data["credentials"]
    assert args["report_config"]["type"] == "inventory"
    
    # Verify result
    assert result["success"] is True
    assert result["file_path"] == "inventory_report.csv"
    assert result["metadata"]["type"] == "inventory"

@pytest.mark.asyncio
async def test_error_handling(collector, test_data):
    """Test error handling in collector methods."""
    # Configure mock to raise an exception
    collector.nova.collect_data.side_effect = Exception("Test error")
    
    # Test error handling in each collection method
    methods = [
        collector.collect_deal_summary,
        collector.collect_lead_activity,
        collector.collect_sales_performance,
        collector.collect_inventory_report
    ]
    
    for method in methods:
        result = await method(test_data["credentials"])
        assert result["success"] is False
        assert "error" in result
        assert "Test error" in result["error"]

@pytest.mark.asyncio
async def test_selector_validation(collector, test_data):
    """Test that all required selectors are present."""
    # Test each report type
    report_types = ["deals", "leads", "sales", "inventory"]
    methods = {
        "deals": collector.collect_deal_summary,
        "leads": collector.collect_lead_activity,
        "sales": collector.collect_sales_performance,
        "inventory": collector.collect_inventory_report
    }
    
    for report_type, method in methods.items():
        # Configure mock response
        collector.nova.collect_data.return_value = {
            "success": True,
            "file_path": f"{report_type}_report.csv",
            "metadata": {"type": report_type}
        }
        
        # Test collection
        result = await method(test_data["credentials"])
        
        # Verify selectors in report config
        args = collector.nova.collect_data.call_args[1]
        config = args["report_config"]
        
        assert "selectors" in config
        assert collector.selectors["login"] == config["selectors"]["login"]
        assert collector.selectors["reports"] == config["selectors"]["reports"]
        assert collector.selectors["download"] == config["selectors"]["download"]

if __name__ == "__main__":
    pytest.main([__file__])