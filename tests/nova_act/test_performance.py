"""
Performance tests for Nova Act integration.
Tests response times, concurrent operations, and resource usage.
"""

import pytest
import asyncio
import time
import psutil
import os
from typing import Dict, Any, List
from unittest.mock import MagicMock, patch

from src.nova_act.core import NovaActManager
from src.nova_act.collectors.vinsolutions import VinSolutionsCollector
from src.nova_act.collectors.dealersocket import DealerSocketCollector

@pytest.fixture
def mock_nova_manager():
    """Provide a mock Nova Act manager with timing capabilities."""
    with patch('src.nova_act.core.NovaActManager') as mock:
        instance = mock.return_value
        
        # Add timing to collect_data
        async def timed_collect_data(*args, **kwargs):
            await asyncio.sleep(0.1)  # Simulate network delay
            return {
                "success": True,
                "file_path": "test_report.csv",
                "metadata": {"type": "test"}
            }
        
        instance.collect_data = timed_collect_data
        yield instance

@pytest.fixture
def collectors(mock_nova_manager):
    """Provide configured collectors for testing."""
    return {
        "vinsolutions": VinSolutionsCollector(mock_nova_manager),
        "dealersocket": DealerSocketCollector(mock_nova_manager)
    }

@pytest.mark.asyncio
async def test_single_collector_performance(collectors, test_data):
    """Test performance of a single collector's operations."""
    collector = collectors["vinsolutions"]
    
    # Measure single operation time
    start_time = time.time()
    result = await collector.collect_sales_report(test_data["credentials"])
    end_time = time.time()
    
    # Verify response time is within acceptable range
    response_time = end_time - start_time
    assert response_time < 1.0, f"Single operation took too long: {response_time:.2f}s"
    
    # Measure memory usage
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / 1024 / 1024  # Convert to MB
    assert memory_usage < 500, f"Memory usage too high: {memory_usage:.2f}MB"

@pytest.mark.asyncio
async def test_concurrent_collector_performance(collectors, test_data):
    """Test performance of concurrent collector operations."""
    # Create a mix of collection tasks
    tasks = []
    for collector_name, collector in collectors.items():
        tasks.extend([
            collector.collect_sales_report(test_data["credentials"]),
            collector.collect_inventory_report(test_data["credentials"])
        ])
    
    # Measure concurrent execution time
    start_time = time.time()
    results = await asyncio.gather(*tasks)
    end_time = time.time()
    
    # Verify all operations succeeded
    assert all(result["success"] for result in results)
    
    # Verify total time is reasonable for concurrent operations
    total_time = end_time - start_time
    assert total_time < 2.0, f"Concurrent operations took too long: {total_time:.2f}s"
    
    # Verify memory usage during concurrent operations
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / 1024 / 1024  # Convert to MB
    assert memory_usage < 1000, f"Memory usage too high during concurrent ops: {memory_usage:.2f}MB"

@pytest.mark.asyncio
async def test_collector_scalability(collectors, test_data):
    """Test collector performance under increasing load."""
    collector = collectors["vinsolutions"]
    
    # Test with increasing numbers of concurrent operations
    concurrent_levels = [2, 5, 10]
    
    for num_concurrent in concurrent_levels:
        # Create multiple collection tasks
        tasks = [
            collector.collect_sales_report(test_data["credentials"])
            for _ in range(num_concurrent)
        ]
        
        # Measure execution time
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        # Verify all operations succeeded
        assert all(result["success"] for result in results)
        
        # Verify execution time scales reasonably
        total_time = end_time - start_time
        assert total_time < num_concurrent * 0.5, \
            f"Scalability test with {num_concurrent} operations took too long: {total_time:.2f}s"

@pytest.mark.asyncio
async def test_error_recovery_performance(collectors, test_data):
    """Test performance of error recovery mechanisms."""
    collector = collectors["vinsolutions"]
    
    # Configure mock to fail on first attempt
    attempt_count = 0
    
    async def failing_collect_data(*args, **kwargs):
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count == 1:
            raise Exception("First attempt failed")
        await asyncio.sleep(0.1)
        return {
            "success": True,
            "file_path": "test_report.csv",
            "metadata": {"type": "test"}
        }
    
    collector.nova.collect_data = failing_collect_data
    
    # Measure recovery time
    start_time = time.time()
    result = await collector.collect_sales_report(test_data["credentials"])
    end_time = time.time()
    
    # Verify recovery time is reasonable
    recovery_time = end_time - start_time
    assert recovery_time < 1.0, f"Error recovery took too long: {recovery_time:.2f}s"
    assert result["success"] is True

@pytest.mark.asyncio
async def test_resource_cleanup(collectors, test_data):
    """Test resource cleanup performance after operations."""
    collector = collectors["vinsolutions"]
    
    # Record initial resource usage
    initial_process = psutil.Process(os.getpid())
    initial_memory = initial_process.memory_info().rss
    initial_handles = initial_process.num_handles() if os.name == 'nt' else 0
    
    # Perform multiple operations
    tasks = [
        collector.collect_sales_report(test_data["credentials"])
        for _ in range(5)
    ]
    await asyncio.gather(*tasks)
    
    # Allow time for cleanup
    await asyncio.sleep(0.5)
    
    # Check final resource usage
    final_process = psutil.Process(os.getpid())
    final_memory = final_process.memory_info().rss
    final_handles = final_process.num_handles() if os.name == 'nt' else 0
    
    # Verify resources were properly cleaned up
    memory_diff = (final_memory - initial_memory) / 1024 / 1024  # Convert to MB
    assert memory_diff < 50, f"Memory not properly cleaned up: {memory_diff:.2f}MB retained"
    
    if os.name == 'nt':
        handle_diff = final_handles - initial_handles
        assert handle_diff < 10, f"Resource handles not properly cleaned up: {handle_diff} handles retained"

if __name__ == "__main__":
    pytest.main([__file__])