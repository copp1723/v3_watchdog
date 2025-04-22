"""
Integration tests for the fallback renderer with real dealership data scenarios.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from fallback_renderer import FallbackRenderer, ErrorCode, ErrorContext

@pytest.fixture
def sample_dealership_data():
    """Create sample dealership data for testing."""
    # Generate sample dates
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    
    # Create sample data
    data = {
        'date': dates,
        'dealership_id': np.random.randint(1, 100, size=len(dates)),
        'sales_amount': np.random.uniform(1000, 50000, size=len(dates)),
        'customer_satisfaction': np.random.uniform(1, 5, size=len(dates)),
        'service_appointments': np.random.randint(0, 20, size=len(dates)),
        'inventory_count': np.random.randint(50, 200, size=len(dates))
    }
    
    return pd.DataFrame(data)

@pytest.fixture
def dealership_schema():
    """Create sample dealership schema for testing."""
    return {
        'sales': 'Daily sales amount in dollars',
        'satisfaction': 'Customer satisfaction rating (1-5)',
        'appointments': 'Number of service appointments',
        'inventory': 'Current vehicle inventory count',
        'revenue': 'Total revenue including sales and service'
    }

def test_column_missing_error(sample_dealership_data, dealership_schema):
    """Test handling of missing column error with real data."""
    renderer = FallbackRenderer()
    
    # Try to access a non-existent column
    try:
        sample_dealership_data['non_existent_column']
    except KeyError:
        error_context = ErrorContext(
            error_code=ErrorCode.COLUMN_MISSING,
            error_message="Column 'non_existent_column' not found",
            details={"column_name": "non_existent_column"},
            timestamp=datetime.now().isoformat(),
            user_query="Show me non_existent_column trends",
            affected_columns=["non_existent_column"]
        )
        
        result = renderer.render_fallback(error_context)
        
        assert result["type"] == "error"
        assert result["error_code"] == "column_missing"
        assert "non_existent_column" in result["message"]
        assert "check your data file" in result["action"].lower()

def test_invalid_data_type_error(sample_dealership_data, dealership_schema):
    """Test handling of invalid data type error with real data."""
    renderer = FallbackRenderer()
    
    # Create a copy with invalid data type
    data = sample_dealership_data.copy()
    data['sales_amount'] = data['sales_amount'].astype(str)
    
    try:
        # Try to perform numeric operation on string data
        data['sales_amount'].mean()
    except Exception:
        error_context = ErrorContext(
            error_code=ErrorCode.INVALID_DATA_TYPE,
            error_message="Invalid data type in sales_amount column",
            details={
                "column_name": "sales_amount",
                "expected_type": "numeric",
                "actual_type": "string"
            },
            timestamp=datetime.now().isoformat(),
            user_query="Calculate average sales",
            affected_columns=["sales_amount"]
        )
        
        result = renderer.render_fallback(error_context)
        
        assert result["type"] == "error"
        assert result["error_code"] == "invalid_data_type"
        assert "sales_amount" in result["message"]
        assert "numeric" in result["action"].lower()

def test_no_matching_data_error(sample_dealership_data, dealership_schema):
    """Test handling of no matching data error with real data."""
    renderer = FallbackRenderer()
    
    # Filter for non-existent condition
    filtered_data = sample_dealership_data[sample_dealership_data['sales_amount'] > 1000000]
    
    if len(filtered_data) == 0:
        error_context = ErrorContext(
            error_code=ErrorCode.NO_MATCHING_DATA,
            error_message="No data found matching criteria",
            details={"criteria": "sales_amount > 1000000"},
            timestamp=datetime.now().isoformat(),
            user_query="Show me sales over $1M"
        )
        
        result = renderer.render_fallback(error_context)
        
        assert result["type"] == "error"
        assert result["error_code"] == "no_matching_data"
        assert "couldn't find any data" in result["message"].lower()
        assert "adjust your query" in result["action"].lower()

def test_error_rate_tracking(sample_dealership_data, dealership_schema):
    """Test error rate tracking functionality."""
    renderer = FallbackRenderer()
    
    # Generate multiple errors
    error_codes = [
        ErrorCode.COLUMN_MISSING,
        ErrorCode.INVALID_DATA_TYPE,
        ErrorCode.NO_MATCHING_DATA,
        ErrorCode.COLUMN_MISSING  # Duplicate to test counting
    ]
    
    for error_code in error_codes:
        error_context = ErrorContext(
            error_code=error_code,
            error_message=f"Test {error_code.value} error",
            details={},
            timestamp=datetime.now().isoformat()
        )
        renderer.render_fallback(error_context)
    
    # Check error statistics
    stats = renderer.get_error_stats()
    
    assert stats["column_missing"] == 2
    assert stats["invalid_data_type"] == 1
    assert stats["no_matching_data"] == 1
    assert len(stats) == 3  # Only three unique error types

def test_performance_monitoring(sample_dealership_data, dealership_schema):
    """Test performance monitoring functionality."""
    renderer = FallbackRenderer()
    
    error_context = ErrorContext(
        error_code=ErrorCode.SYSTEM_ERROR,
        error_message="Test system error",
        details={"error_details": "Test error"},
        timestamp=datetime.now().isoformat()
    )
    
    result = renderer.render_fallback(error_context)
    
    assert "render_time" in result
    assert isinstance(result["render_time"], float)
    assert result["render_time"] > 0  # Should take some time to render

def test_complex_error_scenario(sample_dealership_data, dealership_schema):
    """Test handling of a complex error scenario with real data."""
    renderer = FallbackRenderer()
    
    # Create a complex scenario with multiple issues
    data = sample_dealership_data.copy()
    
    # 1. Missing required column
    if 'revenue' not in data.columns:
        error_context = ErrorContext(
            error_code=ErrorCode.COLUMN_MISSING,
            error_message="Required column 'revenue' not found",
            details={"column_name": "revenue"},
            timestamp=datetime.now().isoformat(),
            user_query="Calculate revenue trends",
            affected_columns=["revenue"]
        )
        
        result = renderer.render_fallback(error_context)
        
        assert result["type"] == "error"
        assert result["error_code"] == "column_missing"
        assert "revenue" in result["message"]
    
    # 2. Invalid data type in existing column
    data['customer_satisfaction'] = data['customer_satisfaction'].astype(str)
    
    try:
        data['customer_satisfaction'].mean()
    except Exception:
        error_context = ErrorContext(
            error_code=ErrorCode.INVALID_DATA_TYPE,
            error_message="Invalid data type in customer_satisfaction column",
            details={
                "column_name": "customer_satisfaction",
                "expected_type": "numeric",
                "actual_type": "string"
            },
            timestamp=datetime.now().isoformat(),
            user_query="Calculate average customer satisfaction",
            affected_columns=["customer_satisfaction"]
        )
        
        result = renderer.render_fallback(error_context)
        
        assert result["type"] == "error"
        assert result["error_code"] == "invalid_data_type"
        assert "customer_satisfaction" in result["message"]
    
    # 3. No matching data for specific criteria
    filtered_data = data[data['sales_amount'] > 1000000]
    
    if len(filtered_data) == 0:
        error_context = ErrorContext(
            error_code=ErrorCode.NO_MATCHING_DATA,
            error_message="No data found matching criteria",
            details={"criteria": "sales_amount > 1000000"},
            timestamp=datetime.now().isoformat(),
            user_query="Show me sales over $1M"
        )
        
        result = renderer.render_fallback(error_context)
        
        assert result["type"] == "error"
        assert result["error_code"] == "no_matching_data"
        assert "couldn't find any data" in result["message"].lower() 