"""
End-to-end tests for KPI dashboard.
"""

import pytest
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
from src.ui.pages.kpi_dashboard import kpi_dashboard

@pytest.fixture
def mock_session_state():
    """Setup mock session state."""
    # Store original session state
    original_state = st.session_state
    
    # Create test data
    sales_data = pd.DataFrame({
        'SalesRepName': ['Alice', 'Bob', 'Charlie'] * 3,
        'TotalGross': [1000, 2000, 1500, 2000, 1000, 1500, 1000, 2000, 1500],
        'SaleDate': pd.date_range(start='2023-01-01', periods=9, freq='D'),
        'LeadSource': ['Web', 'Phone', 'Walk-in'] * 3,
        'LeadStatus': ['Sold', 'Open', 'Sold', 'Lost', 'Sold', 'Open', 'Sold', 'Lost', 'Sold'],
        'DaysInStock': [15, 25, 45, 55, 75, 85, 95, 105, 115],
        'location': ['North', 'South', 'East'] * 3,
        'vehicle_type': ['New', 'Used', 'Used'] * 3
    })
    
    # Create mock state
    mock_state = {
        'validated_data': sales_data,
        'validation_summary': {
            'total_records': len(sales_data),
            'quality_score': 95
        },
        'is_authenticated': True,
        'llm_client': MockLLMClient()
    }
    
    # Replace session state
    st.session_state = mock_state
    
    yield mock_state
    
    # Restore original state
    st.session_state = original_state

class MockLLMClient:
    """Mock LLM client for testing."""
    def generate(self, prompt):
        return "Sales rep Alice's gross margin increased by 15% month-over-month."

def test_kpi_dashboard_rendering(mock_session_state):
    """Test that the dashboard renders without errors."""
    try:
        # Render the dashboard
        kpi_dashboard()
        
        # Check that key metrics were calculated
        assert 'validated_data' in st.session_state
        df = st.session_state.validated_data
        assert len(df) == 9
        
        # Verify data quality
        assert st.session_state['validation_summary']['quality_score'] > 90
        
        # Check for summary text
        assert any(
            element.type == "info" and "gross margin" in element.body
            for element in st.elements
        )
        
    except Exception as e:
        pytest.fail(f"Dashboard rendering failed: {str(e)}")

def test_kpi_dashboard_filters(mock_session_state):
    """Test dashboard filters."""
    try:
        # Set filter values
        st.session_state["date_filter"] = "Last 7 Days"
        st.session_state["location_filter"] = "North"
        st.session_state["vehicle_type_filter"] = "New"
        
        # Render dashboard
        kpi_dashboard()
        
        # Check filtered data
        filtered_elements = [
            element for element in st.elements
            if element.type == "dataframe"
        ]
        
        assert len(filtered_elements) > 0
        filtered_data = filtered_elements[0].data
        
        # Verify filters applied
        assert all(filtered_data['location'] == 'North')
        assert all(filtered_data['vehicle_type'] == 'New')
        assert len(filtered_data) < len(mock_session_state['validated_data'])
        
    except Exception as e:
        pytest.fail(f"Filter testing failed: {str(e)}")

def test_kpi_dashboard_downloads(mock_session_state):
    """Test data download functionality."""
    try:
        # Render dashboard
        kpi_dashboard()
        
        # Check for download buttons
        download_buttons = [
            element for element in st.elements
            if element.type == "download_button"
        ]
        
        assert len(download_buttons) >= 4  # Should have at least 4 download options
        
        # Test each download
        for button in download_buttons:
            assert button.data is not None
            assert isinstance(button.data, bytes)
            assert len(button.data) > 0
        
    except Exception as e:
        pytest.fail(f"Download testing failed: {str(e)}")

def test_kpi_dashboard_chart_interaction(mock_session_state):
    """Test chart interaction functionality."""
    try:
        # Render dashboard
        kpi_dashboard()
        
        # Simulate chart click
        st.query_params["selected_chart"] = "rep_performance"
        st.query_params["selected_value"] = "Alice"
        
        # Re-render dashboard
        kpi_dashboard()
        
        # Check for drill-down content
        assert any(
            element.type == "markdown" and "Details for Alice" in element.body
            for element in st.elements
        )
        
    except Exception as e:
        pytest.fail(f"Chart interaction testing failed: {str(e)}")

def test_kpi_dashboard_authentication(mock_session_state):
    """Test authentication requirements."""
    try:
        # Remove authentication
        st.session_state['is_authenticated'] = False
        
        # Render dashboard
        kpi_dashboard()
        
        # Should show warning
        assert any(
            element.type == "warning" and "Please log in" in element.body
            for element in st.elements
        )
        
    except Exception as e:
        pytest.fail(f"Authentication testing failed: {str(e)}")

def test_kpi_dashboard_error_handling(mock_session_state):
    """Test error handling in dashboard."""
    try:
        # Create invalid data
        st.session_state.validated_data = pd.DataFrame()
        
        # Render dashboard
        kpi_dashboard()
        
        # Should show warning
        assert any(
            element.type == "warning" and "Please upload data" in element.body
            for element in st.elements
        )
        
    except Exception as e:
        pytest.fail(f"Error handling testing failed: {str(e)}")

def test_kpi_dashboard_custom_date_range(mock_session_state):
    """Test custom date range functionality."""
    try:
        # Set custom date range
        st.session_state["date_filter"] = "Custom Range"
        st.session_state["start_date"] = datetime(2023, 1, 1)
        st.session_state["end_date"] = datetime(2023, 1, 5)
        
        # Render dashboard
        kpi_dashboard()
        
        # Check filtered data
        filtered_elements = [
            element for element in st.elements
            if element.type == "dataframe"
        ]
        
        assert len(filtered_elements) > 0
        filtered_data = filtered_elements[0].data
        
        # Verify date filter applied
        assert len(filtered_data) == 5  # Should only show 5 days of data
        
    except Exception as e:
        pytest.fail(f"Custom date range testing failed: {str(e)}")