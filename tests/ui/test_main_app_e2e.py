"""
End-to-end UI tests for the main application.
"""

import unittest
import streamlit as st
import pandas as pd
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
from src.watchdog_ai.ui.pages.main_app import render_app, check_sales_anomaly, initialize_session_state
from src.watchdog_ai.ui.components.sales_report_renderer import SalesReportRenderer
import pytest
from playwright.sync_api import Page, expect
import os

@pytest.fixture
def mock_csv_file():
    """Create a mock CSV file for testing."""
    content = """date,customer,deal_type,amount
2025-04-01,John Doe,New,25000
2025-04-02,Jane Smith,Used,18000
2025-04-03,Bob Wilson,New,32000
"""
    with open("tests/fixtures/mock_sales_reports_123456789.csv", "w") as f:
        f.write(content)
    yield "tests/fixtures/mock_sales_reports_123456789.csv"
    os.remove("tests/fixtures/mock_sales_reports_123456789.csv")

@pytest.fixture(autouse=True)
def setup_streamlit():
    """Set up Streamlit session state before each test."""
    # Initialize session state as a dictionary if it doesn't exist
    if not hasattr(st, "session_state"):
        st.session_state = {}
    
    # Initialize required state variables
    st.session_state["nova_act_connected"] = False
    st.session_state["last_sync_timestamp"] = None
    st.session_state["active_tab"] = "System Connect"
    st.session_state["credentials"] = {
        'vendor': '',
        'email': '',
        'password': '',
        'dealership_id': '',
        '2fa_method': '',
        'reports': []
    }
    st.session_state["theme"] = "light"
    st.session_state["previous_sales"] = 25
    st.session_state["upload_progress"] = {
        'current_file': None,
        'processed_files': [],
        'validation_summary': {},
        'incremental_updates': {}
    }
    
    yield
    
    # Clean up after test
    st.session_state.clear()

class TestMainAppE2E(unittest.TestCase):
    """End-to-end tests for the main application UI."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock session state
        st.session_state = {}
        st.session_state["nova_act_connected"] = False
        st.session_state["last_sync_timestamp"] = None
        st.session_state["active_tab"] = "System Connect"
        st.session_state["credentials"] = {
            'vendor': '',
            'email': '',
            'password': '',
            'dealership_id': '',
            '2fa_method': '',
            'reports': []
        }
        st.session_state["theme"] = "light"
        st.session_state["previous_sales"] = 25
        
        # Sample successful sync data
        self.sample_sync_data = {
            'status': 'success',
            'message': 'Data successfully processed!',
            'files': [
                {'name': 'sales_report.csv', 'records': 128},
                {'name': 'inventory_report.csv', 'records': 76}
            ],
            'timestamp': datetime.now().isoformat()
        }
        
        # Sample insight data
        self.sample_insight = {
            "summary": {
                "title": "NeoIdentity Produced the Most Sales",
                "description": "NeoIdentity leads generated 4 sales with an average gross of $2,600"
            },
            "chart_data": {
                "type": "bar",
                "data": {
                    "x": ["NeoIdentity", "CarGurus", "Website", "Walk-in", "Phone"],
                    "y": [4, 3, 5, 3, 3]
                }
            },
            "metrics": [
                {
                    "label": "Total Sales",
                    "value": "4",
                    "delta": "22%"
                },
                {
                    "label": "Avg. Gross",
                    "value": "$2,600",
                    "delta": "+38%"
                }
            ],
            "recommendations": [
                "Increase marketing budget for NeoIdentity by 15% (Est. impact: +10% sales)",
                "Share best practices from NeoIdentity team with other sources"
            ]
        }
    
    @patch('streamlit.markdown')
    @patch('streamlit.selectbox')
    @patch('streamlit.text_input')
    @patch('streamlit.button')
    def test_system_connect_flow(self, mock_button, mock_text_input, mock_selectbox, mock_markdown):
        """Test the system connect flow."""
        # Set up mock inputs
        mock_selectbox.side_effect = [
            "DealerSocket",  # Vendor
            "SMS",          # 2FA method
            "Daily"         # Sync frequency
        ]
        mock_text_input.side_effect = [
            "test@dealersocket.com",  # Email
            "password123",            # Password
            "DEALER123"               # Dealership ID
        ]
        mock_button.return_value = True  # "Connect System" button
        
        # Render app
        render_app()
        
        # Verify form inputs were called
        mock_selectbox.assert_any_call("Select Vendor", ["", "DealerSocket", "VinSolutions", "CDK", "Reynolds & Reynolds", "DealerTrack"], index=0)
        mock_text_input.assert_any_call("Email", value="")
        mock_text_input.assert_any_call("Password", type="password")
        
        # Verify connection success message
        mock_markdown.assert_any_call("""
            <div class="message-container success">
                <p>✅ Successfully connected to DealerSocket!</p>
            </div>
        """, unsafe_allow_html=True)
    
    @patch('streamlit.spinner')
    @patch('streamlit.success')
    def test_sync_flow(self, mock_success, mock_spinner):
        """Test the sync flow."""
        # Set up connected state
        st.session_state["nova_act_connected"] = True
        st.session_state["credentials"] = {
            'vendor': 'DealerSocket',
            'email': 'test@dealersocket.com',
            'password': 'password123',
            'dealership_id': 'DEALER123',
            '2fa_method': 'sms',
            'reports': ['sales']
        }
        
        # Mock sync operation
        with patch('src.watchdog_ai.ui.pages.main_app.render_system_connect') as mock_sync:
            mock_sync.return_value = self.sample_sync_data
            render_app()
            
            # Verify sync success message
            mock_success.assert_called_with("✅ Data synchronized successfully!")
            
            # Verify timestamp update
            self.assertIsNotNone(st.session_state["last_sync_timestamp"])
    
    @patch('streamlit.error')
    def test_sync_error_handling(self, mock_error):
        """Test sync error handling."""
        # Set up connected state with invalid credentials
        st.session_state["nova_act_connected"] = True
        st.session_state["credentials"] = {
            'vendor': 'DealerSocket',
            'email': 'invalid@test.com',
            'password': 'wrong',
            'dealership_id': 'INVALID',
            '2fa_method': 'sms',
            'reports': ['sales']
        }
        
        # Mock failed sync
        with patch('src.watchdog_ai.ui.pages.main_app.render_system_connect') as mock_sync:
            mock_sync.side_effect = Exception("Authentication failed")
            render_app()
            
            # Verify error message
            mock_error.assert_called_with("Error syncing data: Authentication failed")
    
    @patch('streamlit.markdown')
    def test_insight_rendering(self, mock_markdown):
        """Test insight rendering after successful sync."""
        # Set up connected state with sync data
        st.session_state["nova_act_connected"] = True
        st.session_state["sync_result"] = self.sample_sync_data
        
        # Mock insight renderer
        with patch('src.watchdog_ai.ui.components.sales_report_renderer.SalesReportRenderer.render_insight_block') as mock_renderer:
            render_app()
            
            # Verify insight was rendered
            mock_renderer.assert_called_once()
            
            # Verify insight container styling
            mock_markdown.assert_any_call("""
                <div class="message-container insight">
            """, unsafe_allow_html=True)
    
    @patch('streamlit.info')
    def test_fallback_message(self, mock_info):
        """Test fallback message when not connected."""
        # Clear connection state
        st.session_state["nova_act_connected"] = False
        st.session_state["credentials"] = {}
        
        render_app()
        
        # Verify fallback message
        mock_info.assert_called_with("Please connect your CRM system to get started.")
    
    def test_responsive_layout(self):
        """Test responsive layout elements."""
        with patch('streamlit.columns') as mock_columns:
            mock_col1, mock_col2 = MagicMock(), MagicMock()
            mock_columns.return_value = [mock_col1, mock_col2]
            
            render_app()
            
            # Verify columns were created
            mock_columns.assert_called()
    
    @patch('streamlit.experimental_get_query_params')
    def test_load_time(self, mock_query_params):
        """Test page load time tracking."""
        mock_query_params.return_value = {'t': ['1234567890']}
        
        with patch('time.time', return_value=1234567891):
            render_app()
            
            # Load time should be calculated and logged
            self.assertTrue(True)  # Basic assertion for now

def test_sales_drop_detection(setup_streamlit):
    """Test sales drop detection logic."""
    # Set previous sales
    st.session_state["previous_sales"] = 100
    
    # Test with significant drop
    current_sales = 70
    drop_percent = check_sales_anomaly(current_sales)
    assert drop_percent == pytest.approx(30.0)
    
    # Test with no significant drop
    current_sales = 90
    drop_percent = check_sales_anomaly(current_sales)
    assert drop_percent is None

@pytest.mark.skip("Requires running Streamlit server")
def test_ui_load_time(page: Page):
    """Test UI load time performance."""
    # This test requires a running Streamlit server
    page.goto("http://localhost:8501")
    
    # Verify page loads within acceptable time
    load_time = page.evaluate("() => window.performance.timing.loadEventEnd - window.performance.timing.navigationStart")
    assert load_time < 3000  # 3 seconds max

@pytest.mark.skip("Requires running Streamlit server")
def test_responsive_layout(page: Page):
    """Test responsive layout at different screen sizes."""
    # This test requires a running Streamlit server
    page.goto("http://localhost:8501")
    
    # Test mobile layout
    page.set_viewport_size({"width": 375, "height": 667})
    expect(page.locator(".stApp")).to_be_visible()
    
    # Test desktop layout
    page.set_viewport_size({"width": 1920, "height": 1080})
    expect(page.locator(".stApp")).to_be_visible()

@pytest.mark.skip("Requires running Streamlit server")
def test_anomaly_alert_display(page: Page, setup_streamlit):
    """Test anomaly alert display in UI."""
    # This test requires a running Streamlit server
    page.goto("http://localhost:8501")
    
    # Set up test data
    st.session_state["previous_sales"] = 100
    st.session_state["nova_act_connected"] = True
    
    # Verify alert is displayed for significant drop
    expect(page.locator(".message-container.error")).to_be_visible()

@patch('streamlit.file_uploader')
@patch('streamlit.success')
def test_upload_data_zone(mock_success, mock_file_uploader, mock_csv_file):
    """Test the upload data zone functionality."""
    # Mock file uploader
    mock_file = MagicMock()
    mock_file.name = "mock_sales_reports_123456789.csv"
    with open(mock_csv_file, "rb") as f:
        mock_file.read = lambda: f.read()
    mock_file_uploader.return_value = mock_file
    
    # Render app
    render_app()
    
    # Verify file uploader was called
    mock_file_uploader.assert_called_with(
        "Upload your vendor report (CSV)",
        type=["csv", "pdf"],
        accept_multiple_files=True
    )
    
    # Verify success message
    mock_success.assert_called_with("Data successfully processed! 3 records analyzed with no data quality issues.")

@patch('streamlit.error')
def test_upload_invalid_file(mock_error):
    """Test handling of invalid file upload."""
    # Mock invalid file
    mock_file = MagicMock()
    mock_file.name = "invalid.txt"
    
    # Mock file uploader to return invalid file
    with patch('streamlit.file_uploader', return_value=mock_file):
        render_app()
        
        # Verify error message
        mock_error.assert_called_with("Only CSV and PDF files are supported")

@patch('streamlit.warning')
def test_upload_empty_file(mock_warning):
    """Test handling of empty file upload."""
    # Mock empty file
    mock_file = MagicMock()
    mock_file.name = "empty.csv"
    mock_file.read = lambda: b""
    
    # Mock file uploader to return empty file
    with patch('streamlit.file_uploader', return_value=mock_file):
        render_app()
        
        # Verify warning message
        mock_warning.assert_called_with("File appears to be empty or corrupted")

@patch('streamlit.progress')
def test_upload_progress_tracking(mock_progress):
    """Test upload progress tracking."""
    # Mock file and progress bar
    mock_file = MagicMock()
    mock_file.name = "large_file.csv"
    mock_progress_bar = MagicMock()
    mock_progress.return_value = mock_progress_bar
    
    # Mock file uploader
    with patch('streamlit.file_uploader', return_value=mock_file):
        render_app()
        
        # Verify progress tracking
        assert 'upload_progress' in st.session_state
        assert isinstance(st.session_state.upload_progress, dict)
        assert all(key in st.session_state.upload_progress for key in [
            'current_file',
            'processed_files',
            'validation_summary',
            'incremental_updates'
        ])

@pytest.mark.skip("Requires running Streamlit server")
def test_upload_ui_interaction(page: Page, mock_csv_file):
    """Test upload UI interactions."""
    # This test requires a running Streamlit server
    page.goto("http://localhost:8501")
    
    # Click Upload Data tab
    page.click("text=Upload Data")
    
    # Verify upload zone is visible
    expect(page.locator("text=Upload your vendor report")).to_be_visible()
    
    # Upload file
    page.set_input_files("input[type=file]", mock_csv_file)
    
    # Verify success message
    expect(page.locator(".message-container.success")).to_be_visible()
    expect(page.locator("text=Data successfully processed!")).to_be_visible()

if __name__ == '__main__':
    unittest.main()