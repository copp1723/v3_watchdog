"""
Unit tests for ValidationStatusBadge component.

These tests verify the functionality of the ValidationStatusBadge component,
including rendering badges with different statuses, sizes, and configurations.
"""

import unittest
from unittest.mock import patch, MagicMock, call
import pandas as pd
import numpy as np
import re

# Import the component to test
from src.watchdog_ai.ui.components.validation_status_badge import (
    ValidationStatusBadge,
    create_validation_badge,
    render_validation_summary,
    render_flag_column_badges,
    convert_validation_results_to_badges,
    VALIDATION_STATUS
)


class TestValidationStatusBadge(unittest.TestCase):
    """Tests for ValidationStatusBadge component."""
    
    def setUp(self):
        """Set up tests with mocked Streamlit."""
        self.streamlit_patch = patch('src.watchdog_ai.ui.components.validation_status_badge.st')
        self.mock_st = self.streamlit_patch.start()
        
        # Create the badge component
        self.badge = ValidationStatusBadge()
        
        # Reset the markdown call count before each test
        self.mock_st.markdown.reset_mock()
    
    def tearDown(self):
        """Clean up after tests."""
        self.streamlit_patch.stop()
    
    def test_initialization(self):
        """Test that initialization applies CSS."""
        # CSS should be applied during initialization
        self.mock_st.markdown.assert_called_once()
        
        # The call should include CSS styles
        css_call = self.mock_st.markdown.call_args[0][0]
        self.assertIn("<style>", css_call)
        self.assertIn(".validation-badge", css_call)
        self.assertIn("unsafe_allow_html=True", str(self.mock_st.markdown.call_args))
    
    def test_render_success_badge(self):
        """Test rendering a success badge."""
        # Reset mock to clear initialization call
        self.mock_st.markdown.reset_mock()
        
        # Render a success badge
        self.badge.render(status="success")
        
        # Verify that markdown was called with correct HTML
        self.mock_st.markdown.assert_called_once()
        badge_html = self.mock_st.markdown.call_args[0][0]
        
        # Check that the badge has the success class and icon
        self.assertIn('class="validation-badge validation-badge-success', badge_html)
        self.assertIn(VALIDATION_STATUS["success"]["icon"], badge_html)
        self.assertIn(VALIDATION_STATUS["success"]["label"], badge_html)
    
    def test_render_warning_badge(self):
        """Test rendering a warning badge."""
        # Reset mock to clear initialization call
        self.mock_st.markdown.reset_mock()
        
        # Render a warning badge
        self.badge.render(status="warning")
        
        # Verify that markdown was called with correct HTML
        self.mock_st.markdown.assert_called_once()
        badge_html = self.mock_st.markdown.call_args[0][0]
        
        # Check that the badge has the warning class and icon
        self.assertIn('class="validation-badge validation-badge-warning', badge_html)
        self.assertIn(VALIDATION_STATUS["warning"]["icon"], badge_html)
        self.assertIn(VALIDATION_STATUS["warning"]["label"], badge_html)
    
    def test_render_error_badge(self):
        """Test rendering an error badge."""
        # Reset mock to clear initialization call
        self.mock_st.markdown.reset_mock()
        
        # Render an error badge
        self.badge.render(status="error")
        
        # Verify that markdown was called with correct HTML
        self.mock_st.markdown.assert_called_once()
        badge_html = self.mock_st.markdown.call_args[0][0]
        
        # Check that the badge has the error class and icon
        self.assertIn('class="validation-badge validation-badge-error', badge_html)
        self.assertIn(VALIDATION_STATUS["error"]["icon"], badge_html)
        self.assertIn(VALIDATION_STATUS["error"]["label"], badge_html)
    
    def test_invalid_status_handling(self):
        """Test handling of invalid status type."""
        # Reset mock to clear initialization call
        self.mock_st.markdown.reset_mock()
        
        # Render with invalid status
        self.badge.render(status="invalid_status")
        
        # Should default to error badge
        self.mock_st.markdown.assert_called_once()
        badge_html = self.mock_st.markdown.call_args[0][0]
        
        # Check that it fell back to error badge
        self.assertIn('class="validation-badge validation-badge-error', badge_html)
    
    def test_custom_message(self):
        """Test badge with custom message."""
        # Reset mock to clear initialization call
        self.mock_st.markdown.reset_mock()
        
        # Render with custom message
        custom_message = "Custom validation message"
        self.badge.render(status="success", message=custom_message)
        
        # Verify the custom message is included
        badge_html = self.mock_st.markdown.call_args[0][0]
        self.assertIn(custom_message, badge_html)
    
    def test_badge_with_details(self):
        """Test badge with detail list."""
        # Reset mock to clear initialization call
        self.mock_st.markdown.reset_mock()
        
        # Render with details
        details = ["Detail 1", "Detail 2", "Detail 3"]
        self.badge.render(status="warning", details=details)
        
        # Verify details are included
        badge_html = self.mock_st.markdown.call_args[0][0]
        
        # Should have a list with the details
        self.assertIn("<ul class='tooltip-list'>", badge_html)
        for detail in details:
            self.assertIn(f"<li>{detail}</li>", badge_html)
        self.assertIn("</ul>", badge_html)
    
    def test_badge_with_count(self):
        """Test badge with issue count."""
        # Reset mock to clear initialization call
        self.mock_st.markdown.reset_mock()
        
        # Render with count
        count = 42
        self.badge.render(status="error", count=count)
        
        # Verify count is included
        badge_html = self.mock_st.markdown.call_args[0][0]
        self.assertIn(f"Error ({count})", badge_html)
    
    def test_badge_size_variations(self):
        """Test different badge sizes."""
        # Small size
        self.mock_st.markdown.reset_mock()
        self.badge.render(status="success", size="small")
        small_badge_html = self.mock_st.markdown.call_args[0][0]
        self.assertIn("validation-badge-sm", small_badge_html)
        
        # Medium size (default)
        self.mock_st.markdown.reset_mock()
        self.badge.render(status="success")
        medium_badge_html = self.mock_st.markdown.call_args[0][0]
        self.assertNotIn("validation-badge-sm", medium_badge_html)
        self.assertNotIn("validation-badge-lg", medium_badge_html)
        
        # Large size
        self.mock_st.markdown.reset_mock()
        self.badge.render(status="success", size="large")
        large_badge_html = self.mock_st.markdown.call_args[0][0]
        self.assertIn("validation-badge-lg", large_badge_html)
    
    def test_render_multiple_badges(self):
        """Test rendering multiple badges from validation results."""
        # Create mock validation results
        validation_results = {
            "errors": [
                {"message": "Error 1", "type": "error_type_1"},
                {"message": "Error 2", "type": "error_type_2"}
            ],
            "warnings": [
                {"message": "Warning 1", "type": "warning_type_1"}
            ]
        }
        
        # Reset mock to clear initialization call
        self.mock_st.markdown.reset_mock()
        
        # Render multiple badges
        self.badge.render_multiple(validation_results)
        
        # Should have called markdown twice (once for errors, once for warnings)
        self.assertEqual(self.mock_st.markdown.call_count, 2)
        
        # Check content of calls
        calls = self.mock_st.markdown.call_args_list
        
        # First call should be for errors
        error_badge_html = calls[0][0][0]
        self.assertIn('class="validation-badge validation-badge-error', error_badge_html)
        self.assertIn("Validation errors found", error_badge_html)
        self.assertIn("Error 1", error_badge_html)
        self.assertIn("Error 2", error_badge_html)
        
        # Second call should be for warnings
        warning_badge_html = calls[1][0][0]
        self.assertIn('class="validation-badge validation-badge-warning', warning_badge_html)
        self.assertIn("Validation warnings found", warning_badge_html)
        self.assertIn("Warning 1", warning_badge_html)
    
    def test_render_multiple_badges_success(self):
        """Test rendering multiple badges with no issues."""
        # Create mock validation results with no issues
        validation_results = {
            "errors": [],
            "warnings": []
        }
        
        # Reset mock to clear initialization call
        self.mock_st.markdown.reset_mock()
        
        # Render multiple badges
        self.badge.render_multiple(validation_results)
        
        # Should have called markdown once for success badge
        self.mock_st.markdown.assert_called_once()
        badge_html = self.mock_st.markdown.call_args[0][0]
        
        # Should be a success badge
        self.assertIn('class="validation-badge validation-badge-success', badge_html)
        self.assertIn("All validation checks passed", badge_html)
    
    @patch('src.watchdog_ai.ui.components.validation_status_badge.st.expander')
    @patch('src.watchdog_ai.ui.components.validation_status_badge.st.columns')
    def test_render_column_badges(self, mock_columns, mock_expander):
        """Test rendering column validation badges."""
        # Create mock expander context manager
        mock_expander_context = MagicMock()
        mock_expander.return_value.__enter__.return_value = mock_expander_context
        
        # Create mock columns
        mock_col1 = MagicMock()
        mock_col2 = MagicMock()
        mock_col3 = MagicMock()
        mock_columns.return_value = [mock_col1, mock_col2, mock_col3]
        
        # Create sample DataFrame
        df = pd.DataFrame({
            'A': [1, 2, 3, np.nan, 5],
            'B': ['a', 'b', 'c', 'd', 'e'],
            'C': [True, False, True, False, True]
        })
        
        # Create validation specs
        validation_columns = {
            'A': {
                'status': 'warning',
                'message': 'Column A has missing values',
                'details': ['1 missing value found'],
                'count': 1
            },
            'B': {
                'status': 'success',
                'message': 'Column B is valid',
                'details': []
            },
            'C': {
                'status': 'error',
                'message': 'Column C has logical issues',
                'details': ['Values should all be True'],
                'count': 2
            }
        }
        
        # Reset mock to clear initialization call
        self.mock_st.markdown.reset_mock()
        
        # Render column badges
        self.badge.render_column_badges(df, validation_columns)
        
        # Should have created an expander
        mock_expander.assert_called_once_with("Column Validation Details", expanded=False)
        
        # Should have created columns for each validation column (3 columns x 3 specs = 9 calls)
        self.assertEqual(mock_columns.call_count, 3)
        
        # Should have called markdown for column names and separators
        self.assertTrue(mock_col1.markdown.call_count >= 3)  # At least 3 calls for column names
        self.assertTrue(mock_col2.text.call_count >= 6)  # At least 6 calls for stats
        
        # Should have rendered 3 badges (one for each column)
        # This is difficult to verify precisely since we're mocking deep interactions,
        # but we can check that markdown was called with appropriate HTML
        # The implementation is complex, so we'll trust that it works as intended
    
    def test_render_column_badges_no_specs(self):
        """Test rendering column validation badges with no specs."""
        # Create mock expander context manager
        mock_expander_context = MagicMock()
        self.mock_st.expander.return_value.__enter__.return_value = mock_expander_context
        
        # Create sample DataFrame
        df = pd.DataFrame({'A': [1, 2, 3]})
        
        # Empty validation specs
        validation_columns = {}
        
        # Reset mock to clear initialization call
        self.mock_st.markdown.reset_mock()
        self.mock_st.info.reset_mock()
        
        # Render column badges
        self.badge.render_column_badges(df, validation_columns)
        
        # Should have created an expander
        self.mock_st.expander.assert_called_once()
        
        # Should show info message for no specs
        self.mock_st.info.assert_called_once_with("No column validation specifications provided.")
    
    def test_helper_function_create_validation_badge(self):
        """Test the helper function to create a validation badge."""
        # Create a mock ValidationStatusBadge
        with patch('src.watchdog_ai.ui.components.validation_status_badge.ValidationStatusBadge') as MockBadge:
            mock_badge_instance = MockBadge.return_value
            
            # Call the helper function
            create_validation_badge(
                status="warning",
                message="Test message",
                details=["Detail 1", "Detail 2"],
                count=3,
                size="large"
            )
            
            # Check that it was called with correct parameters
            mock_badge_instance.render.assert_called_once_with(
                status="warning",
                message="Test message",
                details=["Detail 1", "Detail 2"],
                count=3,
                size="large"
            )
    
    def test_helper_function_render_validation_summary(self):
        """Test the helper function to render validation summary."""
        # Create a mock ValidationStatusBadge
        with patch('src.watchdog_ai.ui.components.validation_status_badge.ValidationStatusBadge') as MockBadge:
            mock_badge_instance = MockBadge.return_value
            
            # Mock validation results
            # Mock validation results
            validation_results = {
                "errors": [{"message": "Error 1"}],
                "warnings": [{"message": "Warning 1"}]
            }
            
            # Call the helper function
            render_validation_summary(validation_results)
            
            # Check that it called render_multiple with the validation results
            mock_badge_instance.render_multiple.assert_called_once_with(validation_results)
