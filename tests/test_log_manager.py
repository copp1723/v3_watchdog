"""
Tests for the LogManager component.

These tests cover the functionality of the LogManager class,
which provides interfaces for viewing, managing, and downloading log files.
"""

import unittest
from unittest.mock import patch, MagicMock, mock_open, call
import os
import tempfile
import pandas as pd
import base64
from datetime import datetime
import io
import re

from src.watchdog_ai.ui.components.log_manager import (
    LogManager,
    create_download_link,
    render_settings_page_logs_section
)


class TestLogManager(unittest.TestCase):
    """Tests for the LogManager component."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test log files
        self.test_dir = tempfile.mkdtemp()
        self.log_manager = LogManager(log_dir=self.test_dir)
        
        # Mock streamlit
        self.streamlit_patch = patch('src.watchdog_ai.ui.components.log_manager.st')
        self.mock_st = self.streamlit_patch.start()
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Stop streamlit mock
        self.streamlit_patch.stop()
        
        # Clean up temporary directory
        for filename in os.listdir(self.test_dir):
            os.remove(os.path.join(self.test_dir, filename))
        os.rmdir(self.test_dir)
    
    def create_test_log_files(self):
        """Create test log files in the temporary directory."""
        # Create main log file
        with open(os.path.join(self.test_dir, "app.log"), "w") as f:
            f.write("Main log file content\n" * 10)
        
        # Create rotated log files
        for i in range(1, 4):
            with open(os.path.join(self.test_dir, f"app.log.{i}"), "w") as f:
                f.write(f"Rotated log file {i} content\n" * 5)
        
        # Create another log series
        with open(os.path.join(self.test_dir, "debug.log"), "w") as f:
            f.write("Debug log file content\n" * 8)
        
        with open(os.path.join(self.test_dir, "debug.log.1"), "w") as f:
            f.write("Debug rotated log file content\n" * 4)
    
    def test_get_log_files(self):
        """Test listing log files with metadata."""
        # Create test log files
        self.create_test_log_files()
        
        # Get log files
        log_files = self.log_manager.get_log_files()
        
        # Check that all log files were found
        self.assertEqual(len(log_files), 6)
        
        # Check that files are sorted correctly
        self.assertEqual(log_files[0]['filename'], "app.log")
        self.assertEqual(log_files[1]['filename'], "app.log.1")
        
        # Check metadata
        for log_file in log_files:
            self.assertTrue('size' in log_file)
            self.assertTrue('created' in log_file)
            self.assertTrue('modified' in log_file)
            self.assertTrue('is_main' in log_file)
            self.assertTrue('rotation' in log_file)
            self.assertTrue('base_name' in log_file)
        
        # Check that main log files are identified correctly
        app_log = next(log for log in log_files if log['filename'] == 'app.log')
        app_log_1 = next(log for log in log_files if log['filename'] == 'app.log.1')
        self.assertTrue(app_log['is_main'])
        self.assertFalse(app_log_1['is_main'])
        
        # Check rotation numbers
        self.assertEqual(app_log['rotation'], 0)
        self.assertEqual(app_log_1['rotation'], 1)
    
    def test_get_log_files_empty_directory(self):
        """Test listing log files from an empty directory."""
        # No files created, should return empty list
        log_files = self.log_manager.get_log_files()
        self.assertEqual(len(log_files), 0)
    
    def test_get_log_files_error_handling(self):
        """Test error handling when listing log files."""
        # Mock os.listdir to raise an error
        with patch('os.listdir', side_effect=PermissionError("Access denied")):
            with patch('logging.error') as mock_error:
                log_files = self.log_manager.get_log_files()
                # Should return empty list on error
                self.assertEqual(len(log_files), 0)
                # Should log the error
                mock_error.assert_called_once()
                self.assertIn("Error reading log directory", mock_error.call_args[0][0])
    
    def test_get_log_file_groups(self):
        """Test grouping log files by base name."""
        # Create test log files
        self.create_test_log_files()
        
        # Get log file groups
        groups = self.log_manager.get_log_file_groups()
        
        # Should have two groups (app.log and debug.log)
        self.assertEqual(len(groups), 2)
        self.assertTrue('app.log' in groups)
        self.assertTrue('debug.log' in groups)
        
        # Check app.log group
        app_group = groups['app.log']
        self.assertEqual(len(app_group), 4)  # Main file + 3 rotated files
        
        # Check debug.log group
        debug_group = groups['debug.log']
        self.assertEqual(len(debug_group), 2)  # Main file + 1 rotated file
    
    def test_format_file_size(self):
        """Test formatting file sizes in human-readable format."""
        # Test different sizes
        test_cases = [
            (0, "0.0 B"),
            (100, "100.0 B"),
            (1023, "1023.0 B"),
            (1024, "1.0 KB"),
            (1500, "1.5 KB"),
            (1024 * 1024, "1.0 MB"),
            (1024 * 1024 * 2.5, "2.5 MB"),
            (1024 * 1024 * 1024, "1.0 GB"),
            (1024 * 1024 * 1024 * 5.2, "5.2 GB")
        ]
        
        for size, expected in test_cases:
            self.assertEqual(self.log_manager.format_file_size(size), expected)
    
    def test_read_log_file(self):
        """Test reading log file contents."""
        # Create a test log file
        test_file = os.path.join(self.test_dir, "test.log")
        with open(test_file, "w") as f:
            f.write("Line 1\nLine 2\nLine 3\nLine 4\nLine 5\n")
        
        # Read the file
        content = self.log_manager.read_log_file(test_file)
        self.assertEqual(content, "Line 1\nLine 2\nLine 3\nLine 4\nLine 5\n")
        
        # Test with max_lines
        content = self.log_manager.read_log_file(test_file, max_lines=2)
        self.assertIn("Line 4\nLine 5\n", content)
        self.assertIn("showing last 2 lines", content)
    
    def test_read_log_file_not_found(self):
        """Test reading a non-existent log file."""
        content = self.log_manager.read_log_file("nonexistent.log")
        self.assertTrue(content.startswith("File not found:"))
    
    def test_read_log_file_error(self):
        """Test error handling when reading log files."""
        # Mock open to raise an error
        with patch('builtins.open', side_effect=PermissionError("Access denied")):
            with patch('logging.error') as mock_error:
                content = self.log_manager.read_log_file("any.log")
                # Should return error message
                self.assertTrue(content.startswith("Error reading log file:"))
                # Should log the error
                mock_error.assert_called_once()
                self.assertIn("Error reading log file", mock_error.call_args[0][0])
    
    def test_get_file_download_link(self):
        """Test generating download links for log files."""
        # Create a test log file
        test_file = os.path.join(self.test_dir, "test.log")
        test_content = "Test log content"
        with open(test_file, "w") as f:
            f.write(test_content)
        
        # Get download link
        link = self.log_manager.get_file_download_link(test_file)
        
        # Verify link format
        self.assertIn('<a href="data:file/txt;base64,', link)
        self.assertIn('download="test.log"', link)
        self.assertIn('>Download</a>', link)
        
        # Verify the encoded content
        encoded_content = re.search(r'base64,([^"]+)"', link).group(1)
        decoded_content = base64.b64decode(encoded_content).decode()
        self.assertEqual(decoded_content, test_content)
        
        # Test with custom link text
        link = self.log_manager.get_file_download_link(test_file, "Custom Text")
        self.assertIn('>Custom Text</a>', link)
    
    def test_get_file_download_link_error(self):
        """Test error handling when generating download links."""
        # Mock open to raise an error
        with patch('builtins.open', side_effect=PermissionError("Access denied")):
            with patch('logging.error') as mock_error:
                link = self.log_manager.get_file_download_link("any.log")
                # Should return error message
                self.assertTrue(link.startswith("<span style='color: red'>Error:"))
                # Should log the error
                mock_error.assert_called_once()
                self.assertIn("Error creating download link", mock_error.call_args[0][0])
    
    def test_render_log_list(self):
        """Test rendering log file list UI."""
        # Create test log files
        self.create_test_log_files()
        
        # Call the render method
        self.log_manager.render_log_list()
        
        # Check that streamlit functions were called
        self.mock_st.subheader.assert_called_once_with("Application Logs")
        self.mock_st.write.assert_any_call("These logs contain detailed information about the application's operation and any errors encountered.")
        self.mock_st.dataframe.assert_called_once()
        self.mock_st.expander.assert_called()
        
        # Should call markdown with HTML for download links
        self.assertTrue(any("<a href=" in call_args[0][0] for call_args in self.mock_st.markdown.call_args_list))
    
    def test_render_log_list_empty(self):
        """Test rendering log list when no files exist."""
        # Call the render method without creating any files
        self.log_manager.render_log_list()
        
        # Should show info message
        self.mock_st.info.assert_called_once()
        self.assertIn("No log files found", self.mock_st.info.call_args[0][0])
    
    def test_render_log_rotation_visualization(self):
        """Test rendering log rotation visualization UI."""
        # Create test log files
        self.create_test_log_files()
        
        # Call the render method
        self.log_manager.render_log_rotation_visualization()
        
        # Check that streamlit functions were called
        self.mock_st.subheader.assert_called_once_with("Log Rotation Status")
        self.mock_st.progress.assert_called()
        
        # Should call write with rotation info
        self.assertTrue(any("Total size:" in call_args[0][0] for call_args in self.mock_st.write.call_args_list))
        self.assertTrue(any("Rotations:" in call_args[0][0] for call_args in self.mock_st.write.call_args_list))
    
    def test_render_log_preview(self):
        """Test rendering log file preview UI."""
        # Create test log files
        self.create_test_log_files()
        
        # Mock selectbox to return the first option (index 0)
        self.mock_st.selectbox.return_value = 0
        
        # Call the render method
        self.log_manager.render_log_preview()
        
        # Check that streamlit functions were called
        self.mock_st.subheader.assert_called_once_with("Log File Preview")
        self.mock_st.selectbox.assert_called_once()
        self.mock_st.text.assert_called_once()
        self.mock_st.expander.assert_called_once()
        
        # Should call markdown with HTML for download link
        self.assertTrue(any("<a href=" in call_args[0][0] for call_args in self.mock_st.markdown.call_args_list))
    
    def test_render_log_management(self):
        """Test rendering complete log management UI."""
        # Mock tabs
        mock_tab1 = MagicMock()
        mock_tab2 = MagicMock()
        mock_tab3 = MagicMock()
        self.mock_st.tabs.return_value = [mock_tab1, mock_tab2, mock_tab3]
        
        # Create context managers for tabs
        mock_tab1.__enter__.return_value = mock_tab1
        mock_tab2.__enter__.return_value = mock_tab2
        mock_tab3.__enter__.return_value = mock_tab3
        
        # Mock render methods
        with patch.object(self.log_manager, 'render_log_list') as mock_render_list, \
             patch.object(self.log_manager, 'render_log_rotation_visualization') as mock_render_rotation, \
             patch.object(self.log_manager, 'render_log_preview') as mock_render_preview:
            
            # Call the render method
            self.log_manager.render_log_management()
            
            # Check that streamlit functions were called
            self.mock_st.title.assert_called_once_with("Log Management")
            self.mock_st.tabs.assert_called_once_with(["Log Files", "Rotation Status", "Log Preview"])
            
            # Check that render methods were called in the correct tabs
            mock_render_list.assert_called_once()
            mock_render_rotation.assert_called_once()
            mock_render_preview.assert_called_once()
            
            # Check info text
            self.mock_st.info.assert_called_once()
            self.assertIn("Logs are automatically rotated", self.mock_st.info.call_args[0][0])
    
    def test_create_download_link(self):
        """Test the create_download_link helper function."""
        content = "Test content"
        filename = "test.txt"
        link_text = "Download test"

