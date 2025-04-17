import unittest
import os
import sys
import pandas as pd
import tempfile
from unittest.mock import patch, MagicMock

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import components to test
from src.nova_act.scheduler import NovaScheduler
from src.nova_act.fallback import NovaFallback
from src.nova_act.watchdog_upload import upload_to_watchdog, _get_file_type, _validate_processed_data


class TestNovaScheduler(unittest.TestCase):
    """Tests for the Nova Act scheduler component"""
    
    def setUp(self):
        # Create scheduler with mock task
        self.scheduler = NovaScheduler()
        self.mock_task = MagicMock()
        self.mock_task.return_value = True
    
    def test_schedule_task(self):
        """Test task scheduling functionality"""
        # Schedule a task
        task_id = self.scheduler.schedule_task(
            task_func=self.mock_task,
            schedule="daily",
            schedule_time="12:00",
            task_args=["arg1", "arg2"],
            task_kwargs={"key1": "value1"}
        )
        
        # Verify task was added to schedule
        self.assertIn(task_id, self.scheduler.tasks)
        task = self.scheduler.tasks[task_id]
        self.assertEqual(task["schedule"], "daily")
        self.assertEqual(task["schedule_time"], "12:00")
        self.assertEqual(task["task_args"], ["arg1", "arg2"])
        self.assertEqual(task["task_kwargs"], {"key1": "value1"})
    
    def test_cancel_task(self):
        """Test task cancellation"""
        # Schedule and then cancel a task
        task_id = self.scheduler.schedule_task(
            task_func=self.mock_task,
            schedule="daily"
        )
        
        # Verify task exists
        self.assertIn(task_id, self.scheduler.tasks)
        
        # Cancel the task
        result = self.scheduler.cancel_task(task_id)
        self.assertTrue(result)
        
        # Verify task was removed
        self.assertNotIn(task_id, self.scheduler.tasks)
    
    @patch('src.nova_act.scheduler.time')
    def test_run_pending_tasks(self, mock_time):
        """Test running of pending tasks"""
        # Mock time to ensure task is due
        mock_time.time.return_value = 1000  # Unix timestamp
        
        # Schedule a task that is due
        task_id = self.scheduler.schedule_task(
            task_func=self.mock_task,
            schedule="once",  # Run once
            next_run=900  # Earlier than current time
        )
        
        # Run pending tasks
        self.scheduler.run_pending_tasks()
        
        # Verify task was executed
        self.mock_task.assert_called_once()
        
        # Verify task was removed (since it's a "once" task)
        self.assertNotIn(task_id, self.scheduler.tasks)
    
    def test_task_execution_with_args(self):
        """Test that tasks execute with correct arguments"""
        # Schedule task with arguments
        self.scheduler.schedule_task(
            task_func=self.mock_task,
            schedule="once",
            next_run=0,  # Run immediately
            task_args=["test_arg"],
            task_kwargs={"test_key": "test_value"}
        )
        
        # Run pending tasks
        self.scheduler.run_pending_tasks()
        
        # Verify task was called with correct arguments
        self.mock_task.assert_called_once_with("test_arg", test_key="test_value")


class TestNovaFallback(unittest.TestCase):
    """Tests for the Nova Act fallback handling component"""
    
    def setUp(self):
        # Create a mock session state
        self.mock_session_state = {}
        
        # Create a fallback handler with the mock session state
        self.patcher = patch('src.nova_act.fallback.st')
        self.mock_st = self.patcher.start()
        self.mock_st.session_state = self.mock_session_state
        
        self.fallback = NovaFallback()
    
    def tearDown(self):
        self.patcher.stop()
    
    def test_handle_login_friction_2fa(self):
        """Test handling of 2FA login friction"""
        # Call the method with 2FA friction
        self.fallback.handle_login_friction("2fa", "Test Website")
        
        # Check if session state was properly updated
        self.assertIn("nova_error_type", self.mock_session_state)
        self.assertEqual(self.mock_session_state["nova_error_type"], "2fa")
        self.assertIn("nova_error_site", self.mock_session_state)
        self.assertEqual(self.mock_session_state["nova_error_site"], "Test Website")
    
    def test_handle_login_friction_captcha(self):
        """Test handling of CAPTCHA login friction"""
        # Call the method with CAPTCHA friction
        self.fallback.handle_login_friction("captcha", "Test Website")
        
        # Check if session state was properly updated
        self.assertIn("nova_error_type", self.mock_session_state)
        self.assertEqual(self.mock_session_state["nova_error_type"], "captcha")
        self.assertIn("nova_error_site", self.mock_session_state)
        self.assertEqual(self.mock_session_state["nova_error_site"], "Test Website")
    
    def test_clear_login_friction(self):
        """Test clearing of login friction state"""
        # Set up some friction state
        self.mock_session_state["nova_error_type"] = "2fa"
        self.mock_session_state["nova_error_site"] = "Test Website"
        
        # Clear the friction
        self.fallback.clear_login_friction()
        
        # Check if session state was properly cleared
        self.assertNotIn("nova_error_type", self.mock_session_state)
        self.assertNotIn("nova_error_site", self.mock_session_state)


class TestWatchdogUpload(unittest.TestCase):
    """Tests for the Nova Act watchdog upload component"""
    
    def setUp(self):
        # Create a mock session state
        self.mock_session_state = {}
        
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create a sample CSV file
        self.csv_file = os.path.join(self.temp_dir.name, "test_data.csv")
        with open(self.csv_file, 'w') as f:
            f.write("header1,header2,header3\n")
            f.write("value1,value2,value3\n")
        
        # Create an empty file
        self.empty_file = os.path.join(self.temp_dir.name, "empty.csv")
        with open(self.empty_file, 'w') as f:
            pass
        
        # Mock streamlit
        self.patcher = patch('src.nova_act.watchdog_upload.st')
        self.mock_st = self.patcher.start()
        self.mock_st.session_state = self.mock_session_state
    
    def tearDown(self):
        self.patcher.stop()
        self.temp_dir.cleanup()
    
    def test_get_file_type(self):
        """Test file type detection"""
        # Test CSV detection
        self.assertEqual(_get_file_type(self.csv_file), "csv")
        
        # Test Excel detection
        excel_file = os.path.join(self.temp_dir.name, "test.xlsx")
        with open(excel_file, 'w') as f:
            f.write("dummy excel content")
        self.assertEqual(_get_file_type(excel_file), "excel")
        
        # Test text file detection
        text_file = os.path.join(self.temp_dir.name, "test.txt")
        with open(text_file, 'w') as f:
            f.write("dummy text content")
        self.assertEqual(_get_file_type(text_file), "text")
    
    def test_validate_processed_data(self):
        """Test validation of processed data"""
        # Valid dataframe and summary
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        summary = {"status": "success"}
        self.assertTrue(_validate_processed_data(df, summary))
        
        # Empty dataframe
        empty_df = pd.DataFrame()
        self.assertFalse(_validate_processed_data(empty_df, summary))
        
        # None dataframe
        self.assertFalse(_validate_processed_data(None, summary))
        
        # Error summary
        error_summary = {"status": "error", "message": "Test error"}
        self.assertFalse(_validate_processed_data(df, error_summary))
    
    @patch('src.nova_act.watchdog_upload.process_uploaded_file')
    def test_upload_to_watchdog_file_not_found(self, mock_process):
        """Test handling of non-existent files"""
        # Attempt to upload a non-existent file
        result = upload_to_watchdog("non_existent_file.csv")
        
        # Check that it failed
        self.assertFalse(result)
        self.assertIn("nova_act_error", self.mock_session_state)
        self.assertIn("File not found", self.mock_session_state["nova_act_error"])
        
        # Verify the processing function was not called
        mock_process.assert_not_called()
    
    @patch('src.nova_act.watchdog_upload.process_uploaded_file')
    def test_upload_to_watchdog_empty_file(self, mock_process):
        """Test handling of empty files"""
        # Attempt to upload an empty file
        result = upload_to_watchdog(self.empty_file)
        
        # Check that it failed
        self.assertFalse(result)
        self.assertIn("nova_act_error", self.mock_session_state)
        self.assertIn("File is empty", self.mock_session_state["nova_act_error"])
        
        # Verify the processing function was not called
        mock_process.assert_not_called()
    
    @patch('src.nova_act.watchdog_upload.process_uploaded_file')
    def test_upload_to_watchdog_success(self, mock_process):
        """Test successful file upload and processing"""
        # Mock successful processing
        mock_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        mock_summary = {"status": "success"}
        mock_report = {"report": "test"}
        mock_schema = {"schema": "test"}
        mock_process.return_value = (mock_df, mock_summary, mock_report, mock_schema)
        
        # Attempt to upload a valid file
        result = upload_to_watchdog(self.csv_file)
        
        # Check that it succeeded
        self.assertTrue(result)
        
        # Verify session state was updated correctly
        self.assertEqual(self.mock_session_state["validated_data"], mock_df)
        self.assertEqual(self.mock_session_state["validation_summary"], mock_summary)
        self.assertEqual(self.mock_session_state["validation_report"], mock_report)
        self.assertEqual(self.mock_session_state["schema_info"], mock_schema)
        self.assertEqual(self.mock_session_state["last_uploaded_file"], "test_data.csv")
        
        # Verify the processing function was called
        mock_process.assert_called_once()
    
    @patch('src.nova_act.watchdog_upload.process_uploaded_file')
    def test_upload_to_watchdog_processing_error(self, mock_process):
        """Test handling of processing errors"""
        # Mock processing error
        mock_summary = {"status": "error", "message": "Test processing error"}
        mock_process.return_value = (None, mock_summary, None, None)
        
        # Attempt to upload a valid file
        result = upload_to_watchdog(self.csv_file)
        
        # Check that it failed
        self.assertFalse(result)
        self.assertIn("nova_act_error", self.mock_session_state)
        self.assertIn("Processing failed", self.mock_session_state["nova_act_error"])
        
        # Verify the processing function was called
        mock_process.assert_called_once()


if __name__ == "__main__":
    unittest.main() 