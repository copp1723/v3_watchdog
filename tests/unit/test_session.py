import unittest
import json
from unittest.mock import patch, MagicMock

import streamlit as st

from src.utils.session import get_session_id, record_action

class TestSessionManagement(unittest.TestCase):
    def setUp(self):
        # Ensure a clean session state for each test
        st.session_state.clear()

    def tearDown(self):
        st.session_state.clear()

    @patch.dict('streamlit.st.session_state', {}, clear=True)
    def test_get_session_id_creates_new_session(self):
        # When session_state is empty, get_session_id should create a new session id
        session_id = get_session_id()
        self.assertTrue(isinstance(session_id, str))
        self.assertNotEqual(session_id, "")
        self.assertIn('session_id', st.session_state)
        self.assertEqual(st.session_state['session_id'], session_id)

    @patch.dict('streamlit.st.session_state', {}, clear=True)
    def test_get_session_id_consistency(self):
        # Multiple calls should return the same session id
        session_id1 = get_session_id()
        session_id2 = get_session_id()
        self.assertEqual(session_id1, session_id2)

    @patch('src.utils.session.log_audit_event')
    @patch.dict('streamlit.st.session_state', {}, clear=True)
    def test_record_action_calls_audit_log_and_updates_history(self, mock_log_audit):
        event_name = "data_upload"
        metadata = {"filename": "test.csv"}
        
        # Ensure action_history is not present initially
        self.assertNotIn('action_history', st.session_state)
        
        record = record_action(event_name, metadata)
        
        # Verify that action_history is now in session_state and contains the new record
        self.assertIn('action_history', st.session_state)
        self.assertEqual(len(st.session_state['action_history']), 1)
        self.assertEqual(st.session_state['action_history'][0]['event'], event_name)
        self.assertEqual(st.session_state['action_history'][0]['metadata'], metadata)
        self.assertIn('timestamp', st.session_state['action_history'][0])
        self.assertIn('session_id', st.session_state['action_history'][0])

        # Verify that the log_audit_event function was called with the correct arguments
        session_id = st.session_state['session_id']
        mock_log_audit.assert_called_with(event_name, session_id, session_id, details=metadata)

    @patch('src.utils.session.log_audit_event')
    @patch.dict('streamlit.st.session_state', {}, clear=True)
    def test_record_action_graceful_initialization(self, mock_log_audit):
        # If action_history is missing, record_action should initialize it without error
        # Simulate st.session_state not having 'action_history'
        if 'action_history' in st.session_state:
            del st.session_state['action_history']
        
        event_name = "normalization_step"
        metadata = {"step": "column_rename"}
        
        record = record_action(event_name, metadata)
        
        self.assertIn('action_history', st.session_state)
        self.assertEqual(len(st.session_state['action_history']), 1)
        self.assertEqual(st.session_state['action_history'][0]['event'], event_name)
        
        # Even if log_audit_event fails, record_action should complete gracefully
        mock_log_audit.side_effect = Exception("Redis error")
        try:
            record = record_action("insight_run", {"detail": "test insight"})
        except Exception as e:
            self.fail(f"record_action should handle exceptions silently, but raised: {e}")

if __name__ == '__main__':
    unittest.main() 