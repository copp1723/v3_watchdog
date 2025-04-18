import os
import time
import json
import unittest

from src.utils.session import get_session_id, record_action

AUDIT_LOG_FILE = 'audit.log'

class TestInsightAuditLogging(unittest.TestCase):
    def setUp(self):
        # Remove audit.log if exists, to ensure a clean state
        if os.path.exists(AUDIT_LOG_FILE):
            os.remove(AUDIT_LOG_FILE)

    def tearDown(self):
        # Optionally clean up audit.log after the test
        if os.path.exists(AUDIT_LOG_FILE):
            os.remove(AUDIT_LOG_FILE)

    def test_insight_audit_logging(self):
        # Get current session ID
        session_id = get_session_id()

        # Simulate an insight run
        record_action("insight_start", {"insight": "Lead Conversion"})
        time.sleep(0.1)  # simulate processing delay
        record_action("insight_end", {"insight": "Lead Conversion", "duration_ms": 100, "result_count": 42})

        # Give some time for file writing to occur
        time.sleep(0.1)

        # Read the audit log file and parse JSON entries
        self.assertTrue(os.path.exists(AUDIT_LOG_FILE), "Audit log file does not exist")
        with open(AUDIT_LOG_FILE, 'r') as f:
            lines = f.readlines()

        # Parse lines expecting each line is a JSON dump
        events = []
        for line in lines:
            try:
                event = json.loads(line.strip())
                events.append(event)
            except Exception as e:
                self.fail(f"Failed to parse audit log line as JSON: {line.strip()} Error: {e}")

        # Check that at least one insight_start and one insight_end event exist with the correct session_id
        insight_start_events = [e for e in events if e.get('event') == 'insight_start' and e.get('session_id') == session_id]
        insight_end_events = [e for e in events if e.get('event') == 'insight_end' and e.get('session_id') == session_id]

        self.assertTrue(len(insight_start_events) >= 1, "No insight_start event found with the correct session_id")
        self.assertTrue(len(insight_end_events) >= 1, "No insight_end event found with the correct session_id")

if __name__ == '__main__':
    unittest.main() 