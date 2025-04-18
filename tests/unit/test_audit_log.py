import unittest
import json
from unittest.mock import MagicMock, patch

from src.utils import audit_log


class TestAuditLog(unittest.TestCase):
    @patch('src.utils.audit_log.redis_client')
    @patch('src.utils.audit_log.sentry_sdk')
    @patch('src.utils.audit_log.audit_logger')
    def test_log_audit_event_success(self, mock_logger, mock_sentry, mock_redis):
        # Setup the Redis mock
        mock_redis.rpush = MagicMock()
        
        event_name = "file_upload_success"
        user_id = "user_001"
        session_id = "session_001"
        details = {"filename": "data.csv", "status": "success"}
        
        audit_log.log_audit_event(event_name, user_id, session_id, details)
        
        # Validate that audit_logger.info was called with valid JSON
        self.assertTrue(mock_logger.info.called, "Logger info not called")
        log_args = mock_logger.info.call_args[0][0]
        log_entry = json.loads(log_args)
        self.assertEqual(log_entry["event"], event_name)
        self.assertEqual(log_entry["user_id"], user_id)
        self.assertEqual(log_entry["session_id"], session_id)
        self.assertEqual(log_entry["details"], details)
        self.assertIn("timestamp", log_entry)
        
        # Validate Sentry set_tag call
        mock_sentry.set_tag.assert_any_call("audit_event", event_name)
        
        # Validate that Redis rpush was called with the log entry
        mock_redis.rpush.assert_called_with("watchdog:audit_logs", json.dumps(log_entry))
    
    @patch('src.utils.audit_log.audit_logger')
    def test_log_audit_event_no_redis(self, mock_logger):
        # Backup original redis_client
        from src.utils.audit_log import redis_client
        original_redis = redis_client
        
        # Simulate Redis client as unavailable
        from src.utils import audit_log as al
        al.redis_client = None
        
        event_name = "schema_validation_failure"
        user_id = "user_002"
        session_id = "session_002"
        details = {"error": "invalid schema"}
        try:
            audit_log.log_audit_event(event_name, user_id, session_id, details)
            self.assertTrue(mock_logger.info.called, "Logger info should still be called even if Redis is unavailable")
        finally:
            # Restore original redis_client
            al.redis_client = original_redis


if __name__ == '__main__':
    unittest.main() 