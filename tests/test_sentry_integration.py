"""
Test module for Sentry SDK integration.

Run this directly with:
python -m tests.test_sentry_integration
"""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# Add project root to path for imports to work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the Sentry initialization function
from src.app import initialize_sentry, before_send


class TestSentryIntegration(unittest.TestCase):
    """Test cases for Sentry SDK integration."""
    
    @patch('sentry_sdk.init')
    @patch('os.getenv')
    def test_initialize_sentry(self, mock_getenv, mock_sentry_init):
        """Test Sentry SDK initialization."""
        # Mock environment variables
        mock_getenv.side_effect = lambda key, default: {
            "SENTRY_DSN": "https://test@test.ingest.sentry.io/12345",
            "ENVIRONMENT": "test",
            "VERSION": "0.1.0-test"
        }.get(key, default)
        
        # Call the function
        initialize_sentry()
        
        # Check that sentry.init was called with the right arguments
        mock_sentry_init.assert_called_once()
        
        # Extract the call arguments
        args, kwargs = mock_sentry_init.call_args
        
        # Verify DSN, environment, and release
        self.assertEqual(kwargs['dsn'], "https://test@test.ingest.sentry.io/12345")
        self.assertEqual(kwargs['environment'], "test")
        self.assertEqual(kwargs['release'], "0.1.0-test")
        
        # Verify integrations were set
        self.assertTrue('integrations' in kwargs)
        self.assertTrue(len(kwargs['integrations']) >= 3)  # At least 3 integrations
        
        # Verify before_send handler
        self.assertEqual(kwargs['before_send'], before_send)
    
    def test_before_send_filters_sensitive_data(self):
        """Test that the before_send handler filters sensitive data."""
        # Create a mock event with sensitive data
        event = {
            'exception': {
                'values': [{
                    'stacktrace': {
                        'frames': [{
                            'vars': {
                                'username': 'testuser',
                                'password': 'secret123',
                                'api_key': 'abcdef123456',
                                'normal_var': 'non-sensitive data'
                            }
                        }]
                    }
                }]
            }
        }
        
        # Process the event with before_send
        processed_event = before_send(event, {})
        
        # Check that sensitive data was filtered
        vars_dict = processed_event['exception']['values'][0]['stacktrace']['frames'][0]['vars']
        self.assertEqual(vars_dict['username'], 'testuser')  # Should not be filtered
        self.assertEqual(vars_dict['password'], '[FILTERED]')  # Should be filtered
        self.assertEqual(vars_dict['api_key'], '[FILTERED]')  # Should be filtered
        self.assertEqual(vars_dict['normal_var'], 'non-sensitive data')  # Should not be filtered
    
    def test_before_send_adds_tags(self):
        """Test that before_send adds appropriate tags."""
        # Create a mock event with no tags
        event = {}
        
        # Process the event with before_send
        processed_event = before_send(event, {})
        
        # Check that tags were added
        self.assertIn('tags', processed_event)
        self.assertEqual(processed_event['tags']['app'], 'watchdog_ai')


def generate_test_error():
    """Generate a test error that will be sent to Sentry."""
    try:
        # Intentional error for testing
        1 / 0
    except Exception as e:
        import sentry_sdk
        sentry_sdk.capture_exception(e)
        return "Test error generated and sent to Sentry"


def main():
    """Run the tests."""
    print("Testing Sentry SDK integration...")
    unittest.main()
    
    # Enable this to test actual Sentry reporting (requires valid DSN)
    # print(generate_test_error())


if __name__ == "__main__":
    main()