"""
Unit tests for the Alert Escalation Router module.
"""

import unittest
import pandas as pd
import json
import os
import tempfile
import shutil
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from src.notifications.escalation import AlertEscalationRouter, EscalationLevel, EscalationChannel
from src.scheduler.notification_service import NotificationService
from src.ml.lead_model import LeadOutcomePredictor


class TestAlertEscalationRouter(unittest.TestCase):
    """Test cases for AlertEscalationRouter."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for test config and logs
        self.test_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.test_dir, "test_config.json")
        self.log_path = os.path.join(self.test_dir, "test_log.json")
        
        # Mock notification service
        self.mock_notification = MagicMock(spec=NotificationService)
        self.mock_notification.send_alert_email.return_value = "test_email_id"
        self.mock_notification.send_slack_notification.return_value = None
        self.mock_notification.send_webhook.return_value = None
        
        # Mock lead predictor
        self.mock_predictor = MagicMock(spec=LeadOutcomePredictor)
        self.mock_predictor.predict.return_value = pd.DataFrame({
            "sale_probability_medium": [0.2],
            "predicted_outcome_medium": [0]
        })
        
        # Create router with mocks
        self.router = AlertEscalationRouter(
            config_path=self.config_path,
            log_path=self.log_path,
            notification_service=self.mock_notification,
            predictor=self.mock_predictor
        )
        
        # Create test lead data
        self.lead_data = {
            "lead_id": "TEST_LEAD_123",
            "rep": "Test Rep",
            "rep_email": "test.rep@example.com",
            "source": "Test Source",
            "vehicle": "Test Vehicle",
            "created_date": datetime.now() - timedelta(days=2),
            "contacted_date": datetime.now() - timedelta(days=1),
            "sale_probability_medium": 0.2,
            "vehicle_price": 25000
        }
        
        # Create high-risk lead data
        self.high_risk_lead = self.lead_data.copy()
        self.high_risk_lead["lead_id"] = "HIGH_RISK_LEAD_123"
        self.high_risk_lead["sale_probability_medium"] = 0.05
        self.high_risk_lead["vehicle_price"] = 45000
    
    def tearDown(self):
        """Clean up after tests."""
        # Shutdown router
        self.router.shutdown()
        
        # Remove temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_initialization(self):
        """Test router initialization."""
        # Check paths
        self.assertEqual(self.router.config_path, self.config_path)
        self.assertEqual(self.router.log_path, self.log_path)
        
        # Check mocks
        self.assertEqual(self.router.notification_service, self.mock_notification)
        self.assertEqual(self.router.predictor, self.mock_predictor)
        
        # Check config
        self.assertIsNotNone(self.router.config)
        self.assertTrue("thresholds" in self.router.config)
        self.assertTrue("escalation_rules" in self.router.config)
        self.assertTrue("routing" in self.router.config)
        
        # Check log
        self.assertIsNotNone(self.router.escalation_log)
        self.assertTrue("escalations" in self.router.escalation_log)
        self.assertTrue("stats" in self.router.escalation_log)
    
    def test_process_lead_no_escalation(self):
        """Test processing a lead that doesn't need escalation."""
        # Modify threshold to avoid escalation
        self.router.config["thresholds"]["low_probability"] = 0.1
        
        # Process lead
        result = self.router.process_lead(self.lead_data)
        
        # Check result
        self.assertEqual(result["status"], "no_escalation")
        self.assertEqual(result["lead_id"], self.lead_data["lead_id"])
        self.assertEqual(result["probability"], self.lead_data["sale_probability_medium"])
        self.assertEqual(result["threshold"], 0.1)
    
    def test_process_lead_with_escalation(self):
        """Test processing a lead that needs escalation."""
        # Modify threshold to trigger escalation
        self.router.config["thresholds"]["low_probability"] = 0.3
        
        # Process lead
        result = self.router.process_lead(self.lead_data)
        
        # Check result
        self.assertEqual(result["status"], "scheduled")
        self.assertTrue("escalation_id" in result)
        self.assertEqual(result["lead_id"], self.lead_data["lead_id"])
        self.assertTrue("scheduled_for" in result)
    
    def test_immediate_escalation(self):
        """Test immediate escalation of a lead."""
        # Execute immediate escalation
        result = self.router.escalate_lead(self.high_risk_lead, immediate=True)
        
        # Check result
        self.assertIn(result["status"], ["completed", "failed"])
        self.assertTrue("escalation_id" in result)
        self.assertEqual(result["lead_id"], self.high_risk_lead["lead_id"])
        self.assertTrue("results" in result)
        
        # Check notification service calls
        self.mock_notification.send_alert_email.assert_called_once()
    
    def test_get_escalation_config(self):
        """Test getting escalation configuration for different leads."""
        # Get config for standard lead
        standard_config = self.router._get_escalation_config(self.lead_data)
        
        # Get config for high-risk lead
        high_risk_config = self.router._get_escalation_config(self.high_risk_lead)
        
        # Check that they are different
        self.assertNotEqual(standard_config.get("level"), "critical")
        self.assertEqual(high_risk_config.get("level"), "critical")
    
    def test_get_recipients(self):
        """Test getting recipients for escalation."""
        # Add test recipients to config
        self.router.config["routing"]["manager_recipients"] = [
            {"name": "Test Manager", "email": "test.manager@example.com", "type": "manager"}
        ]
        self.router.config["routing"]["fallback_recipients"] = [
            {"name": "Test Fallback", "email": "test.fallback@example.com", "type": "fallback"}
        ]
        
        # Get standard escalation config
        standard_config = self.router._get_escalation_config(self.lead_data)
        
        # Get recipients for standard lead
        recipients = self.router._get_recipients(self.lead_data, standard_config)
        
        # Check that rep is included
        rep_found = False
        for recipient in recipients:
            if recipient.get("email") == "test.rep@example.com":
                rep_found = True
                break
        
        self.assertTrue(rep_found, "Sales rep should be included in recipients")
        
        # Get critical escalation config
        critical_config = {
            "level": "critical",
            "recipients": []
        }
        
        # Get recipients for critical lead
        critical_recipients = self.router._get_recipients(self.high_risk_lead, critical_config)
        
        # Check that manager is included
        manager_found = False
        for recipient in critical_recipients:
            if recipient.get("type") == "manager":
                manager_found = True
                break
        
        self.assertTrue(manager_found, "Manager should be included in critical recipients")
    
    def test_after_hours_routing(self):
        """Test after-hours routing."""
        # Mock _is_after_hours to return True
        original_method = self.router._is_after_hours
        self.router._is_after_hours = MagicMock(return_value=True)
        
        # Add after-hours recipient
        self.router.config["routing"]["fallback_recipients"] = [
            {"name": "After Hours", "email": "after.hours@example.com", "type": "fallback", "after_hours": True}
        ]
        
        # Get standard escalation config
        standard_config = self.router._get_escalation_config(self.lead_data)
        
        # Get recipients for standard lead
        recipients = self.router._get_recipients(self.lead_data, standard_config)
        
        # Check that after-hours recipient is included
        after_hours_found = False
        for recipient in recipients:
            if recipient.get("email") == "after.hours@example.com":
                after_hours_found = True
                break
        
        self.assertTrue(after_hours_found, "After-hours recipient should be included")
        
        # Restore original method
        self.router._is_after_hours = original_method
    
    def test_update_config(self):
        """Test updating configuration."""
        # Original threshold
        original_threshold = self.router.config["thresholds"]["low_probability"]
        
        # New config
        new_config = {
            "thresholds": {
                "low_probability": 0.4
            }
        }
        
        # Update config
        updated_config = self.router.update_config(new_config)
        
        # Check that threshold changed
        self.assertEqual(updated_config["thresholds"]["low_probability"], 0.4)
        self.assertEqual(self.router.config["thresholds"]["low_probability"], 0.4)
        self.assertNotEqual(original_threshold, self.router.config["thresholds"]["low_probability"])
    
    def test_get_stats(self):
        """Test getting escalation statistics."""
        # Process some leads to generate stats
        self.router.process_lead(self.lead_data)
        self.router.escalate_lead(self.high_risk_lead, immediate=True)
        
        # Get stats
        stats = self.router.get_escalation_stats()
        
        # Check stats
        self.assertTrue("total_escalations" in stats)
        self.assertTrue("by_level" in stats)
        self.assertTrue("by_channel" in stats)
        self.assertTrue("successful" in stats)
        self.assertTrue("failed" in stats)
        
        # Check counts
        self.assertTrue(stats["total_escalations"] >= 2)
    
    def test_get_recent_escalations(self):
        """Test getting recent escalations."""
        # Process some leads to generate escalations
        self.router.process_lead(self.lead_data)
        self.router.escalate_lead(self.high_risk_lead, immediate=True)
        
        # Get recent escalations
        recent = self.router.get_recent_escalations(limit=5)
        
        # Check that we have escalations
        self.assertTrue(len(recent) >= 2)
        
        # Check escalation properties
        for escalation in recent:
            self.assertTrue("id" in escalation)
            self.assertTrue("lead_id" in escalation)
            self.assertTrue("level" in escalation)
            self.assertTrue("status" in escalation)
            self.assertTrue("created_at" in escalation)


if __name__ == "__main__":
    unittest.main()