"""
Alert Escalation Router for V3 Watchdog AI.

Provides functionality for escalating high-risk and high-priority leads to appropriate 
team members based on configurable rules and thresholds.
"""

import json
import logging
import os
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
import pandas as pd
import uuid
import requests

# Scheduler for delayed escalations
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.date import DateTrigger

# Local imports
from ..scheduler.notification_service import NotificationService
from ..ml.lead_model import LeadOutcomePredictor

logger = logging.getLogger(__name__)

# Constants
DEFAULT_ESCALATION_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "config",
    "escalation_config.json"
)

DEFAULT_ESCALATION_LOG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "logs",
    "escalation_log.json"
)


class EscalationLevel(Enum):
    """Escalation priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EscalationChannel(Enum):
    """Available escalation channels."""
    EMAIL = "email"
    SMS = "sms"
    SLACK = "slack"
    WEBHOOK = "webhook"
    NOTIFICATION = "notification"


class AlertEscalationRouter:
    """Router for managing lead alert escalations."""
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 log_path: Optional[str] = None,
                 notification_service: Optional[NotificationService] = None,
                 predictor: Optional[LeadOutcomePredictor] = None):
        """
        Initialize the escalation router.
        
        Args:
            config_path: Path to configuration file
            log_path: Path to log file
            notification_service: Optional notification service instance
            predictor: Optional lead predictor instance
        """
        self.config_path = config_path or DEFAULT_ESCALATION_CONFIG_PATH
        self.log_path = log_path or DEFAULT_ESCALATION_LOG_PATH
        
        # Initialize notification service
        self.notification_service = notification_service or NotificationService()
        
        # Initialize lead predictor
        self.predictor = predictor or LeadOutcomePredictor()
        
        # Create scheduler for delayed escalations
        self.scheduler = BackgroundScheduler()
        self.scheduler.start()
        
        # Load config
        self.config = self._load_config()
        
        # Load or initialize escalation log
        self._init_log()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load escalation configuration.
        
        Returns:
            Configuration dictionary
        """
        default_config = {
            "thresholds": {
                "low_probability": 0.3,
                "critical_probability": 0.1
            },
            "escalation_rules": {
                "default": {
                    "level": "medium",
                    "channels": ["email"],
                    "recipients": [],
                    "delay_minutes": 60,
                    "message_template": "Potential at-risk lead: {lead_id} requires attention"
                },
                "high_value": {
                    "level": "high",
                    "channels": ["email", "notification"],
                    "recipients": [],
                    "delay_minutes": 30,
                    "message_template": "High-value lead {lead_id} at risk of loss"
                },
                "critical": {
                    "level": "critical",
                    "channels": ["email", "sms", "notification"],
                    "recipients": [],
                    "delay_minutes": 0,
                    "message_template": "URGENT: Critical lead {lead_id} requires immediate action"
                }
            },
            "routing": {
                "fallback_recipients": [],
                "manager_recipients": [],
                "auto_reassign": True,
                "reassign_after_hours": True
            },
            "webhook_urls": {
                "default": "",
                "high_priority": ""
            },
            "working_hours": {
                "start_hour": 8,
                "end_hour": 18,
                "work_days": [0, 1, 2, 3, 4, 5]  # Monday-Saturday (0=Monday)
            }
        }
        
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    loaded_config = json.load(f)
                
                # Update default config with loaded values
                self._update_nested_dict(default_config, loaded_config)
                logger.info(f"Loaded escalation config from {self.config_path}")
            else:
                # Create default config
                os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
                with open(self.config_path, 'w') as f:
                    json.dump(default_config, f, indent=2)
                logger.info(f"Created default escalation config at {self.config_path}")
        except Exception as e:
            logger.error(f"Error loading escalation config: {e}")
        
        return default_config
    
    def _update_nested_dict(self, d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update a nested dictionary with values from another dictionary.
        
        Args:
            d: Original dictionary to update
            u: Dictionary with update values
            
        Returns:
            Updated dictionary
        """
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._update_nested_dict(d[k], v)
            else:
                d[k] = v
        return d
    
    def _init_log(self) -> None:
        """Initialize or load the escalation log."""
        try:
            if os.path.exists(self.log_path):
                # Load existing log
                with open(self.log_path, 'r') as f:
                    self.escalation_log = json.load(f)
            else:
                # Create new log
                self.escalation_log = {
                    "escalations": [],
                    "stats": {
                        "total_escalations": 0,
                        "by_level": {
                            "low": 0,
                            "medium": 0,
                            "high": 0,
                            "critical": 0
                        },
                        "by_channel": {
                            "email": 0,
                            "sms": 0,
                            "slack": 0,
                            "webhook": 0,
                            "notification": 0
                        },
                        "successful": 0,
                        "failed": 0
                    },
                    "last_updated": datetime.now().isoformat()
                }
                
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
                
                # Save the initialized log
                with open(self.log_path, 'w') as f:
                    json.dump(self.escalation_log, f, indent=2)
        except Exception as e:
            logger.error(f"Error initializing escalation log: {e}")
            # Create an in-memory log
            self.escalation_log = {
                "escalations": [],
                "stats": {
                    "total_escalations": 0,
                    "by_level": {
                        "low": 0,
                        "medium": 0,
                        "high": 0,
                        "critical": 0
                    },
                    "by_channel": {
                        "email": 0,
                        "sms": 0,
                        "slack": 0,
                        "webhook": 0,
                        "notification": 0
                    },
                    "successful": 0,
                    "failed": 0
                },
                "last_updated": datetime.now().isoformat()
            }
    
    def _save_log(self) -> None:
        """Save the escalation log to disk."""
        try:
            # Update timestamp
            self.escalation_log["last_updated"] = datetime.now().isoformat()
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            
            # Save log
            with open(self.log_path, 'w') as f:
                json.dump(self.escalation_log, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving escalation log: {e}")
    
    def _get_escalation_config(self, lead_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get appropriate escalation configuration based on lead data.
        
        Args:
            lead_data: Lead information dictionary
            
        Returns:
            Dictionary with escalation configuration
        """
        # Default configuration
        config = self.config["escalation_rules"]["default"].copy()
        
        # Check for high value lead
        is_high_value = False
        if "vehicle_price" in lead_data:
            try:
                price = float(lead_data["vehicle_price"])
                # Assuming high value is over $30,000
                if price >= 30000:
                    is_high_value = True
            except (ValueError, TypeError):
                pass
        
        # Check for critical lead
        is_critical = False
        if "sale_probability_medium" in lead_data:
            try:
                probability = float(lead_data["sale_probability_medium"])
                critical_threshold = self.config["thresholds"]["critical_probability"]
                if probability <= critical_threshold:
                    is_critical = True
            except (ValueError, TypeError):
                pass
        
        # Apply specific configuration
        if is_critical and "critical" in self.config["escalation_rules"]:
            config.update(self.config["escalation_rules"]["critical"])
        elif is_high_value and "high_value" in self.config["escalation_rules"]:
            config.update(self.config["escalation_rules"]["high_value"])
        
        return config
    
    def _get_recipients(self, lead_data: Dict[str, Any], 
                      escalation_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get appropriate recipients for the escalation.
        
        Args:
            lead_data: Lead information dictionary
            escalation_config: Escalation configuration
            
        Returns:
            List of recipient dictionaries with contact information
        """
        recipients = []
        
        # Add configured recipients from escalation config
        for recipient in escalation_config.get("recipients", []):
            recipients.append(recipient)
        
        # Add assigned sales rep if available
        if "rep_email" in lead_data and lead_data["rep_email"]:
            rep_recipient = {
                "name": lead_data.get("rep", "Sales Rep"),
                "email": lead_data["rep_email"],
                "phone": lead_data.get("rep_phone", ""),
                "type": "rep"
            }
            recipients.append(rep_recipient)
        
        # Add fallback recipients if no other recipients
        if not recipients:
            for recipient in self.config["routing"].get("fallback_recipients", []):
                recipients.append(recipient)
        
        # Add manager recipients for high level escalations
        level = escalation_config.get("level", "medium")
        if level in ["high", "critical"]:
            for recipient in self.config["routing"].get("manager_recipients", []):
                # Check if manager is already in recipients list
                if not any(r.get("email", "") == recipient.get("email", "") for r in recipients):
                    recipients.append(recipient)
        
        # Handle after-hours routing
        if self._is_after_hours() and self.config["routing"].get("reassign_after_hours", True):
            # Filter to only keep recipients marked as after-hours
            after_hours_recipients = [r for r in recipients if r.get("after_hours", False)]
            if after_hours_recipients:
                return after_hours_recipients
        
        return recipients
    
    def _is_after_hours(self) -> bool:
        """
        Check if current time is outside of working hours.
        
        Returns:
            True if outside working hours, False otherwise
        """
        now = datetime.now()
        start_hour = self.config["working_hours"].get("start_hour", 8)
        end_hour = self.config["working_hours"].get("end_hour", 18)
        work_days = self.config["working_hours"].get("work_days", [0, 1, 2, 3, 4])  # Mon-Fri by default
        
        # Check if current weekday is a work day (0 = Monday in Python's datetime)
        if now.weekday() not in work_days:
            return True
        
        # Check if current hour is outside working hours
        if now.hour < start_hour or now.hour >= end_hour:
            return True
        
        return False
    
    def _update_stats(self, level: str, channels: List[str], success: bool) -> None:
        """
        Update escalation statistics.
        
        Args:
            level: Escalation level
            channels: Channels used
            success: Whether escalation was successful
        """
        stats = self.escalation_log["stats"]
        
        # Update total count
        stats["total_escalations"] += 1
        
        # Update level count
        if level in stats["by_level"]:
            stats["by_level"][level] += 1
        
        # Update channel counts
        for channel in channels:
            if channel in stats["by_channel"]:
                stats["by_channel"][channel] += 1
        
        # Update success/failure
        if success:
            stats["successful"] += 1
        else:
            stats["failed"] += 1
    
    def _format_message(self, template: str, lead_data: Dict[str, Any]) -> str:
        """
        Format message template with lead data.
        
        Args:
            template: Message template string
            lead_data: Lead information dictionary
            
        Returns:
            Formatted message
        """
        try:
            return template.format(**lead_data)
        except KeyError:
            # Fallback for missing fields
            safe_data = {
                "lead_id": lead_data.get("lead_id", "Unknown Lead"),
                "rep": lead_data.get("rep", "Unassigned"),
                "source": lead_data.get("source", "Unknown Source"),
                "vehicle": lead_data.get("vehicle", "Unknown Vehicle"),
                "probability": lead_data.get("sale_probability_medium", "Unknown")
            }
            return template.format(**safe_data)
        except Exception as e:
            logger.error(f"Error formatting message: {e}")
            return f"Alert for lead {lead_data.get('lead_id', 'Unknown')}"
    
    def process_lead(self, lead_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a lead to determine if escalation is needed.
        
        Args:
            lead_data: Lead information dictionary
            
        Returns:
            Dictionary with processing results
        """
        # Ensure lead_id exists
        if "lead_id" not in lead_data:
            lead_data["lead_id"] = f"LEAD_{uuid.uuid4().hex[:8]}"
        
        # Get lead probability if not already present
        if "sale_probability_medium" not in lead_data and self.predictor:
            try:
                # Convert to DataFrame for prediction
                lead_df = pd.DataFrame([lead_data])
                
                # Make prediction
                pred_df = self.predictor.predict(lead_df)
                
                # Extract probability
                if "sale_probability_medium" in pred_df.columns:
                    lead_data["sale_probability_medium"] = float(pred_df["sale_probability_medium"].iloc[0])
                    lead_data["predicted_outcome_medium"] = int(pred_df["predicted_outcome_medium"].iloc[0])
            except Exception as e:
                logger.error(f"Error predicting lead outcome: {e}")
        
        # Check if escalation is needed
        threshold = self.config["thresholds"]["low_probability"]
        needs_escalation = False
        
        if "sale_probability_medium" in lead_data:
            probability = float(lead_data["sale_probability_medium"])
            needs_escalation = probability <= threshold
        
        # Process escalation if needed
        if needs_escalation:
            return self.escalate_lead(lead_data)
        else:
            return {
                "status": "no_escalation",
                "lead_id": lead_data.get("lead_id"),
                "probability": lead_data.get("sale_probability_medium"),
                "threshold": threshold,
                "timestamp": datetime.now().isoformat()
            }
    
    def escalate_lead(self, lead_data: Dict[str, Any], 
                     immediate: bool = False) -> Dict[str, Any]:
        """
        Escalate a lead to appropriate recipients.
        
        Args:
            lead_data: Lead information dictionary
            immediate: Whether to escalate immediately or respect delay settings
            
        Returns:
            Dictionary with escalation results
        """
        # Get escalation configuration based on lead data
        escalation_config = self._get_escalation_config(lead_data)
        
        # Get appropriate recipients
        recipients = self._get_recipients(lead_data, escalation_config)
        
        # Get escalation level
        level = escalation_config.get("level", "medium")
        
        # Get channels
        channels = escalation_config.get("channels", ["email"])
        
        # Get delay time
        delay_minutes = 0 if immediate else escalation_config.get("delay_minutes", 60)
        
        # Format message
        template = escalation_config.get("message_template", "Alert for lead {lead_id}")
        message = self._format_message(template, lead_data)
        
        # Create escalation record
        escalation_id = f"ESC_{uuid.uuid4().hex[:8]}"
        escalation = {
            "id": escalation_id,
            "lead_id": lead_data.get("lead_id"),
            "level": level,
            "channels": channels,
            "recipients": recipients,
            "message": message,
            "created_at": datetime.now().isoformat(),
            "scheduled_for": (datetime.now() + timedelta(minutes=delay_minutes)).isoformat(),
            "status": "scheduled",
            "results": {}
        }
        
        # Add to log
        self.escalation_log["escalations"].append(escalation)
        self._save_log()
        
        if delay_minutes > 0 and not immediate:
            # Schedule for later
            scheduled_time = datetime.now() + timedelta(minutes=delay_minutes)
            self.scheduler.add_job(
                func=self._execute_escalation,
                trigger=DateTrigger(run_date=scheduled_time),
                args=[escalation_id],
                id=f"escalation_{escalation_id}",
                replace_existing=True
            )
            
            return {
                "status": "scheduled",
                "escalation_id": escalation_id,
                "lead_id": lead_data.get("lead_id"),
                "level": level,
                "scheduled_for": scheduled_time.isoformat(),
                "delay_minutes": delay_minutes
            }
        else:
            # Execute immediately
            return self._execute_escalation(escalation_id)
    
    def _execute_escalation(self, escalation_id: str) -> Dict[str, Any]:
        """
        Execute a scheduled escalation.
        
        Args:
            escalation_id: ID of the escalation to execute
            
        Returns:
            Dictionary with execution results
        """
        # Find escalation in log
        escalation = None
        for e in self.escalation_log["escalations"]:
            if e["id"] == escalation_id:
                escalation = e
                break
        
        if not escalation:
            return {"status": "error", "message": f"Escalation {escalation_id} not found"}
        
        # Update status
        escalation["status"] = "executing"
        escalation["executed_at"] = datetime.now().isoformat()
        
        # Get escalation data
        level = escalation["level"]
        channels = escalation["channels"]
        recipients = escalation["recipients"]
        message = escalation["message"]
        lead_id = escalation["lead_id"]
        
        # Prepare result tracking
        results = {}
        success = True
        
        # Execute each channel
        for channel in channels:
            channel_result = self._send_via_channel(
                channel=channel,
                recipients=recipients,
                message=message,
                escalation=escalation
            )
            
            results[channel] = channel_result
            
            # Track overall success
            if channel_result.get("status") != "success":
                success = False
        
        # Update escalation record
        escalation["status"] = "completed" if success else "failed"
        escalation["completed_at"] = datetime.now().isoformat()
        escalation["results"] = results
        
        # Update stats
        self._update_stats(level, channels, success)
        
        # Save log
        self._save_log()
        
        return {
            "status": escalation["status"],
            "escalation_id": escalation_id,
            "lead_id": lead_id,
            "level": level,
            "channels": channels,
            "results": results
        }
    
    def _send_via_channel(self, channel: str, recipients: List[Dict[str, Any]], 
                        message: str, escalation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send escalation message via specified channel.
        
        Args:
            channel: Channel to use
            recipients: List of recipient dictionaries
            message: Message to send
            escalation: Full escalation record
            
        Returns:
            Dictionary with sending results
        """
        result = {
            "status": "unknown",
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            if channel == EscalationChannel.EMAIL.value:
                result = self._send_email(recipients, message, escalation)
            elif channel == EscalationChannel.SMS.value:
                result = self._send_sms(recipients, message, escalation)
            elif channel == EscalationChannel.SLACK.value:
                result = self._send_slack(recipients, message, escalation)
            elif channel == EscalationChannel.WEBHOOK.value:
                result = self._send_webhook(recipients, message, escalation)
            elif channel == EscalationChannel.NOTIFICATION.value:
                result = self._send_notification(recipients, message, escalation)
            else:
                result = {
                    "status": "error",
                    "message": f"Unknown channel: {channel}",
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"Error sending via {channel}: {e}")
            result = {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }
        
        return result
    
    def _send_email(self, recipients: List[Dict[str, Any]], 
                  message: str, escalation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send escalation email.
        
        Args:
            recipients: List of recipient dictionaries
            message: Message to send
            escalation: Full escalation record
            
        Returns:
            Dictionary with sending results
        """
        # Extract email addresses
        emails = [r.get("email") for r in recipients if r.get("email")]
        
        if not emails:
            return {
                "status": "error",
                "message": "No valid email recipients found",
                "timestamp": datetime.now().isoformat()
            }
        
        # Prepare alert data
        lead_id = escalation.get("lead_id", "Unknown")
        level = escalation.get("level", "medium")
        
        alert = {
            "title": f"Lead Alert: {lead_id}",
            "description": message,
            "recommendations": [
                "Review lead details and contact history",
                "Reach out to the customer immediately",
                "Consider offering incentives to close the sale"
            ],
            "metrics": [
                {"label": "Lead ID", "value": lead_id},
                {"label": "Priority", "value": level.upper()},
                {"label": "Probability", "value": f"{escalation.get('probability', 'Unknown')}"}
            ]
        }
        
        # Send email using notification service
        subject = f"LEAD ALERT [{level.upper()}]: {lead_id}"
        
        try:
            message_id = self.notification_service.send_alert_email(
                recipients=emails,
                alert=alert,
                subject=subject
            )
            
            return {
                "status": "success",
                "message_id": message_id,
                "recipients": emails,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _send_sms(self, recipients: List[Dict[str, Any]], 
                message: str, escalation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send escalation SMS.
        
        Args:
            recipients: List of recipient dictionaries
            message: Message to send
            escalation: Full escalation record
            
        Returns:
            Dictionary with sending results
        """
        # Extract phone numbers
        phones = [r.get("phone") for r in recipients if r.get("phone")]
        
        if not phones:
            return {
                "status": "error",
                "message": "No valid SMS recipients found",
                "timestamp": datetime.now().isoformat()
            }
        
        # In a real implementation, you would use a service like Twilio or AWS SNS
        # For this example, we'll just log what would happen
        logger.info(f"Would send SMS to {', '.join(phones)}: {message}")
        
        return {
            "status": "success",
            "message": "SMS would be sent (simulation)",
            "recipients": phones,
            "timestamp": datetime.now().isoformat()
        }
    
    def _send_slack(self, recipients: List[Dict[str, Any]], 
                  message: str, escalation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send escalation to Slack.
        
        Args:
            recipients: List of recipient dictionaries
            message: Message to send
            escalation: Full escalation record
            
        Returns:
            Dictionary with sending results
        """
        # Get slack channels
        channels = [r.get("slack_channel") for r in recipients if r.get("slack_channel")]
        
        # Add default channel if configured
        if not channels and "slack_channel" in self.config.get("routing", {}):
            channels.append(self.config["routing"]["slack_channel"])
        
        if not channels:
            return {
                "status": "error",
                "message": "No valid Slack channels found",
                "timestamp": datetime.now().isoformat()
            }
        
        # Get escalation level
        level = escalation.get("level", "medium")
        
        # Create formatted message
        lead_id = escalation.get("lead_id", "Unknown")
        formatted_message = f"*LEAD ALERT [{level.upper()}]*: {message}"
        
        # In a real implementation, you would use Slack API
        # For this example, we'll use the notification service
        try:
            for channel in channels:
                self.notification_service.send_slack_notification(
                    channel=channel,
                    message=formatted_message
                )
            
            return {
                "status": "success",
                "channels": channels,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error sending Slack notification: {e}")
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _send_webhook(self, recipients: List[Dict[str, Any]], 
                    message: str, escalation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send escalation to webhook.
        
        Args:
            recipients: List of recipient dictionaries
            message: Message to send
            escalation: Full escalation record
            
        Returns:
            Dictionary with sending results
        """
        # Get webhook URLs
        level = escalation.get("level", "medium")
        webhook_url = None
        
        # Use level-specific webhook if available
        if level in ["high", "critical"] and self.config["webhook_urls"].get("high_priority"):
            webhook_url = self.config["webhook_urls"]["high_priority"]
        elif self.config["webhook_urls"].get("default"):
            webhook_url = self.config["webhook_urls"]["default"]
        
        if not webhook_url:
            return {
                "status": "error",
                "message": "No webhook URL configured",
                "timestamp": datetime.now().isoformat()
            }
        
        # Prepare payload
        payload = {
            "action": "lead_escalation",
            "lead_id": escalation.get("lead_id"),
            "level": level,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "escalation_id": escalation.get("id")
        }
        
        # In a real implementation, you would use requests to send the webhook
        # For this example, we'll use the notification service
        try:
            self.notification_service.send_webhook(
                webhook_url=webhook_url,
                payload=payload
            )
            
            return {
                "status": "success",
                "webhook_url": webhook_url,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error sending webhook: {e}")
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _send_notification(self, recipients: List[Dict[str, Any]], 
                         message: str, escalation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send in-app notification.
        
        Args:
            recipients: List of recipient dictionaries
            message: Message to send
            escalation: Full escalation record
            
        Returns:
            Dictionary with sending results
        """
        # Get user IDs
        user_ids = [r.get("user_id") for r in recipients if r.get("user_id")]
        
        if not user_ids:
            return {
                "status": "error",
                "message": "No valid notification recipients found",
                "timestamp": datetime.now().isoformat()
            }
        
        # In a real implementation, you would store the notification in a database
        # For this example, we'll just log what would happen
        logger.info(f"Would send in-app notification to {', '.join(user_ids)}: {message}")
        
        return {
            "status": "success",
            "message": "Notification would be sent (simulation)",
            "recipients": user_ids,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_recent_escalations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent escalations.
        
        Args:
            limit: Maximum number of escalations to return
            
        Returns:
            List of recent escalation records
        """
        escalations = self.escalation_log.get("escalations", [])
        
        # Sort by created timestamp (most recent first)
        sorted_escalations = sorted(
            escalations,
            key=lambda e: e.get("created_at", ""),
            reverse=True
        )
        
        return sorted_escalations[:limit]
    
    def get_escalation_stats(self) -> Dict[str, Any]:
        """
        Get escalation statistics.
        
        Returns:
            Dictionary with escalation statistics
        """
        return self.escalation_log.get("stats", {})
    
    def update_config(self, new_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update escalation configuration.
        
        Args:
            new_config: New configuration dictionary
            
        Returns:
            Updated configuration
        """
        # Update config
        self._update_nested_dict(self.config, new_config)
        
        # Save to file
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
        
        return self.config

    def shutdown(self) -> None:
        """Shutdown the router and clean up resources."""
        # Shutdown the scheduler
        self.scheduler.shutdown()
        # Final save of log
        self._save_log()


# Test function
def run_test() -> Dict[str, Any]:
    """Run a test of the escalation router with sample data."""
    # Create router
    router = AlertEscalationRouter()
    
    # Create sample lead data
    lead_data = {
        "lead_id": f"LEAD_{uuid.uuid4().hex[:8]}",
        "rep": "John Smith",
        "rep_email": "john.smith@example.com",
        "source": "Website",
        "vehicle": "SUV Pro",
        "created_date": datetime.now() - timedelta(days=2),
        "contacted_date": datetime.now() - timedelta(days=1),
        "sale_probability_medium": 0.25,  # Just above threshold to trigger escalation
        "vehicle_price": 28000
    }
    
    # Process lead
    result = router.process_lead(lead_data)
    print("Lead Process Result:", result)
    
    # Create a high-risk lead
    high_risk_lead = lead_data.copy()
    high_risk_lead["lead_id"] = f"LEAD_{uuid.uuid4().hex[:8]}"
    high_risk_lead["sale_probability_medium"] = 0.05  # Very low probability
    high_risk_lead["vehicle_price"] = 45000  # High value
    
    # Process high-risk lead with immediate escalation
    high_risk_result = router.escalate_lead(high_risk_lead, immediate=True)
    print("High Risk Lead Result:", high_risk_result)
    
    # Get recent escalations
    recent = router.get_recent_escalations(5)
    print(f"\nRecent Escalations ({len(recent)}):")
    for esc in recent:
        print(f"- {esc['id']}: {esc['level']} priority, status: {esc['status']}")
    
    # Get stats
    stats = router.get_escalation_stats()
    print("\nEscalation Stats:")
    print(f"- Total: {stats.get('total_escalations', 0)}")
    print(f"- By Level: {stats.get('by_level', {})}")
    print(f"- Success Rate: {stats.get('successful', 0)}/{stats.get('total_escalations', 0)}")
    
    return {
        "standard_lead": result,
        "high_risk_lead": high_risk_result,
        "recent_escalations": len(recent),
        "stats": stats
    }


if __name__ == "__main__":
    run_test()