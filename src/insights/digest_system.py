"""
Digest System Module for V3 Watchdog AI.

Provides functionality for generating and delivering automated executive summaries
(insight digests) on a scheduled basis via Slack and email.
"""

import os
import json
import uuid
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set, Union
from enum import Enum
from dataclasses import dataclass, field
import threading
import time

from src.insights_digest import InsightsDigest, DigestEntry, Insight, Recommendation, create_insights_digest
from src.digest_bot import DigestBot
from src.scheduler.base_scheduler import BaseScheduler, ScheduledReport, ReportFrequency, DeliveryMethod
from src.scheduler.notification_service import NotificationService

# Configure logging
logger = logging.getLogger(__name__)

class DigestFrequency(str, Enum):
    """Digest scheduling frequency options."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"

class DigestType(str, Enum):
    """Types of digests that can be generated."""
    SALES_SUMMARY = "sales_summary"
    PERFORMANCE_OVERVIEW = "performance_overview"
    TREND_ALERTS = "trend_alerts"
    CUSTOM = "custom"

class DigestFormat(str, Enum):
    """Digest output format options."""
    SLACK = "slack"
    EMAIL = "email"
    DASHBOARD = "dashboard"

@dataclass
class DigestRecipient:
    """Configuration for a digest recipient."""
    user_id: str
    name: str
    email: str
    slack_id: Optional[str] = None
    frequency: DigestFrequency = DigestFrequency.DAILY
    digest_types: List[DigestType] = field(default_factory=lambda: [
        DigestType.SALES_SUMMARY,
        DigestType.PERFORMANCE_OVERVIEW,
        DigestType.TREND_ALERTS
    ])
    preferred_format: DigestFormat = DigestFormat.EMAIL
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_delivered: Optional[str] = None
    feedback: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DigestDelivery:
    """Record of a digest delivery."""
    delivery_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    recipient_id: str = ""
    digest_type: DigestType = DigestType.SALES_SUMMARY
    format: DigestFormat = DigestFormat.EMAIL
    delivered_at: str = field(default_factory=lambda: datetime.now().isoformat())
    status: str = "delivered"
    error_message: Optional[str] = None
    engagement: Dict[str, Any] = field(default_factory=dict)

class DigestSystem:
    """Manages the generation and delivery of insight digests."""
    
    def __init__(self, data_dir: str = None):
        """
        Initialize the digest system.
        
        Args:
            data_dir: Directory to store digest data
        """
        self.data_dir = data_dir or os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "digests")
        self.recipients_file = os.path.join(self.data_dir, "recipients.json")
        self.deliveries_file = os.path.join(self.data_dir, "deliveries.json")
        self.digests_file = os.path.join(self.data_dir, "digests.json")
        
        # Create directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize data structures
        self.recipients: Dict[str, DigestRecipient] = {}
        self.deliveries: List[DigestDelivery] = []
        self.digests: Dict[str, Dict[str, Any]] = {}
        
        # Load existing data
        self._load_data()
        
        # Initialize notification service
        self.notification_service = NotificationService()
        
        # Thread for running the scheduler
        self.scheduler_thread = None
        self.running = False
        self.lock = threading.Lock()
    
    def _load_data(self) -> None:
        """Load data from files."""
        # Load recipients
        if os.path.exists(self.recipients_file):
            try:
                with open(self.recipients_file, 'r') as f:
                    data = json.load(f)
                    self.recipients = {
                        user_id: DigestRecipient(**recipient_data)
                        for user_id, recipient_data in data.items()
                    }
            except Exception as e:
                logger.error(f"Error loading recipients: {e}")
        
        # Load deliveries
        if os.path.exists(self.deliveries_file):
            try:
                with open(self.deliveries_file, 'r') as f:
                    data = json.load(f)
                    self.deliveries = [DigestDelivery(**delivery_data) for delivery_data in data]
            except Exception as e:
                logger.error(f"Error loading deliveries: {e}")
        
        # Load digests
        if os.path.exists(self.digests_file):
            try:
                with open(self.digests_file, 'r') as f:
                    self.digests = json.load(f)
            except Exception as e:
                logger.error(f"Error loading digests: {e}")
    
    def _save_data(self) -> None:
        """Save data to files."""
        # Save recipients
        try:
            with open(self.recipients_file, 'w') as f:
                json.dump(
                    {user_id: recipient.__dict__ for user_id, recipient in self.recipients.items()},
                    f,
                    indent=2
                )
        except Exception as e:
            logger.error(f"Error saving recipients: {e}")
        
        # Save deliveries
        try:
            with open(self.deliveries_file, 'w') as f:
                json.dump(
                    [delivery.__dict__ for delivery in self.deliveries],
                    f,
                    indent=2
                )
        except Exception as e:
            logger.error(f"Error saving deliveries: {e}")
        
        # Save digests
        try:
            with open(self.digests_file, 'w') as f:
                json.dump(self.digests, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving digests: {e}")
    
    def add_recipient(self, 
                     user_id: str,
                     name: str,
                     email: str,
                     slack_id: Optional[str] = None,
                     frequency: DigestFrequency = DigestFrequency.DAILY,
                     digest_types: Optional[List[DigestType]] = None,
                     preferred_format: DigestFormat = DigestFormat.EMAIL) -> bool:
        """
        Add a new recipient for digests.
        
        Args:
            user_id: Unique identifier for the user
            name: Display name for the user
            email: Email address for the user
            slack_id: Optional Slack ID for the user
            frequency: How often to send digests
            digest_types: Types of digests to send
            preferred_format: Preferred delivery format
            
        Returns:
            True if successful, False otherwise
        """
        if user_id in self.recipients:
            logger.warning(f"Recipient {user_id} already exists. Use update_recipient to modify.")
            return False
        
        recipient = DigestRecipient(
            user_id=user_id,
            name=name,
            email=email,
            slack_id=slack_id,
            frequency=frequency,
            digest_types=digest_types or [
                DigestType.SALES_SUMMARY,
                DigestType.PERFORMANCE_OVERVIEW,
                DigestType.TREND_ALERTS
            ],
            preferred_format=preferred_format
        )
        
        self.recipients[user_id] = recipient
        self._save_data()
        return True
    
    def update_recipient(self,
                        user_id: str,
                        name: Optional[str] = None,
                        email: Optional[str] = None,
                        slack_id: Optional[str] = None,
                        frequency: Optional[DigestFrequency] = None,
                        digest_types: Optional[List[DigestType]] = None,
                        preferred_format: Optional[DigestFormat] = None) -> bool:
        """
        Update an existing recipient's configuration.
        
        Args:
            user_id: Unique identifier for the user
            name: New display name for the user
            email: New email address for the user
            slack_id: New Slack ID for the user
            frequency: New digest frequency
            digest_types: New digest types
            preferred_format: New preferred format
            
        Returns:
            True if successful, False otherwise
        """
        if user_id not in self.recipients:
            logger.warning(f"Recipient {user_id} not found. Use add_recipient to create.")
            return False
        
        recipient = self.recipients[user_id]
        
        if name is not None:
            recipient.name = name
        if email is not None:
            recipient.email = email
        if slack_id is not None:
            recipient.slack_id = slack_id
        if frequency is not None:
            recipient.frequency = frequency
        if digest_types is not None:
            recipient.digest_types = digest_types
        if preferred_format is not None:
            recipient.preferred_format = preferred_format
        
        self._save_data()
        return True
    
    def remove_recipient(self, user_id: str) -> bool:
        """
        Remove a recipient from the digest system.
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            True if successful, False otherwise
        """
        if user_id not in self.recipients:
            logger.warning(f"Recipient {user_id} not found.")
            return False
        
        del self.recipients[user_id]
        self._save_data()
        return True
    
    def get_recipient(self, user_id: str) -> Optional[DigestRecipient]:
        """
        Get a recipient's configuration.
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            Recipient configuration or None if not found
        """
        return self.recipients.get(user_id)
    
    def get_all_recipients(self) -> List[DigestRecipient]:
        """
        Get all recipient configurations.
        
        Returns:
            List of all recipient configurations
        """
        return list(self.recipients.values())
    
    def generate_digest(self, digest_type: DigestType, data: Dict[str, Any]) -> str:
        """
        Generate a digest of the specified type.
        
        Args:
            digest_type: Type of digest to generate
            data: Data to use for generating the digest
            
        Returns:
            Digest ID
        """
        digest_id = str(uuid.uuid4())
        
        # Create a timestamp for the digest
        timestamp = datetime.now().isoformat()
        
        # Generate the digest content based on type
        if digest_type == DigestType.SALES_SUMMARY:
            content = self._generate_sales_summary(data)
        elif digest_type == DigestType.PERFORMANCE_OVERVIEW:
            content = self._generate_performance_overview(data)
        elif digest_type == DigestType.TREND_ALERTS:
            content = self._generate_trend_alerts(data)
        else:
            content = self._generate_custom_digest(data)
        
        # Store the digest
        self.digests[digest_id] = {
            "digest_id": digest_id,
            "digest_type": digest_type,
            "content": content,
            "generated_at": timestamp,
            "data": data
        }
        
        self._save_data()
        return digest_id
    
    def _generate_sales_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a daily sales summary digest.
        
        Args:
            data: Sales data to summarize
            
        Returns:
            Digest content
        """
        # This is a placeholder implementation
        # In a real implementation, this would analyze sales data and generate insights
        
        # Create a sample digest
        problems = [
            {
                "title": "Sales Decline in SUV Segment",
                "detail": {
                    "description": "SUV sales have declined by 15% compared to last month.",
                    "impact_area": "Sales",
                    "severity": "Medium",
                    "metric_value": -15.0,
                    "metric_unit": "%",
                    "benchmark": 0.0
                },
                "recommendations": [
                    {
                        "action": "Review SUV pricing strategy",
                        "priority": "High",
                        "estimated_impact": "Potential 5% increase in sales"
                    }
                ],
                "tags": ["sales", "suv", "pricing"]
            }
        ]
        
        opportunities = [
            {
                "title": "Increased Interest in Electric Vehicles",
                "detail": {
                    "description": "Customer inquiries for electric vehicles have increased by 25%.",
                    "impact_area": "Sales",
                    "severity": "High",
                    "metric_value": 25.0,
                    "metric_unit": "%",
                    "benchmark": 5.0
                },
                "recommendations": [
                    {
                        "action": "Increase EV inventory",
                        "priority": "High",
                        "estimated_impact": "Potential 10% increase in sales"
                    }
                ],
                "tags": ["sales", "ev", "inventory"]
            }
        ]
        
        # Create the digest
        digest = create_insights_digest(problems, opportunities)
        
        # Convert to dictionary for storage
        return digest.to_dict()
    
    def _generate_performance_overview(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a weekly performance overview digest.
        
        Args:
            data: Performance data to summarize
            
        Returns:
            Digest content
        """
        # This is a placeholder implementation
        # In a real implementation, this would analyze performance data and generate insights
        
        # Create a sample digest
        problems = [
            {
                "title": "Service Department Efficiency Decline",
                "detail": {
                    "description": "Service department efficiency has decreased by 8% this week.",
                    "impact_area": "Service",
                    "severity": "Medium",
                    "metric_value": -8.0,
                    "metric_unit": "%",
                    "benchmark": 0.0
                },
                "recommendations": [
                    {
                        "action": "Review service scheduling process",
                        "priority": "Medium",
                        "estimated_impact": "Potential 5% improvement in efficiency"
                    }
                ],
                "tags": ["service", "efficiency", "scheduling"]
            }
        ]
        
        opportunities = [
            {
                "title": "Parts Department Profitability Increase",
                "detail": {
                    "description": "Parts department profitability has increased by 12% this week.",
                    "impact_area": "Parts",
                    "severity": "High",
                    "metric_value": 12.0,
                    "metric_unit": "%",
                    "benchmark": 5.0
                },
                "recommendations": [
                    {
                        "action": "Expand parts inventory based on current trends",
                        "priority": "Medium",
                        "estimated_impact": "Potential 3% additional increase in profitability"
                    }
                ],
                "tags": ["parts", "profitability", "inventory"]
            }
        ]
        
        # Create the digest
        digest = create_insights_digest(problems, opportunities)
        
        # Convert to dictionary for storage
        return digest.to_dict()
    
    def _generate_trend_alerts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trend alerts digest.
        
        Args:
            data: Trend data to analyze
            
        Returns:
            Digest content
        """
        # This is a placeholder implementation
        # In a real implementation, this would analyze trend data and generate insights
        
        # Create a sample digest
        problems = [
            {
                "title": "Anomaly: Unusual Drop in Customer Satisfaction",
                "detail": {
                    "description": "Customer satisfaction scores have dropped significantly in the last 24 hours.",
                    "impact_area": "Customer Experience",
                    "severity": "High",
                    "metric_value": -15.0,
                    "metric_unit": "points",
                    "benchmark": 0.0
                },
                "recommendations": [
                    {
                        "action": "Investigate recent customer feedback",
                        "priority": "High",
                        "estimated_impact": "Identify and address root cause"
                    }
                ],
                "tags": ["customer", "satisfaction", "anomaly"]
            }
        ]
        
        opportunities = [
            {
                "title": "Top Gainer: Increased Web Traffic",
                "detail": {
                    "description": "Website traffic has increased by 30% compared to the previous week.",
                    "impact_area": "Marketing",
                    "severity": "High",
                    "metric_value": 30.0,
                    "metric_unit": "%",
                    "benchmark": 5.0
                },
                "recommendations": [
                    {
                        "action": "Analyze traffic sources to capitalize on successful channels",
                        "priority": "Medium",
                        "estimated_impact": "Potential 10% increase in lead generation"
                    }
                ],
                "tags": ["marketing", "web", "traffic"]
            }
        ]
        
        # Create the digest
        digest = create_insights_digest(problems, opportunities)
        
        # Convert to dictionary for storage
        return digest.to_dict()
    
    def _generate_custom_digest(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a custom digest.
        
        Args:
            data: Data to use for the custom digest
            
        Returns:
            Digest content
        """
        # This is a placeholder implementation
        # In a real implementation, this would use the provided data to generate a custom digest
        
        # For now, just return a simple digest
        problems = []
        opportunities = []
        
        # Create the digest
        digest = create_insights_digest(problems, opportunities)
        
        # Convert to dictionary for storage
        return digest.to_dict()
    
    def deliver_digest(self, digest_id: str, recipient_id: str, format: Optional[DigestFormat] = None) -> bool:
        """
        Deliver a digest to a recipient.
        
        Args:
            digest_id: ID of the digest to deliver
            recipient_id: ID of the recipient
            format: Optional format override
            
        Returns:
            True if successful, False otherwise
        """
        # Check if digest exists
        if digest_id not in self.digests:
            logger.error(f"Digest {digest_id} not found.")
            return False
        
        # Check if recipient exists
        if recipient_id not in self.recipients:
            logger.error(f"Recipient {recipient_id} not found.")
            return False
        
        digest = self.digests[digest_id]
        recipient = self.recipients[recipient_id]
        
        # Use recipient's preferred format if not specified
        if format is None:
            format = recipient.preferred_format
        
        # Create a delivery record
        delivery = DigestDelivery(
            recipient_id=recipient_id,
            digest_type=digest["digest_type"],
            format=format
        )
        
        try:
            # Format the digest content
            if format == DigestFormat.SLACK:
                content = self._format_for_slack(digest["content"])
                self.notification_service.send_slack_notification(
                    channel=recipient.slack_id or "general",
                    message=content
                )
            elif format == DigestFormat.EMAIL:
                content = self._format_for_email(digest["content"])
                # In a real implementation, this would send an email
                # For now, just log it
                logger.info(f"Would send email to {recipient.email} with subject 'Digest: {digest['digest_type']}'")
            else:  # DASHBOARD
                content = self._format_for_dashboard(digest["content"])
                # In a real implementation, this would update a dashboard
                # For now, just log it
                logger.info(f"Would update dashboard for {recipient.name} with digest {digest_id}")
            
            # Update the delivery record
            delivery.status = "delivered"
            
            # Update the recipient's last_delivered timestamp
            recipient.last_delivered = datetime.now().isoformat()
            
            # Add the delivery to the list
            self.deliveries.append(delivery)
            
            # Save the data
            self._save_data()
            
            return True
        except Exception as e:
            logger.error(f"Error delivering digest {digest_id} to {recipient_id}: {e}")
            delivery.status = "failed"
            delivery.error_message = str(e)
            self.deliveries.append(delivery)
            self._save_data()
            return False
    
    def _format_for_slack(self, digest_content: Dict[str, Any]) -> str:
        """
        Format digest content for Slack.
        
        Args:
            digest_content: Digest content to format
            
        Returns:
            Formatted Slack message
        """
        # Create a DigestBot instance
        bot = DigestBot([], max_insights=5)
        
        # Convert the digest content to a format the bot can use
        insights = []
        
        # Add problems
        for problem in digest_content.get("top_problems", []):
            insight = {
                "summary": problem.get("title", ""),
                "recommendation": problem.get("recommendations", [{}])[0].get("action", ""),
                "risk_flag": True,
                "timestamp": datetime.now().isoformat()
            }
            insights.append(insight)
        
        # Add opportunities
        for opportunity in digest_content.get("top_opportunities", []):
            insight = {
                "summary": opportunity.get("title", ""),
                "recommendation": opportunity.get("recommendations", [{}])[0].get("action", ""),
                "risk_flag": False,
                "timestamp": datetime.now().isoformat()
            }
            insights.append(insight)
        
        # Update the bot's insights
        bot.insights = insights
        
        # Generate the Slack message
        return bot.format_slack_message(insights)
    
    def _format_for_email(self, digest_content: Dict[str, Any]) -> str:
        """
        Format digest content for email.
        
        Args:
            digest_content: Digest content to format
            
        Returns:
            Formatted email content
        """
        # Create a DigestBot instance
        bot = DigestBot([], max_insights=5)
        
        # Convert the digest content to a format the bot can use
        insights = []
        
        # Add problems
        for problem in digest_content.get("top_problems", []):
            insight = {
                "summary": problem.get("title", ""),
                "recommendation": problem.get("recommendations", [{}])[0].get("action", ""),
                "risk_flag": True,
                "timestamp": datetime.now().isoformat()
            }
            insights.append(insight)
        
        # Add opportunities
        for opportunity in digest_content.get("top_opportunities", []):
            insight = {
                "summary": opportunity.get("title", ""),
                "recommendation": opportunity.get("recommendations", [{}])[0].get("action", ""),
                "risk_flag": False,
                "timestamp": datetime.now().isoformat()
            }
            insights.append(insight)
        
        # Update the bot's insights
        bot.insights = insights
        
        # Generate the email content
        return bot.format_email_content(insights)
    
    def _format_for_dashboard(self, digest_content: Dict[str, Any]) -> str:
        """
        Format digest content for dashboard.
        
        Args:
            digest_content: Digest content to format
            
        Returns:
            Formatted dashboard content
        """
        # Create a DigestBot instance
        bot = DigestBot([], max_insights=5)
        
        # Convert the digest content to a format the bot can use
        insights = []
        
        # Add problems
        for problem in digest_content.get("top_problems", []):
            insight = {
                "summary": problem.get("title", ""),
                "recommendation": problem.get("recommendations", [{}])[0].get("action", ""),
                "risk_flag": True,
                "timestamp": datetime.now().isoformat()
            }
            insights.append(insight)
        
        # Add opportunities
        for opportunity in digest_content.get("top_opportunities", []):
            insight = {
                "summary": opportunity.get("title", ""),
                "recommendation": opportunity.get("recommendations", [{}])[0].get("action", ""),
                "risk_flag": False,
                "timestamp": datetime.now().isoformat()
            }
            insights.append(insight)
        
        # Update the bot's insights
        bot.insights = insights
        
        # Generate the dashboard content
        return bot.format_dashboard_card(insights)
    
    def record_feedback(self, delivery_id: str, feedback: Dict[str, Any]) -> bool:
        """
        Record feedback for a digest delivery.
        
        Args:
            delivery_id: ID of the delivery
            feedback: Feedback data
            
        Returns:
            True if successful, False otherwise
        """
        # Find the delivery
        delivery = None
        for d in self.deliveries:
            if d.delivery_id == delivery_id:
                delivery = d
                break
        
        if delivery is None:
            logger.error(f"Delivery {delivery_id} not found.")
            return False
        
        # Update the delivery's engagement data
        delivery.engagement.update(feedback)
        
        # Update the recipient's feedback
        recipient = self.recipients.get(delivery.recipient_id)
        if recipient:
            recipient.feedback[delivery_id] = feedback
        
        # Save the data
        self._save_data()
        
        return True
    
    def get_delivery_stats(self) -> Dict[str, Any]:
        """
        Get statistics about digest deliveries.
        
        Returns:
            Delivery statistics
        """
        stats = {
            "total_deliveries": len(self.deliveries),
            "successful_deliveries": sum(1 for d in self.deliveries if d.status == "delivered"),
            "failed_deliveries": sum(1 for d in self.deliveries if d.status == "failed"),
            "deliveries_by_format": {},
            "deliveries_by_type": {},
            "deliveries_by_recipient": {},
            "feedback_stats": {
                "thumbs_up": 0,
                "thumbs_down": 0
            }
        }
        
        # Count deliveries by format
        for delivery in self.deliveries:
            format_key = delivery.format
            stats["deliveries_by_format"][format_key] = stats["deliveries_by_format"].get(format_key, 0) + 1
            
            type_key = delivery.digest_type
            stats["deliveries_by_type"][type_key] = stats["deliveries_by_type"].get(type_key, 0) + 1
            
            recipient_key = delivery.recipient_id
            stats["deliveries_by_recipient"][recipient_key] = stats["deliveries_by_recipient"].get(recipient_key, 0) + 1
        
        # Count feedback
        for delivery in self.deliveries:
            if "thumbs_up" in delivery.engagement:
                stats["feedback_stats"]["thumbs_up"] += 1
            if "thumbs_down" in delivery.engagement:
                stats["feedback_stats"]["thumbs_down"] += 1
        
        return stats
    
    def start_scheduler(self, interval: int = 60) -> None:
        """
        Start the digest scheduler.
        
        Args:
            interval: How often to check for due digests (in seconds)
        """
        if self.running:
            logger.warning("Scheduler is already running.")
            return
        
        self.running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, args=(interval,))
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()
        
        logger.info(f"Digest scheduler started with interval {interval} seconds.")
    
    def stop_scheduler(self) -> None:
        """Stop the digest scheduler."""
        if not self.running:
            logger.warning("Scheduler is not running.")
            return
        
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
            self.scheduler_thread = None
        
        logger.info("Digest scheduler stopped.")
    
    def _scheduler_loop(self, interval: int) -> None:
        """
        Main scheduler loop.
        
        Args:
            interval: How often to check for due digests (in seconds)
        """
        while self.running:
            try:
                # Check for due digests
                self._process_due_digests()
                
                # Sleep for the interval
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                # Sleep for a short time before retrying
                time.sleep(5)
    
    def _process_due_digests(self) -> None:
        """Process digests that are due for delivery."""
        now = datetime.now()
        
        # Check each recipient
        for recipient in self.recipients.values():
            # Skip if no last delivery (first delivery)
            if recipient.last_delivered is None:
                self._deliver_digests_for_recipient(recipient)
                continue
            
            # Parse the last delivery time
            last_delivery = datetime.fromisoformat(recipient.last_delivered)
            
            # Check if it's time for the next delivery
            if recipient.frequency == DigestFrequency.DAILY:
                if (now - last_delivery).days >= 1:
                    self._deliver_digests_for_recipient(recipient)
            elif recipient.frequency == DigestFrequency.WEEKLY:
                if (now - last_delivery).days >= 7:
                    self._deliver_digests_for_recipient(recipient)
            elif recipient.frequency == DigestFrequency.MONTHLY:
                if (now - last_delivery).days >= 30:
                    self._deliver_digests_for_recipient(recipient)
    
    def _deliver_digests_for_recipient(self, recipient: DigestRecipient) -> None:
        """
        Deliver digests for a recipient.
        
        Args:
            recipient: Recipient to deliver digests to
        """
        # Generate digests for each type
        for digest_type in recipient.digest_types:
            # Generate a new digest
            digest_id = self.generate_digest(digest_type, {})
            
            # Deliver the digest
            self.deliver_digest(digest_id, recipient.user_id, recipient.preferred_format)
            
            logger.info(f"Delivered {digest_type} digest to {recipient.name} ({recipient.user_id})") 