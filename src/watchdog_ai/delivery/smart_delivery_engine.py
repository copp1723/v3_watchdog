"""
Smart Delivery Logic Engine for Watchdog AI.

Provides dynamic triggers and routing logic for insight delivery.
"""

import logging
import json
import os
from typing import Dict, List, Any, Optional, Union, Set, Callable
from enum import Enum
from datetime import datetime, timedelta
import threading
import time
import queue
import uuid

from ..utils.upload_tracker import UploadTracker
from ..scheduler.notification_service import NotificationService, EmailMessage

logger = logging.getLogger(__name__)

class DeliveryPriority(str, Enum):
    """Priority levels for insight delivery."""
    IMMEDIATE = "immediate"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    SCHEDULED = "scheduled"

class DeliveryChannel(str, Enum):
    """Delivery channels for insights."""
    EMAIL = "email"
    SLACK = "slack"
    SMS = "sms"
    DASHBOARD = "dashboard"

class DeliveryTrigger(str, Enum):
    """Trigger types for insight delivery."""
    UPLOAD = "upload"
    CRITICAL_INSIGHT = "critical_insight"
    SCHEDULED = "scheduled"
    MANUAL = "manual"
    THRESHOLD = "threshold"

class DeliveryStatus(str, Enum):
    """Status codes for delivery tracking."""
    QUEUED = "queued"
    PROCESSING = "processing"
    DELIVERED = "delivered"
    FAILED = "failed"
    CANCELLED = "cancelled"

class DeliveryWindow:
    """Represents a time window for scheduled deliveries."""
    
    def __init__(self, 
                start_hour: int, 
                end_hour: int, 
                days: Optional[List[int]] = None):
        """
        Initialize a delivery window.
        
        Args:
            start_hour: Start hour (0-23)
            end_hour: End hour (0-23)
            days: Optional list of days (0=Monday, 6=Sunday)
        """
        self.start_hour = start_hour
        self.end_hour = end_hour
        self.days = days or list(range(7))  # Default to all days
    
    def is_active(self, dt: Optional[datetime] = None) -> bool:
        """
        Check if the current time is within the delivery window.
        
        Args:
            dt: Optional datetime to check (defaults to now)
            
        Returns:
            True if within delivery window
        """
        dt = dt or datetime.now()
        
        # Check day
        if dt.weekday() not in self.days:
            return False
        
        # Check hour
        return self.start_hour <= dt.hour < self.end_hour
    
    def next_window_start(self, dt: Optional[datetime] = None) -> datetime:
        """
        Get the start time of the next delivery window.
        
        Args:
            dt: Optional datetime to start from (defaults to now)
            
        Returns:
            Datetime of next window start
        """
        dt = dt or datetime.now()
        
        # If we're already in a window, and it's the right day, return current time
        if self.is_active(dt):
            return dt
        
        # Find the next available day
        days_to_check = 7  # Check up to a week ahead
        for i in range(days_to_check):
            check_date = dt + timedelta(days=i)
            
            # Skip if not an allowed day
            if check_date.weekday() not in self.days:
                continue
            
            # If it's today but before start hour
            if i == 0 and check_date.hour < self.start_hour:
                return datetime(
                    check_date.year, 
                    check_date.month, 
                    check_date.day, 
                    self.start_hour
                )
            
            # If it's a future day
            if i > 0:
                return datetime(
                    check_date.year, 
                    check_date.month, 
                    check_date.day, 
                    self.start_hour
                )
        
        # If no window found in the next week, default to tomorrow at start hour
        tomorrow = dt + timedelta(days=1)
        return datetime(
            tomorrow.year, 
            tomorrow.month, 
            tomorrow.day, 
            self.start_hour
        )


class DeliveryPreferences:
    """User preferences for insight delivery."""
    
    def __init__(self, 
                user_id: str,
                email: Optional[str] = None,
                slack_channel: Optional[str] = None,
                phone: Optional[str] = None,
                delivery_windows: Optional[List[DeliveryWindow]] = None,
                preferred_channel: DeliveryChannel = DeliveryChannel.EMAIL,
                delivery_frequency: str = "daily",
                critical_alerts_enabled: bool = True,
                digest_enabled: bool = True):
        """
        Initialize delivery preferences.
        
        Args:
            user_id: User ID
            email: Email address
            slack_channel: Slack channel
            phone: Phone number for SMS
            delivery_windows: List of delivery windows
            preferred_channel: Preferred delivery channel
            delivery_frequency: Delivery frequency (daily, weekly, etc.)
            critical_alerts_enabled: Whether critical alerts are enabled
            digest_enabled: Whether digest emails are enabled
        """
        self.user_id = user_id
        self.email = email
        self.slack_channel = slack_channel
        self.phone = phone
        self.delivery_windows = delivery_windows or [
            # Default to business hours on weekdays
            DeliveryWindow(9, 17, [0, 1, 2, 3, 4])
        ]
        self.preferred_channel = preferred_channel
        self.delivery_frequency = delivery_frequency
        self.critical_alerts_enabled = critical_alerts_enabled
        self.digest_enabled = digest_enabled
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DeliveryPreferences':
        """Create from dictionary."""
        # Parse delivery windows
        delivery_windows = []
        for window_data in data.get('delivery_windows', []):
            delivery_windows.append(DeliveryWindow(
                start_hour=window_data.get('start_hour', 9),
                end_hour=window_data.get('end_hour', 17),
                days=window_data.get('days', [0, 1, 2, 3, 4])
            ))
        
        # Create preferences
        return cls(
            user_id=data['user_id'],
            email=data.get('email'),
            slack_channel=data.get('slack_channel'),
            phone=data.get('phone'),
            delivery_windows=delivery_windows,
            preferred_channel=data.get('preferred_channel', DeliveryChannel.EMAIL),
            delivery_frequency=data.get('delivery_frequency', 'daily'),
            critical_alerts_enabled=data.get('critical_alerts_enabled', True),
            digest_enabled=data.get('digest_enabled', True)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'user_id': self.user_id,
            'email': self.email,
            'slack_channel': self.slack_channel,
            'phone': self.phone,
            'delivery_windows': [
                {
                    'start_hour': window.start_hour,
                    'end_hour': window.end_hour,
                    'days': window.days
                }
                for window in self.delivery_windows
            ],
            'preferred_channel': self.preferred_channel,
            'delivery_frequency': self.delivery_frequency,
            'critical_alerts_enabled': self.critical_alerts_enabled,
            'digest_enabled': self.digest_enabled
        }
    
    def get_next_delivery_window(self) -> datetime:
        """Get the start time of the next delivery window."""
        now = datetime.now()
        
        # Find the earliest next window from all windows
        next_windows = [window.next_window_start(now) for window in self.delivery_windows]
        return min(next_windows)


class DeliveryItem:
    """Represents an item to be delivered."""
    
    def __init__(self, 
                content: Dict[str, Any],
                recipient_id: str,
                priority: DeliveryPriority = DeliveryPriority.NORMAL,
                trigger: DeliveryTrigger = DeliveryTrigger.MANUAL,
                scheduled_time: Optional[datetime] = None,
                metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a delivery item.
        
        Args:
            content: Content to deliver
            recipient_id: Recipient user ID
            priority: Delivery priority
            trigger: What triggered this delivery
            scheduled_time: When to deliver (for scheduled items)
            metadata: Additional metadata
        """
        self.id = str(uuid.uuid4())
        self.content = content
        self.recipient_id = recipient_id
        self.priority = priority
        self.trigger = trigger
        self.created_at = datetime.now()
        self.scheduled_time = scheduled_time
        self.delivered_at = None
        self.status = DeliveryStatus.QUEUED
        self.metadata = metadata or {}
        self.error = None
        self.delivery_attempts = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'content_type': self.content.get('type', 'unknown'),
            'recipient_id': self.recipient_id,
            'priority': self.priority,
            'trigger': self.trigger,
            'created_at': self.created_at.isoformat(),
            'scheduled_time': self.scheduled_time.isoformat() if self.scheduled_time else None,
            'delivered_at': self.delivered_at.isoformat() if self.delivered_at else None,
            'status': self.status,
            'metadata': self.metadata,
            'error': self.error,
            'delivery_attempts': self.delivery_attempts
        }


class DeliveryMetrics:
    """Tracks delivery performance metrics."""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize delivery metrics.
        
        Args:
            db_path: Optional path to SQLite database
        """
        self.metrics = {
            'total_deliveries': 0,
            'successful_deliveries': 0,
            'failed_deliveries': 0,
            'delivery_times': [],  # List of delivery times in seconds
            'delivery_by_priority': {
                DeliveryPriority.IMMEDIATE: 0,
                DeliveryPriority.HIGH: 0,
                DeliveryPriority.NORMAL: 0,
                DeliveryPriority.LOW: 0,
                DeliveryPriority.SCHEDULED: 0
            },
            'delivery_by_trigger': {
                DeliveryTrigger.UPLOAD: 0,
                DeliveryTrigger.CRITICAL_INSIGHT: 0,
                DeliveryTrigger.SCHEDULED: 0,
                DeliveryTrigger.MANUAL: 0,
                DeliveryTrigger.THRESHOLD: 0
            },
            'delivery_by_channel': {
                DeliveryChannel.EMAIL: 0,
                DeliveryChannel.SLACK: 0,
                DeliveryChannel.SMS: 0,
                DeliveryChannel.DASHBOARD: 0
            }
        }
        
        # Initialize metrics file
        self.metrics_file = db_path or "data/metrics/delivery_metrics.json"
        os.makedirs(os.path.dirname(self.metrics_file), exist_ok=True)
        
        # Load existing metrics if available
        self._load_metrics()
    
    def _load_metrics(self) -> None:
        """Load metrics from file."""
        if os.path.exists(self.metrics_file):
            try:
                with open(self.metrics_file, 'r') as f:
                    self.metrics = json.load(f)
            except Exception as e:
                logger.error(f"Error loading metrics: {str(e)}")
    
    def _save_metrics(self) -> None:
        """Save metrics to file."""
        try:
            with open(self.metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving metrics: {str(e)}")
    
    def record_delivery(self, 
                      item: DeliveryItem, 
                      success: bool, 
                      delivery_time: float,
                      channel: DeliveryChannel) -> None:
        """
        Record a delivery attempt.
        
        Args:
            item: Delivery item
            success: Whether delivery was successful
            delivery_time: Time taken for delivery (seconds)
            channel: Delivery channel used
        """
        # Update counters
        self.metrics['total_deliveries'] += 1
        
        if success:
            self.metrics['successful_deliveries'] += 1
        else:
            self.metrics['failed_deliveries'] += 1
        
        # Record delivery time
        self.metrics['delivery_times'].append(delivery_time)
        
        # Update priority counter
        if item.priority in self.metrics['delivery_by_priority']:
            self.metrics['delivery_by_priority'][item.priority] += 1
        
        # Update trigger counter
        if item.trigger in self.metrics['delivery_by_trigger']:
            self.metrics['delivery_by_trigger'][item.trigger] += 1
        
        # Update channel counter
        if channel in self.metrics['delivery_by_channel']:
            self.metrics['delivery_by_channel'][channel] += 1
        
        # Save updated metrics
        self._save_metrics()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        return self.metrics
    
    def get_success_rate(self) -> float:
        """Get delivery success rate."""
        total = self.metrics['total_deliveries']
        if total == 0:
            return 100.0
        
        return (self.metrics['successful_deliveries'] / total) * 100
    
    def get_average_delivery_time(self) -> float:
        """Get average delivery time in seconds."""
        times = self.metrics['delivery_times']
        if not times:
            return 0.0
        
        return sum(times) / len(times)


class SmartDeliveryEngine:
    """
    Smart delivery engine with dynamic triggers and routing logic.
    """
    
    def __init__(self, 
                notification_service: Optional[NotificationService] = None,
                upload_tracker: Optional[UploadTracker] = None,
                preferences_file: str = "config/delivery_preferences.json",
                worker_count: int = 2):
        """
        Initialize the delivery engine.
        
        Args:
            notification_service: Optional notification service
            upload_tracker: Optional upload tracker
            preferences_file: Path to preferences file
            worker_count: Number of delivery workers
        """
        self.notification_service = notification_service or NotificationService()
        self.upload_tracker = upload_tracker or UploadTracker()
        self.preferences_file = preferences_file
        self.metrics = DeliveryMetrics()
        
        # Ensure preferences directory exists
        os.makedirs(os.path.dirname(preferences_file), exist_ok=True)
        
        # Load user preferences
        self.user_preferences = self._load_preferences()
        
        # Initialize delivery queue
        self.queue = queue.PriorityQueue()
        self.delivery_items = {}  # Map of item ID to DeliveryItem
        
        # Initialize workers
        self.worker_count = worker_count
        self.workers = []
        self.stop_event = threading.Event()
        
        # Start workers
        self._start_workers()
        
        # Register upload trigger
        if self.upload_tracker:
            self._register_upload_trigger()
    
    def _load_preferences(self) -> Dict[str, DeliveryPreferences]:
        """
        Load user delivery preferences.
        
        Returns:
            Dictionary of user ID to DeliveryPreferences
        """
        preferences = {}
        
        if os.path.exists(self.preferences_file):
            try:
                with open(self.preferences_file, 'r') as f:
                    data = json.load(f)
                
                for user_id, pref_data in data.items():
                    preferences[user_id] = DeliveryPreferences.from_dict({
                        'user_id': user_id,
                        **pref_data
                    })
                
                logger.info(f"Loaded delivery preferences for {len(preferences)} users")
            except Exception as e:
                logger.error(f"Error loading preferences: {str(e)}")
        
        return preferences
    
    def _save_preferences(self) -> None:
        """Save user preferences to file."""
        try:
            data = {
                user_id: prefs.to_dict() 
                for user_id, prefs in self.user_preferences.items()
            }
            
            with open(self.preferences_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved delivery preferences for {len(data)} users")
        except Exception as e:
            logger.error(f"Error saving preferences: {str(e)}")
    
    def set_user_preferences(self, preferences: DeliveryPreferences) -> None:
        """
        Set preferences for a user.
        
        Args:
            preferences: User delivery preferences
        """
        self.user_preferences[preferences.user_id] = preferences
        self._save_preferences()
    
    def get_user_preferences(self, user_id: str) -> Optional[DeliveryPreferences]:
        """
        Get preferences for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            DeliveryPreferences or None if not found
        """
        return self.user_preferences.get(user_id)
    
    def _start_workers(self) -> None:
        """Start delivery worker threads."""
        if self.workers:
            logger.warning("Workers already running")
            return
        
        self.stop_event.clear()
        
        for i in range(self.worker_count):
            worker = threading.Thread(
                target=self._delivery_worker,
                name=f"delivery-worker-{i}",
                daemon=True
            )
            self.workers.append(worker)
            worker.start()
        
        logger.info(f"Started {self.worker_count} delivery workers")
    
    def _stop_workers(self) -> None:
        """Stop delivery worker threads."""
        logger.info("Stopping delivery workers...")
        self.stop_event.set()
        
        for worker in self.workers:
            worker.join(timeout=5)
        
        self.workers = []
        logger.info("Delivery workers stopped")
    
    def _delivery_worker(self) -> None:
        """Worker thread for processing the delivery queue."""
        logger.info(f"Delivery worker started: {threading.current_thread().name}")
        
        while not self.stop_event.is_set():
            try:
                # Get next delivery with timeout
                try:
                    priority, item_id = self.queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Get the delivery item
                item = self.delivery_items.get(item_id)
                if not item:
                    logger.warning(f"Delivery item not found: {item_id}")
                    self.queue.task_done()
                    continue
                
                # Check if scheduled time has arrived
                if item.scheduled_time and item.scheduled_time > datetime.now():
                    # Put back in queue with same priority
                    self.queue.put((priority, item_id))
                    self.queue.task_done()
                    time.sleep(1.0)  # Avoid tight loop
                    continue
                
                # Process the delivery
                self._process_delivery(item)
                
                # Mark task as done
                self.queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in delivery worker: {str(e)}")
        
        logger.info(f"Delivery worker stopped: {threading.current_thread().name}")
    
    def _process_delivery(self, item: DeliveryItem) -> None:
        """
        Process a delivery item.
        
        Args:
            item: DeliveryItem to process
        """
        logger.info(f"Processing delivery {item.id} (priority: {item.priority})")
        
        # Update status
        item.status = DeliveryStatus.PROCESSING
        item.delivery_attempts += 1
        
        # Get user preferences
        preferences = self.get_user_preferences(item.recipient_id)
        if not preferences:
            logger.warning(f"No delivery preferences found for user {item.recipient_id}")
            item.status = DeliveryStatus.FAILED
            item.error = "No delivery preferences found"
            return
        
        # Determine delivery channel
        channel = self._select_delivery_channel(item, preferences)
        
        # Deliver based on channel
        start_time = time.time()
        success = False
        
        try:
            if channel == DeliveryChannel.EMAIL:
                success = self._deliver_email(item, preferences)
            elif channel == DeliveryChannel.SLACK:
                success = self._deliver_slack(item, preferences)
            elif channel == DeliveryChannel.SMS:
                success = self._deliver_sms(item, preferences)
            elif channel == DeliveryChannel.DASHBOARD:
                success = self._deliver_dashboard(item, preferences)
            else:
                logger.warning(f"Unsupported delivery channel: {channel}")
                item.error = f"Unsupported delivery channel: {channel}"
        except Exception as e:
            logger.error(f"Error delivering item {item.id}: {str(e)}")
            item.error = str(e)
            success = False
        
        # Calculate delivery time
        delivery_time = time.time() - start_time
        
        # Update item status
        if success:
            item.status = DeliveryStatus.DELIVERED
            item.delivered_at = datetime.now()
        else:
            item.status = DeliveryStatus.FAILED
        
        # Record metrics
        self.metrics.record_delivery(item, success, delivery_time, channel)
        
        logger.info(f"Delivery {item.id} {'succeeded' if success else 'failed'}")
    
    def _select_delivery_channel(self, 
                               item: DeliveryItem, 
                               preferences: DeliveryPreferences) -> DeliveryChannel:
        """
        Select the best delivery channel based on priority and preferences.
        
        Args:
            item: DeliveryItem to deliver
            preferences: User preferences
            
        Returns:
            Selected delivery channel
        """
        # For critical/immediate items, use the most reliable channel
        if item.priority in [DeliveryPriority.IMMEDIATE, DeliveryPriority.HIGH]:
            # Prefer email if available
            if preferences.email:
                return DeliveryChannel.EMAIL
            # Fall back to other channels
            elif preferences.slack_channel:
                return DeliveryChannel.SLACK
            elif preferences.phone:
                return DeliveryChannel.SMS
            else:
                return DeliveryChannel.DASHBOARD
        
        # For normal priority, use preferred channel
        return preferences.preferred_channel
    
    def _deliver_email(self, 
                     item: DeliveryItem, 
                     preferences: DeliveryPreferences) -> bool:
        """
        Deliver an item via email.
        
        Args:
            item: DeliveryItem to deliver
            preferences: User preferences
            
        Returns:
            True if successful
        """
        if not preferences.email:
            logger.warning(f"No email address for user {preferences.user_id}")
            return False
        
        try:
            # Determine email content based on item type
            content_type = item.content.get('type', 'unknown')
            
            if content_type == 'insight':
                subject = item.content.get('title', 'New Insight')
                html_content = self._format_insight_email(item.content)
            elif content_type == 'alert':
                subject = f"ALERT: {item.content.get('title', 'Alert Notification')}"
                html_content = self._format_alert_email(item.content)
            elif content_type == 'digest':
                subject = item.content.get('title', 'Insight Digest')
                html_content = self._format_digest_email(item.content)
            else:
                subject = f"Notification: {content_type}"
                html_content = f"<html><body><p>{json.dumps(item.content)}</p></body></html>"
            
            # Create email message
            email = EmailMessage(
                recipients=[preferences.email],
                subject=subject,
                html_content=html_content
            )
            
            # Queue for delivery
            message_id = self.notification_service.queue.add_email(email)
            
            # Store message ID in metadata
            item.metadata['email_message_id'] = message_id
            
            return True
            
        except Exception as e:
            logger.error(f"Error delivering email: {str(e)}")
            item.error = f"Email delivery error: {str(e)}"
            return False
    
    def _deliver_slack(self, 
                     item: DeliveryItem, 
                     preferences: DeliveryPreferences) -> bool:
        """
        Deliver an item via Slack.
        
        Args:
            item: DeliveryItem to deliver
            preferences: User preferences
            
        Returns:
            True if successful
        """
        if not preferences.slack_channel:
            logger.warning(f"No Slack channel for user {preferences.user_id}")
            return False
        
        try:
            # Format message based on content type
            content_type = item.content.get('type', 'unknown')
            
            if content_type == 'insight':
                message = f"*{item.content.get('title', 'New Insight')}*\n{item.content.get('summary', '')}"
            elif content_type == 'alert':
                message = f"*ALERT: {item.content.get('title', 'Alert')}*\n{item.content.get('description', '')}"
            else:
                message = f"*Notification: {content_type}*\n{json.dumps(item.content)}"
            
            # Send to Slack
            self.notification_service.send_slack_notification(
                channel=preferences.slack_channel,
                message=message
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error delivering to Slack: {str(e)}")
            item.error = f"Slack delivery error: {str(e)}"
            return False
    
    def _deliver_sms(self, 
                   item: DeliveryItem, 
                   preferences: DeliveryPreferences) -> bool:
        """
        Deliver an item via SMS.
        
        Args:
            item: DeliveryItem to deliver
            preferences: User preferences
            
        Returns:
            True if successful
        """
        if not preferences.phone:
            logger.warning(f"No phone number for user {preferences.user_id}")
            return False
        
        # SMS delivery not yet implemented
        logger.warning("SMS delivery not implemented")
        item.error = "SMS delivery not implemented"
        return False
    
    def _deliver_dashboard(self, 
                         item: DeliveryItem, 
                         preferences: DeliveryPreferences) -> bool:
        """
        Deliver an item to the dashboard.
        
        Args:
            item: DeliveryItem to deliver
            preferences: User preferences
            
        Returns:
            True if successful
        """
        # Dashboard delivery is always successful
        # In a real implementation, this would store the item for dashboard display
        return True
    
    def _format_insight_email(self, insight: Dict[str, Any]) -> str:
        """
        Format an insight for email delivery.
        
        Args:
            insight: Insight data
            
        Returns:
            HTML content for email
        """
        # In a real implementation, this would use a template
        html = f"""
        <html>
        <body>
            <h1>{insight.get('title', 'New Insight')}</h1>
            <p>{insight.get('summary', '')}</p>
            
            <h2>Details</h2>
            <p>{insight.get('details', '')}</p>
            
            <h2>Recommendations</h2>
            <ul>
        """
        
        for rec in insight.get('recommendations', []):
            html += f"<li>{rec}</li>"
        
        html += """
            </ul>
        </body>
        </html>
        """
        
        return html
    
    def _format_alert_email(self, alert: Dict[str, Any]) -> str:
        """
        Format an alert for email delivery.
        
        Args:
            alert: Alert data
            
        Returns:
            HTML content for email
        """
        # In a real implementation, this would use a template
        html = f"""
        <html>
        <body>
            <h1 style="color: red;">ALERT: {alert.get('title', 'Alert')}</h1>
            <p>{alert.get('description', '')}</p>
            
            <h2>Actions</h2>
            <ul>
        """
        
        for action in alert.get('actions', []):
            html += f"<li>{action}</li>"
        
        html += """
            </ul>
        </body>
        </html>
        """
        
        return html
    
    def _format_digest_email(self, digest: Dict[str, Any]) -> str:
        """
        Format a digest for email delivery.
        
        Args:
            digest: Digest data
            
        Returns:
            HTML content for email
        """
        # In a real implementation, this would use a template
        html = f"""
        <html>
        <body>
            <h1>{digest.get('title', 'Insight Digest')}</h1>
            <p>{digest.get('summary', '')}</p>
            
            <h2>Insights</h2>
        """
        
        for insight in digest.get('insights', []):
            html += f"""
            <div style="margin-bottom: 20px; border-left: 4px solid #007bff; padding-left: 10px;">
                <h3>{insight.get('title', 'Insight')}</h3>
                <p>{insight.get('summary', '')}</p>
            </div>
            """
        
        html += """
        </body>
        </html>
        """
        
        return html
    
    def _register_upload_trigger(self) -> None:
        """Register a trigger for new uploads."""
        # This would be implemented with an event listener in a real system
        pass
    
    def queue_delivery(self, 
                     content: Dict[str, Any],
                     recipient_id: str,
                     priority: DeliveryPriority = DeliveryPriority.NORMAL,
                     trigger: DeliveryTrigger = DeliveryTrigger.MANUAL,
                     scheduled_time: Optional[datetime] = None) -> str:
        """
        Queue an item for delivery.
        
        Args:
            content: Content to deliver
            recipient_id: Recipient user ID
            priority: Delivery priority
            trigger: What triggered this delivery
            scheduled_time: When to deliver (for scheduled items)
            
        Returns:
            Delivery item ID
        """
        # Get user preferences
        preferences = self.get_user_preferences(recipient_id)
        
        # If no preferences found, create default
        if not preferences:
            preferences = DeliveryPreferences(
                user_id=recipient_id,
                email=f"{recipient_id}@example.com"  # Default email
            )
            self.set_user_preferences(preferences)
        
        # For scheduled priority, set scheduled time if not provided
        if priority == DeliveryPriority.SCHEDULED and not scheduled_time:
            scheduled_time = preferences.get_next_delivery_window()
        
        # Create delivery item
        item = DeliveryItem(
            content=content,
            recipient_id=recipient_id,
            priority=priority,
            trigger=trigger,
            scheduled_time=scheduled_time
        )
        
        # Store item
        self.delivery_items[item.id] = item
        
        # Determine queue priority (lower number = higher priority)
        queue_priority = {
            DeliveryPriority.IMMEDIATE: 0,
            DeliveryPriority.HIGH: 1,
            DeliveryPriority.NORMAL: 2,
            DeliveryPriority.LOW: 3,
            DeliveryPriority.SCHEDULED: 4
        }.get(priority, 2)  # Default to NORMAL priority
        
        # Add to queue
        self.queue.put((queue_priority, item.id))
        
        logger.info(f"Queued delivery {item.id} with priority {priority}")
        
        return item.id
    
    def cancel_delivery(self, item_id: str) -> bool:
        """
        Cancel a queued delivery.
        
        Args:
            item_id: Delivery item ID
            
        Returns:
            True if cancelled successfully
        """
        item = self.delivery_items.get(item_id)
        if not item:
            logger.warning(f"Delivery item not found: {item_id}")
            return False
        
        # Can only cancel if not already delivered or failed
        if item.status in [DeliveryStatus.DELIVERED, DeliveryStatus.FAILED]:
            logger.warning(f"Cannot cancel delivery {item_id} with status {item.status}")
            return False
        
        # Update status
        item.status = DeliveryStatus.CANCELLED
        
        logger.info(f"Cancelled delivery {item_id}")
        return True
    
    def get_delivery_status(self, item_id: str) -> Dict[str, Any]:
        """
        Get the status of a delivery.
        
        Args:
            item_id: Delivery item ID
            
        Returns:
            Dictionary with delivery status
        """
        item = self.delivery_items.get(item_id)
        if not item:
            return {"status": "unknown", "error": "Delivery item not found"}
        
        return item.to_dict()
    
    def get_pending_deliveries(self, recipient_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get pending deliveries.
        
        Args:
            recipient_id: Optional recipient ID to filter by
            
        Returns:
            List of pending delivery items
        """
        pending = []
        
        for item in self.delivery_items.values():
            if item.status in [DeliveryStatus.QUEUED, DeliveryStatus.PROCESSING]:
                if recipient_id is None or item.recipient_id == recipient_id:
                    pending.append(item.to_dict())
        
        return pending
    
    def trigger_on_upload(self, upload_record_id: str) -> List[str]:
        """
        Trigger deliveries based on a new upload.
        
        Args:
            upload_record_id: Upload record ID
            
        Returns:
            List of queued delivery item IDs
        """
        # Get upload record
        record = self.upload_tracker.get_upload_by_id(upload_record_id)
        if not record:
            logger.warning(f"Upload record not found: {upload_record_id}")
            return []
        
        # Find users who should be notified
        delivery_ids = []
        
        # In a real implementation, this would query user preferences
        # For now, just notify all users with upload trigger enabled
        for user_id, preferences in self.user_preferences.items():
            # Create content
            content = {
                "type": "upload_notification",
                "title": f"New Data Upload: {record.file_name}",
                "summary": f"A new file has been uploaded: {record.file_name}",
                "details": {
                    "file_name": record.file_name,
                    "upload_time": record.upload_time.isoformat(),
                    "file_size": record.file_size,
                    "row_count": record.row_count,
                    "column_count": record.column_count
                }
            }
            
            # Queue delivery
            delivery_id = self.queue_delivery(
                content=content,
                recipient_id=user_id,
                priority=DeliveryPriority.NORMAL,
                trigger=DeliveryTrigger.UPLOAD
            )
            
            delivery_ids.append(delivery_id)
        
        return delivery_ids
    
    def trigger_on_critical_insight(self, insight: Dict[str, Any], 
                                  recipient_ids: List[str]) -> List[str]:
        """
        Trigger deliveries based on a critical insight.
        
        Args:
            insight: Insight data
            recipient_ids: List of recipient user IDs
            
        Returns:
            List of queued delivery item IDs
        """
        delivery_ids = []
        
        for recipient_id in recipient_ids:
            # Get user preferences
            preferences = self.get_user_preferences(recipient_id)
            if not preferences or not preferences.critical_alerts_enabled:
                continue
            
            # Queue delivery
            delivery_id = self.queue_delivery(
                content={
                    "type": "insight",
                    **insight
                },
                recipient_id=recipient_id,
                priority=DeliveryPriority.HIGH,
                trigger=DeliveryTrigger.CRITICAL_INSIGHT
            )
            
            delivery_ids.append(delivery_id)
        
        return delivery_ids
    
    def schedule_digest(self, insights: List[Dict[str, Any]], 
                      recipient_id: str,
                      scheduled_time: Optional[datetime] = None) -> str:
        """
        Schedule a digest delivery.
        
        Args:
            insights: List of insights to include
            recipient_id: Recipient user ID
            scheduled_time: When to deliver
            
        Returns:
            Delivery item ID
        """
        # Get user preferences
        preferences = self.get_user_preferences(recipient_id)
        if not preferences or not preferences.digest_enabled:
            logger.warning(f"Digest not enabled for user {recipient_id}")
            return ""
        
        # If no scheduled time, use next delivery window
        if not scheduled_time:
            scheduled_time = preferences.get_next_delivery_window()
        
        # Create digest content
        content = {
            "type": "digest",
            "title": "Your Insight Digest",
            "summary": f"A summary of {len(insights)} insights",
            "insights": insights
        }
        
        # Queue delivery
        return self.queue_delivery(
            content=content,
            recipient_id=recipient_id,
            priority=DeliveryPriority.SCHEDULED,
            trigger=DeliveryTrigger.SCHEDULED,
            scheduled_time=scheduled_time
        )
    
    def shutdown(self) -> None:
        """Shutdown the delivery engine."""
        logger.info("Shutting down delivery engine...")
        self._stop_workers()
        logger.info("Delivery engine shutdown complete")
