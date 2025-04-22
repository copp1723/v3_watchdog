"""
Event handlers for insight delivery.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from .event_emitter import EventType, Event, EventHandler
from .insight_delivery_manager import InsightDeliveryManager
from ..utils.config import get_user_preferences

logger = logging.getLogger(__name__)

class DataNormalizedHandler(EventHandler):
    """Handles data normalization events to trigger daily insights."""
    
    def __init__(self, delivery_manager: Optional[InsightDeliveryManager] = None):
        """
        Initialize the handler.
        
        Args:
            delivery_manager: Optional delivery manager instance
        """
        super().__init__([EventType.DATA_NORMALIZED])
        self.delivery_manager = delivery_manager or InsightDeliveryManager()
    
    def handle_event(self, event: Event) -> None:
        """
        Handle a data normalized event.
        
        Args:
            event: Event to handle
        """
        try:
            # Get dealer info from event
            dealer_id = event.data.get('dealer_id')
            if not dealer_id:
                logger.warning("No dealer_id in event data")
                return
            
            # Get user preferences
            preferences = get_user_preferences(dealer_id)
            if not preferences:
                logger.info(f"No delivery preferences found for dealer {dealer_id}")
                return
            
            # Check if daily summary is enabled
            if not preferences.get('types', {}).get('daily_summary'):
                logger.info(f"Daily summary disabled for dealer {dealer_id}")
                return
            
            # Get recipient
            recipient = preferences.get('email')
            if not recipient:
                logger.warning(f"No email configured for dealer {dealer_id}")
                return
            
            # Queue daily summary
            self.delivery_manager.schedule_daily_summary(
                recipient=recipient,
                insights=event.data.get('insights', [])
            )
            
            logger.info(f"Scheduled daily summary for dealer {dealer_id}")
            
        except Exception as e:
            logger.error(f"Error handling data normalized event: {e}")

class AlertHandler(EventHandler):
    """Handles alert events for immediate delivery."""
    
    def __init__(self, delivery_manager: Optional[InsightDeliveryManager] = None):
        """
        Initialize the handler.
        
        Args:
            delivery_manager: Optional delivery manager instance
        """
        super().__init__([EventType.ALERT_TRIGGERED])
        self.delivery_manager = delivery_manager or InsightDeliveryManager()
    
    def handle_event(self, event: Event) -> None:
        """
        Handle an alert event.
        
        Args:
            event: Event to handle
        """
        try:
            # Get dealer info from event
            dealer_id = event.data.get('dealer_id')
            if not dealer_id:
                logger.warning("No dealer_id in event data")
                return
            
            # Get user preferences
            preferences = get_user_preferences(dealer_id)
            if not preferences:
                logger.info(f"No delivery preferences found for dealer {dealer_id}")
                return
            
            # Check if alerts are enabled
            if not preferences.get('types', {}).get('critical_alerts'):
                logger.info(f"Alerts disabled for dealer {dealer_id}")
                return
            
            # Get recipients based on alert severity
            recipients = self._get_alert_recipients(
                dealer_id,
                preferences,
                event.data.get('severity', 'medium')
            )
            
            if not recipients:
                logger.warning(f"No recipients configured for dealer {dealer_id}")
                return
            
            # Send alert to each recipient
            for recipient in recipients:
                self.delivery_manager.send_alert(
                    recipient=recipient,
                    alert=event.data.get('alert', {})
                )
            
            logger.info(f"Sent alert to {len(recipients)} recipients for dealer {dealer_id}")
            
        except Exception as e:
            logger.error(f"Error handling alert event: {e}")
    
    def _get_alert_recipients(self, dealer_id: str, preferences: Dict[str, Any],
                            severity: str) -> List[str]:
        """
        Get alert recipients based on severity.
        
        Args:
            dealer_id: Dealer ID
            preferences: User preferences
            severity: Alert severity
            
        Returns:
            List of recipient email addresses
        """
        recipients = []
        
        # Add email recipient if configured
        if preferences.get('channels', {}).get('email'):
            email = preferences.get('email')
            if email:
                recipients.append(email)
        
        # Add SMS recipient for high severity
        if severity == 'high' and preferences.get('channels', {}).get('sms'):
            phone = preferences.get('phone')
            if phone:
                # In production, this would be an SMS gateway email
                recipients.append(f"{phone}@sms.gateway.com")
        
        return recipients

class DeliveryStatusHandler(EventHandler):
    """Handles delivery status events for monitoring."""
    
    def __init__(self):
        """Initialize the handler."""
        super().__init__([
            EventType.DELIVERY_COMPLETED,
            EventType.DELIVERY_FAILED
        ])
        self.status_counts = {
            'completed': 0,
            'failed': 0
        }
        self.recent_failures: List[Dict[str, Any]] = []
    
    def handle_event(self, event: Event) -> None:
        """
        Handle a delivery status event.
        
        Args:
            event: Event to handle
        """
        try:
            if event.event_type == EventType.DELIVERY_COMPLETED:
                self.status_counts['completed'] += 1
                
            elif event.event_type == EventType.DELIVERY_FAILED:
                self.status_counts['failed'] += 1
                
                # Track failure details
                self.recent_failures.append({
                    'timestamp': event.timestamp,
                    'delivery_id': event.data.get('delivery_id'),
                    'error': event.data.get('error'),
                    'recipient': event.data.get('recipient')
                })
                
                # Keep only recent failures
                self.recent_failures = self.recent_failures[-100:]
            
            logger.debug(f"Updated delivery status counts: {self.status_counts}")
            
        except Exception as e:
            logger.error(f"Error handling delivery status event: {e}")
    
    def get_status_metrics(self) -> Dict[str, Any]:
        """Get current status metrics."""
        return {
            'counts': self.status_counts.copy(),
            'recent_failures': self.recent_failures[-10:],  # Last 10 failures
            'success_rate': self._calculate_success_rate()
        }
    
    def _calculate_success_rate(self) -> float:
        """Calculate the delivery success rate."""
        total = sum(self.status_counts.values())
        if total == 0:
            return 100.0
        return (self.status_counts['completed'] / total) * 100