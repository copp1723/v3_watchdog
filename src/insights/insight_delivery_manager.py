"""
Insight delivery management system for automated distribution.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
import json
from dataclasses import dataclass
import threading
import queue
import time
from enum import Enum

from .insight_formatter import InsightFormatter
from .models import InsightResponse
from ..scheduler.notification_service import NotificationService, EmailMessage

logger = logging.getLogger(__name__)

class DeliveryStatus(str, Enum):
    """Delivery status codes."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRYING = "retrying"

@dataclass
class DeliveryAttempt:
    """Records a delivery attempt."""
    timestamp: datetime
    status: DeliveryStatus
    error: Optional[str] = None
    delivery_id: Optional[str] = None

class DeliveryRecord:
    """Tracks the delivery of an insight package."""
    
    def __init__(self, recipient: str, insights: List[Dict[str, Any]], 
                 delivery_type: str):
        """
        Initialize a delivery record.
        
        Args:
            recipient: Email address of recipient
            insights: List of insights to deliver
            delivery_type: Type of delivery (daily, weekly, alert)
        """
        self.recipient = recipient
        self.insights = insights
        self.delivery_type = delivery_type
        self.attempts: List[DeliveryAttempt] = []
        self.created_at = datetime.now()
        self.completed_at: Optional[datetime] = None
        self.delivery_id = self._generate_delivery_id()
    
    def add_attempt(self, status: DeliveryStatus, error: Optional[str] = None,
                   delivery_id: Optional[str] = None) -> None:
        """Add a delivery attempt."""
        self.attempts.append(DeliveryAttempt(
            timestamp=datetime.now(),
            status=status,
            error=error,
            delivery_id=delivery_id
        ))
        
        if status in [DeliveryStatus.DELIVERED, DeliveryStatus.FAILED]:
            self.completed_at = datetime.now()
    
    def _generate_delivery_id(self) -> str:
        """Generate a unique delivery ID."""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"DEL{timestamp}"
    
    @property
    def status(self) -> DeliveryStatus:
        """Get the current delivery status."""
        return self.attempts[-1].status if self.attempts else DeliveryStatus.PENDING
    
    @property
    def attempt_count(self) -> int:
        """Get the number of delivery attempts."""
        return len(self.attempts)

class InsightDeliveryManager:
    """Manages the delivery of insights via various channels."""
    
    def __init__(self, notification_service: Optional[NotificationService] = None,
                 formatter: Optional[InsightFormatter] = None):
        """
        Initialize the delivery manager.
        
        Args:
            notification_service: Optional notification service
            formatter: Optional insight formatter
        """
        self.notification_service = notification_service or NotificationService()
        self.formatter = formatter or InsightFormatter()
        
        # Initialize delivery queue and worker
        self.delivery_queue = queue.Queue()
        self.delivery_records: Dict[str, DeliveryRecord] = {}
        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(
            target=self._delivery_worker,
            daemon=True
        )
        self.worker_thread.start()
        
        # Delivery settings
        self.max_retries = 3
        self.retry_delays = [5, 30, 120]  # Seconds between retries
    
    def schedule_daily_summary(self, recipient: str, 
                             insights: List[Dict[str, Any]]) -> str:
        """
        Schedule a daily summary delivery.
        
        Args:
            recipient: Email address of recipient
            insights: List of insights to include
            
        Returns:
            Delivery record ID
        """
        record = DeliveryRecord(recipient, insights, "daily")
        self.delivery_records[record.delivery_id] = record
        
        # Queue for delivery
        self.delivery_queue.put(record)
        
        return record.delivery_id
    
    def schedule_weekly_executive(self, recipient: str,
                                insights: List[Dict[str, Any]],
                                kpis: List[Dict[str, Any]],
                                summary: str,
                                date_range: str) -> str:
        """
        Schedule a weekly executive summary delivery.
        
        Args:
            recipient: Email address of recipient
            insights: List of insights to include
            kpis: List of KPIs to include
            summary: Executive summary text
            date_range: Date range string
            
        Returns:
            Delivery record ID
        """
        record = DeliveryRecord(recipient, {
            "insights": insights,
            "kpis": kpis,
            "summary": summary,
            "date_range": date_range
        }, "weekly")
        self.delivery_records[record.delivery_id] = record
        
        # Queue for delivery
        self.delivery_queue.put(record)
        
        return record.delivery_id
    
    def send_alert(self, recipient: str, alert: Dict[str, Any]) -> str:
        """
        Send an immediate alert.
        
        Args:
            recipient: Email address of recipient
            alert: Alert data
            
        Returns:
            Delivery record ID
        """
        record = DeliveryRecord(recipient, alert, "alert")
        self.delivery_records[record.delivery_id] = record
        
        # Queue for immediate delivery
        self.delivery_queue.put(record)
        
        return record.delivery_id
    
    def get_delivery_status(self, delivery_id: str) -> Dict[str, Any]:
        """
        Get the status of a delivery.
        
        Args:
            delivery_id: Delivery record ID
            
        Returns:
            Dictionary with delivery status information
        """
        if delivery_id not in self.delivery_records:
            return {
                "status": "unknown",
                "error": "Delivery ID not found"
            }
        
        record = self.delivery_records[delivery_id]
        
        return {
            "delivery_id": delivery_id,
            "status": record.status,
            "recipient": record.recipient,
            "type": record.delivery_type,
            "created_at": record.created_at.isoformat(),
            "completed_at": record.completed_at.isoformat() if record.completed_at else None,
            "attempt_count": record.attempt_count,
            "attempts": [
                {
                    "timestamp": attempt.timestamp.isoformat(),
                    "status": attempt.status,
                    "error": attempt.error,
                    "delivery_id": attempt.delivery_id
                }
                for attempt in record.attempts
            ]
        }
    
    def _delivery_worker(self) -> None:
        """Worker thread for processing the delivery queue."""
        while not self.stop_event.is_set():
            try:
                # Get next delivery with timeout
                try:
                    record = self.delivery_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Process the delivery
                self._process_delivery(record)
                
                # Mark task as done
                self.delivery_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in delivery worker: {e}")
    
    def _process_delivery(self, record: DeliveryRecord) -> None:
        """
        Process a delivery record.
        
        Args:
            record: DeliveryRecord to process
        """
        logger.info(f"Processing delivery {record.delivery_id} ({record.delivery_type})")
        
        # Update status
        record.add_attempt(DeliveryStatus.IN_PROGRESS)
        
        try:
            # Format content based on delivery type
            if record.delivery_type == "daily":
                content = self.formatter.format_daily_summary(record.insights)
                subject = "Daily Insights Summary"
            elif record.delivery_type == "weekly":
                content = self.formatter.format_weekly_executive(
                    record.insights["insights"],
                    record.insights["kpis"],
                    record.insights["summary"],
                    record.insights["date_range"]
                )
                subject = "Weekly Executive Summary"
            else:  # alert
                content = self.formatter.format_alert(record.insights)
                subject = f"ALERT: {record.insights.get('title', 'Alert Notification')}"
            
            # Create email message
            email = EmailMessage(
                recipients=[record.recipient],
                subject=subject,
                html_content=content
            )
            
            # Send with retries
            for attempt in range(self.max_retries):
                try:
                    message_id = self.notification_service.queue.add_email(email)
                    
                    # Update record with success
                    record.add_attempt(
                        DeliveryStatus.DELIVERED,
                        delivery_id=message_id
                    )
                    logger.info(f"Delivery {record.delivery_id} sent successfully")
                    return
                    
                except Exception as e:
                    error = str(e)
                    logger.warning(f"Delivery attempt {attempt + 1} failed: {error}")
                    
                    # Update status and retry if not last attempt
                    if attempt < self.max_retries - 1:
                        record.add_attempt(
                            DeliveryStatus.RETRYING,
                            error=error
                        )
                        delay = self.retry_delays[min(attempt, len(self.retry_delays)-1)]
                        time.sleep(delay)
                    else:
                        # Final failure
                        record.add_attempt(
                            DeliveryStatus.FAILED,
                            error=f"Failed after {self.max_retries} attempts: {error}"
                        )
                        logger.error(f"Delivery {record.delivery_id} failed permanently")
            
        except Exception as e:
            error = str(e)
            logger.error(f"Error processing delivery {record.delivery_id}: {error}")
            record.add_attempt(DeliveryStatus.FAILED, error=error)
    
    def shutdown(self) -> None:
        """Shutdown the delivery manager."""
        logger.info("Shutting down delivery manager...")
        self.stop_event.set()
        self.worker_thread.join(timeout=5)
        logger.info("Delivery manager shutdown complete")