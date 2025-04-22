"""
Event emitter system for data pipeline integration.
"""

import logging
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
from enum import Enum
import queue
import threading
import time

logger = logging.getLogger(__name__)

class EventType(str, Enum):
    """Types of events that can be emitted."""
    DATA_NORMALIZED = "data_normalized"
    INSIGHT_GENERATED = "insight_generated"
    DELIVERY_COMPLETED = "delivery_completed"
    DELIVERY_FAILED = "delivery_failed"
    ALERT_TRIGGERED = "alert_triggered"

class Event:
    """Represents a system event."""
    
    def __init__(self, event_type: EventType, data: Dict[str, Any],
                 source: str, timestamp: Optional[datetime] = None):
        """
        Initialize an event.
        
        Args:
            event_type: Type of event
            data: Event data
            source: Component that generated the event
            timestamp: Optional event timestamp
        """
        self.event_type = event_type
        self.data = data
        self.source = source
        self.timestamp = timestamp or datetime.now()
        self.event_id = self._generate_event_id()
    
    def _generate_event_id(self) -> str:
        """Generate a unique event ID."""
        timestamp = self.timestamp.strftime("%Y%m%d%H%M%S%f")
        return f"EVT{timestamp}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "data": self.data,
            "source": self.source,
            "timestamp": self.timestamp.isoformat()
        }

class EventHandler:
    """Base class for event handlers."""
    
    def __init__(self, event_types: List[EventType]):
        """
        Initialize the handler.
        
        Args:
            event_types: List of event types to handle
        """
        self.event_types = event_types
    
    def handle_event(self, event: Event) -> None:
        """
        Handle an event.
        
        Args:
            event: Event to handle
        """
        raise NotImplementedError("Subclasses must implement handle_event")

class EventEmitter:
    """Event emitter for system events."""
    
    _instance = None
    
    def __new__(cls):
        """Create or return the singleton instance."""
        if cls._instance is None:
            cls._instance = super(EventEmitter, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the event emitter."""
        if self._initialized:
            return
            
        self._initialized = True
        self.handlers: Dict[EventType, List[EventHandler]] = {}
        self.event_queue = queue.Queue()
        self.stop_event = threading.Event()
        
        # Start worker thread
        self.worker_thread = threading.Thread(
            target=self._event_worker,
            daemon=True
        )
        self.worker_thread.start()
        
        logger.info("EventEmitter initialized")
    
    def register_handler(self, handler: EventHandler) -> None:
        """
        Register an event handler.
        
        Args:
            handler: Handler to register
        """
        for event_type in handler.event_types:
            if event_type not in self.handlers:
                self.handlers[event_type] = []
            self.handlers[event_type].append(handler)
        
        logger.info(f"Registered handler for events: {handler.event_types}")
    
    def emit(self, event: Event) -> None:
        """
        Emit an event.
        
        Args:
            event: Event to emit
        """
        self.event_queue.put(event)
        logger.debug(f"Queued event: {event.event_id} ({event.event_type})")
    
    def _event_worker(self) -> None:
        """Worker thread for processing events."""
        logger.info("Event worker started")
        
        while not self.stop_event.is_set():
            try:
                # Get event from queue with timeout
                try:
                    event = self.event_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Process event
                self._process_event(event)
                
                # Mark task as done
                self.event_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in event worker: {e}")
        
        logger.info("Event worker stopped")
    
    def _process_event(self, event: Event) -> None:
        """
        Process an event.
        
        Args:
            event: Event to process
        """
        logger.debug(f"Processing event: {event.event_id} ({event.event_type})")
        
        # Get handlers for this event type
        handlers = self.handlers.get(event.event_type, [])
        
        # Call each handler
        for handler in handlers:
            try:
                handler.handle_event(event)
            except Exception as e:
                logger.error(f"Error in handler for event {event.event_id}: {e}")
    
    def shutdown(self) -> None:
        """Shutdown the event emitter."""
        logger.info("Shutting down EventEmitter...")
        self.stop_event.set()
        self.worker_thread.join(timeout=5)
        logger.info("EventEmitter shutdown complete")