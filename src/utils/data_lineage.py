"""
Data lineage tracking system for Watchdog AI.
Tracks the journey of data from source files through schema mapping to ingestion.
"""

import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
import redis
from dataclasses import dataclass
import uuid

logger = logging.getLogger(__name__)

@dataclass
class LineageEvent:
    """Represents a single data lineage event."""
    event_id: str
    event_type: str
    source_id: str
    target_id: str
    timestamp: str
    metadata: Dict[str, Any]
    vendor: Optional[str] = None
    user_id: Optional[str] = None

@dataclass
class MappingLineage:
    """Tracks column mapping lineage."""
    mapping_id: str
    source_column: str
    target_column: str
    confidence: float
    vendor: Optional[str]
    timestamp: str
    metadata: Dict[str, Any]

class LineageStore:
    """Persistent storage for data lineage using Redis."""
    
    def __init__(self, redis_url: Optional[str] = None):
        """Initialize the lineage store."""
        try:
            self.redis = redis.from_url(redis_url) if redis_url else redis.Redis()
            self.prefix = "watchdog:lineage:"
            self.ttl = 60 * 60 * 24 * 90  # 90 days retention
        except redis.ConnectionError:
            logger.warning("Redis connection failed, falling back to in-memory storage")
            self.redis = None
            self._memory_store = {}
    
    def _get_key(self, event_type: str, id: str) -> str:
        """Get Redis key for a lineage event."""
        return f"{self.prefix}{event_type}:{id}"
    
    def record_event(self, event: LineageEvent) -> None:
        """
        Record a lineage event.
        
        Args:
            event: LineageEvent to record
        """
        key = self._get_key(event.event_type, event.event_id)
        data = {
            "event_id": event.event_id,
            "event_type": event.event_type,
            "source_id": event.source_id,
            "target_id": event.target_id,
            "timestamp": event.timestamp,
            "metadata": event.metadata,
            "vendor": event.vendor,
            "user_id": event.user_id
        }
        
        try:
            if self.redis:
                self.redis.set(key, json.dumps(data), ex=self.ttl)
            else:
                self._memory_store[key] = data
        except Exception as e:
            logger.error(f"Error recording lineage event: {e}")
    
    def record_mapping(self, mapping: MappingLineage) -> None:
        """
        Record a column mapping lineage event.
        
        Args:
            mapping: MappingLineage to record
        """
        event = LineageEvent(
            event_id=mapping.mapping_id,
            event_type="column_mapping",
            source_id=mapping.source_column,
            target_id=mapping.target_column,
            timestamp=mapping.timestamp,
            metadata={
                "confidence": mapping.confidence,
                **mapping.metadata
            },
            vendor=mapping.vendor
        )
        self.record_event(event)
    
    def get_column_history(self, column_name: str) -> List[Dict[str, Any]]:
        """
        Get historical lineage for a column.
        
        Args:
            column_name: Name of the column
            
        Returns:
            List of lineage events
        """
        events = []
        pattern = f"{self.prefix}column_mapping:*"
        
        try:
            if self.redis:
                # Get all mapping keys
                keys = self.redis.keys(pattern)
                for key in keys:
                    data = self.redis.get(key)
                    if data:
                        event = json.loads(data)
                        if (event["source_id"] == column_name or 
                            event["target_id"] == column_name):
                            events.append(event)
            else:
                # Search in-memory store
                for key, data in self._memory_store.items():
                    if key.startswith(f"{self.prefix}column_mapping:"):
                        if (data["source_id"] == column_name or 
                            data["target_id"] == column_name):
                            events.append(data)
        except Exception as e:
            logger.error(f"Error getting column history: {e}")
        
        return sorted(events, key=lambda x: x["timestamp"], reverse=True)
    
    def get_vendor_mappings(self, vendor: str) -> List[Dict[str, Any]]:
        """
        Get all mappings for a specific vendor.
        
        Args:
            vendor: Vendor name
            
        Returns:
            List of mapping events
        """
        events = []
        pattern = f"{self.prefix}column_mapping:*"
        
        try:
            if self.redis:
                keys = self.redis.keys(pattern)
                for key in keys:
                    data = self.redis.get(key)
                    if data:
                        event = json.loads(data)
                        if event.get("vendor") == vendor:
                            events.append(event)
            else:
                for key, data in self._memory_store.items():
                    if (key.startswith(f"{self.prefix}column_mapping:") and
                        data.get("vendor") == vendor):
                        events.append(data)
        except Exception as e:
            logger.error(f"Error getting vendor mappings: {e}")
        
        return sorted(events, key=lambda x: x["timestamp"], reverse=True)

class DataLineage:
    """
    Manages data lineage tracking throughout the ingestion pipeline.
    """
    
    def __init__(self, redis_url: Optional[str] = None):
        """Initialize the data lineage system."""
        self.store = LineageStore(redis_url)
    
    def track_file_ingestion(self, file_id: str, vendor: str,
                           metadata: Dict[str, Any]) -> None:
        """
        Track ingestion of a source file.
        
        Args:
            file_id: Unique identifier for the file
            vendor: Vendor name
            metadata: Additional metadata about the file
        """
        event = LineageEvent(
            event_id=str(uuid.uuid4()),
            event_type="file_ingestion",
            source_id=file_id,
            target_id="",  # Empty for initial ingestion
            timestamp=datetime.now().isoformat(),
            metadata=metadata,
            vendor=vendor
        )
        self.store.record_event(event)
    
    def track_column_mapping(self, source_column: str, target_column: str,
                           confidence: float, vendor: Optional[str] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Track a column mapping operation.
        
        Args:
            source_column: Original column name
            target_column: Mapped column name
            confidence: Mapping confidence score
            vendor: Optional vendor name
            metadata: Optional additional metadata
        """
        mapping = MappingLineage(
            mapping_id=str(uuid.uuid4()),
            source_column=source_column,
            target_column=target_column,
            confidence=confidence,
            vendor=vendor,
            timestamp=datetime.now().isoformat(),
            metadata=metadata or {}
        )
        self.store.record_mapping(mapping)
    
    def track_data_transformation(self, source_id: str, target_id: str,
                                transform_type: str,
                                metadata: Dict[str, Any]) -> None:
        """
        Track a data transformation operation.
        
        Args:
            source_id: Source data identifier
            target_id: Target data identifier
            transform_type: Type of transformation
            metadata: Transformation metadata
        """
        event = LineageEvent(
            event_id=str(uuid.uuid4()),
            event_type="data_transform",
            source_id=source_id,
            target_id=target_id,
            timestamp=datetime.now().isoformat(),
            metadata={
                "transform_type": transform_type,
                **metadata
            }
        )
        self.store.record_event(event)
    
    def get_column_lineage(self, column_name: str) -> List[Dict[str, Any]]:
        """
        Get lineage history for a column.
        
        Args:
            column_name: Name of the column
            
        Returns:
            List of lineage events
        """
        return self.store.get_column_history(column_name)
    
    def get_vendor_column_mappings(self, vendor: str) -> List[Dict[str, Any]]:
        """
        Get all column mappings for a vendor.
        
        Args:
            vendor: Vendor name
            
        Returns:
            List of mapping events
        """
        return self.store.get_vendor_mappings(vendor)