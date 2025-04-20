"""
Tests for the data lineage tracking system.
"""

import pytest
from datetime import datetime, timedelta
import json
from src.utils.data_lineage import (
    DataLineage,
    LineageStore,
    LineageEvent,
    MappingLineage
)

@pytest.fixture
def lineage_store():
    """Create LineageStore instance with in-memory storage."""
    return LineageStore(redis_url=None)

@pytest.fixture
def data_lineage():
    """Create DataLineage instance."""
    return DataLineage(redis_url=None)

def test_record_event(lineage_store):
    """Test recording a lineage event."""
    event = LineageEvent(
        event_id="test-event-1",
        event_type="file_ingestion",
        source_id="file-1",
        target_id="",
        timestamp=datetime.now().isoformat(),
        metadata={"file_size": 1000},
        vendor="test-vendor"
    )
    
    lineage_store.record_event(event)
    
    # Check in-memory store
    key = lineage_store._get_key(event.event_type, event.event_id)
    stored_event = lineage_store._memory_store.get(key)
    
    assert stored_event is not None
    assert stored_event["event_id"] == event.event_id
    assert stored_event["vendor"] == event.vendor
    assert stored_event["metadata"]["file_size"] == 1000

def test_record_mapping(lineage_store):
    """Test recording a mapping lineage event."""
    mapping = MappingLineage(
        mapping_id="test-mapping-1",
        source_column="source_col",
        target_column="target_col",
        confidence=0.9,
        vendor="test-vendor",
        timestamp=datetime.now().isoformat(),
        metadata={"mapping_type": "automatic"}
    )
    
    lineage_store.record_mapping(mapping)
    
    # Check in-memory store
    key = lineage_store._get_key("column_mapping", mapping.mapping_id)
    stored_mapping = lineage_store._memory_store.get(key)
    
    assert stored_mapping is not None
    assert stored_mapping["source_id"] == mapping.source_column
    assert stored_mapping["target_id"] == mapping.target_column
    assert stored_mapping["metadata"]["confidence"] == mapping.confidence

def test_get_column_history(lineage_store):
    """Test retrieving column history."""
    # Record multiple mappings
    mappings = [
        MappingLineage(
            mapping_id=f"mapping-{i}",
            source_column="original_col",
            target_column="mapped_col",
            confidence=0.8 + (i/10),
            vendor="test-vendor",
            timestamp=(datetime.now() - timedelta(days=i)).isoformat(),
            metadata={"iteration": i}
        )
        for i in range(3)
    ]
    
    for mapping in mappings:
        lineage_store.record_mapping(mapping)
    
    # Get history
    history = lineage_store.get_column_history("original_col")
    
    assert len(history) == 3
    assert history[0]["metadata"]["iteration"] == 0  # Most recent first
    assert history[-1]["metadata"]["iteration"] == 2  # Oldest last

def test_get_vendor_mappings(lineage_store):
    """Test retrieving vendor-specific mappings."""
    # Record mappings for different vendors
    vendors = ["vendor1", "vendor2"]
    for vendor in vendors:
        for i in range(2):
            mapping = MappingLineage(
                mapping_id=f"{vendor}-mapping-{i}",
                source_column=f"source_{i}",
                target_column=f"target_{i}",
                confidence=0.9,
                vendor=vendor,
                timestamp=datetime.now().isoformat(),
                metadata={}
            )
            lineage_store.record_mapping(mapping)
    
    # Get mappings for vendor1
    vendor1_mappings = lineage_store.get_vendor_mappings("vendor1")
    assert len(vendor1_mappings) == 2
    assert all(m["vendor"] == "vendor1" for m in vendor1_mappings)

def test_track_file_ingestion(data_lineage):
    """Test tracking file ingestion."""
    data_lineage.track_file_ingestion(
        file_id="test-file-1",
        vendor="test-vendor",
        metadata={"format": "csv", "rows": 1000}
    )
    
    # Get events from store
    events = [
        e for e in data_lineage.store._memory_store.values()
        if e["event_type"] == "file_ingestion"
    ]
    
    assert len(events) == 1
    assert events[0]["source_id"] == "test-file-1"
    assert events[0]["metadata"]["format"] == "csv"

def test_track_column_mapping(data_lineage):
    """Test tracking column mapping."""
    data_lineage.track_column_mapping(
        source_column="original_name",
        target_column="canonical_name",
        confidence=0.95,
        vendor="test-vendor",
        metadata={"mapping_type": "fuzzy"}
    )
    
    # Get mappings from store
    mappings = data_lineage.get_column_lineage("original_name")
    
    assert len(mappings) == 1
    assert mappings[0]["source_id"] == "original_name"
    assert mappings[0]["target_id"] == "canonical_name"
    assert mappings[0]["metadata"]["confidence"] == 0.95

def test_track_data_transformation(data_lineage):
    """Test tracking data transformation."""
    data_lineage.track_data_transformation(
        source_id="raw_data",
        target_id="processed_data",
        transform_type="normalization",
        metadata={"changes": ["removed_nulls", "standardized_dates"]}
    )
    
    # Get events from store
    events = [
        e for e in data_lineage.store._memory_store.values()
        if e["event_type"] == "data_transform"
    ]
    
    assert len(events) == 1
    assert events[0]["source_id"] == "raw_data"
    assert events[0]["target_id"] == "processed_data"
    assert "removed_nulls" in events[0]["metadata"]["changes"]

def test_error_handling(lineage_store):
    """Test error handling in lineage store."""
    # Simulate Redis failure
    lineage_store.redis = None
    
    # Should fall back to in-memory storage
    event = LineageEvent(
        event_id="test-error-1",
        event_type="error_test",
        source_id="source",
        target_id="target",
        timestamp=datetime.now().isoformat(),
        metadata={},
        vendor="test"
    )
    
    # Should not raise exception
    lineage_store.record_event(event)
    
    # Should still be able to retrieve
    key = lineage_store._get_key(event.event_type, event.event_id)
    assert key in lineage_store._memory_store