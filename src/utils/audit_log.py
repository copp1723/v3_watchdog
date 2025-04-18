"""
Audit logging utilities for Watchdog AI.
Provides a function to log critical events such as ingestion and normalization.
Implements a Redis-based audit log with 90-day TTL by default.
"""

import logging
import os
from datetime import datetime
import redis
import json

import sentry_sdk

# Configure audit logger
audit_logger = logging.getLogger("audit")
if not audit_logger.handlers:
    handler = logging.FileHandler("audit.log")
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    audit_logger.addHandler(handler)
    audit_logger.setLevel(logging.INFO)

# Default TTL for audit logs in Redis: 90 days (in seconds)
# 90 days × 24 hours × 60 minutes × 60 seconds = 7,776,000 seconds
DEFAULT_AUDIT_LOG_TTL = 7776000

# Get TTL from environment variable or use default
AUDIT_LOG_TTL_SECONDS = int(os.environ.get('WATCHDOG_AUDIT_LOG_TTL_SECONDS', DEFAULT_AUDIT_LOG_TTL))

# Redis key for audit logs
AUDIT_LOG_KEY = "watchdog:audit_logs"

# Initialize Redis client from environment variables
REDIS_HOST = os.environ.get('WATCHDOG_REDIS_HOST', 'localhost')
REDIS_PORT = int(os.environ.get('WATCHDOG_REDIS_PORT', 6379))
REDIS_DB = int(os.environ.get('WATCHDOG_REDIS_DB', 0))
REDIS_PASSWORD = os.environ.get('WATCHDOG_REDIS_PASSWORD', None)

try:
    redis_client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        password=REDIS_PASSWORD,
        socket_timeout=5,
        decode_responses=False  # Keep binary for JSON serialization
    )
    # Test connection
    redis_client.ping()
    
    # Check if the audit log key has a TTL set
    ttl = redis_client.ttl(AUDIT_LOG_KEY)
    if ttl < 0:  # -1 means no TTL, -2 means key doesn't exist
        # Set the TTL on the existing key if it exists
        if ttl == -1:
            redis_client.expire(AUDIT_LOG_KEY, AUDIT_LOG_TTL_SECONDS)
            audit_logger.info(f"Set TTL on existing audit log key to {AUDIT_LOG_TTL_SECONDS} seconds")
    
    audit_logger.info(f"Redis audit logging initialized with TTL={AUDIT_LOG_TTL_SECONDS} seconds")
    
except Exception as e:
    redis_client = None
    audit_logger.error(f"Redis client initialization failed: {e}")


def log_audit_event(event_name: str, user_id: str, session_id: str, details: dict = None) -> None:
    """
    Persist a timestamped audit log entry.
    Logs the event to a local file and pushes the log entry into Redis with TTL.
    Also sets Sentry tags for the event.
    
    Args:
        event_name: The name of the event being logged
        user_id: The ID of the user who performed the action
        session_id: The session ID in which the action was performed
        details: Optional dictionary with additional event details
    """
    log_entry = {
        "event": event_name,
        "user_id": user_id,
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(),
        "ip_address": details.get("ip_address", "unknown"),
        "resource_type": details.get("resource_type", "unknown"),
        "resource_id": details.get("resource_id", "unknown"),
        "status": details.get("status", "success"),
        "details": details or {}
    }
    
    # Log to local file
    audit_logger.info(json.dumps(log_entry))
    
    # Mirror Sentry tags
    sentry_sdk.set_tag("audit_event", event_name)
    sentry_sdk.set_tag("audit_status", log_entry["status"])
    
    # Push the log entry to Redis
    if redis_client:
        try:
            # Add log entry to the Redis list
            redis_client.rpush(AUDIT_LOG_KEY, json.dumps(log_entry))
            
            # Apply TTL to the key if not already set
            if redis_client.ttl(AUDIT_LOG_KEY) == -1:  # -1 means no TTL is set
                redis_client.expire(AUDIT_LOG_KEY, AUDIT_LOG_TTL_SECONDS)
                
            # Log success
            audit_logger.debug(f"Audit log for event '{event_name}' stored in Redis with TTL={AUDIT_LOG_TTL_SECONDS}s")
        except Exception as e:
            audit_logger.error(f"Failed to push audit log to Redis: {e}")
            sentry_sdk.capture_exception(e)
            
            
def get_audit_log_ttl() -> int:
    """
    Get the current TTL setting for audit logs.
    
    Returns:
        TTL in seconds
    """
    return AUDIT_LOG_TTL_SECONDS