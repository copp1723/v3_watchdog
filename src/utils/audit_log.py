"""
Audit logging module for Watchdog AI.
"""

import os
import redis
import logging
import json
import sentry_sdk
from datetime import datetime
from typing import Dict, Any

# Configure Redis
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
REDIS_DB = 0
REDIS_PASSWORD = None
AUDIT_LOG_KEY = "watchdog:audit_logs"

# Configure logger
logger = logging.getLogger(__name__)

# Initialize Redis client
redis_client = None
try:
    redis_client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        password=REDIS_PASSWORD,
        decode_responses=True
    )
    redis_client.ping()
except Exception as e:
    logger.error(f"Failed to connect to Redis: {e}")

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
        "ip_address": details.get("ip_address", "unknown") if details else "unknown",
        "resource_type": details.get("resource_type", "unknown") if details else "unknown",
        "resource_id": details.get("resource_id", "unknown") if details else "unknown",
        "status": details.get("status", "success") if details else "success",
        "details": details or {}
    }
    
    # Log to local file
    logger.info(json.dumps(log_entry))
    
    # Mirror Sentry tags
    sentry_sdk.set_tag("audit_event", event_name)
    sentry_sdk.set_tag("audit_status", log_entry["status"])
    
    # Push the log entry to Redis
    if redis_client:
        try:
            # Add log entry to the Redis list
            redis_client.rpush(AUDIT_LOG_KEY, json.dumps(log_entry))
            
            # Apply TTL to the key if not already set
            ttl = int(os.getenv("AUDIT_LOG_TTL_SECONDS", "7776000"))  # Default 90 days
            redis_client.expire(AUDIT_LOG_KEY, ttl)
            
            # Log success
            logger.debug(f"Audit log for event '{event_name}' stored in Redis with TTL={ttl}s")
        except Exception as e:
            logger.error(f"Failed to push audit log to Redis: {e}")
            sentry_sdk.capture_exception(e)