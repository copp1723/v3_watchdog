"""
Audit logging module for Watchdog AI.
"""

import redis
import logging
import json
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

def log_audit_event(event_type: str, details: Dict[str, Any]) -> None:
    """Log an audit event to Redis."""
    if not redis_client:
        logger.error("Redis client not available for audit logging")
        return

    try:
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "details": details
        }
        
        redis_client.rpush(AUDIT_LOG_KEY, json.dumps(event))
    except Exception as e:
        logger.error(f"Failed to log audit event: {e}")