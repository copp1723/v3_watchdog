"""
Session management module for Watchdog AI.
"""

import redis
import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional

# Configure Redis
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
REDIS_DB = 0
REDIS_PASSWORD = None
SESSION_KEY_PREFIX = "watchdog:session:"

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

def get_session_id() -> str:
    """Get the current session ID."""
    return datetime.utcnow().strftime("%Y%m%d-%H%M%S")

def record_action(session_id: str, action: str, details: Optional[Dict[str, Any]] = None) -> None:
    """Record a user action in the session."""
    if not redis_client:
        logger.error("Redis client not available for session recording")
        return

    try:
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": action,
            "details": details or {}
        }
        
        key = f"{SESSION_KEY_PREFIX}{session_id}"
        redis_client.rpush(key, json.dumps(event))
    except Exception as e:
        logger.error(f"Failed to record session action: {e}")