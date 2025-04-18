"""
Session management utilities for Watchdog AI.
Provides functions to get a unique session ID and record user actions in session state.
"""

import uuid
import streamlit as st
from datetime import datetime
import os
import json
import logging
import redis
from sentry_sdk import capture_exception

import sentry_sdk
from src.utils.audit_log import log_audit_event


def get_session_id() -> str:
    """Return a unique session ID, persisting it in session state if not already present."""
    if 'session_id' not in st.session_state:
        st.session_state['session_id'] = uuid.uuid4().hex
    return st.session_state['session_id']


def record_action(session_id: str, action_type: str, metadata: dict = None) -> bool:
    """
    Record a user action in the session log with automatic TTL enforcement.
    
    Args:
        session_id: Unique session identifier
        action_type: Type of action being recorded
        metadata: Additional context about the action
        
    Returns:
        bool: True if recording succeeded, False otherwise
    """
    try:
        session_key = f"session:{session_id}"
        action_data = {
            "timestamp": datetime.now().isoformat(),
            "action": action_type,
            "metadata": metadata or {}
        }
        
        # Record action in Redis
        redis_client.rpush(session_key, json.dumps(action_data))
        
        # Enforce TTL
        ttl = int(os.getenv("SESSION_TTL_SECONDS", "31536000"))  # Default 365 days
        redis_client.expire(session_key, ttl)
        
        return True
    except Exception as e:
        logging.error(f"Failed to record session action: {str(e)}")
        capture_exception(e)
        return False


def get_all_session_ids() -> list:
    """Retrieve all session IDs from Redis set 'watchdog:sessions'. Fallback to session state if Redis is unavailable."""
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0)
        sessions = r.smembers("watchdog:sessions")
        return [s.decode('utf-8') for s in sessions] if sessions else []
    except Exception as e:
        # Fallback: try to return current session ID if present
        if 'session_id' in st.session_state:
            return [st.session_state['session_id']]
        return [] 