"""
Session management utilities for Watchdog AI.
Provides functions to get a unique session ID and record user actions in session state.
"""

import uuid
import streamlit as st
from datetime import datetime

import sentry_sdk
from src.utils.audit_log import log_audit_event


def get_session_id() -> str:
    """Return a unique session ID, persisting it in session state if not already present."""
    if 'session_id' not in st.session_state:
        st.session_state['session_id'] = uuid.uuid4().hex
    return st.session_state['session_id']


def record_action(event_name: str, metadata: dict = None) -> dict:
    """Record an action with event name, metadata, timestamp, and session ID in session state.
    Sets Sentry tags and calls log_audit_event to persist the action as an audit log entry.
    Also, pushes the session ID into Redis set 'watchdog:sessions' for session tracking.
    """
    if 'action_history' not in st.session_state:
        st.session_state['action_history'] = []

    session_id = get_session_id()
    
    # Set Sentry tags for pipeline action
    sentry_sdk.set_tag("pipeline_session", session_id)
    sentry_sdk.set_tag("pipeline_action", event_name)
    
    record = {
        'event': event_name,
        'metadata': metadata or {},
        'timestamp': datetime.now().isoformat(),
        'session_id': session_id
    }
    st.session_state['action_history'].append(record)
    
    # Call audit logging (using session_id as user_id, since no separate user_id is provided)
    try:
        log_audit_event(event_name, session_id, session_id, details=metadata or {})
    except Exception as e:
        # Log the exception silently; we do not want to break the pipeline
        pass
    
    # Push the session ID into Redis set 'watchdog:sessions' for session tracking
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0)
        r.sadd("watchdog:sessions", session_id)
    except Exception as e:
        # Log the error if needed, but do not break the flow
        pass
    
    return record 


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