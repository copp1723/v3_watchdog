"""
AgentOps configuration and utilities for Watchdog AI.
Provides centralized management of AgentOps initialization and handlers.
"""

import os
import logging
from typing import Optional, Dict, Any
import agentops
import sentry_sdk
import hashlib

logger = logging.getLogger(__name__)

class AgentOpsConfig:
    def __init__(self):
        self.handler = None
        self.enabled = False
        api_key = os.environ.get("AGENTOPS_API_KEY")
        if api_key:
            try:
                agentops.init(api_key=api_key, tags=["watchdog-ai"])
                self.handler = agentops.get_handler()
                self.enabled = True
                sentry_sdk.capture_message("AgentOps initialized successfully", level="info")
                logger.info("AgentOps initialized successfully")
            except Exception as e:
                sentry_sdk.capture_exception(e)
                sentry_sdk.capture_message("AgentOps failed to init", level="warning")
                logger.error(f"AgentOps failed to initialize: {e}")
        else:
            sentry_sdk.capture_message("No AgentOps API key found", level="info")
            logger.info("No AgentOps API key found, monitoring disabled")

    def get_handler(self):
        return self.handler

    def track(self, operation_type, session_id=None, query=None):
        if not self.enabled:
            # Return a context manager that does nothing
            class NoOpContextManager:
                def __enter__(self):
                    return self
                def __exit__(self, exc_type, exc_val, exc_tb):
                    pass
            return NoOpContextManager()
        
        # Build tags for this operation
        tags = {
            "operation_type": operation_type,
            "service": "watchdog-ai"
        }
        
        if session_id:
            tags["session_id"] = session_id
        
        if query:
            # Hash the query for privacy while still allowing identification of similar queries
            tags["query_hash"] = hashlib.sha256(query.encode()).hexdigest()
            # Add a length tag for performance analysis
            tags["query_length"] = len(query)
        
        # Get a tracking context manager
        try:
            return agentops.track(tags=tags)
        except Exception as e:
            logger.error(f"Failed to create AgentOps tracking context: {e}")
            # Return no-op context manager on error
            class NoOpContextManager:
                def __enter__(self):
                    return self
                def __exit__(self, exc_type, exc_val, exc_tb):
                    pass
            return NoOpContextManager()

def init_agentops() -> bool:
    """
    Initialize AgentOps with configuration from environment.
    
    Returns:
        bool: True if initialization successful, False otherwise
    """
    api_key = os.getenv("AGENTOPS_API_KEY")
    if not api_key:
        logger.info("AGENTOPS_API_KEY not set, monitoring disabled")
        return False
        
    try:
        agentops.init(api_key, tags=["watchdog-ai"])
        logger.info("AgentOps initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize AgentOps: {e}")
        return False

def get_handler(
    session_id: Optional[str] = None,
    query_type: Optional[str] = None,
    additional_tags: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """
    Get an AgentOps handler with configured tags.
    
    Args:
        session_id: Optional session identifier
        query_type: Optional query type for categorization
        additional_tags: Optional additional tags to include
        
    Returns:
        AgentOps handler if available, None otherwise
    """
    if not os.getenv("AGENTOPS_API_KEY"):
        return None
        
    try:
        tags = {"service": "watchdog-ai"}
        
        if session_id:
            tags["session_id"] = session_id
        if query_type:
            tags["query_type"] = query_type
        if additional_tags:
            tags.update(additional_tags)
            
        return {"tags": tags}  # Simple dict for now
    except Exception as e:
        logger.error(f"Failed to create AgentOps handler: {e}")
        return None