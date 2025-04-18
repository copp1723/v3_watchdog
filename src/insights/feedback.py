"""
Feedback system for Watchdog AI insights.

Captures and persists executive feedback on generated insights,
with Redis storage and Sentry monitoring.
"""

import redis
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
import sentry_sdk
from ..utils.audit_log import log_audit_event

# Configure logger
logger = logging.getLogger(__name__)

class FeedbackManager:
    """
    Manages insight feedback collection and persistence.
    
    Stores feedback in Redis and emits events to Sentry for monitoring.
    """
    
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379):
        """
        Initialize the feedback manager.
        
        Args:
            redis_host: Redis server hostname
            redis_port: Redis server port
        """
        self.redis_key = "watchdog:insight_feedback"
        
        # Initialize Redis client
        try:
            self.redis = redis.Redis(host=redis_host, port=redis_port, db=0)
            self.redis.ping()  # Test connection
            self.storage_available = True
        except Exception as e:
            logger.warning(f"Redis connection failed, falling back to in-memory storage: {str(e)}")
            self.storage_available = False
            self._memory_storage = []
    
    def record_feedback(
        self,
        insight_id: str,
        feedback_type: str,
        user_id: str,
        session_id: str,
        comment: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Record user feedback for an insight.
        
        Args:
            insight_id: Unique identifier for the insight
            feedback_type: Type of feedback (e.g., 'helpful', 'not_helpful')
            user_id: ID of the user providing feedback
            session_id: Current session ID
            comment: Optional feedback comment
            metadata: Optional additional metadata
            
        Returns:
            True if feedback was recorded successfully
        """
        try:
            # Create feedback entry
            feedback = {
                "insight_id": insight_id,
                "feedback_type": feedback_type,
                "user_id": user_id,
                "session_id": session_id,
                "comment": comment,
                "metadata": metadata or {},
                "timestamp": datetime.now().isoformat()
            }
            
            # Add Sentry breadcrumb
            sentry_sdk.add_breadcrumb(
                category="feedback",
                message=f"Feedback received for insight {insight_id}",
                data={
                    "feedback_type": feedback_type,
                    "session_id": session_id
                },
                level="info"
            )
            
            # Set Sentry tags
            sentry_sdk.set_tag("feedback_type", feedback_type)
            sentry_sdk.set_tag("insight_id", insight_id)
            
            # Store feedback
            if self.storage_available:
                self.redis.rpush(self.redis_key, json.dumps(feedback))
            else:
                self._memory_storage.append(feedback)
            
            # Log audit event
            log_audit_event(
                event_name="insight_feedback_recorded",
                user_id=user_id,
                session_id=session_id,
                details={
                    "insight_id": insight_id,
                    "feedback_type": feedback_type
                }
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error recording feedback: {str(e)}")
            sentry_sdk.capture_exception(e)
            return False
    
    def get_feedback(
        self,
        insight_id: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Retrieve recorded feedback with optional filtering.
        
        Args:
            insight_id: Optional insight ID to filter by
            session_id: Optional session ID to filter by
            limit: Maximum number of feedback entries to return
            
        Returns:
            List of feedback entries
        """
        try:
            # Get raw feedback entries
            if self.storage_available:
                entries = [
                    json.loads(entry)
                    for entry in self.redis.lrange(self.redis_key, 0, limit - 1)
                ]
            else:
                entries = self._memory_storage[:limit]
            
            # Apply filters if provided
            if insight_id:
                entries = [e for e in entries if e["insight_id"] == insight_id]
            if session_id:
                entries = [e for e in entries if e["session_id"] == session_id]
            
            return entries
            
        except Exception as e:
            logger.error(f"Error retrieving feedback: {str(e)}")
            sentry_sdk.capture_exception(e)
            return []
    
    def get_feedback_stats(self, insight_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get aggregated feedback statistics.
        
        Args:
            insight_id: Optional insight ID to filter stats by
            
        Returns:
            Dictionary with feedback statistics
        """
        try:
            # Get relevant feedback entries
            entries = self.get_feedback(insight_id=insight_id)
            
            # Count feedback types
            feedback_counts = {}
            for entry in entries:
                feedback_type = entry["feedback_type"]
                feedback_counts[feedback_type] = feedback_counts.get(feedback_type, 0) + 1
            
            # Calculate percentages
            total = len(entries)
            feedback_percentages = {
                k: (v / total * 100) if total > 0 else 0
                for k, v in feedback_counts.items()
            }
            
            return {
                "total_feedback": total,
                "counts": feedback_counts,
                "percentages": feedback_percentages
            }
            
        except Exception as e:
            logger.error(f"Error calculating feedback stats: {str(e)}")
            sentry_sdk.capture_exception(e)
            return {
                "total_feedback": 0,
                "counts": {},
                "percentages": {}
            }

# Create singleton instance
feedback_manager = FeedbackManager()