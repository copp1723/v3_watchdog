"""
Rate limiting middleware for Watchdog AI.
Provides rate limiting functionality to protect API endpoints and resources.
"""

import time
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import threading
from collections import defaultdict
import logging

from ..utils.errors import ValidationError
from ..utils.logging_config import get_logger

logger = get_logger(__name__)

class RateLimiter:
    """Thread-safe rate limiter implementation."""
    
    def __init__(self, window_size: int = 300, max_requests: int = 100):
        """
        Initialize the rate limiter.
        
        Args:
            window_size: Time window in seconds
            max_requests: Maximum requests per window
        """
        self.window_size = window_size
        self.max_requests = max_requests
        self.requests = defaultdict(list)
        self.lock = threading.Lock()
    
    def _clean_old_requests(self, key: str):
        """Remove requests outside the current window."""
        now = time.time()
        with self.lock:
            self.requests[key] = [
                req_time for req_time in self.requests[key]
                if now - req_time < self.window_size
            ]
    
    def is_allowed(self, key: str) -> bool:
        """
        Check if a request is allowed.
        
        Args:
            key: Rate limit key (e.g., IP address)
            
        Returns:
            bool indicating if request is allowed
        """
        self._clean_old_requests(key)
        
        with self.lock:
            if len(self.requests[key]) >= self.max_requests:
                return False
            
            self.requests[key].append(time.time())
            return True
    
    def get_remaining(self, key: str) -> Dict[str, Any]:
        """
        Get remaining requests information.
        
        Args:
            key: Rate limit key
            
        Returns:
            Dict with remaining requests info
        """
        self._clean_old_requests(key)
        
        with self.lock:
            used = len(self.requests[key])
            remaining = self.max_requests - used
            reset_time = time.time() + self.window_size
            
            return {
                "remaining": remaining,
                "limit": self.max_requests,
                "used": used,
                "reset": datetime.fromtimestamp(reset_time).isoformat()
            }


class RateLimitMiddleware:
    """Middleware for applying rate limits to requests."""
    
    def __init__(self):
        """Initialize the rate limit middleware."""
        # Different rate limits for different actions
        self.limiters = {
            "file_upload": RateLimiter(window_size=300, max_requests=10),  # 10 uploads per 5 minutes
            "api_call": RateLimiter(window_size=60, max_requests=60),      # 60 requests per minute
            "insight_generation": RateLimiter(window_size=60, max_requests=10)  # 10 insights per minute
        }
    
    def check_rate_limit(self, action: str, key: str) -> Dict[str, Any]:
        """
        Check rate limit for an action.
        
        Args:
            action: Action type to check
            key: Rate limit key
            
        Returns:
            Dict with rate limit info
            
        Raises:
            ValidationError if rate limit exceeded
        """
        if action not in self.limiters:
            logger.warning(f"Unknown rate limit action: {action}")
            return {"allowed": True}  # Default to allowed if unknown action
        
        limiter = self.limiters[action]
        
        if not limiter.is_allowed(key):
            info = limiter.get_remaining(key)
            raise ValidationError(
                f"Rate limit exceeded for {action}",
                details={
                    "action": action,
                    "rate_limit_info": info
                }
            )
        
        return limiter.get_remaining(key)


# Global rate limit middleware instance
rate_limiter = RateLimitMiddleware()