"""
Rate limiting implementation for Nova Act operations.
"""

import time
import asyncio
from collections import deque
from typing import Dict, Deque
from .constants import RATE_LIMITS
from .logging_config import log_warning, log_info

class RateLimiter:
    """Rate limiter for Nova Act operations."""
    
    def __init__(self):
        """Initialize the rate limiter."""
        self.minute_requests: Dict[str, Deque[float]] = {}
        self.hour_requests: Dict[str, Deque[float]] = {}
        self.active_sessions = 0
        self.lock = asyncio.Lock()
    
    async def acquire(self, vendor: str) -> bool:
        """
        Attempt to acquire permission for an operation.
        
        Args:
            vendor: The vendor system making the request
            
        Returns:
            bool indicating if the request is allowed
        """
        async with self.lock:
            current_time = time.time()
            
            # Initialize request queues if needed
            if vendor not in self.minute_requests:
                self.minute_requests[vendor] = deque()
            if vendor not in self.hour_requests:
                self.hour_requests[vendor] = deque()
            
            # Clean up old requests
            self._cleanup_old_requests(vendor, current_time)
            
            # Check rate limits
            if not self._check_limits(vendor):
                log_warning(
                    f"Rate limit exceeded for {vendor}",
                    vendor,
                    "rate_limit_check"
                )
                return False
            
            # Record new request
            self.minute_requests[vendor].append(current_time)
            self.hour_requests[vendor].append(current_time)
            self.active_sessions += 1
            
            log_info(
                f"Rate limit check passed for {vendor}",
                vendor,
                "rate_limit_check"
            )
            return True
    
    async def release(self):
        """Release an active session."""
        async with self.lock:
            self.active_sessions = max(0, self.active_sessions - 1)
    
    def _cleanup_old_requests(self, vendor: str, current_time: float):
        """Remove expired requests from tracking."""
        # Clean up minute requests
        while (self.minute_requests[vendor] and 
               current_time - self.minute_requests[vendor][0] > 60):
            self.minute_requests[vendor].popleft()
        
        # Clean up hour requests
        while (self.hour_requests[vendor] and 
               current_time - self.hour_requests[vendor][0] > 3600):
            self.hour_requests[vendor].popleft()
    
    def _check_limits(self, vendor: str) -> bool:
        """Check if current request would exceed rate limits."""
        # Check concurrent sessions
        if self.active_sessions >= RATE_LIMITS["max_concurrent_sessions"]:
            return False
        
        # Check requests per minute
        if len(self.minute_requests[vendor]) >= RATE_LIMITS["requests_per_minute"]:
            return False
        
        # Check requests per hour
        if len(self.hour_requests[vendor]) >= RATE_LIMITS["requests_per_hour"]:
            return False
        
        return True

# Global rate limiter instance
rate_limiter = RateLimiter()