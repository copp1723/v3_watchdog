"""
Metrics collection for Nova Act operations.
"""

import time
from typing import Dict, Any, Optional
from collections import defaultdict
from datetime import datetime, timedelta
import asyncio
from .logging_config import log_info, log_warning
from .constants import RATE_LIMITS

class MetricsCollector:
    """Collects and tracks metrics for Nova Act operations."""
    
    def __init__(self):
        """Initialize the metrics collector."""
        self.metrics = defaultdict(lambda: defaultdict(float))
        self.operation_counts = defaultdict(lambda: defaultdict(int))
        self.error_counts = defaultdict(lambda: defaultdict(int))
        self.response_times = defaultdict(lambda: defaultdict(list))
        self.lock = asyncio.Lock()
        
        # Track rate limit metrics
        self.rate_limit_hits = defaultdict(int)
        self.last_reset = datetime.now()
    
    async def record_operation(self,
                             vendor: str,
                             operation: str,
                             duration: float,
                             success: bool,
                             error_type: Optional[str] = None):
        """
        Record metrics for an operation.
        
        Args:
            vendor: The vendor system
            operation: The operation performed
            duration: Time taken in seconds
            success: Whether the operation succeeded
            error_type: Type of error if operation failed
        """
        async with self.lock:
            # Record operation count
            self.operation_counts[vendor][operation] += 1
            
            # Record response time
            self.response_times[vendor][operation].append(duration)
            
            # Keep only last 1000 response times
            if len(self.response_times[vendor][operation]) > 1000:
                self.response_times[vendor][operation] = self.response_times[vendor][operation][-1000:]
            
            # Record error if operation failed
            if not success and error_type:
                self.error_counts[vendor][error_type] += 1
            
            # Calculate and update metrics
            self._update_metrics(vendor, operation)
            
            # Log significant metrics
            if duration > 5.0:  # Log slow operations
                log_warning(
                    f"Slow operation detected: {operation} for {vendor} took {duration:.2f}s",
                    vendor,
                    operation
                )
    
    async def record_rate_limit_hit(self, vendor: str):
        """Record when a rate limit is hit."""
        async with self.lock:
            self.rate_limit_hits[vendor] += 1
            
            # Log if rate limits are frequently hit
            if self.rate_limit_hits[vendor] > RATE_LIMITS["requests_per_minute"] // 2:
                log_warning(
                    f"High rate limit hits for {vendor}",
                    vendor,
                    "rate_limiting"
                )
    
    def _update_metrics(self, vendor: str, operation: str):
        """Update calculated metrics for a vendor/operation pair."""
        times = self.response_times[vendor][operation]
        if times:
            self.metrics[vendor][f"{operation}_avg_time"] = sum(times) / len(times)
            self.metrics[vendor][f"{operation}_max_time"] = max(times)
            self.metrics[vendor][f"{operation}_min_time"] = min(times)
        
        total_ops = self.operation_counts[vendor][operation]
        total_errors = sum(
            count for error_type, count in self.error_counts[vendor].items()
            if operation in error_type.lower()
        )
        
        if total_ops > 0:
            self.metrics[vendor][f"{operation}_success_rate"] = \
                (total_ops - total_errors) / total_ops
    
    async def get_vendor_metrics(self, vendor: str) -> Dict[str, Any]:
        """Get all metrics for a vendor."""
        async with self.lock:
            return {
                "metrics": dict(self.metrics[vendor]),
                "operation_counts": dict(self.operation_counts[vendor]),
                "error_counts": dict(self.error_counts[vendor]),
                "rate_limit_hits": self.rate_limit_hits[vendor]
            }
    
    async def reset_daily_metrics(self):
        """Reset daily metrics. Should be called once per day."""
        async with self.lock:
            self.last_reset = datetime.now()
            self.rate_limit_hits.clear()
            
            # Log metrics before clearing
            for vendor in self.metrics:
                log_info(
                    f"Daily metrics for {vendor}: {dict(self.metrics[vendor])}",
                    vendor,
                    "metrics_reset"
                )
            
            self.metrics.clear()
            self.operation_counts.clear()
            self.error_counts.clear()
            self.response_times.clear()

# Global metrics collector instance
metrics_collector = MetricsCollector()