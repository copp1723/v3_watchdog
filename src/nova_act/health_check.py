"""
Health checking for Nova Act integration.
"""

import asyncio
import aiohttp
from typing import Dict, Any, List
from datetime import datetime, timedelta
from .constants import HEALTH_CHECK, VENDOR_CONFIGS
from .logging_config import log_error, log_warning, log_info
from .metrics import metrics_collector

class HealthChecker:
    """Monitors health of Nova Act integration and vendor systems."""
    
    def __init__(self):
        """Initialize the health checker."""
        self.health_status = {}
        self.last_check = {}
        self.unhealthy_count = {}
        self.lock = asyncio.Lock()
    
    async def check_vendor_health(self, vendor: str) -> Dict[str, Any]:
        """
        Check health of a vendor's system.
        
        Args:
            vendor: The vendor to check
            
        Returns:
            Dictionary containing health status
        """
        if vendor not in VENDOR_CONFIGS:
            return {
                "status": "unknown",
                "message": f"Unknown vendor: {vendor}",
                "timestamp": datetime.now().isoformat()
            }
        
        try:
            # Get vendor-specific endpoint
            endpoint = HEALTH_CHECK["endpoints"].get(vendor)
            if not endpoint:
                return {
                    "status": "unknown",
                    "message": f"No health check endpoint for {vendor}",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Build full URL
            base_url = VENDOR_CONFIGS[vendor]["base_url"]
            if "{store_id}" in base_url:
                base_url = base_url.format(store_id="health")
            if "{region}" in base_url:
                base_url = base_url.format(region="us")
            
            url = f"{base_url}{endpoint}"
            
            # Perform health check
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    timeout=HEALTH_CHECK["timeout"]
                ) as response:
                    status = "healthy" if response.status == 200 else "unhealthy"
                    
                    async with self.lock:
                        # Update health tracking
                        self.last_check[vendor] = datetime.now()
                        
                        if status == "unhealthy":
                            self.unhealthy_count[vendor] = \
                                self.unhealthy_count.get(vendor, 0) + 1
                            if self.unhealthy_count[vendor] >= HEALTH_CHECK["unhealthy_threshold"]:
                                status = "critical"
                        else:
                            self.unhealthy_count[vendor] = 0
                        
                        self.health_status[vendor] = {
                            "status": status,
                            "message": f"Health check returned {response.status}",
                            "timestamp": datetime.now().isoformat(),
                            "metrics": await metrics_collector.get_vendor_metrics(vendor)
                        }
                        
                        return self.health_status[vendor]
                        
        except asyncio.TimeoutError:
            message = f"Health check timed out for {vendor}"
            log_warning(message, vendor, "health_check")
            return await self._handle_check_failure(vendor, message)
            
        except Exception as e:
            message = f"Health check failed for {vendor}: {str(e)}"
            log_error(e, vendor, "health_check")
            return await self._handle_check_failure(vendor, message)
    
    async def _handle_check_failure(self, vendor: str, message: str) -> Dict[str, Any]:
        """Handle a failed health check."""
        async with self.lock:
            self.unhealthy_count[vendor] = \
                self.unhealthy_count.get(vendor, 0) + 1
            
            status = "critical" if \
                self.unhealthy_count[vendor] >= HEALTH_CHECK["unhealthy_threshold"] \
                else "unhealthy"
            
            self.health_status[vendor] = {
                "status": status,
                "message": message,
                "timestamp": datetime.now().isoformat()
            }
            
            return self.health_status[vendor]
    
    async def get_all_health_status(self) -> Dict[str, Any]:
        """Get health status for all vendors."""
        async with self.lock:
            return {
                "vendors": dict(self.health_status),
                "timestamp": datetime.now().isoformat()
            }
    
    async def start_monitoring(self):
        """Start continuous health monitoring."""
        while True:
            try:
                for vendor in VENDOR_CONFIGS:
                    await self.check_vendor_health(vendor)
                    
                # Wait for next check interval
                await asyncio.sleep(HEALTH_CHECK["interval"])
                
            except Exception as e:
                log_error(e, "system", "health_monitoring")
                await asyncio.sleep(60)  # Wait a minute before retrying

# Global health checker instance
health_checker = HealthChecker()