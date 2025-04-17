"""
Core Nova Act integration module for automated data collection.
"""

import os
import asyncio
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
from playwright.async_api import async_playwright, Browser, Page, TimeoutError
from concurrent.futures import ThreadPoolExecutor
import queue
import threading

from .constants import ErrorType, TwoFactorMethod, TIMEOUTS, VENDOR_CONFIGS
from .logging_config import log_error, log_info, log_warning
from .rate_limiter import rate_limiter
from .metrics import metrics_collector
from .credentials import CredentialManager

class NovaActClient:
    """Core client for browser automation and data collection."""
    
    def __init__(self, headless: bool = True, max_concurrent: int = 3):
        """
        Initialize the Nova Act client.
        
        Args:
            headless: Whether to run browsers in headless mode
            max_concurrent: Maximum number of concurrent browser sessions
        """
        self.headless = headless
        self.max_concurrent = max_concurrent
        self.browser_pool: Dict[str, Browser] = {}
        self.page_pool: Dict[str, Page] = {}
        self.session_locks: Dict[str, asyncio.Lock] = {}
        self.credential_manager = CredentialManager()
        
        # Thread pool for background tasks
        self.thread_pool = ThreadPoolExecutor(max_workers=max_concurrent)
        
        # Task queue for managing concurrent operations
        self.task_queue = queue.Queue()
        self.task_results = {}
        
        # Start background worker
        self.worker_thread = threading.Thread(target=self._process_task_queue)
        self.worker_thread.daemon = True
        self.worker_thread.start()
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.shutdown()
    
    async def start(self):
        """Initialize the browser automation system."""
        log_info("Starting Nova Act client", "system", "initialization")
        self.playwright = await async_playwright().start()
        
        # Pre-launch browser instances
        for _ in range(self.max_concurrent):
            browser = await self.playwright.chromium.launch(
                headless=self.headless,
                args=['--no-sandbox', '--disable-dev-shm-usage']
            )
            session_id = f"session_{len(self.browser_pool)}"
            self.browser_pool[session_id] = browser
            self.session_locks[session_id] = asyncio.Lock()
    
    async def shutdown(self):
        """Clean up resources."""
        log_info("Shutting down Nova Act client", "system", "shutdown")
        
        # Close all pages and browsers
        for page in self.page_pool.values():
            await page.close()
        for browser in self.browser_pool.values():
            await browser.close()
        
        await self.playwright.stop()
        self.thread_pool.shutdown(wait=True)
    
    async def _acquire_session(self) -> tuple[str, Browser, Page]:
        """Acquire an available browser session."""
        while True:
            for session_id, lock in self.session_locks.items():
                if lock.locked():
                    continue
                
                async with lock:
                    browser = self.browser_pool[session_id]
                    if session_id not in self.page_pool:
                        self.page_pool[session_id] = await browser.new_page()
                    
                    return session_id, browser, self.page_pool[session_id]
            
            await asyncio.sleep(1)  # Wait before retrying
    
    async def _handle_navigation(self, page: Page, url: str) -> bool:
        """Handle page navigation with timeout and error checking."""
        try:
            response = await page.goto(url, timeout=TIMEOUTS["navigation"] * 1000)
            if not response or not response.ok:
                log_warning(
                    f"Navigation failed or returned non-200 status: {response.status if response else 'No response'}",
                    "system",
                    "navigation"
                )
                return False
            
            # Wait for network idle
            await page.wait_for_load_state("networkidle")
            return True
            
        except TimeoutError:
            log_error(
                TimeoutError("Navigation timeout"),
                "system",
                "navigation"
            )
            return False
        except Exception as e:
            log_error(e, "system", "navigation")
            return False
    
    async def _handle_login(self,
                          page: Page,
                          credentials: Dict[str, Any],
                          selectors: Dict[str, str]) -> bool:
        """Handle the login process including 2FA."""
        try:
            # Fill username
            await page.fill(selectors["username"], credentials["username"])
            await page.fill(selectors["password"], credentials["password"])
            
            # Click login button
            await page.click(selectors["submit"])
            
            # Wait for navigation
            await page.wait_for_load_state("networkidle")
            
            # Check for 2FA
            if credentials.get("2fa_method"):
                return await self._handle_2fa(
                    page,
                    credentials["2fa_method"],
                    credentials.get("2fa_config", {})
                )
            
            return True
            
        except Exception as e:
            log_error(e, "system", "login")
            return False
    
    async def _handle_2fa(self,
                         page: Page,
                         method: str,
                         config: Dict[str, Any]) -> bool:
        """Handle 2FA verification."""
        try:
            if method == TwoFactorMethod.SMS.value:
                # Wait for SMS code input field
                code_input = await page.wait_for_selector(
                    '[name="sms-code"], [id*="sms"], [class*="sms"]',
                    timeout=TIMEOUTS["2fa"] * 1000
                )
                if not code_input:
                    return False
                
                # TODO: Implement SMS code retrieval
                return False
                
            elif method == TwoFactorMethod.EMAIL.value:
                # Wait for email code input
                code_input = await page.wait_for_selector(
                    '[name="email-code"], [id*="email"], [class*="email"]',
                    timeout=TIMEOUTS["2fa"] * 1000
                )
                if not code_input:
                    return False
                
                # TODO: Implement email code retrieval
                return False
                
            elif method == TwoFactorMethod.AUTHENTICATOR.value:
                # Wait for authenticator code input
                code_input = await page.wait_for_selector(
                    '[name="totp"], [id*="authenticator"], [class*="authenticator"]',
                    timeout=TIMEOUTS["2fa"] * 1000
                )
                if not code_input:
                    return False
                
                # TODO: Implement authenticator code generation
                return False
            
            return False
            
        except Exception as e:
            log_error(e, "system", "2fa")
            return False
    
    async def _handle_download(self,
                             page: Page,
                             download_selector: str,
                             file_pattern: str) -> Optional[str]:
        """Handle file download process."""
        try:
            # Create download promise before clicking
            async with page.expect_download(timeout=TIMEOUTS["download"] * 1000) as download_info:
                await page.click(download_selector)
                download = await download_info.value
            
            # Wait for download to complete
            path = await download.path()
            
            # Move to final location
            save_path = os.path.join("downloads", download.suggested_filename)
            os.makedirs("downloads", exist_ok=True)
            await download.save_as(save_path)
            
            return save_path
            
        except Exception as e:
            log_error(e, "system", "download")
            return None
    
    def _process_task_queue(self):
        """Background worker for processing tasks."""
        while True:
            try:
                task_id, coro = self.task_queue.get()
                if task_id is None:  # Shutdown signal
                    break
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    result = loop.run_until_complete(coro)
                    self.task_results[task_id] = result
                except Exception as e:
                    self.task_results[task_id] = e
                finally:
                    loop.close()
                    
            except Exception as e:
                log_error(e, "system", "task_processing")
    
    async def collect_report(self,
                           vendor: str,
                           credentials: Dict[str, Any],
                           report_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collect a report from a vendor system.
        
        Args:
            vendor: Vendor system name
            credentials: Login credentials and selectors
            report_config: Report navigation and download configuration
            
        Returns:
            Dictionary containing collection results
        """
        start_time = time.time()
        session_id = None
        
        # Import health_checker here
        from .health_check import health_checker

        try:
            # Check vendor health
            health_status = await health_checker.check_vendor_health(vendor)
            if health_status["status"] == "critical":
                raise Exception(f"Vendor system unhealthy: {health_status['message']}")
            
            # Acquire rate limit permission
            if not await rate_limiter.acquire(vendor):
                await metrics_collector.record_rate_limit_hit(vendor)
                raise Exception("Rate limit exceeded")
            
            # Acquire browser session
            session_id, browser, page = await self._acquire_session()
            
            # Navigate to login page
            if not await self._handle_navigation(page, credentials["url"]):
                raise Exception("Failed to navigate to login page")
            
            # Handle login
            if not await self._handle_login(page, credentials, report_config["selectors"]):
                raise Exception("Login failed")
            
            # Navigate through report menu
            for step in report_config["path"]:
                await page.click(report_config["selectors"][step])
                await page.wait_for_load_state("networkidle")
            
            # Download report
            file_path = await self._handle_download(
                page,
                report_config["download_selector"],
                report_config["file_pattern"]
            )
            
            if not file_path:
                raise Exception("Failed to download report")
            
            # Record success metrics
            duration = time.time() - start_time
            await metrics_collector.record_operation(
                vendor=vendor,
                operation="collect_report",
                duration=duration,
                success=True
            )
            
            return {
                "success": True,
                "file_path": file_path,
                "duration": duration,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            # Record error metrics
            duration = time.time() - start_time
            await metrics_collector.record_operation(
                vendor=vendor,
                operation="collect_report",
                duration=duration,
                success=False,
                error_type=str(e)
            )
            
            return {
                "success": False,
                "error": str(e),
                "duration": duration,
                "timestamp": datetime.now().isoformat()
            }
            
        finally:
            # Release rate limit
            await rate_limiter.release()
            
            # Release session lock if acquired
            if session_id and session_id in self.session_locks:
                self.session_locks[session_id].release()