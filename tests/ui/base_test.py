"""
Base class for visual regression tests.
"""

import os
import pytest
from playwright.sync_api import Page, expect
from typing import Dict, Any, Optional, Callable

class BaseVisualTest:
    """Base class for visual regression tests."""
    
    @pytest.fixture(autouse=True)
    def setup(self, page: Page):
        """Setup fixture that runs before each test."""
        self.page = page
        self.screenshot_dir = os.path.join(os.path.dirname(__file__), "screenshots")
        os.makedirs(self.screenshot_dir, exist_ok=True)
        
        # Navigate to app and wait for it to load
        self.page.goto("http://localhost:8503")
        self.page.wait_for_load_state("networkidle")
        self.page.wait_for_selector("[data-testid='stApp']", state="visible", timeout=60000)
    
    def assert_screenshot(
        self,
        name: str,
        selector: str = "[data-testid='stApp']",
        setup_func: Optional[Callable] = None,
        **kwargs: Dict[str, Any]
    ):
        """
        Take a screenshot and compare it with the baseline.
        
        Args:
            name: Name of the screenshot
            selector: CSS selector to capture (defaults to full app)
            setup_func: Optional function to run before taking screenshot
            **kwargs: Additional arguments passed to page.screenshot()
        """
        if setup_func:
            setup_func()
            
        screenshot_path = os.path.join(self.screenshot_dir, f"{name}.png")
        
        # Ensure the element is visible
        self.page.wait_for_selector(selector, state="visible", timeout=60000)
        
        # Take the screenshot
        element = self.page.locator(selector)
        expect(element).to_be_visible()
        
        # Capture the screenshot
        if os.path.exists(screenshot_path):
            # Compare with baseline if it exists
            with open(screenshot_path, "rb") as f:
                expected_screenshot = f.read()
            actual_screenshot = element.screenshot()
            assert actual_screenshot == expected_screenshot, f"Screenshot {name} does not match baseline"
        else:
            # Create new baseline if it doesn't exist
            element.screenshot(path=screenshot_path)
            print(f"Created new baseline screenshot: {screenshot_path}")
    
    def assert_visual_regression(
        self,
        name: str,
        setup_func: Optional[Callable] = None,
        **kwargs: Dict[str, Any]
    ):
        """
        Decorator to handle visual regression testing with optional setup.
        
        Args:
            name: Name of the screenshot
            setup_func: Optional function to run before taking screenshot
            **kwargs: Additional arguments passed to page.screenshot()
        """
        if setup_func:
            setup_func()
        self.assert_screenshot(name, **kwargs) 