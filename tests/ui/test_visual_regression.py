"""
Visual regression tests for UI components.
"""

import os
import pytest
from pathlib import Path
from playwright.sync_api import Page, expect
from typing import Generator
import logging
from .base_test import BaseVisualTest

# Configure logging
logger = logging.getLogger(__name__)

# Constants
SCREENSHOT_DIR = Path("tests/ui/screenshots")
THEMES = ["light", "dark"]
VIEWPORT_SIZES = {
    "desktop": {"width": 1440, "height": 900},
    "mobile": {"width": 360, "height": 640}
}
APP_URL = "http://localhost:8503"  # Using port 8503 as shown in run_app.py output

@pytest.fixture(scope="session", autouse=True)
def setup_screenshot_dirs():
    """Create screenshot directories if they don't exist."""
    for theme in THEMES:
        theme_dir = SCREENSHOT_DIR / theme
        theme_dir.mkdir(parents=True, exist_ok=True)

@pytest.fixture
def setup_page(page: Page):
    """Configure the page for testing."""
    page.goto(APP_URL)
    # Wait for the main content to be visible
    page.wait_for_selector("[data-testid='stApp']")
    return page

@pytest.mark.visual
@pytest.mark.parametrize("theme", THEMES)
@pytest.mark.parametrize("device", VIEWPORT_SIZES.keys())
def test_landing_page_visual(setup_page, theme, device):
    """Test visual appearance of the landing page in different themes and viewport sizes."""
    page = setup_page
    viewport = VIEWPORT_SIZES[device]
    
    # Set viewport size
    page.set_viewport_size(viewport)
    
    # Apply theme
    if theme == "light":
        # Click theme toggle button
        page.click(".theme-toggle")
        # Wait for light theme to be applied
        page.wait_for_selector("body.light-theme")
    
    # Take screenshot
    screenshot_path = SCREENSHOT_DIR / theme / f"{viewport['width']}.png"
    page.screenshot(path=str(screenshot_path))
    
    # Verify screenshot was created and is not empty
    assert screenshot_path.exists(), f"Screenshot not created at {screenshot_path}"
    assert screenshot_path.stat().st_size > 0, f"Screenshot is empty at {screenshot_path}"
    
    # Verify critical UI elements are visible
    expect(page.locator("[data-testid='stApp']")).to_be_visible()
    expect(page.locator(".theme-toggle")).to_be_visible()
    
    # Add more specific UI element checks as needed
    expect(page.locator(".header-container")).to_be_visible()
    expect(page.locator(".welcome-container")).to_be_visible()

@pytest.mark.visual
def test_screenshot_directory_structure():
    """Test that screenshot directory structure is correct."""
    for theme in THEMES:
        theme_dir = SCREENSHOT_DIR / theme
        assert theme_dir.exists(), f"Theme directory {theme_dir} does not exist"
        assert theme_dir.is_dir(), f"{theme_dir} is not a directory"

@pytest.mark.visual
class TestVisualRegression(BaseVisualTest):
    """Visual regression tests using base test class."""

    def test_theme_toggle(self):
        """Test that theme toggle changes the theme correctly."""
        # First ensure we're in dark mode
        self.page.wait_for_selector("body:not(.light-theme)")
        
        # Take a screenshot of dark theme
        self.assert_screenshot("dark_theme", selector="body")
        
        # Click the theme toggle
        self.page.click(".theme-toggle")
        
        # Verify light theme is applied
        self.page.wait_for_selector("body.light-theme")
        
        # Take a screenshot of light theme
        self.assert_screenshot("light_theme", selector="body")
        
        # Verify the theme class was applied
        assert "light-theme" in self.page.evaluate("document.body.className")
    
    def test_header_component(self):
        """Test the header component appearance."""
        self.assert_screenshot("header", selector=".header-container")
    
    def test_welcome_panel(self):
        """Test the welcome panel appearance."""
        self.assert_screenshot("welcome_panel", selector=".welcome-container")
    
    def test_mobile_responsive(self):
        """Test responsive layout at mobile size."""
        # Set viewport to mobile size
        self.page.set_viewport_size(VIEWPORT_SIZES["mobile"])
        
        # Test the entire page
        self.assert_screenshot("mobile_layout", selector="body")
        
        # Test specific components
        self.assert_screenshot("mobile_header", selector=".header-container")
        self.assert_screenshot("mobile_welcome", selector=".welcome-container")