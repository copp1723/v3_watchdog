"""
End-to-end tests for the navigation tabs in the main application.
"""

import pytest
from playwright.sync_api import expect, Page
import streamlit as st
from src.watchdog_ai.ui.pages.main_app import render_app

@pytest.mark.skip("Requires running Streamlit server")
def test_navigation_tabs(page: Page):
    """Test the navigation tabs in the main application."""
    # Navigate to the app
    page.goto("http://localhost:8501")
    
    # Check for the three tabs
    tabs = page.locator(".stTabs button")
    expect(tabs).to_have_count(3)
    
    # Check the tab text
    expect(tabs.nth(0)).to_have_text("Insight Engine")
    expect(tabs.nth(1)).to_have_text("System Connect") 
    expect(tabs.nth(2)).to_have_text("Settings")

@pytest.mark.skip("Requires running Streamlit server")
def test_tab_switching(page: Page):
    """Test switching between tabs."""
    # Navigate to the app
    page.goto("http://localhost:8501")
    
    # Access the tabs
    tabs = page.locator(".stTabs button")
    
    # Click on System Connect tab
    tabs.nth(1).click()
    expect(page.locator("text=System Connect")).to_be_visible()
    
    # Click on Settings tab
    tabs.nth(2).click()
    expect(page.locator("text=Upload Configuration")).to_be_visible()
    
    # Click on Insight Engine tab
    tabs.nth(0).click()
    expect(page.locator("text=Data Upload")).to_be_visible()

@pytest.mark.skip("Requires running Streamlit server")
def test_tab_content_loaded(page: Page):
    """Test that the content for each tab is properly loaded."""
    # Navigate to the app
    page.goto("http://localhost:8501")
    
    # Check Insight Engine tab content
    expect(page.locator("text=Welcome to Watchdog AI!")).to_be_visible()
    expect(page.locator("text=Data Upload")).to_be_visible()
    expect(page.locator("text=Ask Questions & Get Insights")).to_be_visible()
    
    # Click on System Connect tab and check content
    page.locator(".stTabs button").nth(1).click()
    expect(page.locator("text=System Connect")).to_be_visible()
    
    # Click on Settings tab and check content
    page.locator(".stTabs button").nth(2).click()
    expect(page.locator("text=Theme")).to_be_visible()
    expect(page.locator("text=Upload Configuration")).to_be_visible()
    expect(page.locator("text=Update Frequency")).to_be_visible()