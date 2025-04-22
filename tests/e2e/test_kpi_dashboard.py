"""
End-to-end tests for KPI dashboard.
"""

import pytest
import streamlit as st
from src.watchdog_ai.ui.pages.kpi_dashboard import kpi_dashboard

def test_kpi_dashboard():
    """Test KPI dashboard rendering."""
    kpi_dashboard()
    assert True  # If we got here without errors, the test passes