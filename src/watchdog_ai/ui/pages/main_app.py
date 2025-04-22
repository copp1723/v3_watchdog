"""
Watchdog-AI Streamlit Front-End
--------------------------------
A cleaned and reorganized implementation of the main application UI.

This module provides the main application interface for Watchdog AI,
including tab navigation, welcome panel, and integration of all components.

EXECUTIVE SUMMARY:
-----------------
Purpose:
  Streamlit-based web application for a dealership intelligence platform
  that helps analyze data and get AI-powered insights.

Core Features:
  - Data Upload & Processing (via render_data_uploader())
  - System Integration (via render_system_connect_tab())
  - Settings Management (via render_settings_tab())
  - Interactive Dashboard (via render_insight_engine_tab())
  - AI-Powered Insights (via ChatInterface)

Technical Organization:
  - Clean, modular code structure with clear function separation
  - Consistent error handling (try/except blocks with logging)
  - Theme-aware styling system
  - Streamlit session state management for persistence

Developer Notes:
  - HTML templates are defined as constants at the top of the file
  - CSS is consolidated in a single COMMON_CSS constant
  - Session state is initialized in initialize_session_state()
  - Main app rendering happens in render_app()

Code Review Summary:
  - Eliminated duplicate HTML/CSS definitions
  - Consolidated styling into a single COMMON_CSS constant
  - Implemented missing functions (render_quick_start_guide, render_system_connect_tab, render_settings_tab)
  - Fixed structural issues with proper function boundaries
  - Enhanced error handling throughout the application
  - Improved overall code organization and readability
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import logging
import time
import traceback
import os
import io
from typing import Optional, List, Dict, Any, Tuple, Callable, Union

# Analytics and Visualization libraries
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
import numpy as np
from scipy import stats
from watchdog_ai.core.constants import (
    DEFAULT_THEME,
    DEFAULT_TAB,
    TABS,
    DEFAULT_PREVIOUS_SALES,
    PAGE_CONFIG
)

# Import components using absolute imports
from watchdog_ai.ui.components.header import render_header
from watchdog_ai.ui.components.data_uploader import render_data_uploader
from watchdog_ai.ui.components.chat_interface import ChatInterface
from watchdog_ai.ui.components.sales_report_renderer import render_insight_block
from watchdog_ai.ui.components.nova_act_panel import render_nova_act_panel

# Import theme system
from watchdog_ai.ui.utils.ui_theme import Theme, ColorMode, Spacing, ColorSystem, Typography

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# HTML TEMPLATES
# =============================================================================

QUICK_START_GUIDE_HTML = """
<div class="quick-start-container">
    <div class="quick-start-steps">
        <div class="step">
st.header("Data Upload", "📤")
            <div class="step-content">
                <h4>Upload Your Data</h4>
                <p>Upload your dealership data or connect to your DMS/CRM.</p>
            </div>
        </div>
        <div class="step">
            <div class="step-number">2</div>
            <div class="step-content">
                <h4>Process & Validate</h4>
                <p>We'll analyze and prepare your data for insights.</p>
            </div>
        </div>
        <div class="step">
            <div class="step-number">3</div>
            <div class="step-content">
                <h4>Get Insights</h4>
                <p>View automated insights or ask questions to our AI.</p>
            </div>
        </div>
    </div>
</div>
"""

SYSTEM_CONNECT_FORM_HTML = """
<div class="system-connect-container">
    <div class="system-connect-header">
        <h3>Connect Your Dealership Systems</h3>
        <p class="system-description">Connect to your DMS or CRM system for automatic data synchronization</p>
    </div>
</div>
"""

FILE_HISTORY_ITEM_HTML = """
<div class="file-history-item">
    <div class="file-name">{filename}</div>
    <div class="file-timestamp">{timestamp_str}</div>
    <div class="file-status {status_class}">{icon} {status}</div>
    <div class="file-actions">
        <button id="view_{i}" class="file-action-btn view-btn">View</button>
        <button id="delete_{i}" class="file-action-btn delete-btn">Delete</button>
    </div>
</div>
"""

SECTION_HEADER_HTML = """
<div class="section-header">
    <h3>{icon} {title}</h3>
    <p class="section-description">{description}</p>
</div>
"""

EMPTY_STATE_HTML = """
<div class="empty-state">
    <div class="empty-state-icon">{icon}</div>
    <div class="empty-state-message">{message}</div>
    <div class="empty-state-hint">{hint}</div>
</div>
"""

MESSAGE_CONTAINER_HTML = """
<div class="message-container {type}">
    <p>{message}</p>
</div>
"""

SETTINGS_GROUP_HTML = """
<div class="settings-group">
    <h3 class="settings-group-title">{title}</h3>
    <div class="settings-group-content">
        {content}
    </div>
</div>
"""

SETTING_ITEM_HTML = """
<div class="setting-item">
    <div class="setting-info">
        <div class="setting-label">{label}</div>
        <div class="setting-description">{description}</div>
    </div>
    <div class="setting-control">
        {control}
    </div>
</div>
"""
# =============================================================================
# CSS Styles as Constants
# =============================================================================

COMMON_CSS = """
/* Common container styles */
.section-container {
    margin-bottom: var(--spacing-lg);
    padding: var(--spacing-md);
    border-radius: 8px;
    background-color: var(--color-bg-primary);
    border: 1px solid var(--color-border);
}

/* Section headers */
.section-header {
    margin-bottom: var(--spacing-md);
    padding-bottom: var(--spacing-xs);
    border-bottom: 1px solid var(--color-border);
}

.section-header h3 {
    font-size: var(--font-size-xl);
    font-weight: var(--font-weight-semibold);
    color: var(--color-text-primary);
    display: flex;
    align-items: center;
    gap: var(--spacing-xs);
}

.section-description {
    font-size: var(--font-size-sm);
    font-weight: var(--font-weight-normal);
    color: var(--color-text-secondary);
    margin-top: var(--spacing-xxs);
}

.loading-text {
    font-size: var(--font-size-md);
    font-weight: var(--font-weight-medium);
    color: var(--color-primary);
}

/* Quick start guide styles */
.quick-start-container {
    margin: var(--spacing-md) 0;
    padding: var(--spacing-md);
    border-radius: 8px;
    background-color: var(--color-bg-tertiary);
}

.quick-start-steps {
    display: flex;
    flex-direction: column;
    gap: var(--spacing-md);
}

.step {
    display: flex;
    gap: var(--spacing-sm);
    align-items: flex-start;
    padding: var(--spacing-sm);
    border-radius: 8px;
    background-color: var(--color-bg-tertiary);
    transition: transform 0.2s ease-in-out;
}

.step:hover {
    transform: translateY(-2px);
}

.step-number {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background-color: var(--color-primary);
    color: white;
    font-weight: var(--font-weight-bold);
}

.step-content h4 {
    font-size: var(--font-size-md);
    font-weight: var(--font-weight-semibold);
    margin: 0 0 var(--spacing-xs) 0;
}

.step-content p {
    font-size: var(--font-size-sm);
    margin: 0;
    color: var(--color-text-secondary);
}

/* Insight card styles */
.insight-container {
    margin-top: var(--spacing-lg);
    margin-bottom: var(--spacing-lg);
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    background-color: var(--color-bg-primary);
    border: 1px solid var(--color-border);
}

.insight-header {
    padding: var(--spacing-sm) var(--spacing-md);
    background-color: var(--color-bg-accent);
    border-bottom: 1px solid var(--color-border);
}

.insight-title {
    font-size: var(--font-size-lg);
    font-weight: var(--font-weight-semibold);
    color: var(--color-primary-dark);
    margin: 0;
}

.insight-body {
    padding: var(--spacing-md);
}

.insight-metric {
    font-size: var(--font-size-2xl);
    font-weight: var(--font-weight-bold);
    color: var(--color-primary);
    margin: var(--spacing-sm) 0;
}

.insight-description {
    font-size: var(--font-size-md);
    font-weight: var(--font-weight-normal);
    margin-top: var(--spacing-sm);
    color: var(--color-text-primary);
}

.insight-actions {
    margin-top: var(--spacing-md);
    display: flex;
    justify-content: flex-end;
    gap: var(--spacing-sm);
}

/* Chat message container */
.chat-message-container {
    margin-bottom: var(--spacing-md);
    max-height: 50vh;
    overflow-y: auto;
    padding: var(--spacing-sm);
    border-radius: 8px;
    background-color: var(--color-bg-secondary);
    border: 1px solid var(--color-border);
}

/* Message containers for alerts and notifications */
.message-container {
    padding: var(--spacing-sm) var(--spacing-md);
    border-radius: 8px;
    margin-bottom: var(--spacing-md);
}

.message-container.success {
    background-color: var(--color-success-bg);
    border: 1px solid var(--color-success);
    color: var(--color-success-text);
}

.message-container.error {
    background-color: var(--color-error-bg);
    border: 1px solid var(--color-error);
    color: var(--color-error-text);
}

.message-container.info {
    background-color: var(--color-info-bg);
    border: 1px solid var(--color-info);
    color: var(--color-info-text);
}

.message-container.warning {
    background-color: var(--color-warning-bg);
    border: 1px solid var(--color-warning);
    color: var(--color-warning-text);
}

/* File history styles */
.file-history-item {
    display: grid;
    grid-template-columns: 3fr 2fr 1fr 1fr;
    padding: var(--spacing-sm);
    border-bottom: 1px solid var(--color-border);
    align-items: center;
}

.file-name {
    font-size: var(--font-size-sm);
    font-weight: var(--font-weight-medium);
}

.file-timestamp {
    font-size: var(--font-size-sm);
    font-weight: var(--font-weight-normal);
    color: var(--color-text-secondary);
}

.file-status {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: var(--font-size-sm);
    font-weight: var(--font-weight-medium);
}

.status-success {
    color: var(--color-success);
}

.status-processing {
    color: var(--color-warning);
}

.status-error {
    color: var(--color-error);
}

.file-actions {
    display: flex;
    gap: 8px;
}

.file-action-btn {
    padding: 4px 8px;
    border-radius: 4px;
    border: none;
    cursor: pointer;
    font-size: var(--font-size-xs);
    font-weight: var(--font-weight-medium);
}

.view-btn {
    background-color: var(--color-neutral-light);
    color: var(--color-text-primary);
}

.delete-btn {
    background-color: var(--color-error-bg);
    color: var(--color-error);
}

/* System connect form styles */
.system-connect-container {
    padding: var(--spacing-md);
    border-radius: 8px;
    background-color: var(--color-bg-secondary);
    margin-bottom: var(--spacing-lg);
}

.system-connect-header {
    margin-bottom: var(--spacing-md);
}

.system-connect-header h3 {
    font-size: var(--font-size-xl);
    font-weight: var(--font-weight-semibold);
    color: var(--color-text-primary);
    margin-bottom: var(--spacing-xs);
}

.system-description {
    font-size: var(--font-size-sm);
    color: var(--color-text-secondary);
}

.form-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: var(--spacing-md);
}

@media (max-width: 768px) {
    .form-grid {
        grid-template-columns: 1fr;
    }
}

.form-actions {
    margin-top: var(--spacing-lg);
    display: flex;
    justify-content: flex-end;
    gap: var(--spacing-md);
}

/* Settings panel styles */
.settings-container {
    padding: var(--spacing-md);
    border-radius: 8px;
    background-color: var(--color-bg-secondary);
    margin-bottom: var(--spacing-lg);
}

.settings-group {
    margin-bottom: var(--spacing-lg);
}

.settings-group-title {
    font-size: var(--font-size-lg);
    font-weight: var(--font-weight-semibold);
    color: var(--color-text-primary);
    margin-bottom: var(--spacing-md);
    padding-bottom: var(--spacing-xs);
    border-bottom: 1px solid var(--color-border);
}

.setting-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: var(--spacing-sm) 0;
    border-bottom: 1px solid var(--color-border-light);
}

.setting-label {
    font-size: var(--font-size-md);
    font-weight: var(--font-weight-medium);
    color: var(--color-text-primary);
}

.setting-description {
    font-size: var(--font-size-sm);
    color: var(--color-text-secondary);
    margin-top: var(--spacing-xxs);
}

.setting-control {
    min-width: 150px;
}

/* Status dots */
.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: #4CAF50;
    display: inline-block;
}

.data-status-indicator {
    margin-top: var(--spacing-md);
    background-color: rgba(255, 255, 255, 0.2);
    padding: var(--spacing-xs) var(--spacing-sm);
    border-radius: 4px;
    display: inline-flex;
    align-items: center;
    gap: var(--spacing-xs);
    font-size: 14px;
}
"""

# =============================================================================
# Session Management Functions
# =============================================================================

def initialize_session_state() -> None:
    """Initialize session state variables with default values."""
    defaults = {
        'nova_act_connected': False,
        'last_sync_timestamp': None,
        'active_tab': DEFAULT_TAB,
        'theme': DEFAULT_THEME,
        'previous_sales': DEFAULT_PREVIOUS_SALES,
        'incremental_updates': True,
        'auto_detect_schema': True,
        'show_upload_history': False,
        'chat_history': [],
        'insights': [],
        'uploaded_files': [],
        'upload_timestamps': {},
        'last_insight_time': None,
        'show_welcome_panel': True,
        'loading_data': False,
        'error_message': None,
        'success_message': None,
        'notification_count': 0,
        'nova_act_available': False,  # Set to True when Nova Act is available
        'validated_data': None,  # Placeholder for data
        
        # Phase 4 additions - Analytics and Charting
        'chart_preferences': {
            'chart_type': 'bar',  # Default chart type (bar, line, scatter, etc.)
            'color_scheme': 'viridis',  # Default color scheme
            'show_legend': True,
            'show_grid': True,
            'animation': True,
            'custom_title': '',  # Custom chart title
            'x_axis_label': '',  # Custom x-axis label
            'y_axis_label': '',  # Custom y-axis label
            'font_size': 'medium',  # Font size for chart text
            'enable_trend_line': False  # Whether to show trend line
        },
        
        # Phase 4 additions - Data Validation
        'validation_status': {
            'is_valid': False,
            'validation_messages': [],
            'validation_timestamp': None,
            'schema_errors': [],
            'data_quality_score': 0.0
        },
        
        # Phase 4 additions - Threshold Settings
        'threshold_settings': {
            'confidence_threshold': 0.75,  # Default confidence threshold for insights
            'anomaly_detection_threshold': 0.05,  # Default anomaly detection threshold
            'similarity_threshold': 0.8,  # Default similarity threshold for matching
            'min_data_points': 100  # Minimum data points required for analysis
        },
        
        # Phase 4 additions - Data Retention
        'data_retention_settings': {
            'retention_period_days': 90,  # Default data retention period in days
            'auto_archive': True,  # Whether to automatically archive old data
            'include_in_analytics': True,  # Whether to include archived data in analytics
            'last_purge_date': None  # Last time old data was purged
        },
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
def show_message(message_type: str, message: str) -> None:
    """Show a message to the user and register it in the session state.
    
    Args:
        message_type: Type of message (error, success, info, warning)
        message: Message content
    """
    if message_type == "error":
        st.session_state.error_message = message
    elif message_type == "success":
        st.session_state.success_message = message
    
    if message_type == "error":
        st.error(message, icon="🚫")
    elif message_type == "success":
        st.success(message, icon="✅")
    elif message_type == "info":
        st.info(message, icon="ℹ️")
    elif message_type == "warning":
        st.warning(message, icon="⚠️")


def display_status_messages() -> None:
    """Display any status messages stored in session state and clear them."""
    if error := st.session_state.get('error_message'):
        st.error(error, icon="🚫")
        st.session_state.error_message = None
        
    if success := st.session_state.get('success_message'):
        st.success(success, icon="✅")
        st.session_state.success_message = None

# =============================================================================
# CSS and Styling Functions
# =============================================================================

def apply_common_css() -> None:
    """Apply common CSS styles used across the application."""
    # Create theme instance for styling
    theme = Theme(mode=ColorMode.DARK if st.session_state.theme == "dark" else ColorMode.LIGHT)
    
    st.markdown(f"""
        <style>
            /* Common container styles */
            .section-container {{
                margin-bottom: {Spacing.LG};
                padding: {Spacing.MD};
                border-radius: 8px;
                background-color: {ColorSystem.NEUTRAL.get(50 if theme.get_color_mode() == ColorMode.LIGHT else 800)};
                border: 1px solid {ColorSystem.NEUTRAL.get(200 if theme.get_color_mode() == ColorMode.LIGHT else 700)};
            }}
            
            /* Section headers */
            .section-header {{
                margin-bottom: {Spacing.MD};
                padding-bottom: {Spacing.XS};
                border-bottom: 1px solid {ColorSystem.NEUTRAL.get(200 if theme.get_color_mode() == ColorMode.LIGHT else 700)};
            }}
            
            .section-header h3 {{
                {Typography.get_font_css(Typography.SIZE_XL, Typography.WEIGHT_SEMIBOLD)}
                color: {ColorSystem.get_text(theme.get_color_mode())};
                display: flex;
                align-items: center;
                gap: {Spacing.XS};
            }}
            
            .section-description {{
                {Typography.get_font_css(Typography.SIZE_SM, Typography.WEIGHT_NORMAL)}
                color: {ColorSystem.NEUTRAL.get(600)};
                margin-top: {Spacing.XXS};
            }}
            
            /* Loading state styles */
            .loading-container {{
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                padding: {Spacing.LG};
            }}
            
            .loading-spinner {{
                margin-bottom: {Spacing.MD};
            }}
            
            .loading-text {{
                {Typography.get_font_css(Typography.SIZE_MD, Typography.WEIGHT_MEDIUM)}
                color: {ColorSystem.get_primary()};
            }}
            
            /* Empty state styles */
            .empty-state {{
                text-align: center;
                padding: {Spacing.LG};
            }}
            
            .empty-state-icon {{
                font-size: 48px;
                margin-bottom: {Spacing.MD};
                color: {ColorSystem.NEUTRAL.get(400)};
            }}
            
            .empty-state-message {{
                {Typography.get_font_css(Typography.SIZE_LG, Typography.WEIGHT_MEDIUM)}
                margin-bottom: {Spacing.SM};
            }}
            
            .empty-state-hint {{
                {Typography.get_font_css(Typography.SIZE_SM, Typography.WEIGHT_NORMAL)}
                color: {ColorSystem.NEUTRAL.get(600)};
            }}
            
            /* Welcome panel styles */
            .welcome-container {{
                text-align: center;
                padding: {Spacing.LAYOUT_SM};
                margin-bottom: {Spacing.LAYOUT_XS};
                background: linear-gradient(135deg, {ColorSystem.get_primary(400)} 0%, {ColorSystem.get_secondary(400)} 100%);
                color: white;
                border-radius: 12px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            }}
            
            .welcome-title {{
                {Typography.get_font_css(Typography.SIZE_3XL, Typography.WEIGHT_BOLD)}
                margin-bottom: {Spacing.MD};
            }}
            
            .welcome-subtitle {{
                {Typography.get_font_css(Typography.SIZE_LG, Typography.WEIGHT_NORMAL)}
                opacity: 0.9;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                background-color: {ColorSystem.NEUTRAL.get(50 if theme.get_color_mode() == ColorMode.LIGHT else 800)};
                border: 1px solid {ColorSystem.NEUTRAL.get(200 if theme.get_color_mode() == ColorMode.LIGHT else 700)};
            }}
            
            .insight-header {{
                padding: {Spacing.SM} {Spacing.MD};
                background-color: {ColorSystem.get_primary(100 if theme.get_color_mode() == ColorMode.LIGHT else 800)};
                border-bottom: 1px solid {ColorSystem.NEUTRAL.get(200 if theme.get_color_mode() == ColorMode.LIGHT else 700)};
            }}
            
            .insight-title {{
                {Typography.get_font_css(Typography.SIZE_LG, Typography.WEIGHT_SEMIBOLD)}
                color: {ColorSystem.get_primary(700)};
                margin: 0;
            }}
            
            .insight-body {{
                padding: {Spacing.MD};
            }}
            
            .insight-metric {{
                {Typography.get_font_css(Typography.SIZE_2XL, Typography.WEIGHT_BOLD)}
                color: {ColorSystem.get_primary()};
                margin: {Spacing.SM} 0;
            }}
            
            .insight-description {{
                {Typography.get_font_css(Typography.SIZE_MD, Typography.WEIGHT_NORMAL)}
                margin-top: {Spacing.SM};
                color: {ColorSystem.get_text(theme.get_color_mode())};
            }}
            
            .insight-actions {{
                margin-top: {Spacing.MD};
                display: flex;
                justify-content: flex-end;
                gap: {Spacing.SM};
            }}
            
            /* Chat message container */
            .chat-message-container {{
                margin-bottom: {Spacing.MD};
                max-height: 50vh;
                overflow-y: auto;
                padding: {Spacing.SM};
                border-radius: 8px;
                background-color: {ColorSystem.NEUTRAL.get(100 if theme.get_color_mode() == ColorMode.LIGHT else 800)};
                border: 1px solid {ColorSystem.NEUTRAL.get(200 if theme.get_color_mode() == ColorMode.LIGHT else 700)};
            }}
            
            /* File history styles */
            .file-history-item {{
                display: grid;
                grid-template-columns: 3fr 2fr 1fr 1fr;
                padding: {Spacing.SM};
                border-bottom: 1px solid {ColorSystem.NEUTRAL.get(200 if theme.get_color_mode() == ColorMode.LIGHT else 700)};
            }}
        """, unsafe_allow_html=True)
# =============================================================================
# UI Rendering Functions
# =============================================================================

def render_quick_start_guide() -> None:
    """Render the quick start guide with step-by-step instructions."""
    # Display status messages
    display_status_messages()
    
    # Apply common CSS
    apply_common_css()
    
    # Create theme instance for styling
    theme = Theme(mode=ColorMode.DARK if st.session_state.theme == "dark" else ColorMode.LIGHT)
    
    # Render section header
    st.markdown(
        SECTION_HEADER_HTML.format(
            icon="🚀",
            title="Quick Start Guide",
            description="Follow these steps to get started with Watchdog AI"
        ),
        unsafe_allow_html=True
    )
    
    # Display quick start guide
    st.markdown(QUICK_START_GUIDE_HTML, unsafe_allow_html=True)
    
    # Add interactive elements
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📤 Upload Data", use_container_width=True):
            st.session_state.active_tab = "Insight Engine"
            st.rerun()
    
    with col2:
        if st.button("🔌 Connect Systems", use_container_width=True):
            st.session_state.active_tab = "System Connect"
            st.rerun()
    
    with col3:
        if st.button("⚙️ Configure Settings", use_container_width=True):
            st.session_state.active_tab = "Settings"
            st.rerun()
    
    # Additional resources section
    st.markdown("### Additional Resources")
    
    resources_col1, resources_col2 = st.columns(2)
    
    with resources_col1:
        st.markdown("""
        * [Documentation](https://docs.watchdog-ai.com)
        * [Video Tutorials](https://watchdog-ai.com/tutorials)
        * [FAQ](https://watchdog-ai.com/faq)
        """)
    
    with resources_col2:
        st.markdown("""
        * [Support](https://support.watchdog-ai.com)
        * [Community Forum](https://community.watchdog-ai.com)
        * [Release Notes](https://watchdog-ai.com/releases)
        """)

def render_system_connect_tab() -> None:
    """Render the System Connect tab for integrating with external systems."""
    # Display status messages
    display_status_messages()
    
    # Apply common CSS
    apply_common_css()
    
    # Create theme instance for styling
    theme = Theme(mode=ColorMode.DARK if st.session_state.theme == "dark" else ColorMode.LIGHT)
    
    # Render section header
    st.markdown(
        SECTION_HEADER_HTML.format(
            icon="🔌",
            title="Connect Your Systems",
            description="Integrate with your dealership management systems for seamless data flow"
        ),
        unsafe_allow_html=True
    )
    
    # Display system connect form
    st.markdown(SYSTEM_CONNECT_FORM_HTML, unsafe_allow_html=True)
    
    # Connection status section
    connection_status = st.session_state.get('nova_act_connected', False)
    
    status_col1, status_col2 = st.columns([1, 3])
    
    with status_col1:
        if connection_status:
            st.success("Connected", icon="✅")
        else:
            st.error("Not Connected", icon="❌")
    
    with status_col2:
        if connection_status:
            last_sync = st.session_state.get('last_sync_timestamp', datetime.now())
            if isinstance(last_sync, datetime):
                sync_str = last_sync.strftime("%b %d, %Y %I:%M %p")
            else:
                sync_str = str(last_sync)
            st.info(f"Last synchronized: {sync_str}", icon="🔄")
        else:
            st.warning("No active connections", icon="⚠️")
    
    # API Key Management
    st.subheader("API Credentials")
    
    with st.form(key="api_key_form"):
        api_key = st.text_input("API Key", type="password", placeholder="Enter your API key")
        api_secret = st.text_input("API Secret", type="password", placeholder="Enter your API secret")
        
        col1, col2 = st.columns(2)
        with col1:
            endpoint = st.text_input("API Endpoint", placeholder="https://api.yourdms.com")
        with col2:
            timeout = st.number_input("Timeout (seconds)", min_value=1, max_value=60, value=30)
        
        submitted = st.form_submit_button("Save Credentials")
        if submitted:
            # In a real app, you'd validate and store these credentials securely
            if api_key and api_secret and endpoint:
                st.session_state.nova_act_connected = True
                st.session_state.last_sync_timestamp = datetime.now()
                show_message("success", "API credentials saved successfully!")
                st.rerun()
            else:
                show_message("error", "Please fill in all required fields")
    
    # Connection history
    st.subheader("Connection History")
    
    # Dummy data for connection history
    history = st.session_state.get('connection_history', [
        {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "status": "Success", "details": "Initial connection"},
        {"timestamp": (datetime.now().replace(hour=datetime.now().hour-1)).strftime("%Y-%m-%d %H:%M:%S"), "status": "Failed", "details": "Network timeout"},
        {"timestamp": (datetime.now().replace(day=datetime.now().day-1)).strftime("%Y-%m-%d %H:%M:%S"), "status": "Success", "details": "Data synchronized"},
    ])
    
    if history:
        history_df = pd.DataFrame(history)
        st.dataframe(history_df, use_container_width=True)
    else:
        st.markdown(
            EMPTY_STATE_HTML.format(
                icon="📊",
                message="No connection history available",
                hint="Connect to your DMS or CRM to start tracking connections"
            ),
            unsafe_allow_html=True
        )
    
    # Manual sync button
    if st.button("Sync Now", type="primary", disabled=not connection_status):
        if connection_status:
            with st.spinner("Synchronizing data..."):
                # Simulate sync operation
                time.sleep(2)
                st.session_state.last_sync_timestamp = datetime.now()
                
                # Add to history
                if 'connection_history' not in st.session_state:
                    st.session_state.connection_history = []
                
                st.session_state.connection_history.insert(0, {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "status": "Success",
                    "details": "Manual sync completed"
                })
                
                show_message("success", "Data synchronized successfully!")
                st.rerun()
        else:
            show_message("error", "Please connect your system first")

def render_settings_tab() -> None:
    """Render the Settings tab for configuring application preferences."""
    # Display status messages
    display_status_messages()
    
    # Apply common CSS
    apply_common_css()
    
    # Create theme instance for styling
    theme = Theme(mode=ColorMode.DARK if st.session_state.theme == "dark" else ColorMode.LIGHT)
    
    # Render section header
    st.markdown(
        SECTION_HEADER_HTML.format(
            icon="⚙️",
            title="Settings",
            description="Configure application preferences and options"
        ),
        unsafe_allow_html=True
    )
    
    # UI Settings Group
    ui_settings_content = ""
    
    # Theme setting
    theme_options = {"light": "Light", "dark": "Dark", "system": "System Default"}
    theme_control = f"""<select id="theme-select" class="stSelectbox">
        {"".join([f'<option value="{k}" {"selected" if st.session_state.theme == k else ""}>{v}</option>' for k, v in theme_options.items()])}
    </select>"""
    
    ui_settings_content += SETTING_ITEM_HTML.format(
        label="Theme",
        description="Choose the application color theme",
        control=theme_control
    )
    
    # Show welcome panel setting
    welcome_panel_control = f"""<input type="checkbox" id="welcome-panel-toggle" {"checked" if st.session_state.show_welcome_panel else ""}>"""
    
    ui_settings_content += SETTING_ITEM_HTML.format(
        label="Show Welcome Panel",
        description="Display the welcome panel on startup",
        control=welcome_panel_control
    )
    
    # Render UI settings group
    st.markdown(
        SETTINGS_GROUP_HTML.format(
            title="User Interface",
            content=ui_settings_content
        ),
        unsafe_allow_html=True
    )
    
    # Data Processing Settings
    st.subheader("Data Processing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        auto_detect = st.checkbox("Auto-detect schema", value=st.session_state.auto_detect_schema)
        if auto_detect != st.session_state.auto_detect_schema:
            st.session_state.auto_detect_schema = auto_detect
            show_message("success", "Settings updated")
            st.rerun()
    
    with col2:
        incremental = st.checkbox("Incremental updates", value=st.session_state.incremental_updates)
        if incremental != st.session_state.incremental_updates:
            st.session_state.incremental_updates = incremental
            show_message("success", "Settings updated")
            st.rerun()
    
    # Data status indicator
    st.subheader("Data Status")
    display_data_status_indicator()
    
    # Data management options
    st.subheader("Data Management")
    
    # Upload history toggle
    show_history = st.checkbox("Show upload history", value=st.session_state.show_upload_history)
    if show_history != st.session_state.show_upload_history:
        st.session_state.show_upload_history = show_history
        st.rerun()
    
    # Clear data button
    if st.button("Clear All Data", type="secondary"):
        if st.session_state.get('validated_data') is not None:
            st.session_state.validated_data = None
            st.session_state.uploaded_files = []
            st.session_state.upload_timestamps = {}
            show_message("success", "All data has been cleared")
            st.rerun()
        else:
            st.info("No data to clear", icon="ℹ️")
def display_data_status_indicator() -> None:
    """Display data status indicator if data is available."""
    # Create theme instance for styling
    theme = Theme(mode=ColorMode.DARK if st.session_state.theme == "dark" else ColorMode.LIGHT)
    
    if st.session_state.get('validated_data') is not None:
        last_update = st.session_state.get('last_sync_timestamp', datetime.now())
        if isinstance(last_update, datetime):
            last_update_str = last_update.strftime("%b %d, %Y %I:%M %p")
        else:
            last_update_str = str(last_update)
            
        st.markdown(f"""
            <div class="data-status-indicator">
                <span class="status-dot active"></span> Data ready for analysis (Last updated: {last_update_str})
            </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No files have been uploaded yet.", icon="ℹ️")

def render_welcome_panel() -> None:
    """Render the welcome panel with overview information."""
    if not st.session_state.get('show_welcome_panel', True):
        return
        
    # Create theme instance for styling
    theme = Theme(mode=ColorMode.DARK if st.session_state.theme == "dark" else ColorMode.LIGHT)
    
    st.markdown(f"""
        <div class="welcome-container">
            <h1 class="welcome-title">Welcome to Watchdog AI</h1>
            <p class="welcome-subtitle">Your dealership intelligence platform</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Close welcome panel button
    if st.button("Hide Welcome Panel", type="secondary", key="close_welcome"):
        st.session_state.show_welcome_panel = False
        st.rerun()

def validate_uploaded_data(data: pd.DataFrame) -> dict:
    """Validate uploaded data for quality and completeness.
    
    Args:
        data: DataFrame containing uploaded data
        
    Returns:
        Dict containing validation results
    """
    validation_result = {
        'is_valid': True,
        'validation_messages': [],
        'schema_errors': [],
        'data_quality_score': 0.0,
        'data_type_issues': [],
        'value_range_issues': [],
        'duplicates': 0,
        'critical_errors': [],
        'warnings': []
    }
    # Basic validation checks
    try:
        # Check for null values
        null_count = data.isnull().sum().sum()
        try:
            null_percentage = (null_count / (data.shape[0] * data.shape[1])) * 100
            validation_result['validation_messages'].append(f"⚠️ Found {null_count} missing values ({null_percentage:.2f}%)")
            
            if null_percentage > 50:
                validation_result['is_valid'] = False
                validation_result['schema_errors'].append("Too many missing values (>50%)")
                validation_result['critical_errors'].append({
                    'error_type': 'missing_values',
                    'message': f"Dataset contains {null_percentage:.2f}% missing values, which exceeds the 50% threshold",
                    'recommendation': "Please clean your data by filling missing values or removing rows/columns with too many nulls."
                })
            elif null_percentage > 20:
                validation_result['warnings'].append({
                    'warning_type': 'missing_values',
                    'message': f"Dataset contains {null_percentage:.2f}% missing values, which may impact analysis quality",
                    'recommendation': "Consider imputing missing values or filtering affected rows for better results."
                })
        except ZeroDivisionError:
            # Handle the case where data is empty
            validation_result['is_valid'] = False
            validation_result['schema_errors'].append("Cannot process empty dataset")
            validation_result['critical_errors'].append({
                'error_type': 'empty_dataset',
                'message': "The uploaded dataset appears to be empty",
                'recommendation': "Please upload a non-empty CSV file with valid data."
            })
            
        # Check for duplicate rows
        duplicate_count = data.duplicated().sum()
        validation_result['duplicates'] = duplicate_count
        
        try:
            if len(data) > 0:  # Guard against empty DataFrame
                if duplicate_count > 0:
                    duplicate_percentage = (duplicate_count / len(data)) * 100
                    validation_result['validation_messages'].append(f"⚠️ Found {duplicate_count} duplicate rows ({duplicate_percentage:.2f}%)")
                    
                    if duplicate_percentage > 20:
                        validation_result['schema_errors'].append("High percentage of duplicate rows (>20%)")
                        validation_result['warnings'].append({
                            'warning_type': 'duplicates',
                            'message': f"Dataset contains {duplicate_percentage:.2f}% duplicate rows",
                            'recommendation': "Consider removing duplicates for more accurate analysis."
                        })
            else:
                validation_result['is_valid'] = False
                validation_result['schema_errors'].append("Dataset contains no rows")
                validation_result['critical_errors'].append({
                    'error_type': 'empty_dataset',
                    'message': "The uploaded dataset contains no rows",
                    'recommendation': "Please upload a valid CSV file with data rows."
                })
        except Exception as e:
            validation_result['schema_errors'].append(f"Error checking for duplicates: {str(e)}")
            logger.error(f"Error checking duplicates: {str(e)}")
            logging.error(traceback.format_exc())
        # Data type consistency checks
        numeric_cols = data.select_dtypes(include=['number', 'float', 'int']).columns.tolist()
        for col in numeric_cols:
            # Check for mixed data types within numeric columns (e.g., strings in numeric columns)
            try:
                pd.to_numeric(data[col])
            except:
                validation_result['data_type_issues'].append(f"Column '{col}' contains mixed data types")
                validation_result['validation_messages'].append(f"⚠️ Column '{col}' contains mixed data types")
            
            # Check for outliers and extreme values
            if len(data[col].dropna()) > 0:
                q1 = data[col].quantile(0.25)
                q3 = data[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - (1.5 * iqr)
                upper_bound = q3 + (1.5 * iqr)
                outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)][col].count()
                
                if outliers > 0:
                    outlier_percentage = (outliers / len(data[col].dropna())) * 100
                    if outlier_percentage > 10:
                        validation_result['value_range_issues'].append(f"Column '{col}' contains {outlier_percentage:.1f}% outliers")
                        validation_result['validation_messages'].append(f"⚠️ Column '{col}' contains {outlier_percentage:.1f}% outliers")
        
        # Check date columns for consistency
        date_cols = []
        for col in data.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    # Attempt to convert to datetime to verify format consistency
                    data[col] = pd.to_datetime(data[col], errors='coerce')
                    null_after_convert = data[col].isna().sum()
                    if null_after_convert > 0:
                        conversion_fail_pct = (null_after_convert / len(data)) * 100
                        validation_result['data_type_issues'].append(f"Column '{col}' has {conversion_fail_pct:.1f}% invalid date formats")
                        validation_result['validation_messages'].append(f"⚠️ Column '{col}' contains {conversion_fail_pct:.1f}% invalid date formats")
                    date_cols.append(col)
                except:
                    pass
        
        # Score the data quality from 0 to 100
        score_components = []
        
        # Missing values component: 100% if no missing values, 0% if all missing
        missing_score = 100 * (1 - (null_count / (data.shape[0] * data.shape[1])))
        score_components.append(missing_score)
        
        # Column presence score
        cols_lower = [col.lower() for col in data.columns]
        expected_cols_present = sum(1 for col in expected_columns if any(exp_col in col_lower for col_lower in cols_lower for exp_col in [col]))
        column_score = 100 * (expected_cols_present / len(expected_columns))
        score_components.append(column_score)
        
        # Row count score (at least 100 rows for good analysis)
        row_score = min(100, (data.shape[0] / 100) * 100)
        score_components.append(row_score)
        
        # Data type consistency score (100% if no issues, less otherwise)
        type_score = 100
        if validation_result['data_type_issues']:
            type_score -= (len(validation_result['data_type_issues']) * 20)  # Deduct 20% per issue
        type_score = max(0, type_score)  # Ensure non-negative
        score_components.append(type_score)
        
        # Duplication score (100% if no duplicates, 0% if all duplicates)
        duplicate_score = 100 * (1 - (validation_result.get('duplicates', 0) / max(1, len(data))))
        score_components.append(duplicate_score)
        
        # Calculate overall score
        validation_result['data_quality_score'] = sum(score_components) / len(score_components)
        
        # Add validation timestamp
        validation_result['validation_timestamp'] = datetime.now()
        
        # Final validation check - minimum required columns and rows
        if len(data.columns) < 2:
            validation_result['is_valid'] = False
            validation_result['critical_errors'].append({
                'error_type': 'insufficient_columns',
                'message': f"Dataset contains only {len(data.columns)} column(s), minimum 2 required for analysis",
                'recommendation': "Please upload a dataset with at least 2 columns for meaningful analysis."
            })
            validation_result['schema_errors'].append(f"Insufficient columns: {len(data.columns)}")
        
        if len(data) < 5:
            validation_result['is_valid'] = False
            validation_result['critical_errors'].append({
                'error_type': 'insufficient_rows',
                'message': f"Dataset contains only {len(data)} row(s), minimum 5 required for analysis",
                'recommendation': "Please upload a dataset with at least 5 rows for meaningful analysis."
            })
            validation_result['schema_errors'].append(f"Insufficient rows: {len(data)}")
            
    except Exception as e:
        validation_result['is_valid'] = False
        error_message = f"❌ Error during validation: {str(e)}"
        validation_result['validation_messages'].append(error_message)
        validation_result['schema_errors'].append(str(e))
        validation_result['data_quality_score'] = 0.0
        validation_result['critical_errors'].append({
            'error_type': 'validation_error',
            'message': error_message,
            'recommendation': "Please check your data format and try again. If the issue persists, contact support."
        })
        
        # Log the full error with traceback for debugging
        logger.error(f"Data validation error: {str(e)}")
        logger.error(traceback.format_exc())
    return validation_result

def generate_sample_analytics(data: pd.DataFrame) -> list:
    """Generate sample analytics from validated data.
    
    
        
        # Time series insights if we have date columns and numeric data
        if date_cols and numeric_cols:
            primary_date = date_cols[0]
            primary_numeric = numeric_cols[0]
            
            # Group by date and calculate statistics
            if len(data) > 0:
                try:
                    # Extract just date part if datetime
                    data['date_part'] = data[primary_date].dt.date
                    grouped = data.groupby('date_part')[primary_numeric].agg(['mean', 'sum', 'count'])
                    
                    insights.append({
                        "type": "time_series",
                        "title": f"{primary_numeric.capitalize()} by {primary_date.capitalize()}",
                        "data": grouped.reset_index().rename(columns={
                            'date_part': 'date',
                            'mean': 'Average',
                            'sum': 'Total',
                            'count': 'Count'
                        }).to_dict('records')
                    })
                except:
                    # If date grouping fails, skip this insight
                    pass
        
        # Most common values for categorical columns
        cat_cols = data.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
        for col in cat_cols[:3]:  # Limit to first 3 categorical columns
            if data[col].nunique() < 20:  # Only if reasonable number of categories
                value_counts = data[col].value_counts().head(5).reset_index()
                value_counts.columns = [col, 'count']
                
                insights.append({
                    "type": "categorical",
                    "title": f"Top {col.capitalize()} Categories",
                    "data": value_counts.to_dict('records')
                })
                
    except Exception as e:
        # Add an error insight
        insights.append({
            "type": "error",
            "title": "Error Generating Insights",
            "data": {"error": str(e)}
        })
        
    return insights

def plot_chart(data: pd.DataFrame, chart_type: str, settings: dict) -> None:
    """Generate and display a chart based on user preferences.
    
    Args:
        data: DataFrame containing validated data
        chart_type: Type of chart to display
        settings: Dictionary of chart settings
        
    Returns:
        None. Displays chart directly using st.plotly_chart
        
    Raises:
        Various exceptions are caught and displayed to the user, not raised.
    """
    # Input validation
    if data is None or len(data) == 0 or len(data.columns) == 0:
        st.error("Cannot generate chart: Empty or invalid dataset provided.", icon="❌")
        return
        
    if not isinstance(chart_type, str) or chart_type.strip() == "":
        st.error("Invalid chart type specified.", icon="❌")
        return
        
    if not isinstance(settings, dict):
        st.error("Invalid chart settings provided.", icon="❌")
        return
    
    try:
        # Find appropriate columns for charts
        numeric_cols = data.select_dtypes(include=['number', 'float', 'int']).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
        date_cols = []
        
        # Try to convert potential date columns
        for col in data.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    data[col] = pd.to_datetime(data[col])
                    date_cols.append(col)
                except:
                    pass
        
        # Default to first columns of each type if available
        x_col = date_cols[0] if date_cols else categorical_cols[0] if categorical_cols else numeric_cols[0] if numeric_cols else None
        y_col = numeric_cols[0] if numeric_cols else None
        color_col = categorical_cols[0] if categorical_cols and len(categorical_cols) > 1 else None
        
        if not numeric_cols:
            st.warning("No numeric columns detected in your dataset. Charts require at least one numeric column for analysis.", icon="⚠️")
            return
            
        if not x_col or not y_col:
            st.warning("Could not identify appropriate columns for charting. Please make sure your data has both numeric and categorical/date columns.", icon="⚠️")
            return
        
        # Create colorscale based on settings
        colorscale = settings.get('color_scheme', 'viridis')
        
        # Create appropriate chart based on chart type
        if chart_type == 'bar':
            if categorical_cols:
                # Group by the categorical column
                if color_col and color_col != x_col:
                    # Stacked bar chart
                    grouped = data.groupby([x_col, color_col])[y_col].sum().reset_index()
                    fig = px.bar(grouped, x=x_col, y=y_col, color=color_col, 
                                title=f"{y_col} by {x_col}", 
                                color_discrete_sequence=px.colors.sequential.get(colorscale, px.colors.sequential.Viridis))
                else:
                    # Simple bar chart
                    grouped = data.groupby(x_col)[y_col].sum().reset_index()
                    fig = px.bar(grouped, x=x_col, y=y_col, 
                                title=f"{y_col} by {x_col}",
                                color_discrete_sequence=px.colors.sequential.get(colorscale, px.colors.sequential.Viridis))
            else:
                st.warning("Bar charts require categorical data. Please select a different chart type.")
                return
            
        elif chart_type == 'line':
            if date_cols:
                # Group by date for time series
                try:
                    data['date_part'] = data[x_col].dt.date
                    grouped = data.groupby('date_part')[y_col].mean().reset_index()
                    fig = px.line(grouped, x='date_part', y=y_col, 
                                title=f"{y_col} over time",
                                color_discrete_sequence=px.colors.sequential.get(colorscale, px.colors.sequential.Viridis))
                except:
                    # If date grouping fails, use regular line chart
                    fig = px.line(data, x=x_col, y=y_col, 
                                title=f"{y_col} by {x_col}",
                                color_discrete_sequence=px.colors.sequential.get(colorscale, px.colors.sequential.Viridis))
            else:
                # Regular line chart
                fig = px.line(data.sort_values(x_col), x=x_col, y=y_col, 
                            title=f"{y_col} by {x_col}",
                            color_discrete_sequence=px.colors.sequential.get(colorscale, px.colors.sequential.Viridis))
                
        elif chart_type == 'scatter':
            if color_col:
                fig = px.scatter(data, x=x_col, y=y_col, color=color_col,
                                title=f"{y_col} vs {x_col} by {color_col}",
                                color_discrete_sequence=px.colors.sequential.get(colorscale, px.colors.sequential.Viridis))
            else:
                fig = px.scatter(data, x=x_col, y=y_col,
                                title=f"{y_col} vs {x_col}",
                                color_discrete_sequence=px.colors.sequential.get(colorscale, px.colors.sequential.Viridis))
                
        elif chart_type == 'histogram':
            fig = px.histogram(data, x=y_col, 
                            title=f"Distribution of {y_col}",
                            color_discrete_sequence=px.colors.sequential.get(colorscale, px.colors.sequential.Viridis))
            
        elif chart_type == 'pie':
            if categorical_cols:
                counts = data[categorical_cols[0]].value_counts().reset_index()
                counts.columns = [categorical_cols[0], 'count']
                fig = px.pie(counts, values='count', names=categorical_cols[0],
                            title=f"Distribution of {categorical_cols[0]}",
                            color_discrete_sequence=px.colors.sequential.get(colorscale, px.colors.sequential.Viridis))
            else:
                st.warning("Pie charts require categorical data.")
                return
            
        elif chart_type == 'box':
            if categorical_cols:
                fig = px.box(data, x=categorical_cols[0], y=y_col,
                            title=f"Distribution of {y_col} by {categorical_cols[0]}",
                            color_discrete_sequence=px.colors.sequential.get(colorscale, px.colors.sequential.Viridis))
            else:
                fig = px.box(data, y=y_col,
                            title=f"Distribution of {y_col}",
                            color_discrete_sequence=px.colors.sequential.get(colorscale, px.colors.sequential.Viridis))
                
        else:
            # Default to bar chart if type not recognized
            grouped = data.groupby(x_col)[y_col].sum().reset_index()
            fig = px.bar(grouped, x=x_col, y=y_col, 
                        title=f"{y_col} by {x_col}",
                        color_discrete_sequence=px.colors.sequential.get(colorscale, px.colors.sequential.Viridis))
        
        # Apply common layout settings
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            showlegend=settings.get('show_legend', True),
            xaxis=dict(showgrid=settings.get('show_grid', True)),
            showlegend=settings.get('show_legend', True),
            xaxis=dict(
                showgrid=settings.get('show_grid', True),
                title=settings.get('x_axis_label', x_col) if settings.get('x_axis_label') else x_col
            ),
            yaxis=dict(
                showgrid=settings.get('show_grid', True),
                title=settings.get('y_axis_label', y_col) if settings.get('y_axis_label') else y_col
            ),
            title=dict(
                text=settings.get('custom_title', f"{y_col} by {x_col}") if settings.get('custom_title') else f"{y_col} by {x_col}",
                font=dict(
                    size=24 if settings.get('font_size') == 'large' else 
                         18 if settings.get('font_size') == 'medium' else 
                         14 if settings.get('font_size') == 'small' else 18
                )
            )
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
        # Add chart export options
        fig_json = fig.to_json()
        st.download_button(
            label="Export Chart Data",
            data=data.to_csv(index=False),
            file_name=f"chart_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            key="export_chart_data"
        )
        
        # Display the chart
        st.plotly_chart(fig, use_container_width=True, config={
            'displayModeBar': True,
            'toImageButtonOptions': {
                'format': 'png',
                'filename': f"chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'height': 800,
                'width': 1200,
                'scale': 2
            }
        })
        
    except KeyError as ke:
        st.error(f"Chart creation failed: Could not find required column '{ke}'", icon="❌")
        st.info("Try selecting a different chart type that's compatible with your data", icon="ℹ️")
        logger.error(f"Chart KeyError: {str(ke)}")
        
    except ValueError as ve:
        st.error(f"Chart creation failed: Invalid value - {str(ve)}", icon="❌")
        st.info("This may be due to incompatible data types for the selected chart", icon="ℹ️")
        logger.error(f"Chart ValueError: {str(ve)}")
        
    except Exception as e:
        st.error(f"Error generating chart: {str(e)}", icon="❌")
        st.info("An unexpected error occurred. Try a different chart type or check your data.", icon="ℹ️")
        
        # Log the detailed error for debugging
        logger.error(f"Chart generation error: {str(e)}")
        logger.error(traceback.format_exc())
def render_analytics_panel(data: pd.DataFrame) -> None:
    """Render analytics panel with insights and visualizations.
    
    Args:
        data: DataFrame containing validated data
    """
    # Generate insights
    insights = generate_sample_analytics(data)
    
    # Display analytics header
    st.markdown(
        SECTION_HEADER_HTML.format(
            icon="📊",
            title="Data Analytics",
            description="Automated insights and visualizations from your data"
        ),
        unsafe_allow_html=True
    )
    
    # Chart selection and customization
    with st.container():
        st.subheader("Visualize Your Data")
        
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            chart_type = st.selectbox(
                "Chart Type",
                options=["bar", "line", "scatter", "histogram", "pie", "box"],
                index=list(["bar", "line", "scatter", "histogram", "pie", "box"]).index(st.session_state.chart_preferences['chart_type']),
                key="chart_type_select"
            )
            if chart_type != st.session_state.chart_preferences['chart_type']:
                st.session_state.chart_preferences['chart_type'] = chart_type
                st.rerun()
        
        with chart_col2:
            color_scheme = st.selectbox(
                "Color Scheme",
                options=["viridis", "plasma", "inferno", "magma", "cividis", "Blues", "Greens", "Reds"],
                index=list(["viridis", "plasma", "inferno", "magma", "cividis", "Blues", "Greens", "Reds"]).index(st.session_state.chart_preferences['color_scheme']),
                key="color_scheme_select"
            )
            if color_scheme != st.session_state.chart_preferences['color_scheme']:
                st.session_state.chart_preferences['color_scheme'] = color_scheme
                st.rerun()
        
        # Advanced chart options in expander
        with st.expander("Chart Options"):
            show_legend = st.checkbox("Show Legend", value=st.session_state.chart_preferences['show_legend'], key="show_legend_cb")
            show_grid = st.checkbox("Show Grid", value=st.session_state.chart_preferences['show_grid'], key="show_grid_cb")
            animation = st.checkbox("Enable Animation", value=st.session_state.chart_preferences['animation'], key="animation_cb")
            
            # Update preferences if changed
            if (show_legend != st.session_state.chart_preferences['show_legend'] or
                show_grid != st.session_state.chart_preferences['show_grid'] or
                animation != st.session_state.chart_preferences['animation']):
                
                st.session_state.chart_preferences['show_legend'] = show_legend
                st.session_state.chart_preferences['show_grid'] = show_grid
                st.session_state.chart_preferences['animation'] = animation
                st.rerun()
    
    # Plot the chart
    plot_chart(data, chart_type, st.session_state.chart_preferences)
    
    # Display insights from data
    if insights:
        st.subheader("Data Insights")
        
        for i, insight in enumerate(insights):
            with st.container():
                if insight['type'] == 'statistic':
                    # Display statistic insight
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Mean", f"{insight['data']['mean']:.2f}")
                    with col2:
                        st.metric("Median", f"{insight['data']['median']:.2f}")
                    with col3:
                        st.metric("Min", f"{insight['data']['min']:.2f}")
                    with col4:
                        st.metric("Max", f"{insight['data']['max']:.2f}")
                    
                elif insight['type'] == 'time_series':
                    # Display time series insight
                    st.subheader(insight['title'])
                    if len(insight['data']) > 0:
                        ts_df = pd.DataFrame(insight['data'])
                        st.line_chart(ts_df.set_index('date')[['Total', 'Average']])
                        
                elif insight['type'] == 'categorical':
                    # Display categorical insight
                    st.subheader(insight['title'])
                    if len(insight['data']) > 0:
                        cat_df = pd.DataFrame(insight['data'])
                        st.bar_chart(cat_df.set_index(cat_df.columns[0]))
                
                elif insight['type'] == 'error':
                    # Display error
                    st.error(f"Error in insight generation: {insight['data']['error']}")
                
                st.markdown("---")
    else:
        st.info("No insights were generated. Try uploading more comprehensive data.")


def render_insight_engine_tab() -> None:
    """Render the Insight Engine tab with chat interface and insights display."""
    # Display any pending status messages
    display_status_messages()
    
    # Apply common CSS
    apply_common_css()
    
    # Create theme instance for styling
    theme = Theme(mode=ColorMode.DARK if st.session_state.theme == "dark" else ColorMode.LIGHT)
    
    # Display welcome panel if enabled
    render_welcome_panel()
    
    # All common CSS is already applied by apply_common_css()
    
    # Data Upload Section
    with st.container():
        st.markdown("""
            <div class="section-header">
                <h3>📤 Data Upload</h3>
                <p class="section-description">Upload your dealership data or connect to your CRM</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Render data uploader
        render_data_uploader()
        
        # Add validation status indicators if data has been uploaded
        if 'uploaded_files' in st.session_state and st.session_state.uploaded_files:
            # Use the existing spinner to indicate processing state
            if st.session_state.get('loading_data', False):
                with st.spinner("Processing and validating your data..."):
                    # Show a progress bar for validation progress
                    progress_bar = st.progress(0)
                    for i in range(100):
                        # Simulate progress for demo purposes
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    
                    # After processing, update validation status
                    if 'validated_data' in st.session_state and st.session_state.validated_data is not None:
                        # Validate the data
                        validation_result = validate_uploaded_data(st.session_state.validated_data)
                        
                        # Store validation results in session state
                        st.session_state.validation_status = validation_result
                        
                        # Clear loading flag
                        st.session_state.loading_data = False
                        st.rerun()
            
            if 'validation_status' in st.session_state:
                status = st.session_state.validation_status
                
                # Create container for validation status
                with st.container():
                    st.markdown("### Data Validation Results")
                    
                    # Quality score with gauge
                    quality_score = status.get('data_quality_score', 0)
                    quality_color = "green" if quality_score > 80 else "orange" if quality_score > 50 else "red"
                    
                    st.metric(
                        "Data Quality Score", 
                        f"{quality_score:.1f}%",
                        delta=None,
                        delta_color="normal"
                    )
                    
                    # Status indicator
                    if status.get('is_valid', False):
                        st.success("✅ Data validation successful", icon="✅")
                    else:
                        st.error("❌ Data validation failed", icon="❌")
                    
                    # Show validation messages in an expander
                    if status.get('validation_messages', []):
                        with st.expander("Validation Messages"):
                            for msg in status['validation_messages']:
                                st.markdown(msg)
                    
                    # Show schema errors if any
                    if status.get('schema_errors', []):
                        with st.expander("Schema Errors"):
                            for error in status['schema_errors']:
                                st.error(error)
                    
                    # Timestamp of validation
                    if status.get('validation_timestamp', None):
                        timestamp = status['validation_timestamp']
                        st.caption(f"Last validated: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Only show analytics and chat if data is validated and valid
    if ('validation_status' in st.session_state and 
        st.session_state.validation_status.get('is_valid', False) and 
        'validated_data' in st.session_state and 
        st.session_state.validated_data is not None):
        
        # Add tabs for Analytics and Chat
        analytics_tab, chat_tab = st.tabs(["📊 Analytics", "💬 Chat Interface"])
        
        with analytics_tab:
            # Render analytics panel with the validated data
            render_analytics_panel(st.session_state.validated_data)
        
        with chat_tab:
            # Render chat interface section
            st.markdown("""
                <div class="section-header">
                    <h3>💬 Chat Interface</h3>
                    <p class="section-description">Ask questions about your data and get AI-powered insights</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Initialize the chat interface with the data from validated_data
            chat_interface = ChatInterface(data=st.session_state.validated_data)
            
            # Render the chat interface
            chat_interface.render()
    
    else:
        # If no validated data or validation failed, show empty state
        if (not 'validation_status' in st.session_state or 
            not st.session_state.validation_status.get('is_valid', False)):
            
            # Only show this if we've attempted to validate data
            if 'validation_status' in st.session_state:
                st.warning("Please correct the validation errors before proceeding to analytics and insights.")
            
            # Show empty state for analytics and chat
            st.markdown(
                EMPTY_STATE_HTML.format(
                    icon="📊",
                    message="No validated data available for analysis",
                    hint="Upload and validate your data to see analytics and insights"
                ),
                unsafe_allow_html=True
            )

def render_app() -> None:
    try:
        # Initialize session state
        initialize_session_state()
        
        # Configure page
        st.set_page_config(**PAGE_CONFIG)
        
        # Render header
        render_header()
        
        # Main navigation
        tabs = st.tabs(TABS)
        
        # Render active tab content
        tab_renderers = {
            "Insight Engine": render_insight_engine_tab,
            "System Connect": render_system_connect_tab,
            "Settings": render_settings_tab
        }
        
        with tabs[TABS.index(st.session_state.active_tab)]:
            tab_renderers[st.session_state.active_tab]()
            
    except Exception as e:
        logger.error(f"Error rendering app: {str(e)}")
        st.error(
            "An error occurred while rendering the application. Please try again or contact support.",
            icon="🚫"
        )

if __name__ == "__main__":
    render_app()
