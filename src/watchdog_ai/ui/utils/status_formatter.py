"""
Status text formatter utility for consistent status indicators across the UI.

This module provides functions and enums for formatting status text consistently,
replacing emoji characters with plain text alternatives to avoid encoding issues.
"""

import streamlit as st
from enum import Enum, auto
from typing import Optional, Dict, Any, Tuple


class StatusType(Enum):
    """Enum defining various status types for consistent formatting."""
    SUCCESS = auto()
    ERROR = auto()
    WARNING = auto()
    INFO = auto()
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    ACTIVE = auto()
    INACTIVE = auto()
    ON = auto()
    OFF = auto()
    UNKNOWN = auto()


class StatusFormatter:
    """
    Utility class for formatting status text consistently across the UI.
    Replaces emoji characters with plain text alternatives.
    """
    
    # Mapping of status types to their display properties
    STATUS_CONFIG = {
        StatusType.SUCCESS: {
            "text": "SUCCESS",
            "color": "#10B981",  # Green
            "description": "Operation completed successfully"
        },
        StatusType.ERROR: {
            "text": "ERROR",
            "color": "#EF4444",  # Red
            "description": "An error occurred during the operation"
        },
        StatusType.WARNING: {
            "text": "WARNING",
            "color": "#F59E0B",  # Yellow/Orange
            "description": "Operation completed with warnings"
        },
        StatusType.INFO: {
            "text": "INFO",
            "color": "#3B82F6",  # Blue
            "description": "Informational status"
        },
        StatusType.PENDING: {
            "text": "PENDING",
            "color": "#9CA3AF",  # Gray
            "description": "Operation is pending"
        },
        StatusType.RUNNING: {
            "text": "RUNNING",
            "color": "#3B82F6",  # Blue
            "description": "Operation is in progress"
        },
        StatusType.COMPLETED: {
            "text": "COMPLETED",
            "color": "#10B981",  # Green
            "description": "Operation has completed"
        },
        StatusType.FAILED: {
            "text": "FAILED",
            "color": "#EF4444",  # Red
            "description": "Operation has failed"
        },
        StatusType.ACTIVE: {
            "text": "ACTIVE",
            "color": "#10B981",  # Green
            "description": "Item is active"
        },
        StatusType.INACTIVE: {
            "text": "INACTIVE",
            "color": "#9CA3AF",  # Gray
            "description": "Item is inactive"
        },
        StatusType.ON: {
            "text": "ON",
            "color": "#10B981",  # Green
            "description": "Feature or system is on"
        },
        StatusType.OFF: {
            "text": "OFF",
            "color": "#9CA3AF",  # Gray
            "description": "Feature or system is off"
        },
        StatusType.UNKNOWN: {
            "text": "UNKNOWN",
            "color": "#9CA3AF",  # Gray
            "description": "Status is unknown"
        }
    }
    
    @staticmethod
    def format_status(
        status_type: StatusType, 
        include_brackets: bool = True,
        custom_text: Optional[str] = None,
        use_color: bool = True
    ) -> str:
        """
        Format a status indicator as plain text, optionally with markdown color.
        
        Args:
            status_type: The type of status to format
            include_brackets: Whether to include square brackets around the status text
            custom_text: Optional custom text to display instead of the default
            use_color: Whether to apply color formatting using markdown
            
        Returns:
            Formatted status text with optional color formatting
        """
        # Get status configuration
        config = StatusFormatter.STATUS_CONFIG.get(status_type)
        if not config:
            config = StatusFormatter.STATUS_CONFIG.get(StatusType.UNKNOWN)
        
        # Get text (either custom or from config)
        text = custom_text if custom_text else config["text"]
        
        # Add brackets if requested
        if include_brackets:
            text = f"[{text}]"
        
        # Apply color formatting if requested
        if use_color:
            text = f"<span style='color:{config['color']}'>{text}</span>"
            
        return text
    
    @staticmethod
    def format_status_badge(
        status_type: StatusType,
        custom_text: Optional[str] = None,
        include_brackets: bool = True
    ) -> str:
        """
        Format a status badge with consistent styling.
        Returns HTML that must be rendered with unsafe_allow_html=True.
        
        Args:
            status_type: The type of status to format
            custom_text: Optional custom text to display instead of the default
            include_brackets: Whether to include square brackets around the status text
            
        Returns:
            HTML string for a styled status badge
        """
        # Get status configuration
        config = StatusFormatter.STATUS_CONFIG.get(status_type)
        if not config:
            config = StatusFormatter.STATUS_CONFIG.get(StatusType.UNKNOWN)
            
        # Get text (either custom or from config)
        text = custom_text if custom_text else config["text"]
        
        # Add brackets if requested
        if include_brackets:
            text = f"[{text}]"
            
        # Create badge HTML
        badge_html = f"""
        <span style="
            background-color: {config['color']}1A;
            color: {config['color']};
            border: 1px solid {config['color']}33;
            border-radius: 4px;
            padding: 2px 8px;
            font-size: 12px;
            font-weight: 500;
            white-space: nowrap;
        ">{text}</span>
        """
        
        return badge_html


def format_status_text(
    status_type: StatusType, 
    include_brackets: bool = True,
    custom_text: Optional[str] = None,
    use_color: bool = True
) -> str:
    """
    Format a status indicator as plain text, optionally with markdown color.
    
    Args:
        status_type: The type of status to format
        include_brackets: Whether to include square brackets around the status text
        custom_text: Optional custom text to display instead of the default
        use_color: Whether to apply color formatting using markdown
        
    Returns:
        Formatted status text with optional color formatting
    """
    return StatusFormatter.format_status(
        status_type, 
        include_brackets=include_brackets,
        custom_text=custom_text,
        use_color=use_color
    )


def format_status_badge(
    status_type: StatusType,
    custom_text: Optional[str] = None,
    include_brackets: bool = True
) -> str:
    """
    Format a status badge with consistent styling.
    Returns HTML that must be rendered with unsafe_allow_html=True.
    
    Args:
        status_type: The type of status to format
        custom_text: Optional custom text to display instead of the default
        include_brackets: Whether to include square brackets around the status text
        
    Returns:
        HTML string for a styled status badge
    """
    return StatusFormatter.format_status_badge(
        status_type, 
        custom_text=custom_text,
        include_brackets=include_brackets
    )


def render_status_badge(
    status_type: StatusType,
    custom_text: Optional[str] = None,
    include_brackets: bool = True
) -> None:
    """
    Render a status badge directly in the Streamlit UI.
    
    Args:
        status_type: The type of status to format
        custom_text: Optional custom text to display instead of the default
        include_brackets: Whether to include square brackets around the status text
    """
    badge_html = format_status_badge(
        status_type, 
        custom_text=custom_text, 
        include_brackets=include_brackets
    )
    st.markdown(badge_html, unsafe_allow_html=True)


def get_status_from_bool(value: bool, success_type: StatusType = StatusType.SUCCESS, 
                         failure_type: StatusType = StatusType.ERROR) -> StatusType:
    """
    Convert a boolean value to a status type.
    
    Args:
        value: Boolean value to convert
        success_type: Status type to use for True values
        failure_type: Status type to use for False values
        
    Returns:
        Appropriate StatusType based on the boolean value
    """
    return success_type if value else failure_type


# Common emoji to status type mappings for reference when replacing emoji
EMOJI_TO_STATUS = {
    "✅": StatusType.SUCCESS,
    "✓": StatusType.SUCCESS,
    "❌": StatusType.ERROR,
    "✗": StatusType.ERROR,
    "⚠️": StatusType.WARNING,
    "🟢": StatusType.ON,
    "🔴": StatusType.OFF,
    "🚀": StatusType.RUNNING,
    "⏳": StatusType.PENDING,
    "⌛": StatusType.PENDING,
    "🔍": StatusType.INFO,
    "ℹ️": StatusType.INFO
}

