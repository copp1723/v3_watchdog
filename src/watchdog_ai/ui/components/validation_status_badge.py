"""
Validation status badge component for showing data validation status.

This component provides visual indicators of validation status (success/warning/error)
with appropriate colors, icons, and tooltips to display validation messages.
"""

import streamlit as st
from typing import Dict, Any, Optional, List, Tuple, Union
import pandas as pd

from watchdog_ai.ui.utils.ui_theme import ColorSystem, Typography, Spacing

# Validation status types and their corresponding colors/icons
VALIDATION_STATUS = {
    "success": {
        "color": ColorSystem.SECONDARY.get(500),  # Green
        "icon": "✓",
        "label": "Success",
        "description": "Validation passed without issues"
    },
    "warning": {
        "color": ColorSystem.WARNING.get(500),  # Yellow
        "icon": "⚠️",
        "label": "Warning",
        "description": "Validation completed with warnings"
    },
    "error": {
        "color": ColorSystem.ALERT.get(500),  # Red
        "icon": "✗",
        "label": "Error",
        "description": "Validation failed with errors"
    }
}

class ValidationStatusBadge:
    """
    Component for displaying validation status badges.
    Provides visual indicators with tooltips for validation results.
    """
    
    def __init__(self):
        """Initialize the validation status badge component."""
        # Apply CSS styles
        self._apply_css()
    
    def _apply_css(self):
        """Apply CSS styles for the validation badges."""
        st.markdown("""
        <style>
            .validation-badge {
                display: flex;
                align-items: center;
                justify-content: center;
                border-radius: 12px;
                padding: 2px 8px;
                font-size: 12px;
                font-weight: 500;
                min-width: 80px;
                text-align: center;
                cursor: help;
                transition: all 0.2s ease;
                margin-right: 8px;
            }
            
            .validation-badge-icon {
                margin-right: 4px;
            }
            
            /* Badge sizes */
            .validation-badge-sm {
                font-size: 10px;
                min-width: 60px;
                padding: 1px 6px;
            }
            
            .validation-badge-lg {
                font-size: 14px;
                min-width: 100px;
                padding: 3px 10px;
            }
            
            /* Badge types */
            .validation-badge-success {
                background-color: rgba(16, 185, 129, 0.1);
                color: rgb(16, 185, 129);
                border: 1px solid rgba(16, 185, 129, 0.2);
            }
            
            .validation-badge-warning {
                background-color: rgba(245, 158, 11, 0.1);
                color: rgb(245, 158, 11);
                border: 1px solid rgba(245, 158, 11, 0.2);
            }
            
            .validation-badge-error {
                background-color: rgba(239, 68, 68, 0.1);
                color: rgb(239, 68, 68);
                border: 1px solid rgba(239, 68, 68, 0.2);
            }
            
            /* Tooltip container */
            .validation-tooltip {
                position: relative;
                display: inline-block;
            }
            
            /* Tooltip text */
            .validation-tooltip .tooltip-text {
                visibility: hidden;
                background-color: rgba(0, 0, 0, 0.8);
                color: #fff;
                text-align: left;
                padding: 8px 12px;
                border-radius: 6px;
                position: absolute;
                z-index: 10;
                bottom: 125%;
                left: 50%;
                transform: translateX(-50%);
                white-space: nowrap;
                opacity: 0;
                transition: opacity 0.3s;
                font-size: 12px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
                min-width: 200px;
            }
            
            /* Tooltip arrow */
            .validation-tooltip .tooltip-text::after {
                content: "";
                position: absolute;
                top: 100%;
                left: 50%;
                margin-left: -5px;
                border-width: 5px;
                border-style: solid;
                border-color: rgba(0, 0, 0, 0.8) transparent transparent transparent;
            }
            
            /* Show tooltip on hover */
            .validation-tooltip:hover .tooltip-text {
                visibility: visible;
                opacity: 1;
            }
            
            /* Tooltip title */
            .tooltip-title {
                font-weight: 600;
                margin-bottom: 4px;
                font-size: 13px;
            }
            
            /* Tooltip message */
            .tooltip-message {
                font-size: 12px;
                color: rgba(255, 255, 255, 0.9);
            }
            
            /* Tooltip list */
            .tooltip-list {
                margin-top: 4px;
                margin-bottom: 0;
                padding-left: 20px;
            }
            
            .tooltip-list li {
                margin-bottom: 2px;
            }
            
            /* Add a bit of space after Streamlit components */
            div.stButton, div.row-widget {
                margin-bottom: 8px;
            }
        </style>
        """, unsafe_allow_html=True)
    
    def render(self, 
              status: str, 
              message: Optional[str] = None, 
              details: Optional[List[str]] = None, 
              count: Optional[int] = None,
              size: str = "medium") -> None:
        """
        Render a validation status badge.
        
        Args:
            status: Status type ('success', 'warning', or 'error')
            message: Optional custom message to display in tooltip
            details: Optional list of details to display in tooltip
            count: Optional count to display (e.g., number of issues)
            size: Badge size ('small', 'medium', 'large')
        """
        # Ensure valid status
        if status not in VALIDATION_STATUS:
            status = "error"
            message = message or "Invalid status type"
        
        # Get status config
        status_config = VALIDATION_STATUS[status]
        
        # Determine size class
        size_class = ""
        if size == "small":
            size_class = "validation-badge-sm"
        elif size == "large":
            size_class = "validation-badge-lg"
        
        # Create badge text
        badge_text = status_config["label"]
        if count is not None:
            badge_text = f"{badge_text} ({count})"
        
        # Create tooltip content
        tooltip_title = message or status_config["description"]
        tooltip_details = ""
        
        if details and len(details) > 0:
            tooltip_details = "<ul class='tooltip-list'>"
            for detail in details:
                tooltip_details += f"<li>{detail}</li>"
            tooltip_details += "</ul>"
        
        # Create tooltip HTML
        tooltip_html = f"""
        <div class="tooltip-title">{tooltip_title}</div>
        <div class="tooltip-message">{status_config.get('description', '')}</div>
        {tooltip_details}
        """
        
        # Create badge HTML
        badge_html = f"""
        <div class="validation-tooltip">
            <div class="validation-badge validation-badge-{status} {size_class}">
                <span class="validation-badge-icon">{status_config['icon']}</span>
                <span>{badge_text}</span>
            </div>
            <span class="tooltip-text">{tooltip_html}</span>
        </div>
        """
        
        # Render badge
        st.markdown(badge_html, unsafe_allow_html=True)
    
    def render_multiple(self, validation_results: Dict[str, Any]) -> None:
        """
        Render multiple validation badges based on validation results.
        
        Args:
            validation_results: Dictionary with validation results
        """
        # Extract validation statuses
        statuses = []
        
        # Check for errors
        if 'errors' in validation_results and validation_results['errors']:
            error_count = len(validation_results['errors'])
            error_details = [error.get('message', 'Unknown error') for error in validation_results['errors']]
            statuses.append(('error', 'Validation errors found', error_details, error_count))
        
        # Check for warnings
        if 'warnings' in validation_results and validation_results['warnings']:
            warning_count = len(validation_results['warnings'])
            warning_details = [warning.get('message', 'Unknown warning') for warning in validation_results['warnings']]
            statuses.append(('warning', 'Validation warnings found', warning_details, warning_count))
        
        # If no errors or warnings, show success
        if not statuses:
            statuses.append(('success', 'All validation checks passed', None, None))
        
        # Render all status badges
        for status, message, details, count in statuses:
            self.render(status, message, details, count)
    
    def render_column_badges(self, df: pd.DataFrame, validation_columns: Dict[str, Dict[str, Any]]) -> None:
        """
        Render validation badges for specific DataFrame columns.
        
        Args:
            df: DataFrame with data to validate
            validation_columns: Dictionary mapping column names to validation specs
                Each spec should contain:
                - 'status': Status to show ('success', 'warning', 'error')
                - 'message': Optional custom message
                - 'details': Optional list of details
        """
        # Create expander for column validation results
        with st.expander("Column Validation Details", expanded=False):
            # Create a data grid to display column validation results
            columns = list(validation_columns.keys())
            
            if not columns:
                st.info("No column validation specifications provided.")
                return
            
            for col_name, validation_spec in validation_columns.items():
                # Check if column exists
                if col_name not in df.columns:
                    st.warning(f"Column '{col_name}' not found in DataFrame.")
                    continue
                
                # Get column stats
                non_null_count = df[col_name].notna().sum()
                null_count = df[col_name].isna().sum()
                unique_count = df[col_name].nunique()
                
                # Create a row for each column
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.markdown(f"**{col_name}**")
                    st.text(f"Type: {df[col_name].dtype}")
                
                with col2:
                    st.text(f"Non-null: {non_null_count}/{len(df)}")
                    st.text(f"Unique: {unique_count}")
                
                with col3:
                    status = validation_spec.get('status', 'success')
                    message = validation_spec.get('message', f"Validation for '{col_name}'")
                    details = validation_spec.get('details', [])
                    count = validation_spec.get('count', None)
                    
                    self.render(status, message, details, count, size="small")
                
                # Add separator
                st.markdown("---")


def create_validation_badge(
    status: str, 
    message: Optional[str] = None, 
    details: Optional[List[str]] = None, 
    count: Optional[int] = None,
    size: str = "medium"
) -> None:
    """
    Create and render a validation status badge.
    
    Args:
        status: Status type ('success', 'warning', or 'error')
        message: Optional custom message to display in tooltip
        details: Optional list of details to display in tooltip
        count: Optional count to display (e.g., number of issues)
        size: Badge size ('small', 'medium', 'large')
    """
    badge = ValidationStatusBadge()
    badge.render(status, message, details, count, size)


def render_validation_summary(validation_results: Dict[str, Any]) -> None:
    """
    Render a summary of validation results with appropriate badges.
    
    Args:
        validation_results: Dictionary with validation results
    """
    badge = ValidationStatusBadge()
    badge.render_multiple(validation_results)


def render_flag_column_badges(df: pd.DataFrame) -> None:
    """
    Render validation badges based on flag columns in the DataFrame.
    
    Args:
        df: DataFrame with flag columns (prefix 'flag_')
    """
    # Identify flag columns
    flag_columns = [col for col in df.columns if col.startswith('flag_')]
    
    if not flag_columns:
        st.info("No validation flag columns found in the DataFrame.")
        return
    
    # Create validation column specs
    validation_columns = {}
    
    for flag_col in flag_columns:
        # Get the flag count
        flag_count = int(df[flag_col].sum())
        
        # Skip if no flags
        if flag_count == 0:
            continue
        
        # Format validation spec
        validation_name = flag_col.replace('flag_', '').replace('_', ' ').title()
        
        validation_columns[flag_col] = {
            'status': 'error' if flag_count > 0 else 'success',
            'message': f"{validation_name} Validation",
            'details': [f"{flag_count} records flagged with {validation_name}"],
            'count': flag_count
        }
    
    # Render badges
    badge = ValidationStatusBadge()
    badge.render_column_badges(df, validation_columns)


def convert_validation_results_to_badges(validation_results: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Convert validation results to badge specifications.
    
    Args:
        validation_results: Validation results dictionary
        
    Returns:
        Dictionary mapping validation keys to badge specifications
    """
    badge_specs = {}
    
    # Process errors
    if 'errors' in validation_results and validation_results['errors']:
        for error in validation_results['errors']:
            key = error.get('rule', 'error')
            badge_specs[key] = {
                'status': 'error',
                'message': error.get('message', 'Validation error'),
                'details': [f"Row {row['row']}: {row['message']}" for row in error.get('invalid_rows', [])],
                'count': len(error.get('invalid_rows', []))
            }
    
    # Process warnings
    if 'warnings' in validation_results and validation_results['warnings']:
        for warning in validation_results['warnings']:
            key = warning.get('rule', 'warning')
            badge_specs[key] = {
                'status': 'warning',
                'message': warning.get('message', 'Validation warning'),
                'details': [f"Row {row['row']}: {row['message']}" for row in warning.get('invalid_rows', [])],
                'count': len(warning.get('invalid_rows', []))
            }

    return badge_specs
