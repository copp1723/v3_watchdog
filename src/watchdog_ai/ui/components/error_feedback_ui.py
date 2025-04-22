"""
Enhanced error feedback UI component for CSV imports.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
import json
import logging
from datetime import datetime

from ...utils.upload_tracker import UploadTracker, UploadRecord

logger = logging.getLogger(__name__)

class ErrorFeedbackUI:
    """
    Enhanced error feedback UI component for CSV imports.
    Provides detailed error reporting, validation summaries, and suggestions.
    """
    
    def __init__(self, upload_tracker: Optional[UploadTracker] = None):
        """
        Initialize the error feedback UI.
        
        Args:
            upload_tracker: Optional UploadTracker instance
        """
        self.upload_tracker = upload_tracker or UploadTracker()
        
        # Initialize session state for error tracking
        if 'csv_import_errors' not in st.session_state:
            st.session_state.csv_import_errors = []
        
        if 'csv_import_warnings' not in st.session_state:
            st.session_state.csv_import_warnings = []
    
    def add_error(self, error_type: str, message: str, 
                details: Optional[Dict[str, Any]] = None,
                file_name: Optional[str] = None,
                record_id: Optional[str] = None) -> None:
        """
        Add an error to the session state.
        
        Args:
            error_type: Type of error
            message: Error message
            details: Optional error details
            file_name: Optional file name
            record_id: Optional upload record ID
        """
        error = {
            "type": error_type,
            "message": message,
            "details": details or {},
            "timestamp": datetime.now().isoformat(),
            "file_name": file_name,
            "record_id": record_id
        }
        
        st.session_state.csv_import_errors.append(error)
        logger.error(f"CSV import error: {message}")
    
    def add_warning(self, warning_type: str, message: str, 
                  details: Optional[Dict[str, Any]] = None,
                  file_name: Optional[str] = None,
                  record_id: Optional[str] = None) -> None:
        """
        Add a warning to the session state.
        
        Args:
            warning_type: Type of warning
            message: Warning message
            details: Optional warning details
            file_name: Optional file name
            record_id: Optional upload record ID
        """
        warning = {
            "type": warning_type,
            "message": message,
            "details": details or {},
            "timestamp": datetime.now().isoformat(),
            "file_name": file_name,
            "record_id": record_id
        }
        
        st.session_state.csv_import_warnings.append(warning)
        logger.warning(f"CSV import warning: {message}")
    
    def clear_errors(self) -> None:
        """Clear all errors from the session state."""
        st.session_state.csv_import_errors = []
    
    def clear_warnings(self) -> None:
        """Clear all warnings from the session state."""
        st.session_state.csv_import_warnings = []
    
    def render_error_summary(self) -> None:
        """Render a summary of errors and warnings."""
        errors = st.session_state.csv_import_errors
        warnings = st.session_state.csv_import_warnings
        
        if not errors and not warnings:
            return
        
        st.markdown("### Import Issues")
        
        # Group errors by type
        error_types = {}
        for error in errors:
            error_type = error["type"]
            if error_type not in error_types:
                error_types[error_type] = []
            error_types[error_type].append(error)
        
        # Display errors by type
        if errors:
            st.error(f"Found {len(errors)} errors during import")
            
            for error_type, type_errors in error_types.items():
                with st.expander(f"{error_type} Errors ({len(type_errors)})"):
                    for i, error in enumerate(type_errors):
                        st.markdown(f"**Error {i+1}:** {error['message']}")
                        
                        if error.get("file_name"):
                            st.caption(f"File: {error['file_name']}")
                        
                        if error.get("details"):
                            with st.expander("Details"):
                                st.json(error["details"])
        
        # Group warnings by type
        warning_types = {}
        for warning in warnings:
            warning_type = warning["type"]
            if warning_type not in warning_types:
                warning_types[warning_type] = []
            warning_types[warning_type].append(warning)
        
        # Display warnings by type
        if warnings:
            st.warning(f"Found {len(warnings)} warnings during import")
            
            for warning_type, type_warnings in warning_types.items():
                with st.expander(f"{warning_type} Warnings ({len(type_warnings)})"):
                    for i, warning in enumerate(type_warnings):
                        st.markdown(f"**Warning {i+1}:** {warning['message']}")
                        
                        if warning.get("file_name"):
                            st.caption(f"File: {warning['file_name']}")
                        
                        if warning.get("details"):
                            with st.expander("Details"):
                                st.json(warning["details"])
        
        # Clear errors/warnings button
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear Errors"):
                self.clear_errors()
                st.rerun()
        
        with col2:
            if st.button("Clear Warnings"):
                self.clear_warnings()
                st.rerun()
    
    def render_validation_results(self, validation_results: Dict[str, Any],
                                file_name: Optional[str] = None) -> None:
        """
        Render validation results for a file.
        
        Args:
            validation_results: Validation results dictionary
            file_name: Optional file name
        """
        if not validation_results:
            return
        
        st.markdown("### Validation Results")
        
        # File info
        if "file_info" in validation_results:
            file_info = validation_results["file_info"]
            
            st.markdown("**File Information**")
            info_cols = st.columns(3)
            
            with info_cols[0]:
                st.metric("Total Rows", file_info.get("total_rows", 0))
            
            with info_cols[1]:
                st.metric("Total Columns", file_info.get("total_columns", 0))
            
            with info_cols[2]:
                memory_mb = file_info.get("memory_usage", 0)
                st.metric("Memory Usage", f"{memory_mb:.2f} MB")
        
        # Column info
        if "column_info" in validation_results:
            st.markdown("**Column Information**")
            
            # Create DataFrame for display
            column_data = []
            for col_info in validation_results["column_info"]:
                column_data.append({
                    "Column": col_info["name"],
                    "Type": col_info["type"],
                    "Non-Null": col_info["non_null_count"],
                    "Null": col_info["null_count"],
                    "Unique": col_info["unique_count"],
                    "Sample Values": ", ".join(str(v) for v in col_info["sample_values"][:2])
                })
            
            if column_data:
                column_df = pd.DataFrame(column_data)
                st.dataframe(column_df, use_container_width=True)
        
        # Data quality
        if "data_quality" in validation_results:
            st.markdown("**Data Quality**")
            
            quality = validation_results["data_quality"]
            
            # Completeness
            if "completeness" in quality:
                st.markdown("**Completeness (% of non-null values)**")
                
                completeness_data = []
                for col, value in quality["completeness"].items():
                    completeness_data.append({
                        "Column": col,
                        "Completeness": f"{value:.1%}"
                    })
                
                if completeness_data:
                    completeness_df = pd.DataFrame(completeness_data)
                    st.dataframe(completeness_df, use_container_width=True)
            
            # Unique ratios
            if "unique_ratios" in quality:
                st.markdown("**Uniqueness (% of unique values)**")
                
                uniqueness_data = []
                for col, value in quality["unique_ratios"].items():
                    uniqueness_data.append({
                        "Column": col,
                        "Uniqueness": f"{value:.1%}"
                    })
                
                if uniqueness_data:
                    uniqueness_df = pd.DataFrame(uniqueness_data)
                    st.dataframe(uniqueness_df, use_container_width=True)
    
    def render_mapping_issues(self, mapping_results: Dict[str, Any],
                            file_name: Optional[str] = None,
                            on_apply_mapping: Optional[Callable] = None) -> None:
        """
        Render column mapping issues and suggestions.
        
        Args:
            mapping_results: Mapping results dictionary
            file_name: Optional file name
            on_apply_mapping: Optional callback for applying a mapping
        """
        if not mapping_results:
            return
        
        # Check for unmapped columns
        unmapped_columns = mapping_results.get("unmapped_columns", [])
        if not unmapped_columns:
            return
        
        st.markdown("### Column Mapping Issues")
        st.warning(f"Found {len(unmapped_columns)} unmapped columns")
        
        # Get mapping suggestions
        suggestions = mapping_results.get("mapping_suggestions", [])
        suggestion_map = {s["source_column"]: s for s in suggestions}
        
        # Display unmapped columns with suggestions
        for i, column in enumerate(unmapped_columns):
            with st.expander(f"Unmapped Column: {column}"):
                suggestion = suggestion_map.get(column)
                
                if suggestion:
                    st.markdown(f"**Suggestion:** Map to '{suggestion['target_column']}'")
                    st.caption(f"Confidence: {suggestion['confidence']:.1%}")
                    st.caption(f"Reason: {suggestion['reason']}")
                    
                    # Show alternatives if available
                    alternatives = suggestion.get("alternatives", [])
                    if alternatives:
                        st.markdown("**Alternatives:**")
                        for alt in alternatives:
                            st.markdown(f"- {alt}")
                    
                    # Apply mapping button
                    if on_apply_mapping:
                        if st.button(f"Apply Suggested Mapping", key=f"apply_mapping_{i}"):
                            on_apply_mapping(column, suggestion["target_column"])
                            st.success(f"Applied mapping: {column} → {suggestion['target_column']}")
                else:
                    st.info("No mapping suggestion available for this column")
    
    def render_error_details(self, error_details: Dict[str, Any],
                           file_name: Optional[str] = None) -> None:
        """
        Render detailed error information.
        
        Args:
            error_details: Error details dictionary
            file_name: Optional file name
        """
        if not error_details:
            return
        
        st.markdown("### Error Details")
        
        # Error message
        if "message" in error_details:
            st.error(error_details["message"])
        
        # Error location
        if "location" in error_details:
            location = error_details["location"]
            
            if "row" in location and "column" in location:
                st.markdown(f"**Location:** Row {location['row']}, Column {location['column']}")
            elif "row" in location:
                st.markdown(f"**Location:** Row {location['row']}")
            elif "column" in location:
                st.markdown(f"**Location:** Column {location['column']}")
        
        # Error context
        if "context" in error_details:
            st.markdown("**Context:**")
            st.code(error_details["context"])
        
        # Suggestions
        if "suggestions" in error_details:
            st.markdown("**Suggestions:**")
            for suggestion in error_details["suggestions"]:
                st.markdown(f"- {suggestion}")
    
    def render_upload_history(self, limit: int = 5) -> None:
        """
        Render recent upload history.
        
        Args:
            limit: Maximum number of records to display
        """
        st.markdown("### Recent Uploads")
        
        # Get upload history
        records = self.upload_tracker.get_upload_history(limit=limit)
        
        if not records:
            st.info("No upload history found")
            return
        
        # Create DataFrame for display
        history_data = []
        for record in records:
            history_data.append({
                "File": record.file_name,
                "Date": record.upload_time.strftime("%Y-%m-%d %H:%M"),
                "Status": record.status,
                "Rows": record.row_count or "-",
                "Columns": record.column_count or "-",
                "Size": f"{record.file_size / 1024:.1f} KB"
            })
        
        if history_data:
            history_df = pd.DataFrame(history_data)
            st.dataframe(history_df, use_container_width=True)
        
        # View full history button
        if st.button("View Full History"):
            st.session_state.show_full_history = True
    
    def render_full_history(self) -> None:
        """Render full upload history with filtering options."""
        st.markdown("### Upload History")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            status_filter = st.selectbox(
                "Status",
                options=["All", "Success", "Error", "Warning"],
                index=0
            )
        
        with col2:
            days_filter = st.selectbox(
                "Time Range",
                options=[7, 30, 90, 365],
                index=1,
                format_func=lambda x: f"Last {x} days"
            )
        
        with col3:
            limit_filter = st.selectbox(
                "Limit",
                options=[10, 25, 50, 100],
                index=1
            )
        
        # Convert filters to parameters
        status = None if status_filter == "All" else status_filter.lower()
        start_date = datetime.now() - pd.Timedelta(days=days_filter)
        
        # Get filtered history
        records = self.upload_tracker.get_upload_history(
            limit=limit_filter,
            status=status,
            start_date=start_date
        )
        
        if not records:
            st.info("No upload history found matching the filters")
            return
        
        # Create DataFrame for display
        history_data = []
        for record in records:
            history_data.append({
                "ID": record.id,
                "File": record.file_name,
                "Date": record.upload_time.strftime("%Y-%m-%d %H:%M"),
                "Status": record.status,
                "Rows": record.row_count or "-",
                "Columns": record.column_count or "-",
                "Size": f"{record.file_size / 1024:.1f} KB",
                "Dealer": record.dealer_id or "-",
                "User": record.user_id or "-"
            })
        
        if history_data:
            history_df = pd.DataFrame(history_data)
            st.dataframe(history_df, use_container_width=True)
        
        # Upload stats
        stats = self.upload_tracker.get_upload_stats(days=days_filter)
        
        if stats:
            st.markdown("### Upload Statistics")
            
            stat_cols = st.columns(4)
            
            with stat_cols[0]:
                st.metric("Total Uploads", stats["total_uploads"])
            
            with stat_cols[1]:
                st.metric("Success Rate", f"{stats['success_rate']:.1f}%")
            
            with stat_cols[2]:
                st.metric("Failed Uploads", stats["failed_uploads"])
            
            with stat_cols[3]:
                total_size_mb = stats["total_size_bytes"] / (1024 * 1024)
                st.metric("Total Size", f"{total_size_mb:.1f} MB")
        
        # Back button
        if st.button("Back"):
            st.session_state.show_full_history = False
    
    def render(self) -> None:
        """Render the error feedback UI."""
        # Check if we should show full history
        if st.session_state.get("show_full_history", False):
            self.render_full_history()
            return
        
        # Render error summary
        self.render_error_summary()
        
        # Render upload history
        self.render_upload_history()
