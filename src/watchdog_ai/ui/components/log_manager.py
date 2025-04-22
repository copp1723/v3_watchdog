"""
Log management component for Watchdog AI.

This component provides a UI for viewing, managing, and downloading log files
from the application. It's designed to be integrated with the Settings page.
"""

import os
import streamlit as st
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import time
import base64
import io
import logging
from pathlib import Path
import re

from watchdog_ai.core.config.logging import LOG_DIR

class LogManager:
    """
    Manages log files and provides UI components for viewing and downloading them.
    """
    
    def __init__(self, log_dir: str = LOG_DIR):
        """
        Initialize the log manager.
        
        Args:
            log_dir: Directory containing log files, defaults to LOG_DIR from logging config
        """
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
    
    def get_log_files(self) -> List[Dict[str, Any]]:
        """
        Get a list of all log files with their metadata.
        
        Returns:
            List of dictionaries containing log file information
        """
        log_files = []
        
        try:
            for filename in os.listdir(self.log_dir):
                if filename.endswith('.log') or re.match(r'.*\.log\.\d+$', filename):
                    file_path = os.path.join(self.log_dir, filename)
                    stat = os.stat(file_path)
                    
                    # Parse rotation number if present (like .log.1, .log.2)
                    rotation_match = re.match(r'.*\.log\.(\d+)$', filename)
                    rotation_num = int(rotation_match.group(1)) if rotation_match else 0
                    
                    # Check if this is the main log file
                    is_main = filename.endswith('.log') and not rotation_match
                    
                    # Extract base name without rotation suffix
                    base_name = filename.split('.log')[0] + '.log'
                    
                    log_files.append({
                        'filename': filename,
                        'path': file_path,
                        'size': stat.st_size,
                        'created': datetime.fromtimestamp(stat.st_ctime),
                        'modified': datetime.fromtimestamp(stat.st_mtime),
                        'is_main': is_main,
                        'rotation': rotation_num,
                        'base_name': base_name
                    })
        except Exception as e:
            logging.error(f"Error reading log directory: {e}")
        
        # Sort by base name and rotation (main file first, then in descending order)
        log_files.sort(key=lambda x: (x['base_name'], -x['rotation']))
        
        return log_files
    
    def get_log_file_groups(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group log files by their base name (for rotation visualization).
        
        Returns:
            Dictionary mapping base log file names to lists of file information
        """
        log_files = self.get_log_files()
        groups = {}
        
        for log_file in log_files:
            base_name = log_file['base_name']
            if base_name not in groups:
                groups[base_name] = []
            groups[base_name].append(log_file)
        
        return groups
    
    def format_file_size(self, size_bytes: int) -> str:
        """
        Format file size in a human-readable format.
        
        Args:
            size_bytes: Size in bytes
            
        Returns:
            Formatted size string (e.g., "2.5 MB")
        """
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024 or unit == 'GB':
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
    
    def read_log_file(self, file_path: str, max_lines: int = 1000) -> str:
        """
        Read a log file and return its contents.
        
        Args:
            file_path: Path to the log file
            max_lines: Maximum number of lines to read
            
        Returns:
            String containing the log file contents
        """
        try:
            if not os.path.exists(file_path):
                return f"File not found: {file_path}"
            
            # For large files, read the last max_lines
            with open(file_path, 'r') as f:
                lines = f.readlines()
                if len(lines) > max_lines:
                    return f"... (showing last {max_lines} lines) ...\n" + ''.join(lines[-max_lines:])
                return ''.join(lines)
        except Exception as e:
            logging.error(f"Error reading log file {file_path}: {e}")
            return f"Error reading log file: {str(e)}"
    
    def get_file_download_link(self, file_path: str, link_text: str = "Download") -> str:
        """
        Generate a download link for a log file.
        
        Args:
            file_path: Path to the log file
            link_text: Text to display for the link
            
        Returns:
            HTML string with a download link
        """
        try:
            with open(file_path, 'r') as f:
                file_content = f.read()
            
            # Create a download link using base64 encoding
            b64 = base64.b64encode(file_content.encode()).decode()
            filename = os.path.basename(file_path)
            href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{link_text}</a>'
            return href
        except Exception as e:
            logging.error(f"Error creating download link for {file_path}: {e}")
            return f"<span style='color: red'>Error: {str(e)}</span>"
    
    def render_log_list(self) -> None:
        """
        Render a list of log files with metadata and download links.
        """
        st.subheader("Application Logs")
        st.write("These logs contain detailed information about the application's operation and any errors encountered.")
        
        # Get log files
        log_files = self.get_log_files()
        
        if not log_files:
            st.info("No log files found. They will be created when the application runs.")
            return
        
        # Create a DataFrame for displaying log files
        logs_data = []
        for log in log_files:
            logs_data.append({
                "Filename": log['filename'],
                "Size": self.format_file_size(log['size']),
                "Last Modified": log['modified'].strftime("%Y-%m-%d %H:%M"),
                "Type": "Current" if log['is_main'] else f"Backup #{log['rotation']}"
            })
        
        logs_df = pd.DataFrame(logs_data)
        st.dataframe(logs_df, use_container_width=True)
        
        # Add download buttons
        st.write("### Download Logs")
        
        # Group logs by base name for organized download
        log_groups = self.get_log_file_groups()
        
        for base_name, files in log_groups.items():
            expander_label = f"{base_name} ({len(files)} file{'s' if len(files) > 1 else ''})"
            with st.expander(expander_label):
                # Display individual files in this group
                for log in files:
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        file_type = "Current log" if log['is_main'] else f"Backup #{log['rotation']}"
                        st.markdown(f"**{log['filename']}** - {file_type}")
                    
                    with col2:
                        st.write(f"Size: {self.format_file_size(log['size'])}")
                    
                    with col3:
                        download_link = self.get_file_download_link(log['path'])
                        st.markdown(download_link, unsafe_allow_html=True)
    
    def render_log_rotation_visualization(self) -> None:
        """
        Render a visualization of log rotation status.
        """
        st.subheader("Log Rotation Status")
        
        # Get log file groups
        log_groups = self.get_log_file_groups()
        
        if not log_groups:
            st.info("No log files found.")
            return
        
        # Display rotation visualization for each log group
        for base_name, files in log_groups.items():
            # Calculate total size and rotation stats
            total_size = sum(f['size'] for f in files)
            max_rotations = max(f['rotation'] for f in files) if files else 0
            
            # Create progress bars for visualizing rotation
            st.write(f"### {base_name}")
            st.write(f"Total size: {self.format_file_size(total_size)}, Rotations: {max_rotations}")
            
            # Display files as progress bars
            for log in files:
                # Calculate percentage of total group size
                percentage = (log['size'] / total_size) * 100 if total_size > 0 else 0
                
                # Determine color based on file type
                color = "#1f77b4" if log['is_main'] else "#ff7f0e"  # Blue for main, orange for backups
                
                # Create label
                file_type = "Current" if log['is_main'] else f"Backup #{log['rotation']}"
                label = f"{log['filename']} ({self.format_file_size(log['size'])})"
                
                # Create a progress bar
                st.write(f"**{file_type}**")
                st.progress(percentage / 100)
                st.write(f"{label} - Last modified: {log['modified'].strftime('%Y-%m-%d %H:%M')}")
    
    def render_log_preview(self, max_lines: int = 100) -> None:
        """
        Render a preview of log file contents.
        
        Args:
            max_lines: Maximum number of lines to show in the preview
        """
        st.subheader("Log File Preview")
        
        # Get log files
        log_files = self.get_log_files()
        
        if not log_files:
            st.info("No log files available for preview.")
            return
        
        # Create a selectbox for choosing which log to preview
        options = [(log['filename'], log['path']) for log in log_files]
        selected = st.selectbox(
            "Select a log file to preview:",
            options=range(len(options)),
            format_func=lambda i: options[i][0]
        )
        
        # Get the selected file path
        selected_path = options[selected][1]
        
        # Show a preview of the log file
        log_content = self.read_log_file(selected_path, max_lines)
        
        with st.expander("Log Preview", expanded=True):
            st.text(log_content)
            
            # Add a download button
            st.markdown(self.get_file_download_link(selected_path, "Download Full Log"), unsafe_allow_html=True)
    
    def render_log_management(self) -> None:
        """
        Render the complete log management UI.
        """
        st.title("Log Management")
        
        # Create tabs for different log management functions
        tab1, tab2, tab3 = st.tabs(["Log Files", "Rotation Status", "Log Preview"])
        
        with tab1:
            self.render_log_list()
        
        with tab2:
            self.render_log_rotation_visualization()
        
        with tab3:
            self.render_log_preview()
        
        # Add settings and information
        st.write("### Log Settings")
        st.write(f"Log directory: `{self.log_dir}`")
        st.info("""
        Logs are automatically rotated when they reach a certain size.
        The most recent logs are in the main log file (without a number suffix),
        while older logs are in files with suffixes like .1, .2, etc.
        """)


def create_download_link(content: str, filename: str, link_text: str = "Download") -> str:
    """
    Create a download link for arbitrary content.
    
    Args:
        content: Content to be downloaded
        filename: Name of the download file
        link_text: Text to display in the link
        
    Returns:
        HTML string with a download link
    """
    b64 = base64.b64encode(content.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{link_text}</a>'
    return href


def render_settings_page_logs_section() -> None:
    """
    Render the logs section for the Settings page.
    This is a simplified view intended to be embedded in a larger Settings page.
    """
    log_manager = LogManager()
    
    st.header("Application Logs")
    st.write("View and download application logs for troubleshooting.")
    
    # Get recent logs
    log_files = log_manager.get_log_files()
    
    if not log_files:
        st.info("No log files found.")
        return
    
    # Show the most recent logs
    recent_logs = log_files[:5]  # Show only the 5 most recent logs
    
    for log in recent_logs:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.write(f"**{log['filename']}**")
            st.caption(f"Modified: {log['modified'].strftime('%Y-%m-%d %H:%M')} - Size: {log_manager.format_file_size(log['size'])}")
        
        with col2:
            download_link = log_manager.get_file_download_link(log['path'])
            st.markdown(download_link, unsafe_allow_html=True)
    
    # Add a link to the full log management interface
    if len(log_files) > 5:
        st.write(f"{len(log_files) - 5} more log files available.")
    
    if st.button("Open Log Manager"):
        st.session_state.show_full_log_manager = True
    
    # Show the full log manager if requested
    if st.session_state.get("show_full_log_manager", False):
        log_manager.render_log_management()
        
        # Add a button to go back
        if st.button("Back to Settings"):
            st.session_state.show_full_log_manager = False
            st.rerun()


if __name__ == "__main__":
    # Simple demo when running this file directly
    st.set_page_config(page_title="Log Manager", layout="wide")
    
    log_manager = LogManager()
    log_manager.render_log_management()

