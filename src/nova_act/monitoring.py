"""
Monitoring module for Nova Act data ingestion pipeline.

This module provides functionality to track and monitor
the health and status of the ingestion pipeline.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional, Tuple, Union
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import io
import base64

from .logging_config import log_error, log_info, log_warning

logger = logging.getLogger(__name__)

# Default data directory
DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data"
)

class IngestionMonitor:
    """
    Monitor for tracking the health and status of the ingestion pipeline.
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize the monitor.
        
        Args:
            data_dir: Optional data directory path
        """
        self.data_dir = data_dir or DEFAULT_DATA_DIR
        self.metadata_dir = os.path.join(self.data_dir, "metadata")
    
    def get_ingestion_status(self, 
                             dealer_id: Optional[str] = None, 
                             vendor_id: Optional[str] = None,
                             report_type: Optional[str] = None,
                             days: int = 7) -> Dict[str, Any]:
        """
        Get the ingestion status for the specified parameters.
        
        Args:
            dealer_id: Optional dealer ID filter
            vendor_id: Optional vendor ID filter
            report_type: Optional report type filter
            days: Number of days to look back
            
        Returns:
            Dictionary with status information
        """
        try:
            # Calculate the cutoff date
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
            
            # Find all metadata files matching the criteria
            metadata_files = self._find_metadata_files(dealer_id, vendor_id, report_type)
            
            # Process each file to collect status information
            status_data = []
            for metadata_file in metadata_files:
                try:
                    # Parse the metadata file
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    # Skip if before the cutoff date
                    processing_time = datetime.fromisoformat(metadata.get('processing_time', ''))
                    if processing_time < cutoff_date:
                        continue
                    
                    # Extract relevant information
                    status_entry = {
                        'dealer_id': metadata.get('dealer_id', ''),
                        'vendor_id': metadata.get('vendor_id', ''),
                        'report_type': metadata.get('report_type', ''),
                        'processing_time': processing_time.isoformat(),
                        'success': metadata.get('pipeline_result', {}).get('success', False),
                        'row_count': metadata.get('file_stats', {}).get('row_count', 0),
                        'input_file': metadata.get('input_file', ''),
                        'output_file': metadata.get('output_file', '')
                    }
                    
                    # Add error or warning if present
                    if not status_entry['success']:
                        status_entry['error'] = metadata.get('pipeline_result', {}).get('error', 'Unknown error')
                    elif 'warning' in metadata.get('pipeline_result', {}):
                        status_entry['warning'] = metadata.get('pipeline_result', {}).get('warning')
                    
                    status_data.append(status_entry)
                except Exception as e:
                    # Log but continue with other files
                    logger.warning(f"Error processing metadata file {metadata_file}: {str(e)}")
            
            # Convert to DataFrame for easier analysis
            if not status_data:
                return {
                    'status': 'no_data',
                    'message': 'No ingestion data found for the specified criteria',
                    'counts': {
                        'total': 0,
                        'success': 0,
                        'failure': 0,
                        'warning': 0
                    }
                }
            
            df = pd.DataFrame(status_data)
            
            # Calculate summary statistics
            total_count = len(df)
            success_count = df['success'].sum()
            failure_count = total_count - success_count
            warning_count = sum('warning' in row for _, row in df.iterrows())
            
            # Calculate success rate
            success_rate = (success_count / total_count) * 100 if total_count > 0 else 0
            
            # Group by dealer, vendor, and report type
            group_counts = df.groupby(['dealer_id', 'vendor_id', 'report_type']).size().reset_index(name='count')
            group_success = df[df['success']].groupby(['dealer_id', 'vendor_id', 'report_type']).size().reset_index(name='success_count')
            
            # Merge to get success rates by group
            group_stats = group_counts.merge(group_success, on=['dealer_id', 'vendor_id', 'report_type'], how='left')
            group_stats['success_count'] = group_stats['success_count'].fillna(0)
            group_stats['success_rate'] = (group_stats['success_count'] / group_stats['count']) * 100
            
            # Sort by success rate
            group_stats = group_stats.sort_values('success_rate')
            
            # Format the results
            return {
                'status': 'success',
                'message': f'Found {total_count} ingestion runs in the last {days} days',
                'counts': {
                    'total': int(total_count),
                    'success': int(success_count),
                    'failure': int(failure_count),
                    'warning': int(warning_count)
                },
                'success_rate': float(success_rate),
                'group_stats': group_stats.to_dict(orient='records'),
                'recent_failures': df[~df['success']].to_dict(orient='records')
            }
            
        except Exception as e:
            log_error(
                e,
                dealer_id or "system",
                "get_ingestion_status"
            )
            
            return {
                'status': 'error',
                'message': f'Error getting ingestion status: {str(e)}',
                'counts': {
                    'total': 0,
                    'success': 0,
                    'failure': 0,
                    'warning': 0
                }
            }
    
    def get_dealer_status(self, dealer_id: str) -> Dict[str, Any]:
        """
        Get detailed status for a specific dealer.
        
        Args:
            dealer_id: Dealer ID
            
        Returns:
            Dictionary with dealer status
        """
        try:
            # Get overall status
            overall_status = self.get_ingestion_status(dealer_id=dealer_id, days=30)
            
            # Find all metadata files for this dealer
            metadata_files = self._find_metadata_files(dealer_id=dealer_id)
            
            # Process each file to collect detailed information
            vendor_data = {}
            report_type_data = {}
            
            for metadata_file in metadata_files:
                try:
                    # Parse the metadata file
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    vendor_id = metadata.get('vendor_id', '')
                    report_type = metadata.get('report_type', '')
                    success = metadata.get('pipeline_result', {}).get('success', False)
                    
                    # Update vendor stats
                    if vendor_id not in vendor_data:
                        vendor_data[vendor_id] = {'total': 0, 'success': 0}
                    vendor_data[vendor_id]['total'] += 1
                    if success:
                        vendor_data[vendor_id]['success'] += 1
                    
                    # Update report type stats
                    report_key = f"{vendor_id}:{report_type}"
                    if report_key not in report_type_data:
                        report_type_data[report_key] = {'total': 0, 'success': 0}
                    report_type_data[report_key]['total'] += 1
                    if success:
                        report_type_data[report_key]['success'] += 1
                    
                except Exception as e:
                    # Log but continue with other files
                    logger.warning(f"Error processing metadata file {metadata_file}: {str(e)}")
            
            # Calculate success rates
            vendor_stats = []
            for vendor_id, stats in vendor_data.items():
                success_rate = (stats['success'] / stats['total']) * 100 if stats['total'] > 0 else 0
                vendor_stats.append({
                    'vendor_id': vendor_id,
                    'total': stats['total'],
                    'success': stats['success'],
                    'success_rate': success_rate
                })
            
            report_stats = []
            for report_key, stats in report_type_data.items():
                vendor_id, report_type = report_key.split(':', 1)
                success_rate = (stats['success'] / stats['total']) * 100 if stats['total'] > 0 else 0
                report_stats.append({
                    'vendor_id': vendor_id,
                    'report_type': report_type,
                    'total': stats['total'],
                    'success': stats['success'],
                    'success_rate': success_rate
                })
            
            # Get latest files
            latest_files = self._get_latest_processed_files(dealer_id)
            
            return {
                'dealer_id': dealer_id,
                'overall_status': overall_status,
                'vendor_stats': vendor_stats,
                'report_stats': report_stats,
                'latest_files': latest_files
            }
            
        except Exception as e:
            log_error(
                e,
                dealer_id,
                "get_dealer_status"
            )
            
            return {
                'status': 'error',
                'message': f'Error getting dealer status: {str(e)}'
            }
    
    def get_vendor_status(self, vendor_id: str) -> Dict[str, Any]:
        """
        Get detailed status for a specific vendor.
        
        Args:
            vendor_id: Vendor ID
            
        Returns:
            Dictionary with vendor status
        """
        try:
            # Get overall status
            overall_status = self.get_ingestion_status(vendor_id=vendor_id, days=30)
            
            # Find all metadata files for this vendor
            metadata_files = self._find_metadata_files(vendor_id=vendor_id)
            
            # Process each file to collect detailed information
            dealer_data = {}
            report_type_data = {}
            
            for metadata_file in metadata_files:
                try:
                    # Parse the metadata file
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    dealer_id = metadata.get('dealer_id', '')
                    report_type = metadata.get('report_type', '')
                    success = metadata.get('pipeline_result', {}).get('success', False)
                    
                    # Update dealer stats
                    if dealer_id not in dealer_data:
                        dealer_data[dealer_id] = {'total': 0, 'success': 0}
                    dealer_data[dealer_id]['total'] += 1
                    if success:
                        dealer_data[dealer_id]['success'] += 1
                    
                    # Update report type stats
                    if report_type not in report_type_data:
                        report_type_data[report_type] = {'total': 0, 'success': 0}
                    report_type_data[report_type]['total'] += 1
                    if success:
                        report_type_data[report_type]['success'] += 1
                    
                except Exception as e:
                    # Log but continue with other files
                    logger.warning(f"Error processing metadata file {metadata_file}: {str(e)}")
            
            # Calculate success rates
            dealer_stats = []
            for dealer_id, stats in dealer_data.items():
                success_rate = (stats['success'] / stats['total']) * 100 if stats['total'] > 0 else 0
                dealer_stats.append({
                    'dealer_id': dealer_id,
                    'total': stats['total'],
                    'success': stats['success'],
                    'success_rate': success_rate
                })
            
            report_stats = []
            for report_type, stats in report_type_data.items():
                success_rate = (stats['success'] / stats['total']) * 100 if stats['total'] > 0 else 0
                report_stats.append({
                    'report_type': report_type,
                    'total': stats['total'],
                    'success': stats['success'],
                    'success_rate': success_rate
                })
            
            return {
                'vendor_id': vendor_id,
                'overall_status': overall_status,
                'dealer_stats': dealer_stats,
                'report_stats': report_stats
            }
            
        except Exception as e:
            log_error(
                e,
                "system",
                "get_vendor_status"
            )
            
            return {
                'status': 'error',
                'message': f'Error getting vendor status: {str(e)}'
            }
    
    def generate_health_dashboard(self) -> Dict[str, Any]:
        """
        Generate a comprehensive health dashboard.
        
        Returns:
            Dictionary with dashboard data
        """
        try:
            # Get overall status for the last 7 days
            overall_status = self.get_ingestion_status(days=7)
            
            # Get overall status for the last 30 days for trend analysis
            trend_status = self.get_ingestion_status(days=30)
            
            # Get list of all dealers and vendors
            dealers = self._get_unique_dealers()
            vendors = self._get_unique_vendors()
            
            # Create a dataframe for all metadata to analyze
            metadata_df = self._load_all_metadata(days=30)
            
            # Generate charts
            charts = {}
            
            if not metadata_df.empty:
                # Success rate by vendor
                vendor_success = metadata_df.groupby('vendor_id')['success'].agg(['count', 'sum'])
                vendor_success['rate'] = (vendor_success['sum'] / vendor_success['count']) * 100
                
                # Success rate by dealer
                dealer_success = metadata_df.groupby('dealer_id')['success'].agg(['count', 'sum'])
                dealer_success['rate'] = (dealer_success['sum'] / dealer_success['count']) * 100
                
                # Success rate by report type
                report_success = metadata_df.groupby('report_type')['success'].agg(['count', 'sum'])
                report_success['rate'] = (report_success['sum'] / report_success['count']) * 100
                
                # Trend of success rate over time
                metadata_df['date'] = pd.to_datetime(metadata_df['processing_time']).dt.date
                time_success = metadata_df.groupby('date')['success'].agg(['count', 'sum'])
                time_success['rate'] = (time_success['sum'] / time_success['count']) * 100
                
                # Create visualizations and convert to base64
                charts['vendor_success'] = self._create_bar_chart(
                    vendor_success.index, vendor_success['rate'], 
                    'Success Rate by Vendor', 'Vendor', 'Success Rate (%)'
                )
                
                charts['dealer_success'] = self._create_bar_chart(
                    dealer_success.index, dealer_success['rate'], 
                    'Success Rate by Dealer', 'Dealer', 'Success Rate (%)'
                )
                
                charts['report_success'] = self._create_bar_chart(
                    report_success.index, report_success['rate'], 
                    'Success Rate by Report Type', 'Report Type', 'Success Rate (%)'
                )
                
                charts['time_trend'] = self._create_line_chart(
                    time_success.index, time_success['rate'], 
                    'Success Rate Trend', 'Date', 'Success Rate (%)'
                )
            
            # Get alerts (failures in the last 24 hours)
            alerts = self._get_recent_alerts(hours=24)
            
            return {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'overall_status': overall_status,
                'trend_status': trend_status,
                'dealer_count': len(dealers),
                'vendor_count': len(vendors),
                'charts': charts,
                'alerts': alerts,
                'recent_activity': self._get_recent_activity(hours=48)
            }
            
        except Exception as e:
            log_error(
                e,
                "system",
                "generate_health_dashboard"
            )
            
            return {
                'status': 'error',
                'message': f'Error generating health dashboard: {str(e)}'
            }
    
    def _find_metadata_files(self, 
                            dealer_id: Optional[str] = None, 
                            vendor_id: Optional[str] = None,
                            report_type: Optional[str] = None) -> List[str]:
        """Find metadata files matching the specified criteria."""
        # Build the path pattern
        path_parts = [self.metadata_dir]
        
        if vendor_id:
            path_parts.append(vendor_id)
            if dealer_id:
                path_parts.append(dealer_id)
                if report_type:
                    path_parts.append(report_type)
                else:
                    path_parts.append("*")
            else:
                path_parts.extend(["*", "*" if not report_type else report_type])
        else:
            if dealer_id:
                path_parts.extend(["*", dealer_id, "*" if not report_type else report_type])
            else:
                path_parts.extend(["*", "*", "*" if not report_type else report_type])
        
        # Add file pattern
        path_parts.append("*.json")
        
        # Build the glob pattern
        pattern = os.path.join(*path_parts)
        
        # Find matching files
        return glob.glob(pattern)
    
    def _get_latest_processed_files(self, dealer_id: str) -> List[Dict[str, Any]]:
        """Get the latest processed files for a dealer."""
        processed_dir = os.path.join(self.data_dir, "processed", "*", dealer_id, "*")
        
        # Find all CSV files
        csv_files = glob.glob(os.path.join(processed_dir, "*.csv"))
        
        # Group by vendor and report type
        file_groups = {}
        for csv_file in csv_files:
            # Extract vendor, report_type from path
            parts = Path(csv_file).parts
            vendor_idx = parts.index("processed") + 1
            
            if vendor_idx < len(parts) and vendor_idx + 2 < len(parts):
                vendor_id = parts[vendor_idx]
                report_type = parts[vendor_idx + 2]
                
                key = f"{vendor_id}:{report_type}"
                
                # Get file timestamp from filename or file stats
                try:
                    timestamp_str = Path(csv_file).stem.split('_')[-1]
                    timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                except:
                    # Fall back to file modification time
                    timestamp = datetime.fromtimestamp(os.path.getmtime(csv_file))
                
                if key not in file_groups or timestamp > file_groups[key]['timestamp']:
                    file_groups[key] = {
                        'file_path': csv_file,
                        'timestamp': timestamp,
                        'vendor_id': vendor_id,
                        'report_type': report_type,
                        'file_size': os.path.getsize(csv_file)
                    }
        
        # Convert to list and sort by timestamp (newest first)
        latest_files = list(file_groups.values())
        latest_files.sort(key=lambda x: x['timestamp'], reverse=True)
        
        # Convert timestamps to ISO format strings
        for file_info in latest_files:
            file_info['timestamp'] = file_info['timestamp'].isoformat()
        
        return latest_files
    
    def _get_unique_dealers(self) -> List[str]:
        """Get a list of all unique dealer IDs."""
        dealer_dirs = glob.glob(os.path.join(self.metadata_dir, "*", "*"))
        return list(set(Path(d).name for d in dealer_dirs))
    
    def _get_unique_vendors(self) -> List[str]:
        """Get a list of all unique vendor IDs."""
        vendor_dirs = glob.glob(os.path.join(self.metadata_dir, "*"))
        return list(set(Path(d).name for d in vendor_dirs))
    
    def _load_all_metadata(self, days: int = 30) -> pd.DataFrame:
        """Load all metadata into a DataFrame for analysis."""
        # Calculate the cutoff date
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        # Find all metadata files
        metadata_files = self._find_metadata_files()
        
        # Process each file
        metadata_records = []
        for metadata_file in metadata_files:
            try:
                # Parse the metadata file
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Skip if before the cutoff date
                processing_time = datetime.fromisoformat(metadata.get('processing_time', ''))
                if processing_time < cutoff_date:
                    continue
                
                # Extract relevant information
                record = {
                    'dealer_id': metadata.get('dealer_id', ''),
                    'vendor_id': metadata.get('vendor_id', ''),
                    'report_type': metadata.get('report_type', ''),
                    'processing_time': processing_time.isoformat(),
                    'success': metadata.get('pipeline_result', {}).get('success', False),
                    'row_count': metadata.get('file_stats', {}).get('row_count', 0)
                }
                
                # Add error or warning if present
                if not record['success']:
                    record['error'] = metadata.get('pipeline_result', {}).get('error', 'Unknown error')
                elif 'warning' in metadata.get('pipeline_result', {}):
                    record['warning'] = metadata.get('pipeline_result', {}).get('warning')
                
                metadata_records.append(record)
            except Exception as e:
                # Log but continue with other files
                logger.warning(f"Error processing metadata file {metadata_file}: {str(e)}")
        
        if not metadata_records:
            return pd.DataFrame()
        
        return pd.DataFrame(metadata_records)
    
    def _create_bar_chart(self, x, y, title, xlabel, ylabel) -> str:
        """Create a bar chart and return as base64 encoded PNG."""
        plt.figure(figsize=(10, 6))
        plt.bar(x, y)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        
        # Save to bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        
        # Convert to base64
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        return f"data:image/png;base64,{img_str}"
    
    def _create_line_chart(self, x, y, title, xlabel, ylabel) -> str:
        """Create a line chart and return as base64 encoded PNG."""
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, marker='o')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.tight_layout()
        
        # Save to bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        
        # Convert to base64
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        return f"data:image/png;base64,{img_str}"
    
    def _get_recent_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent failures as alerts."""
        # Calculate the cutoff time
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        # Find all metadata files
        metadata_files = self._find_metadata_files()
        
        # Process each file to find failures
        alerts = []
        for metadata_file in metadata_files:
            try:
                # Parse the metadata file
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Skip if before the cutoff time
                processing_time = datetime.fromisoformat(metadata.get('processing_time', ''))
                if processing_time < cutoff_time:
                    continue
                
                # Check if it's a failure
                success = metadata.get('pipeline_result', {}).get('success', False)
                if not success:
                    alerts.append({
                        'dealer_id': metadata.get('dealer_id', ''),
                        'vendor_id': metadata.get('vendor_id', ''),
                        'report_type': metadata.get('report_type', ''),
                        'processing_time': processing_time.isoformat(),
                        'error': metadata.get('pipeline_result', {}).get('error', 'Unknown error'),
                        'severity': 'error'
                    })
                elif 'warning' in metadata.get('pipeline_result', {}):
                    alerts.append({
                        'dealer_id': metadata.get('dealer_id', ''),
                        'vendor_id': metadata.get('vendor_id', ''),
                        'report_type': metadata.get('report_type', ''),
                        'processing_time': processing_time.isoformat(),
                        'warning': metadata.get('pipeline_result', {}).get('warning'),
                        'severity': 'warning'
                    })
            except Exception as e:
                # Log but continue with other files
                logger.warning(f"Error processing metadata file {metadata_file}: {str(e)}")
        
        # Sort by time (newest first)
        alerts.sort(key=lambda x: x['processing_time'], reverse=True)
        
        return alerts
    
    def _get_recent_activity(self, hours: int = 48) -> List[Dict[str, Any]]:
        """Get recent activity (both successes and failures)."""
        # Calculate the cutoff time
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        # Find all metadata files
        metadata_files = self._find_metadata_files()
        
        # Process each file
        activities = []
        for metadata_file in metadata_files:
            try:
                # Parse the metadata file
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Skip if before the cutoff time
                processing_time = datetime.fromisoformat(metadata.get('processing_time', ''))
                if processing_time < cutoff_time:
                    continue
                
                # Extract activity information
                success = metadata.get('pipeline_result', {}).get('success', False)
                
                activity = {
                    'dealer_id': metadata.get('dealer_id', ''),
                    'vendor_id': metadata.get('vendor_id', ''),
                    'report_type': metadata.get('report_type', ''),
                    'processing_time': processing_time.isoformat(),
                    'success': success,
                    'status': 'success' if success else 'error'
                }
                
                if not success:
                    activity['error'] = metadata.get('pipeline_result', {}).get('error', 'Unknown error')
                elif 'warning' in metadata.get('pipeline_result', {}):
                    activity['warning'] = metadata.get('pipeline_result', {}).get('warning')
                    activity['status'] = 'warning'
                
                if 'file_stats' in metadata:
                    activity['row_count'] = metadata['file_stats'].get('row_count', 0)
                
                activities.append(activity)
            except Exception as e:
                # Log but continue with other files
                logger.warning(f"Error processing metadata file {metadata_file}: {str(e)}")
        
        # Sort by time (newest first)
        activities.sort(key=lambda x: x['processing_time'], reverse=True)
        
        return activities


# Singleton instance
_monitor = None

def get_monitor() -> IngestionMonitor:
    """
    Get the singleton monitor instance.
    
    Returns:
        The monitor instance
    """
    global _monitor
    if _monitor is None:
        _monitor = IngestionMonitor()
    return _monitor