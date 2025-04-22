"""
Scheduler Bridge for Nova Act integration.

This module provides a bridge between the Watchdog ReportScheduler and the NovaActConnector,
allowing scheduled reports to trigger the collection of dealer data automatically.
"""

import os
import logging
import asyncio
import time
import streamlit as st
from typing import Dict, Any, Optional
from datetime import datetime

from .core import NovaActConnector
from ..scheduler.report_scheduler import ReportScheduler, ScheduledReport, ReportFrequency

logger = logging.getLogger(__name__)

class NovaActSchedulerBridge:
    """
    Connects the Watchdog report scheduler with the Nova Act connector, allowing
    scheduled reports to trigger automated data collection.
    """
    
    def __init__(self, 
                 report_scheduler: Optional[ReportScheduler] = None,
                 connector: Optional[NovaActConnector] = None):
        """
        Initialize the scheduler bridge.
        
        Args:
            report_scheduler: Optional ReportScheduler instance, created if not provided
            connector: Optional NovaActConnector instance, created if not provided
        """
        self.report_scheduler = report_scheduler or ReportScheduler()
        self.connector = connector or NovaActConnector(headless=True)
        
        # Dictionary to track sync status by dealer/vendor
        self.sync_status: Dict[str, Dict[str, Any]] = {}
        
        # Flag to track if the bridge is running
        self.running = False
        
        # Store task mapping from report ID to Nova Act task ID
        self.task_mapping = {}
    
    async def start(self):
        """Start the scheduler bridge."""
        if self.running:
            return
        
        # Start the connector if not already running
        if not hasattr(self.connector, 'playwright') or self.connector.playwright is None:
            await self.connector.start()
        
        # Mark as running
        self.running = True
        
        # Start the background task
        asyncio.create_task(self._run_scheduler_bridge())
        
        logger.info("Nova Act scheduler bridge started")
    
    async def stop(self):
        """Stop the scheduler bridge."""
        if not self.running:
            return
        
        # Mark as not running
        self.running = False
        
        # Shutdown connector if needed
        if hasattr(self.connector, 'playwright') and self.connector.playwright is not None:
            await self.connector.shutdown()
        
        logger.info("Nova Act scheduler bridge stopped")
    
    async def _run_scheduler_bridge(self):
        """Background task that monitors scheduler for collection tasks."""
        while self.running:
            try:
                # Get due reports from the scheduler
                due_reports = self.report_scheduler.get_due_reports()
                
                for report in due_reports:
                    # Check if this is a Nova Act report from metadata
                    if self._is_nova_act_report(report):
                        # Process the report
                        await self._process_nova_act_report(report)
                
                # Process tasks from our own scheduler
                await self._process_scheduled_tasks()
                
                # Sleep for a bit before checking again
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"Error in scheduler bridge: {str(e)}")
                await asyncio.sleep(30)  # Longer sleep on error
    
    def _is_nova_act_report(self, report: ScheduledReport) -> bool:
        """
        Check if a scheduled report is a Nova Act report.
        
        Args:
            report: The scheduled report to check
            
        Returns:
            True if this is a Nova Act report, False otherwise
        """
        # Check for Nova Act metadata
        if not report.parameters:
            return False
        
        # Look for Nova Act keys in parameters
        nova_act_keys = ['vendor_id', 'dealer_id', 'is_nova_act']
        return any(key in report.parameters for key in nova_act_keys) or report.parameters.get('is_nova_act', False)
    
    async def _process_nova_act_report(self, report: ScheduledReport):
        """
        Process a scheduled Nova Act report by triggering data collection.
        
        Args:
            report: The report to process
        """
        logger.info(f"Processing Nova Act report: {report.report_id}")
        
        # Extract parameters
        params = report.parameters or {}
        vendor_id = params.get('vendor_id')
        dealer_id = params.get('dealer_id')
        report_type = params.get('report_type', 'sales')
        
        if not vendor_id or not dealer_id:
            logger.error(f"Missing vendor_id or dealer_id in report {report.report_id}")
            return
        
        # Update report next run time to avoid re-processing
        report.update_next_run()
        self.report_scheduler.save_reports()
        
        try:
            # Collect the report with the connector
            result = await self.connector.collect_report(
                vendor=vendor_id,
                dealer_id=dealer_id,
                report_type=report_type
            )
            
            # Update sync status
            self._update_sync_status(vendor_id, dealer_id, report_type, result)
            
            # Update UI if running in Streamlit
            self._update_ui_status(vendor_id, dealer_id, result)
            
        except Exception as e:
            logger.error(f"Error collecting report for {vendor_id}:{dealer_id}: {str(e)}")
            
            # Update sync status with error
            self._update_sync_status(vendor_id, dealer_id, report_type, {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
    
    async def _process_scheduled_tasks(self):
        """Process scheduled tasks from the Nova Act scheduler."""
        pass  # Implement if needed for custom scheduling
    
    def _update_sync_status(self, vendor_id: str, dealer_id: str, report_type: str, result: Dict[str, Any]):
        """
        Update the sync status for a vendor/dealer.
        
        Args:
            vendor_id: Vendor identifier
            dealer_id: Dealer identifier
            report_type: Type of report
            result: Result of the report collection
        """
        # Create key
        key = f"{vendor_id}:{dealer_id}:{report_type}"
        
        # Update status
        self.sync_status[key] = {
            "vendor_id": vendor_id,
            "dealer_id": dealer_id,
            "report_type": report_type,
            "last_sync": datetime.now().isoformat(),
            "success": result.get("success", False),
            "file_path": result.get("file_path"),
            "error": result.get("error"),
            "duration": result.get("duration")
        }
        
        logger.info(f"Updated sync status for {key}: success={result.get('success', False)}")
    
    def _update_ui_status(self, vendor_id: str, dealer_id: str, result: Dict[str, Any]):
        """
        Update UI status in Streamlit session state.
        
        Args:
            vendor_id: Vendor identifier
            dealer_id: Dealer identifier
            result: Result of the report collection
        """
        # Check if running in Streamlit
        if not hasattr(st, "session_state"):
            return
        
        # Create key for nova_act_sync_status in session state
        if "nova_act_sync_status" not in st.session_state:
            st.session_state.nova_act_sync_status = {}
        
        # Create key
        key = f"{vendor_id}:{dealer_id}"
        
        # Update session state
        st.session_state.nova_act_sync_status[key] = {
            "vendor_id": vendor_id,
            "dealer_id": dealer_id,
            "last_sync": datetime.now().isoformat(),
            "success": result.get("success", False),
            "error": result.get("error", None)
        }
    
    def schedule_report_collection(self, 
                                 vendor_id: str, 
                                 dealer_id: str, 
                                 report_type: str,
                                 frequency: ReportFrequency,
                                 **schedule_params) -> str:
        """
        Schedule regular collection of dealer data with the ReportScheduler.
        
        Args:
            vendor_id: Vendor identifier
            dealer_id: Dealer identifier
            report_type: Type of report to collect
            frequency: How often to collect the report
            **schedule_params: Additional scheduling parameters
            
        Returns:
            Report ID of the scheduled report
        """
        # Create parameters
        params = {
            "vendor_id": vendor_id,
            "dealer_id": dealer_id,
            "report_type": report_type,
            "is_nova_act": True
        }
        
        # Schedule the report
        report_id = self.report_scheduler.create_report(
            name=f"{vendor_id} {report_type} for {dealer_id}",
            template="nova_act",
            frequency=frequency,
            format=None,  # Not used for Nova Act reports
            delivery=None,  # Not used for Nova Act reports
            parameters=params,
            created_by="NovaActBridge"
        )
        
        logger.info(f"Scheduled {report_type} report collection for {vendor_id}:{dealer_id} with frequency {frequency}")
        
        return report_id
    
    def get_sync_status(self, vendor_id: Optional[str] = None, dealer_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get the sync status for a vendor/dealer or all sync statuses.
        
        Args:
            vendor_id: Optional vendor to filter by
            dealer_id: Optional dealer to filter by
            
        Returns:
            Dictionary of sync statuses
        """
        if vendor_id and dealer_id:
            # Look for specific status
            key_prefix = f"{vendor_id}:{dealer_id}"
            return {k: v for k, v in self.sync_status.items() if k.startswith(key_prefix)}
        elif vendor_id:
            # Filter by vendor
            return {k: v for k, v in self.sync_status.items() if v["vendor_id"] == vendor_id}
        elif dealer_id:
            # Filter by dealer
            return {k: v for k, v in self.sync_status.items() if v["dealer_id"] == dealer_id}
        else:
            # Return all
            return self.sync_status
    
    def trigger_sync_now(self, vendor_id: str, dealer_id: str, report_type: str) -> Dict[str, Any]:
        """
        Trigger an immediate sync for a vendor/dealer.
        
        Args:
            vendor_id: Vendor identifier
            dealer_id: Dealer identifier
            report_type: Type of report to collect
            
        Returns:
            Status of the triggered sync
        """
        # Create a one-time task
        task_id = self.connector.schedule_report_collection(
            vendor=vendor_id,
            dealer_id=dealer_id,
            report_type=report_type,
            frequency="once"
        )
        
        logger.info(f"Triggered immediate sync for {vendor_id}:{dealer_id} with task ID {task_id}")
        
        return {
            "task_id": task_id,
            "vendor_id": vendor_id,
            "dealer_id": dealer_id,
            "report_type": report_type,
            "triggered_at": datetime.now().isoformat()
        }


# Singleton instance
_bridge = None

def get_scheduler_bridge() -> NovaActSchedulerBridge:
    """
    Get the singleton scheduler bridge instance.
    
    Returns:
        The scheduler bridge instance
    """
    global _bridge
    if _bridge is None:
        _bridge = NovaActSchedulerBridge()
    return _bridge