#!/usr/bin/env python3
"""
Script to run the Nova Act scheduler as a background process.
Uses APScheduler for real scheduling at specific intervals.
"""

import os
import sys
import logging
import asyncio
import signal
from datetime import datetime, time
import json
from pathlib import Path
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

# Add project root to path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from src.nova_act.scheduler_bridge import NovaActSchedulerBridge
from src.nova_act.core import NovaActConnector
from src.scheduler.report_scheduler import ReportScheduler, ReportFrequency

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - Sync for %(name)s: %(message)s',
    handlers=[
        logging.FileHandler('/tmp/watchdog_nova_act_sync.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class NovaActSchedulerService:
    """Service to run the Nova Act scheduler with real scheduling intervals."""
    
    def __init__(self):
        """Initialize the service."""
        self.scheduler = ReportScheduler()
        self.connector = NovaActConnector(headless=True)
        self.bridge = NovaActSchedulerBridge(
            report_scheduler=self.scheduler,
            connector=self.connector
        )
        self.running = False
        self.ap_scheduler = AsyncIOScheduler()
        
    def _setup_schedules(self):
        """Set up schedules based on report frequencies."""
        reports = self.scheduler.get_all_reports()
        for report in reports:
            if not self.bridge._is_nova_act_report(report):
                continue
                
            # Get frequency from report
            frequency = report.frequency
            report_id = report.report_id
            
            # Create appropriate cron trigger
            if frequency == ReportFrequency.DAILY:
                trigger = CronTrigger(hour=8, minute=0)
                self.ap_scheduler.add_job(
                    self._run_sync,
                    trigger=trigger,
                    args=[report],
                    id=f"daily_{report_id}"
                )
                logger.info(f"Scheduled daily sync at 8 AM for report {report_id}")
                
            elif frequency == ReportFrequency.WEEKLY:
                trigger = CronTrigger(day_of_week='mon', hour=8, minute=0)
                self.ap_scheduler.add_job(
                    self._run_sync,
                    trigger=trigger,
                    args=[report],
                    id=f"weekly_{report_id}"
                )
                logger.info(f"Scheduled weekly sync on Monday 8 AM for report {report_id}")
                
            elif frequency == ReportFrequency.MONTHLY:
                trigger = CronTrigger(day=1, hour=8, minute=0)
                self.ap_scheduler.add_job(
                    self._run_sync,
                    trigger=trigger,
                    args=[report],
                    id=f"monthly_{report_id}"
                )
                logger.info(f"Scheduled monthly sync on 1st day 8 AM for report {report_id}")
    
    async def _run_sync(self, report):
        """Run a sync for a specific report."""
        try:
            params = report.parameters
            vendor_id = params.get('vendor_id')
            dealer_id = params.get('dealer_id')
            report_type = params.get('report_type', 'sales')
            
            logger.info(f"Starting scheduled sync for {vendor_id}:{dealer_id}")
            
            await self.bridge._process_nova_act_report(report)
            
            logger.info(f"Completed scheduled sync for {vendor_id}:{dealer_id}")
            
        except Exception as e:
            logger.error(f"Error in scheduled sync: {str(e)}")
    
    async def run(self):
        """Run the scheduler service."""
        self.running = True
        
        # Register signal handlers
        for sig in (signal.SIGTERM, signal.SIGINT):
            asyncio.get_event_loop().add_signal_handler(sig, self.stop)
        
        logger.info("Starting Nova Act scheduler service")
        
        # Set up schedules
        self._setup_schedules()
        
        # Start APScheduler
        self.ap_scheduler.start()
        
        # Keep service running
        while self.running:
            await asyncio.sleep(60)  # Check every minute for new reports
            self._setup_schedules()  # Update schedules for any new reports
    
    def stop(self):
        """Stop the scheduler service."""
        logger.info("Stopping Nova Act scheduler service")
        self.ap_scheduler.shutdown()
        self.running = False

async def main():
    """Main entry point."""
    service = NovaActSchedulerService()
    await service.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Service stopped by user")
