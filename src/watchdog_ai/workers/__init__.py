"""
Watchdog AI Workers Package

This package contains background workers and scheduled job implementations
for the Watchdog AI system.
"""

__version__ = '0.1.0'
__author__ = 'Watchdog AI Team'

# Export scheduler
from .scheduler import CRMSyncScheduler, start_scheduler

# Public API
__all__ = [
    'CRMSyncScheduler',
    'start_scheduler',
]

# Worker processes for watchdog_ai

