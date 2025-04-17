"""
Nova Act - Advanced Workflow Automation Framework

A comprehensive system for automating tasks with scheduling, credential management,
fallback handling, and file upload capabilities.
"""

# Version information
__version__ = "0.1.0"
__author__ = "Watchdog AI Team"

# Import key components for easy access
from .nova_act import NovaAct
from .scheduler import NovaScheduler
from .task import NovaTask, TaskStatus, TaskPriority
from .fallback import NovaFallback
from .credentials import NovaCredential
from .watchdog_upload import WatchdogUploader

# Export public API
__all__ = [
    'NovaAct',
    'NovaScheduler',
    'NovaTask',
    'TaskStatus',
    'TaskPriority',
    'NovaFallback',
    'NovaCredential',
    'WatchdogUploader',
] 