"""
Celery application configuration for Watchdog AI.

This module sets up the Celery application for handling async tasks.
"""

import os
from celery import Celery

# Get broker URL from environment or use default
CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0')
CELERY_RESULT_BACKEND = os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')

# Create Celery app
app = Celery(
    'watchdog',
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=['src.insights.tasks']  # Register task modules
)

# Configure Celery
app.conf.update(
    result_expires=3600,  # Results expire after 1 hour
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    worker_prefetch_multiplier=1,  # Prevent workers from prefetching too many tasks
)

# Optional: Configure task routes
app.conf.task_routes = {
    'src.insights.tasks.generate_insight': {'queue': 'insights'},
    'src.insights.tasks.run_insight_pipeline': {'queue': 'insights'},
}

if __name__ == '__main__':
    app.start()