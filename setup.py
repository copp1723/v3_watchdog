#!/usr/bin/env python3
"""
Watchdog AI setup script.

This setup script enables installation of the Watchdog AI package
for development or production use.
"""

import os
from setuptools import setup, find_packages

# Dependencies specifically needed for CRM integration
CRM_DEPENDENCIES = [
    "apscheduler>=3.9.0",
    "sqlalchemy>=1.4.0",  # For APScheduler SQLAlchemyJobStore
    "requests>=2.28.0",
]

# Development dependencies
DEV_DEPENDENCIES = [
    "pytest>=7.0.0",
    "black>=22.3.0",
    "flake8>=4.0.1",
    "mypy>=0.950",
]

setup(
    name="watchdog_ai",
    version="0.1.0",
    description="Watchdog AI - Intelligent monitoring and insights system",
    author="Watchdog AI Team",
    author_email="team@watchdogai.example",
    url="https://github.com/watchdogai/v3watchdog_ai",
    
    # Package structure
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    
    # Dependencies
    install_requires=CRM_DEPENDENCIES,
    
    # Optional dependencies
    extras_require={
        "dev": DEV_DEPENDENCIES,
    },
    
    # Python requirements
    python_requires=">=3.8",
    
    # Package metadata
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
