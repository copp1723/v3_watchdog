# Nova Act Integration for Watchdog AI

The Nova Act module provides automated data collection, normalization, and processing for dealer management systems. This module handles secure credential storage, scheduled data collection, and pipeline processing for integrating external data sources into Watchdog AI.

## Key Components

- **Core Automation**: Browser-based automation for data collection
- **Credential Management**: Secure storage of authentication credentials
- **Scheduler**: Flexible task scheduling for data collection
- **Ingestion Pipeline**: Data normalization and validation
- **Monitoring**: Dashboard for tracking pipeline health

## Enhanced Features

### Secure Credential Storage

The enhanced credential manager supports multiple storage backends:

- AWS Secrets Manager (primary)
- HashiCorp Vault (optional)
- Local encrypted storage (fallback)

### Flexible Scheduler

The enhanced scheduler supports multiple schedule types:

- One-time execution
- Interval-based execution
- Daily execution
- Weekly execution
- Monthly execution

### Ingestion Pipeline

The ingestion pipeline processes data through several stages:

1. Column mapping from vendor-specific formats
2. Data cleaning and normalization
3. Schema validation
4. Secure storage of processed data

### Monitoring Dashboard

A comprehensive dashboard for monitoring the health of the ingestion process:

- Success rates by dealer, vendor, and report type
- Recent failures and warnings
- Scheduled tasks status
- Data collection history

## Usage

See the [Ingestion Pipeline Documentation](/docs/ingestion_pipeline.md) for detailed usage instructions.

## Supported Systems

- DealerSocket CRM
- VinSolutions
- eLeads CRM (Beta)