# Secure Ingestion Pipeline with Nova Integration

This document provides documentation for the secure ingestion pipeline with Nova integration and data normalization.

## Overview

The secure ingestion pipeline automatically fetches data from dealership management systems, normalizes formats, validates data against schemas, and prepares it for use in the insight engine. The system handles credentials securely, manages scheduled data pulls, and provides comprehensive monitoring.

![Ingestion Pipeline Flow](https://www.plantuml.com/plantuml/png/hLHDRzim33rFlu8Nxw5v01Pkf0o5aAL5mfnHYa1Zs6S8RxJ5QOV7fGJhw_dlTqvvFvqCnXGHKLHtWUBgR54zt-iqb5uJbM5dDYaNr2OGCPGW4KTsXAXo9EqXUK2Q81KiJQYYPIEpGjOZb8z77a64m2MmZrMYG6L4FuX6Ue3o0-7fWqL8L0PPB6f_Xb4YRNmrAbfgjl3joDNKp6eiLLzQ36-GQoE_Vv4DV3LhXQ5Q9G63B5FLYvEk5PKGpTnJNb2KMjf8KBGjfZT7n4OvxdR_tXk4o0IZwztQiLHlZpMOHcVv0EQPUuPDFTELRgxP_n_AvRZPnxlw3g-UdAqF9LJZcxR3FZOZfgZLzESqXPQD6gf2EKLRS9D-7dD5eXDLXQzg9SQyH1Xrb2CKmb4J_c0IHDV-AO1jgXJx-4D_BaRbRTdmC0nsSVgcZOkQ2FyMNaOy2mh3CvjLPHp9fLVZTCg-0R3lUMhj6Jw_gNfRzgJJ_zTbvlKWlZNF2uKLVuCKkRm6YR9KhMXiikM5PxvZZ9OD-fC-7qk1ERpk1I8W3r0xVl1fAW1z8QykbN3Qb09yNAFZm2y-j-1kOHnIszLu1oZlQFXrfmXu38z5XmV_oxZCX1i9VJCC7lpfgFB-0G00)

## Components

The secure ingestion pipeline consists of several key components:

1. **Credential Management**: Secure storage for vendor credentials using AWS Secrets Manager with local file encryption as a fallback.

2. **Enhanced Scheduler**: Flexible scheduling for data collection tasks with robust error handling and retry logic.

3. **Ingestion Pipeline**: Multi-stage pipeline for data normalization, cleaning, validation, and storage.

4. **Monitoring Dashboard**: Comprehensive monitoring of the ingestion process with alerts and visualization.

## Credential Management

The credential management system uses a tiered approach for security:

1. **Primary Storage**: AWS Secrets Manager (when available)
2. **Secondary Storage**: HashiCorp Vault (when configured)
3. **Fallback Storage**: Local encrypted file system

Credentials are stored in a standardized format that includes all necessary authentication details for each vendor system.

### Usage

```python
from src.nova_act.enhanced_credentials import get_credential_manager

# Get the credential manager
cred_manager = get_credential_manager()

# Store credentials
credential = DealerCredential(
    dealer_id="dealer123",
    vendor_id="dealersocket",
    client_id="client_id_here",
    client_secret="client_secret_here",
    dealer_code="dealer_code_here",
    environment="production"
)
cred_manager.store_credential(credential)

# Retrieve credentials
creds = cred_manager.get_credential("dealer123", "dealersocket")

# List dealers and vendors
dealers = cred_manager.list_dealers()
vendors = cred_manager.list_vendors()
```

## Enhanced Scheduler

The enhanced scheduler manages data collection tasks with various scheduling options:

- **One-time**: Execute once at a specific time
- **Interval**: Execute at regular intervals (e.g., every 30 minutes)
- **Daily**: Execute once per day at a specific time
- **Weekly**: Execute once per week on a specific day and time
- **Monthly**: Execute once per month on a specific day and time

### Usage

```python
from src.nova_act.enhanced_scheduler import get_scheduler

# Get the scheduler
scheduler = get_scheduler()

# Start the scheduler
await scheduler.start()

# Schedule a daily task
scheduler.schedule_task(
    dealer_id="dealer123",
    vendor_id="dealersocket",
    report_type="sales",
    schedule="daily",
    schedule_config={"hour": 1, "minute": 30}
)

# Schedule a weekly task
scheduler.schedule_task(
    dealer_id="dealer123",
    vendor_id="dealersocket",
    report_type="inventory",
    schedule="weekly",
    schedule_config={"day_of_week": 1, "hour": 2, "minute": 0}
)

# Schedule an immediate task
scheduler.schedule_task(
    dealer_id="dealer123",
    vendor_id="dealersocket",
    report_type="leads",
    schedule="once",
    schedule_config={"time": datetime.now(timezone.utc).isoformat()}
)

# Get all scheduled tasks
tasks = scheduler.get_all_tasks()

# Stop the scheduler
await scheduler.stop()
```

## Ingestion Pipeline

The ingestion pipeline processes data through several stages:

1. **Data Collection**: Fetching data from vendor systems
2. **Column Mapping**: Mapping vendor-specific column names to standard names
3. **Data Cleaning**: Cleaning and normalizing data values
4. **Schema Validation**: Validating data against expected schemas
5. **Data Storage**: Storing processed data in a standardized format

### Usage

```python
from src.nova_act.ingestion_pipeline import normalize_and_validate

# Process a data file
result = await normalize_and_validate(
    file_path="/path/to/data.csv",
    vendor_id="dealersocket",
    report_type="sales",
    dealer_id="dealer123"
)

# Check the result
if result["success"]:
    print(f"Data processed successfully: {result['output_path']}")
else:
    print(f"Error processing data: {result.get('error')}")
```

## Monitoring Dashboard

The monitoring dashboard provides visibility into the ingestion pipeline's health and performance:

- Success/failure rates by dealer, vendor, and report type
- Recent failures and warnings
- Scheduled tasks status
- Data collection history

### Usage

To launch the monitoring dashboard:

```bash
cd examples
streamlit run ingestion_dashboard.py
```

## Examples

### Complete Pipeline Example

```python
import asyncio
from datetime import datetime, timezone
from src.nova_act.enhanced_credentials import get_credential_manager, DealerCredential
from src.nova_act.enhanced_scheduler import get_scheduler
from src.nova_act.monitoring import get_monitor

async def main():
    # Set up credentials
    cred_manager = get_credential_manager()
    
    credential = DealerCredential(
        dealer_id="dealer123",
        vendor_id="dealersocket",
        client_id="client_id_here",
        client_secret="client_secret_here",
        dealer_code="dealer_code_here",
        environment="production"
    )
    cred_manager.store_credential(credential)
    
    # Set up scheduler
    scheduler = get_scheduler()
    await scheduler.start()
    
    # Schedule daily tasks
    scheduler.schedule_task(
        dealer_id="dealer123",
        vendor_id="dealersocket",
        report_type="sales",
        schedule="daily",
        schedule_config={"hour": 1, "minute": 30}
    )
    
    scheduler.schedule_task(
        dealer_id="dealer123",
        vendor_id="dealersocket",
        report_type="inventory",
        schedule="daily",
        schedule_config={"hour": 2, "minute": 0}
    )
    
    # Run an immediate task
    task_id = scheduler.schedule_task(
        dealer_id="dealer123",
        vendor_id="dealersocket",
        report_type="leads",
        schedule="once",
        schedule_config={"time": datetime.now(timezone.utc).isoformat()}
    )
    
    # Wait for tasks to complete (in a real application, you would use a proper event loop)
    await asyncio.sleep(120)
    
    # Check task status
    task = scheduler.get_task(task_id)
    print(f"Task status: {task.last_status}")
    
    # Get monitoring information
    monitor = get_monitor()
    status = monitor.get_ingestion_status(dealer_id="dealer123")
    print(f"Success rate: {status['success_rate']}%")
    
    # Stop scheduler
    await scheduler.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

## Security Considerations

The ingestion pipeline implements several security measures:

1. **Credential Protection**:
   - AWS Secrets Manager encryption at rest and in transit
   - Fallback to local encryption with AES-256
   - Key rotation mechanism
   - Minimal access permissions

2. **Data Security**:
   - Original data files stored securely
   - Processed data stored with access controls
   - Schema validation to detect anomalies

3. **Error Handling**:
   - Graceful failure handling
   - Detailed error logging
   - Retry mechanisms with backoff

## Configuration

The ingestion pipeline can be configured through environment variables:

```
# AWS Integration
USE_AWS_SECRETS=true
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key

# HashiCorp Vault (Optional)
USE_VAULT=false
VAULT_ADDR=https://vault.example.com
VAULT_TOKEN=your_vault_token

# Scheduler Configuration
MAX_CONCURRENT_TASKS=5

# Security Settings
CREDENTIAL_CACHE_TTL=300  # seconds
WATCHDOG_CRED_KEY=your_encryption_key
```

## Error Handling and Fallbacks

The pipeline includes comprehensive error handling and fallback mechanisms:

1. **Credential Access**:
   - Try AWS Secrets Manager first
   - Fall back to HashiCorp Vault if AWS fails
   - Fall back to local storage as a last resort

2. **Data Collection**:
   - Retry failed collections up to 3 times
   - Exponential backoff between retries
   - Detailed error reporting

3. **Data Processing**:
   - Continue pipeline even with non-critical errors
   - Store partial results when possible
   - Comprehensive error metadata

## Monitoring and Metrics

The monitoring system tracks several key metrics:

1. **Success Rates**:
   - Overall success rate
   - Success rate by dealer
   - Success rate by vendor
   - Success rate by report type

2. **Timing Metrics**:
   - Average processing time
   - Time spent in each pipeline stage
   - Schedule adherence

3. **Error Metrics**:
   - Error counts by type
   - Error counts by dealer/vendor
   - Retry statistics

## Troubleshooting

Common issues and solutions:

### AWS Secrets Manager Issues

- **Access Denied**: Check IAM permissions for the application
- **Connection Failures**: Check network connectivity and VPC configuration
- **Not Found Errors**: Verify secret names and region settings

### Scheduler Issues

- **Tasks Not Running**: Check if scheduler is started and running
- **Missed Schedules**: Check system time and timezone settings
- **Failed Tasks**: Check credential validity and vendor system availability

### Data Processing Issues

- **Mapping Errors**: Check column mapping definitions for the vendor
- **Validation Errors**: Check if data matches the expected schema
- **Storage Errors**: Check file permissions and disk space

## File and Data Locations

All data is stored in standardized locations:

- **Raw Data**: `/data/collected/{vendor_id}/{dealer_id}/{report_type}/{timestamp}.csv`
- **Processed Data**: `/data/processed/{vendor_id}/{dealer_id}/{report_type}/{timestamp}.csv`
- **Metadata**: `/data/metadata/{vendor_id}/{dealer_id}/{report_type}/{timestamp}.json`
- **Schedule Data**: `/data/scheduler_storage/tasks.json`
- **Local Credentials**: `/data/secrets/credentials/`

## Logging

The system logs important events to aid in debugging and auditing:

- **Info Level**: Routine operations like task scheduling and successful processing
- **Warning Level**: Non-critical issues that might require attention
- **Error Level**: Critical failures that need immediate attention
- **Debug Level**: Detailed information for troubleshooting

Logs include tags for dealer ID, vendor ID, and operation type for easy filtering.

## Best Practices

1. **Credential Management**:
   - Prefer AWS Secrets Manager for production environments
   - Rotate credentials regularly
   - Limit access to credentials to only necessary personnel

2. **Scheduling**:
   - Schedule data collection during off-peak hours
   - Stagger schedules to avoid overwhelming vendor systems
   - Set appropriate retry intervals

3. **Monitoring**:
   - Check the monitoring dashboard regularly
   - Set up alerts for failure thresholds
   - Review error patterns to identify systematic issues