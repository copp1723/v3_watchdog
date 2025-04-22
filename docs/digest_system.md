# Digest System Documentation

## Overview

The Digest System is a module that manages the generation and delivery of automated executive summaries (insight digests) on a scheduled basis. It supports multiple delivery formats (Slack, email, dashboard) and provides comprehensive tracking of digest delivery and engagement.

## Features

- **Scheduled Digest Generation**: Automatically generate and deliver digests based on recipient preferences
- **Multiple Digest Types**: Support for sales summaries, performance overviews, trend alerts, and custom digests
- **Flexible Delivery Formats**: Deliver digests via Slack, email, or dashboard
- **Recipient Management**: Add, update, and remove digest recipients with customizable preferences
- **Engagement Tracking**: Record and analyze recipient feedback on digest content
- **Delivery Statistics**: Track delivery success rates, engagement metrics, and format preferences
- **Data Persistence**: Automatically save and load digest data for reliability

## Components

### Enums

- `DigestFrequency`: Defines scheduling frequency options (DAILY, WEEKLY, MONTHLY)
- `DigestType`: Specifies types of digests (SALES_SUMMARY, PERFORMANCE_OVERVIEW, TREND_ALERTS, CUSTOM)
- `DigestFormat`: Lists supported delivery formats (SLACK, EMAIL, DASHBOARD)

### Data Classes

- `DigestRecipient`: Stores recipient information and preferences
  - `user_id`: Unique identifier for the recipient
  - `name`: Recipient's name
  - `email`: Email address for delivery
  - `slack_id`: Slack user ID for delivery
  - `frequency`: Preferred delivery frequency
  - `digest_types`: List of digest types to receive
  - `preferred_format`: Preferred delivery format
  - `last_delivered`: Timestamp of last delivery
  - `feedback`: Dictionary of feedback on received digests

- `DigestDelivery`: Records information about digest deliveries
  - `delivery_id`: Unique identifier for the delivery
  - `digest_id`: ID of the delivered digest
  - `recipient_id`: ID of the recipient
  - `digest_type`: Type of digest delivered
  - `format`: Delivery format used
  - `status`: Delivery status
  - `delivered_at`: Timestamp of delivery
  - `engagement`: Dictionary of recipient engagement data

### DigestSystem Class

The main class that manages the digest system functionality:

#### Initialization

```python
digest_system = DigestSystem(data_dir="/path/to/data")
```

#### Recipient Management

```python
# Add a recipient
digest_system.add_recipient(
    user_id="user1",
    name="John Doe",
    email="john@example.com",
    slack_id="U123456",
    frequency=DigestFrequency.DAILY,
    digest_types=[DigestType.SALES_SUMMARY],
    preferred_format=DigestFormat.SLACK
)

# Update a recipient
digest_system.update_recipient(
    user_id="user1",
    frequency=DigestFrequency.WEEKLY,
    digest_types=[DigestType.SALES_SUMMARY, DigestType.PERFORMANCE_OVERVIEW]
)

# Remove a recipient
digest_system.remove_recipient("user1")
```

#### Digest Generation and Delivery

```python
# Generate a digest
digest_id = digest_system.generate_digest(
    DigestType.SALES_SUMMARY,
    data={"sales_data": {...}}
)

# Deliver a digest
digest_system.deliver_digest(digest_id, "user1")
```

#### Feedback Recording

```python
# Record feedback for a delivery
digest_system.record_feedback(
    delivery_id="delivery1",
    feedback={
        "thumbs_up": True,
        "comment": "Great insights!"
    }
)
```

#### Statistics and Monitoring

```python
# Get delivery statistics
stats = digest_system.get_delivery_stats()
print(f"Total deliveries: {stats['total_deliveries']}")
print(f"Success rate: {stats['successful_deliveries'] / stats['total_deliveries']}")
```

#### Scheduler Management

```python
# Start the scheduler
digest_system.start_scheduler(interval=300)  # Check every 5 minutes

# Stop the scheduler
digest_system.stop_scheduler()
```

## Usage Examples

### Setting Up a Daily Sales Summary

```python
from src.digest_system import DigestSystem, DigestFrequency, DigestType, DigestFormat

# Initialize the digest system
digest_system = DigestSystem(data_dir="./data")

# Add a recipient for daily sales summaries
digest_system.add_recipient(
    user_id="sales_team",
    name="Sales Team",
    email="sales@example.com",
    slack_id="S123456",
    frequency=DigestFrequency.DAILY,
    digest_types=[DigestType.SALES_SUMMARY],
    preferred_format=DigestFormat.SLACK
)

# Start the scheduler
digest_system.start_scheduler()
```

### Recording and Analyzing Feedback

```python
# Record feedback for a delivery
digest_system.record_feedback(
    delivery_id="delivery1",
    feedback={
        "thumbs_up": True,
        "comment": "Very useful insights!"
    }
)

# Get delivery statistics
stats = digest_system.get_delivery_stats()
print(f"Positive feedback rate: {stats['feedback_stats']['thumbs_up'] / stats['total_deliveries']}")
```

## Best Practices

1. **Recipient Management**
   - Keep recipient information up to date
   - Use appropriate digest types and frequencies
   - Consider recipient time zones for scheduling

2. **Digest Generation**
   - Ensure data is complete and accurate
   - Use appropriate digest types for different insights
   - Include relevant context in digest content

3. **Delivery**
   - Monitor delivery success rates
   - Handle delivery failures gracefully
   - Respect recipient preferences

4. **Feedback**
   - Encourage recipient feedback
   - Analyze feedback patterns
   - Use feedback to improve digest content

5. **Monitoring**
   - Regularly check delivery statistics
   - Monitor system performance
   - Address issues promptly

## Troubleshooting

### Common Issues

1. **Delivery Failures**
   - Check recipient contact information
   - Verify delivery service configuration
   - Review error logs for details

2. **Scheduler Issues**
   - Ensure scheduler is running
   - Check interval settings
   - Verify system time is correct

3. **Data Persistence**
   - Check data directory permissions
   - Verify file integrity
   - Monitor disk space

### Error Handling

The system includes comprehensive error handling:
- Failed deliveries are logged and retried
- Invalid data is caught and reported
- System errors are logged for debugging

## Contributing

When contributing to the digest system:
1. Follow the existing code style
2. Add tests for new functionality
3. Update documentation as needed
4. Use meaningful commit messages

## License

This module is part of the V3Watchdog AI project and is subject to the project's license terms. 