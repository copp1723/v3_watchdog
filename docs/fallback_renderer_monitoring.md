# Fallback Renderer Monitoring Dashboard

This document outlines the configuration for monitoring the Fallback Renderer's error rates and performance metrics.

## Metrics Overview

The Fallback Renderer produces the following metrics:

1. **Error Counts**: Number of occurrences for each error type
2. **Render Time**: Time taken to render error messages
3. **Error Rate**: Errors per minute/hour/day
4. **User Impact**: Number of affected users

## Dashboard Configuration

### Error Rate Dashboard

```
Title: Fallback Renderer Error Rates
Time Range: Last 24 hours
Refresh Interval: 5 minutes

Panels:
1. Error Count by Type (Bar Chart)
   - Query: count(error_code) by error_code
   - Group By: error_code
   - Sort By: Count (Descending)
   - Thresholds:
     - Warning: > 10 errors/hour
     - Critical: > 50 errors/hour

2. Error Rate Trend (Line Chart)
   - Query: rate(error_count[5m])
   - Group By: error_code
   - Thresholds:
     - Warning: > 0.5 errors/minute
     - Critical: > 2 errors/minute

3. Top Error Types (Pie Chart)
   - Query: sum(error_count) by error_code
   - Limit: Top 5
   - Thresholds:
     - Warning: Any error type > 30% of total
     - Critical: Any error type > 50% of total

4. Error Rate by User Query (Table)
   - Query: count(error_count) by user_query, error_code
   - Sort By: Count (Descending)
   - Limit: Top 10
   - Columns:
     - User Query
     - Error Code
     - Count
     - Percentage

5. Error Rate by Time (Heatmap)
   - Query: count(error_count) by error_code, hour
   - Group By: error_code, hour
   - Color Scheme: Red-Yellow-Green
```

### Performance Dashboard

```
Title: Fallback Renderer Performance
Time Range: Last 24 hours
Refresh Interval: 5 minutes

Panels:
1. Render Time (Line Chart)
   - Query: avg(render_time)
   - Group By: error_code
   - Thresholds:
     - Warning: > 50ms
     - Critical: > 100ms

2. Render Time Percentiles (Line Chart)
   - Query: percentiles(render_time, 50, 90, 95, 99)
   - Group By: error_code
   - Thresholds:
     - Warning: p95 > 75ms
     - Critical: p95 > 150ms

3. Render Time Distribution (Histogram)
   - Query: histogram(render_time)
   - Buckets: 10
   - Thresholds:
     - Warning: > 10% in > 50ms bucket
     - Critical: > 10% in > 100ms bucket

4. Slow Renders (Table)
   - Query: render_time > 100ms
   - Sort By: render_time (Descending)
   - Limit: Top 10
   - Columns:
     - Error Code
     - Render Time
     - Timestamp
     - User Query
```

### User Impact Dashboard

```
Title: Fallback Renderer User Impact
Time Range: Last 24 hours
Refresh Interval: 5 minutes

Panels:
1. Affected Users (Line Chart)
   - Query: count_distinct(user_id) by error_code
   - Group By: error_code
   - Thresholds:
     - Warning: > 10 users/hour
     - Critical: > 50 users/hour

2. Error Rate by User Segment (Bar Chart)
   - Query: count(error_count) by user_segment, error_code
   - Group By: user_segment, error_code
   - Sort By: Count (Descending)
   - Thresholds:
     - Warning: Any segment > 20% error rate
     - Critical: Any segment > 40% error rate

3. User Retention Impact (Line Chart)
   - Query: count_distinct(user_id) by error_code, day
   - Group By: error_code, day
   - Thresholds:
     - Warning: > 10% drop in users
     - Critical: > 30% drop in users

4. Most Affected Users (Table)
   - Query: count(error_count) by user_id, error_code
   - Sort By: Count (Descending)
   - Limit: Top 10
   - Columns:
     - User ID
     - Error Code
     - Count
     - Last Error Time
```

## Alerts Configuration

### Error Rate Alerts

```
Alert: High Error Rate
Condition: rate(error_count[5m]) > 2
Duration: 5 minutes
Severity: Critical
Notification: Slack #alerts, Email
Description: Error rate exceeded 2 errors per minute for 5 minutes

Alert: Error Rate Spike
Condition: rate(error_count[5m]) / rate(error_count[1h]) > 3
Duration: 5 minutes
Severity: Warning
Notification: Slack #alerts
Description: Error rate increased by 3x compared to hourly average

Alert: New Error Type
Condition: count_distinct(error_code) > baseline + 1
Duration: 1 hour
Severity: Warning
Notification: Slack #alerts
Description: New error type detected in the system
```

### Performance Alerts

```
Alert: Slow Render Time
Condition: avg(render_time) > 100ms
Duration: 5 minutes
Severity: Warning
Notification: Slack #alerts
Description: Average render time exceeded 100ms for 5 minutes

Alert: Render Time Spike
Condition: avg(render_time) / avg(render_time[1h]) > 2
Duration: 5 minutes
Severity: Warning
Notification: Slack #alerts
Description: Render time increased by 2x compared to hourly average
```

### User Impact Alerts

```
Alert: High User Impact
Condition: count_distinct(user_id) > 50
Duration: 1 hour
Severity: Critical
Notification: Slack #alerts, Email, PagerDuty
Description: More than 50 users affected by errors in the last hour

Alert: User Retention Risk
Condition: count_distinct(user_id) / count_distinct(user_id[1d]) < 0.7
Duration: 1 day
Severity: Warning
Notification: Slack #alerts
Description: User retention dropped by more than 30% compared to previous day
```

## Dashboard Integration

These dashboards can be integrated with:

1. **Grafana**: For visualization and alerting
2. **Prometheus**: For metric collection and storage
3. **ELK Stack**: For log analysis and correlation
4. **Datadog**: For APM and infrastructure monitoring

## Implementation Notes

1. **Metric Collection**:
   - Use the `logger.info()` method with the `extra` parameter to include metrics
   - Ensure thread safety when incrementing error counters
   - Use consistent metric names across the application

2. **Alert Thresholds**:
   - Adjust thresholds based on historical data
   - Consider business impact when setting severity levels
   - Implement alert fatigue prevention (e.g., grouping similar alerts)

3. **Dashboard Maintenance**:
   - Review and update dashboards quarterly
   - Add new error types as they are discovered
   - Remove obsolete metrics and alerts 