# Insights Configuration Guide

## Overview
This guide covers configuration options for the Watchdog AI insights system, including data quality thresholds, schema profiles, and business rules.

## Data Quality Configuration

### Thresholds
Data quality thresholds are defined in `config/insight_quality.yml`. These control when warnings and errors are triggered:

```yaml
thresholds:
  missing_data:
    warning: 10.0  # Percentage that triggers warning
    error: 20.0    # Percentage that triggers error
  sample_size:
    minimum: 30    # Minimum sample size for reliable insights
  outliers:
    max_percent: 5.0  # Maximum percentage of outliers allowed
```

### Quality Badges
Visual indicators for data quality are configured in the same file:

```yaml
badges:
  high:
    icon: "✅"
    color: "#2e7d32"
  medium:
    icon: "⚠️"
    color: "#f9a825"
  low:
    icon: "🔴"
    color: "#c62828"
```

### Feature Flags
Data quality features can be toggled using environment variables:
- `ENABLE_DATA_QUALITY_WARNINGS`: Enable/disable quality warning badges
- `ENABLE_SCHEMA_ADAPTATION`: Enable/disable dynamic schema profile updates

## Schema Profiles

### Base Profiles
Base schema profiles are loaded from `config/schema_profiles/`:
- `default.json`: Default column mappings and rules
- `executive.json`: Executive-specific views
- `analyst.json`: Analyst-specific views

### User Overrides
User-specific schema adjustments are stored in:
```
config/schema_profiles/users/{user_id}/profile.json
```

### Dynamic Updates
The system learns from user feedback to suggest schema updates:
1. Missing column aliases
2. Common query terms
3. Preferred metric names

## Business Rules

### Rule Configuration
Business rules are defined in `BusinessRuleRegistry.yaml`:

```yaml
rules:
  data_quality:
    type: threshold
    column: "*"
    threshold:
      missing_data: 20.0
      sample_size: 30
    action: downgrade_confidence
```

### Custom Rules
Add custom rules by extending the `BusinessRuleEngine`:

```python
engine.add_rule("min_data_quality", {
    "type": "custom",
    "function": "validate_quality",
    "threshold": 0.8,
    "message": "Insufficient data quality"
})
```

## Recommendation Templates
Custom recommendation templates in `config/recommendations.yml`:

```yaml
templates:
  missing_data:
    warning: "Consider collecting more data for {column}"
    error: "High percentage of missing data impacts results"
  sample_size:
    warning: "Limited sample size of {size} records"
```

## Monitoring & Metrics

### Quality Trends
Track data quality metrics over time:
- NaN percentages
- Excluded rows
- Quality scores

### Schema Adaptation
Monitor schema profile changes:
- Suggested column mappings
- User acceptance rates
- Query success rates

## Environment Variables
- `QUALITY_THRESHOLD_WARNING`: Override warning threshold (default: 10.0)
- `QUALITY_THRESHOLD_ERROR`: Override error threshold (default: 20.0)
- `MIN_SAMPLE_SIZE`: Override minimum sample size (default: 30)
- `ENABLE_DATA_QUALITY_WARNINGS`: Enable quality warnings (default: true)
- `ENABLE_SCHEMA_ADAPTATION`: Enable schema learning (default: true)

## API Endpoints
- `GET /api/quality/trends`: Get data quality trends
- `GET /api/quality/thresholds`: Get current thresholds
- `PUT /api/quality/thresholds`: Update thresholds
- `POST /api/schema/suggest`: Submit schema suggestion
- `GET /api/schema/profile`: Get current schema profile