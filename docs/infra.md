# V3 Watchdog AI Infrastructure

This document outlines the infrastructure setup for the V3 Watchdog AI platform, including required services, configuration, and deployment considerations.

## System Architecture

The V3 Watchdog AI platform consists of several components deployed across multiple environments:

```
+---------------------+       +----------------------+       +----------------------+
|                     |       |                      |       |                      |
|  Web UI             |       |  Application Server  |       |  Data Processing     |
|  - Streamlit        |------>|  - FastAPI           |------>|  - Background Tasks  |
|  - React Components |       |  - Authentication    |       |  - Report Generation |
|                     |       |  - API Gateway       |       |  - Nova ACT          |
+---------------------+       +----------------------+       +----------------------+
         |                              |                            |
         |                              |                            |
         v                              v                            v
+-----------------------------------------------------------------------------------+
|                                                                                   |
|                             Shared Infrastructure                                 |
|  - Redis (Caching, Session Management, Background Tasks)                         |
|  - PostgreSQL (User Data, Configuration, Audit Logs)                             |
|  - S3/Object Storage (Reports, Large Data Files)                                 |
|  - Monitoring & Logging                                                          |
|                                                                                   |
+-----------------------------------------------------------------------------------+
```

## Required Services

### Application Components

1. **Web UI Server**
   - Streamlit server for main UI
   - Serves dynamic web interface
   - Requirements: 2 vCPU, 4GB RAM

2. **Application Server**
   - FastAPI-based REST API
   - Authentication and authorization
   - API gateway for services
   - Requirements: 2 vCPU, 4GB RAM

3. **Data Processing Service**
   - Background tasks processing
   - Report generation
   - Data analysis and insight generation
   - Requirements: 4 vCPU, 8GB RAM (scales based on load)

4. **Nova ACT Service**
   - Integration with dealership systems
   - Data collection and normalization
   - Requirements: 2 vCPU, 4GB RAM

### Infrastructure Services

1. **Redis**
   - Used for:
     - Caching (performance improvement)
     - Session management (user sessions)
     - Task queues (background jobs)
   - Requirements: 2GB RAM minimum

2. **PostgreSQL**
   - Main application database
   - Stores:
     - User accounts and profiles
     - Configuration settings
     - Audit logs
     - Report metadata
   - Requirements: 4GB RAM, 50GB storage (minimum)

3. **Object Storage (S3-compatible)**
   - Stores:
     - Generated reports
     - Uploaded data files
     - Trained models
   - Requirements: 100GB storage (scales with usage)

4. **Monitoring & Logging**
   - Prometheus for metrics
   - Grafana for dashboards
   - Centralized logging solution

## Environment Variables

The following environment variables must be configured for proper operation of the V3 Watchdog AI platform:

### Core Configuration

```
# Application Configuration
WATCHDOG_ENV=production|staging|development
WATCHDOG_DEBUG=false
WATCHDOG_LOG_LEVEL=INFO
WATCHDOG_ALLOWED_ORIGINS=https://example.com,https://admin.example.com

# Security
WATCHDOG_SECRET_KEY=${SECRET}
WATCHDOG_JWT_SECRET=${SECRET}
WATCHDOG_ENCRYPTION_KEY=${SECRET}
WATCHDOG_AUTH_COOKIE_SECURE=true
WATCHDOG_AUTH_COOKIE_DOMAIN=example.com

# Database
WATCHDOG_DB_HOST=postgres.example.com
WATCHDOG_DB_PORT=5432
WATCHDOG_DB_NAME=watchdog
WATCHDOG_DB_USER=${SECRET}
WATCHDOG_DB_PASSWORD=${SECRET}
WATCHDOG_DB_SSL=true
WATCHDOG_DB_POOL_SIZE=5
WATCHDOG_DB_MAX_OVERFLOW=10

# Redis
WATCHDOG_REDIS_URL=redis://redis.example.com:6379
WATCHDOG_REDIS_PASSWORD=${SECRET}
WATCHDOG_REDIS_DB=0
WATCHDOG_REDIS_SSL=true

# Storage
WATCHDOG_STORAGE_TYPE=s3
WATCHDOG_S3_ENDPOINT=s3.amazonaws.com
WATCHDOG_S3_REGION=us-west-2
WATCHDOG_S3_BUCKET=watchdog-data
WATCHDOG_S3_ACCESS_KEY=${SECRET}
WATCHDOG_S3_SECRET_KEY=${SECRET}
WATCHDOG_S3_SECURE=true
```

### Feature-Specific Configuration

```
# Audit Logging (Phase 2)
WATCHDOG_AUDIT_ENABLED=true
WATCHDOG_AUDIT_LOG_LEVEL=INFO
WATCHDOG_AUDIT_RETENTION_DAYS=90
WATCHDOG_AUDIT_ENCRYPTION_ENABLED=true

# Session Management (Phase 2)
WATCHDOG_SESSION_STORE=redis
WATCHDOG_SESSION_TIMEOUT=1800
WATCHDOG_SESSION_MAX_LIFETIME=28800
WATCHDOG_SESSION_ENFORCE_SAME_IP=false
WATCHDOG_SESSION_ENFORCE_SAME_AGENT=true
WATCHDOG_SESSION_REFRESH_TOKEN_ROTATION=true

# Predictive Forecasting (Phase 2)
WATCHDOG_FORECAST_ENABLED=true
WATCHDOG_FORECAST_MODELS_PATH=/app/models
WATCHDOG_FORECAST_DEFAULT_ALGORITHM=prophet
WATCHDOG_FORECAST_TRAINING_SCHEDULE=daily
WATCHDOG_FORECAST_MAX_HISTORY=2years
WATCHDOG_FORECAST_CONFIDENCE_LEVEL=0.80

# Nova ACT
WATCHDOG_NOVA_ENABLED=true
WATCHDOG_NOVA_CONFIG_PATH=/app/nova/config
WATCHDOG_NOVA_CREDENTIALS_PATH=/app/nova/credentials
WATCHDOG_NOVA_RATE_LIMIT=100
WATCHDOG_NOVA_TIMEOUT=30
WATCHDOG_NOVA_SCHEDULER_ENABLED=true
```

## Feature Flags

Feature flags control the availability of specific features and can be adjusted without redeployment:

```json
{
  "global": {
    "maintenance_mode": false,
    "read_only_mode": false
  },
  "ui": {
    "modern_theme_enabled": true,
    "mobile_enhancements": true,
    "chat_interface": true,
    "flag_panel_visible": true,
    "insights_dashboard": true
  },
  "insights": {
    "executive_insights_enabled": true,
    "custom_insights_enabled": true,
    "adaptive_learning_enabled": true, 
    "forecasting_enabled": false
  },
  "integrations": {
    "nova_act_enabled": true,
    "webhooks_enabled": true,
    "export_formats": ["csv", "pdf", "json"],
    "import_formats": ["csv", "excel"]
  },
  "security": {
    "audit_logging_enabled": true,
    "detailed_action_logging": true,
    "session_jwt_refresh": true,
    "session_concurrent_limit": 5,
    "ip_geolocation_check": false
  },
  "reporting": {
    "scheduler_enabled": true,
    "email_delivery": true,
    "dashboard_delivery": true,
    "forecast_model_training": true,
    "forecast_anomaly_detection": true
  }
}
```

## Resource Requirements

### Minimum Production Setup

- **Total CPU**: 10 vCPU
- **Total RAM**: 20GB
- **Storage**: 
  - Database: 50GB
  - Object Storage: 100GB

### Scaling Considerations

- The Data Processing Service should be scaled horizontally based on load
- Redis memory should be scaled based on cache hit ratio and queue backlog
- Database IOPS should be monitored and scaled accordingly

## Network Requirements

### Ports

- **Web UI**: 8501 (Streamlit), typically behind reverse proxy on 443
- **API Server**: 8000 (FastAPI), typically behind reverse proxy on 443
- **Redis**: 6379
- **PostgreSQL**: 5432
- **Nova ACT Service**: 8080

### Connectivity

- Web UI → API Server: Internal network
- API Server → Data Processing: Internal network
- All services → Redis: Internal network
- All services → PostgreSQL: Internal network
- All services → S3/Object Storage: Secure connection (VPC endpoint or TLS)
- Nova ACT → External DMS APIs: Secure outbound internet access

## Security Considerations

### Authentication

- JWT-based token authentication
- Redis-backed session management
- Optional OIDC/SAML integration for enterprise deployments

### Data Protection

- Encryption at rest for all databases and object storage
- TLS for all network communication
- PII handled according to data protection regulations
- Audit logging of all data access

### Monitoring

- System metrics (CPU, memory, disk, network)
- Application metrics (request rates, error rates, latency)
- Business metrics (active users, insights generated, reports created)
- Alert thresholds for abnormal conditions

## Backup and Retention

### Redis and Audit Log Retention

- **Redis Audit Logs**: In-memory audit logs use the `WATCHDOG_AUDIT_LOG_TTL_SECONDS` environment variable (default: 7776000 seconds = 90 days)
- **Automated Cleanup**: Redis TTL automatically removes audit log entries older than the configured retention period
- **Backup Before Expiry**: Audit logs are archived to S3 before Redis TTL expiration

### ElastiCache Snapshots

- **Automatic Snapshots**: Daily automatic snapshots of Redis ElastiCache cluster
- **Retention Period**: 7-day snapshot retention (configured via `snapshot_retention_limit = 7`)
- **Snapshot Window**: 00:00-01:00 UTC daily
- **Restoration**: Point-in-time recovery available within the retention period

### S3 Lifecycle Rules

- **Audit Log Archive Bucket**:
  - Transition to Glacier storage class after 30 days (cost optimization)
  - Expire (permanently delete) objects after 365 days (compliance requirement)
  - Versioning enabled for additional data protection
  - Encryption at rest using AES-256
- **Application Data Bucket**:
  - Standard storage class
  - Versioning enabled
  - No automatic expiration

### Database Backups

- Database: Daily full backups, hourly incremental backups
- Retention Period: 30 days for full backups, 7 days for incremental backups
- Encryption: All backups are encrypted at rest

### Disaster Recovery

- Configuration: Infrastructure as code, stored in version control
- Disaster Recovery: RPO 1 hour, RTO 4 hours

## Deployment Strategies

### CI/CD Pipeline

- Automated testing on all commits
- Staging deployment for verification
- Blue/green deployment for production
- Canary releases for high-risk features

### Containerization

- Docker-based deployment
- Orchestration via Kubernetes or similar
- Helm charts for deployment configuration

## Monitoring and Healthchecks

### Health Endpoints

- `/health` - Overall application health
- `/health/db` - Database connectivity
- `/health/redis` - Redis connectivity
- `/health/storage` - Object storage connectivity
- `/health/nova` - Nova ACT service status

### Metrics

- Request rate, latency, and error rate by endpoint
- User session metrics
- Background task throughput and backlog
- Cache hit/miss ratio
- Database query performance

## Troubleshooting

### Common Issues

- Redis connection failures: Check network, credentials, and memory usage
- Database performance: Check query patterns, indexing, and connection pool settings
- API timeouts: Check downstream service health and timeout configurations
- S3 access issues: Check credentials, bucket permissions, and network connectivity

### Logs

- Application logs: `/var/log/watchdog/app.log`
- Access logs: `/var/log/watchdog/access.log`
- Error logs: `/var/log/watchdog/error.log`
- Audit logs: `/var/log/watchdog/audit.log`

### Support Information

- Support email: support@watchdogai.com
- Technical documentation: https://docs.watchdogai.com
- Status page: https://status.watchdogai.com