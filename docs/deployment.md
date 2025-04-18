# V3 Watchdog AI Deployment Guide

This document provides comprehensive instructions for deploying the V3 Watchdog AI platform in various environments.

## Prerequisites

Before deploying the V3 Watchdog AI platform, ensure the following prerequisites are met:

### System Requirements

- **Operating System**: Linux (Ubuntu 20.04+ or RHEL/CentOS 8+)
- **Python**: Version 3.9+
- **Docker**: Version 20.10+ (if using containerized deployment)
- **Kubernetes**: Version 1.23+ (if using Kubernetes)
- **Database**: PostgreSQL 13+
- **Redis**: Version 6.2+
- **Storage**: S3-compatible object storage

### Network Requirements

- Outbound internet access for package installation and API integrations
- Internal network connectivity between all components
- HTTPS certificates for production deployments

### Access Requirements

- Database administration credentials
- Object storage credentials and bucket access
- Docker registry access (if using containerized deployment)
- Domain name and DNS control (for production)

## Deployment Options

The V3 Watchdog AI platform can be deployed using several methods, depending on your infrastructure and requirements.

### 1. Standard Deployment (Python/Virtualenv)

#### Setup

1. Create a Python virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up the configuration file:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. Initialize the database:
   ```bash
   python setup_env.py --init-db
   ```

#### Running the Application

1. Start the web UI:
   ```bash
   ./run.sh --ui
   ```

2. Start the API server:
   ```bash
   ./run.sh --api
   ```

3. Start the data processing service:
   ```bash
   ./run.sh --processing
   ```

4. Start the Nova ACT service (if enabled):
   ```bash
   ./run.sh --nova
   ```

### 2. Docker Deployment

#### Setup

1. Build the Docker images:
   ```bash
   docker-compose build
   ```

2. Set up the configuration file:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

#### Running the Application

1. Start all services:
   ```bash
   docker-compose up -d
   ```

2. View logs:
   ```bash
   docker-compose logs -f
   ```

3. Stop all services:
   ```bash
   docker-compose down
   ```

### 3. Kubernetes Deployment

#### Setup

1. Configure Kubernetes secrets:
   ```bash
   kubectl create secret generic watchdog-secrets \
     --from-file=.env
   ```

2. Apply the Kubernetes configurations:
   ```bash
   kubectl apply -f k8s/namespace.yaml
   kubectl apply -f k8s/configmap.yaml
   kubectl apply -f k8s/services/
   kubectl apply -f k8s/deployments/
   ```

3. Check deployment status:
   ```bash
   kubectl get pods -n watchdog
   ```

## Configuration

### Environment Variables

The following environment variables must be configured for proper operation. See `infra.md` for the comprehensive list.

Critical variables that must be set for any deployment:

```
# Core
WATCHDOG_ENV=production|staging|development
WATCHDOG_SECRET_KEY=<random-secret-key>

# Database
WATCHDOG_DB_HOST=<db-host>
WATCHDOG_DB_NAME=<db-name>
WATCHDOG_DB_USER=<db-user>
WATCHDOG_DB_PASSWORD=<db-password>

# Redis
WATCHDOG_REDIS_URL=<redis-url>
WATCHDOG_REDIS_PASSWORD=<redis-password>

# Storage
WATCHDOG_STORAGE_TYPE=s3|local
WATCHDOG_S3_BUCKET=<bucket-name>
WATCHDOG_S3_ACCESS_KEY=<access-key>
WATCHDOG_S3_SECRET_KEY=<secret-key>

# Feature-specific (Phase 2)
WATCHDOG_AUDIT_ENABLED=true|false
WATCHDOG_SESSION_STORE=redis|db
WATCHDOG_FORECAST_ENABLED=true|false
```

### Feature Flags

Feature flags control the availability of specific features and can be adjusted without redeployment.

Production environments should use Redis or database-backed feature flags:

```bash
# Set a feature flag
python manage.py set_feature_flag --name "insights.forecasting_enabled" --value true

# View current feature flags
python manage.py list_feature_flags
```

For development, feature flags can be configured in a JSON file:

```json
{
  "insights": {
    "forecasting_enabled": true,
    "adaptive_learning_enabled": true
  }
}
```

## Database Initialization

The database must be initialized before the first use:

```bash
# Using the CLI tool
python manage.py initialize_db

# Or using the setup script
python setup_env.py --init-db
```

For upgrades and migrations:

```bash
python manage.py migrate
```

## Data Migration

When migrating data from a previous version:

```bash
# Export data from old system
python manage.py export_data --output old_data.json

# Import data to new system
python manage.py import_data --input old_data.json
```

## SSL/TLS Configuration

For production deployments, configure SSL/TLS:

### Using Nginx as Reverse Proxy

```nginx
server {
    listen 443 ssl;
    server_name watchdog.example.com;

    ssl_certificate /etc/ssl/certs/watchdog.crt;
    ssl_certificate_key /etc/ssl/private/watchdog.key;
    
    # Modern SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305;
    ssl_prefer_server_ciphers off;
    
    # HSTS (optional)
    add_header Strict-Transport-Security "max-age=63072000" always;
    
    # UI service
    location / {
        proxy_pass http://localhost:8501;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # API service
    location /api/ {
        proxy_pass http://localhost:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Using Kubernetes with Cert-Manager

```yaml
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: watchdog-cert
  namespace: watchdog
spec:
  secretName: watchdog-tls
  issuerRef:
    name: letsencrypt-prod
    kind: ClusterIssuer
  dnsNames:
    - watchdog.example.com
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: watchdog-ingress
  namespace: watchdog
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - watchdog.example.com
    secretName: watchdog-tls
  rules:
  - host: watchdog.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: watchdog-ui
            port:
              number: 8501
      - path: /api/
        pathType: Prefix
        backend:
          service:
            name: watchdog-api
            port:
              number: 8000
```

## Monitoring Setup

### Prometheus and Grafana

1. Install Prometheus and Grafana:
   ```bash
   # Using Helm in Kubernetes
   helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
   helm repo update
   helm install prometheus prometheus-community/kube-prometheus-stack \
     --namespace monitoring --create-namespace
   ```

2. Configure Prometheus to scrape watchdog metrics:
   ```yaml
   # prometheus.yml addition
   scrape_configs:
     - job_name: 'watchdog'
       scrape_interval: 15s
       metrics_path: '/metrics'
       static_configs:
         - targets: ['watchdog-api:8000', 'watchdog-processing:8080']
   ```

3. Import the Watchdog dashboard into Grafana:
   ```bash
   curl -X POST -H "Content-Type: application/json" -d @dashboards/watchdog-overview.json \
     http://admin:password@grafana:3000/api/dashboards/db
   ```

### Health Checks

Set up health check monitoring:

```bash
# Check overall health
curl https://watchdog.example.com/api/health

# Check component health
curl https://watchdog.example.com/api/health/db
curl https://watchdog.example.com/api/health/redis
curl https://watchdog.example.com/api/health/storage
```

## Backup Configuration

### Database Backups

Set up automatic database backups:

```bash
# Using cron
0 2 * * * pg_dump -U watchdog -d watchdog | gzip > /backups/watchdog_$(date +\%Y\%m\%d).sql.gz

# Using Kubernetes CronJob
apiVersion: batch/v1
kind: CronJob
metadata:
  name: watchdog-db-backup
  namespace: watchdog
spec:
  schedule: "0 2 * * *"
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup
            image: postgres:13
            command:
            - /bin/sh
            - -c
            - pg_dump -h $(WATCHDOG_DB_HOST) -U $(WATCHDOG_DB_USER) -d $(WATCHDOG_DB_NAME) | gzip > /backups/watchdog_$(date +\%Y\%m\%d).sql.gz
            env:
            - name: PGPASSWORD
              valueFrom:
                secretKeyRef:
                  name: watchdog-secrets
                  key: WATCHDOG_DB_PASSWORD
            volumeMounts:
            - name: backup-volume
              mountPath: /backups
          volumes:
          - name: backup-volume
            persistentVolumeClaim:
              claimName: backup-pvc
          restartPolicy: OnFailure
```

### Object Storage Backups

S3-compatible storage typically provides versioning and replication:

```bash
# Enable versioning on AWS S3
aws s3api put-bucket-versioning \
  --bucket watchdog-data \
  --versioning-configuration Status=Enabled

# Configure cross-region replication
aws s3api put-bucket-replication \
  --bucket watchdog-data \
  --replication-configuration file://replication-config.json
```

## Scaling Configuration

### Horizontal Scaling

For Kubernetes deployments:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: watchdog-processing-hpa
  namespace: watchdog
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: watchdog-processing
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### Vertical Scaling

Adjust resource limits based on usage patterns:

```yaml
resources:
  requests:
    memory: "4Gi"
    cpu: "1"
  limits:
    memory: "8Gi"
    cpu: "2"
```

## Logging Configuration

Configure centralized logging:

```yaml
# logback.xml
<configuration>
  <appender name="FILE" class="ch.qos.logback.core.FileAppender">
    <file>/var/log/watchdog/app.log</file>
    <encoder>
      <pattern>%date %level [%thread] %logger{10} [%file:%line] %msg%n</pattern>
    </encoder>
  </appender>
  
  <appender name="STDOUT" class="ch.qos.logback.core.ConsoleAppender">
    <encoder>
      <pattern>%d{HH:mm:ss.SSS} [%thread] %-5level %logger{36} - %msg%n</pattern>
    </encoder>
  </appender>
  
  <root level="info">
    <appender-ref ref="FILE" />
    <appender-ref ref="STDOUT" />
  </root>
</configuration>
```

For audit logging (Phase 2), enable the appropriate settings:

```
WATCHDOG_AUDIT_ENABLED=true
WATCHDOG_AUDIT_LOG_LEVEL=INFO
WATCHDOG_AUDIT_RETENTION_DAYS=90
```

## Troubleshooting

### Common Issues

#### Database Connection Issues

```bash
# Check database connectivity
python -c "from src.utils.db import get_db; conn = get_db(); print('Connected:', conn.closed == 0)"

# Check database migrations
python manage.py check_migrations
```

#### Redis Connection Issues

```bash
# Check Redis connectivity
python -c "import redis; r = redis.from_url('$WATCHDOG_REDIS_URL'); print('Connected:', r.ping())"

# Check Redis memory usage
redis-cli -u $WATCHDOG_REDIS_URL info memory
```

#### API Service Issues

```bash
# Check API service health
curl -v https://watchdog.example.com/api/health

# Check API logs
kubectl logs -f deployment/watchdog-api -n watchdog
```

#### UI Service Issues

```bash
# Check UI service logs
kubectl logs -f deployment/watchdog-ui -n watchdog

# Restart UI service
kubectl rollout restart deployment/watchdog-ui -n watchdog
```

### Diagnostic Commands

```bash
# Check system status
python manage.py system_check

# View recent logs
python manage.py show_logs --lines 100

# Check feature flags
python manage.py list_feature_flags

# Test email delivery
python manage.py test_email --recipient admin@example.com
```

## Upgrade Procedure

### Standard Upgrade

1. Backup the database:
   ```bash
   pg_dump -U watchdog -d watchdog | gzip > watchdog_backup_before_upgrade.sql.gz
   ```

2. Update the code:
   ```bash
   git pull
   pip install -r requirements.txt
   ```

3. Run migrations:
   ```bash
   python manage.py migrate
   ```

4. Restart services:
   ```bash
   ./run.sh --restart-all
   ```

### Containerized Upgrade

1. Pull new images:
   ```bash
   docker-compose pull
   ```

2. Apply migrations:
   ```bash
   docker-compose run --rm api python manage.py migrate
   ```

3. Restart services:
   ```bash
   docker-compose down
   docker-compose up -d
   ```

### Kubernetes Upgrade

1. Update image versions:
   ```bash
   kubectl set image deployment/watchdog-api api=watchdog-api:new-version -n watchdog
   kubectl set image deployment/watchdog-ui ui=watchdog-ui:new-version -n watchdog
   kubectl set image deployment/watchdog-processing processing=watchdog-processing:new-version -n watchdog
   ```

2. Run database migrations:
   ```bash
   kubectl create job --from=cronjob/database-migration upgrade-migration -n watchdog
   ```

3. Monitor the upgrade:
   ```bash
   kubectl rollout status deployment/watchdog-api -n watchdog
   kubectl rollout status deployment/watchdog-ui -n watchdog
   kubectl rollout status deployment/watchdog-processing -n watchdog
   ```

## Post-Deployment Verification

After deployment, verify the system is functioning correctly:

```bash
# Check all services are running
python manage.py service_status

# Verify database connectivity
python manage.py db_check

# Verify Redis connectivity
python manage.py redis_check

# Verify storage connectivity
python manage.py storage_check

# Run integration tests
python -m pytest tests/integration/

# Verify UI access
curl -I https://watchdog.example.com/

# Verify API access
curl -I https://watchdog.example.com/api/health
```

## Security Hardening

For production deployments, apply these security measures:

```bash
# Set secure file permissions
find . -type f -name "*.py" -exec chmod 644 {} \;
find . -type d -exec chmod 755 {} \;
chmod 600 .env

# Secure configuration files
chmod 600 /path/to/credentials/*

# Set up firewall rules
ufw allow 80/tcp
ufw allow 443/tcp
ufw deny 8000/tcp
ufw deny 8501/tcp
ufw enable

# Set up fail2ban for SSH protection
apt-get install fail2ban
cp /etc/fail2ban/jail.conf /etc/fail2ban/jail.local
# Edit jail.local to configure
service fail2ban restart
```

## Maintenance Procedures

### Routine Maintenance

1. Database maintenance:
   ```bash
   # Vacuum and analyze the database
   python manage.py db_maintenance
   
   # Or directly with PostgreSQL
   psql -U watchdog -d watchdog -c "VACUUM ANALYZE;"
   ```

2. Log rotation:
   ```bash
   # Set up logrotate
   cat > /etc/logrotate.d/watchdog << EOF
   /var/log/watchdog/*.log {
       daily
       missingok
       rotate 14
       compress
       delaycompress
       notifempty
       create 0640 watchdog watchdog
       sharedscripts
       postrotate
           systemctl reload watchdog >/dev/null 2>&1 || true
       endscript
   }
   EOF
   ```

3. Cache clearing:
   ```bash
   # Clear Redis cache
   python manage.py clear_cache --type all|data|session|metrics
   ```

### Scheduled Maintenance

For planned downtime:

1. Enable maintenance mode:
   ```bash
   python manage.py set_feature_flag --name "global.maintenance_mode" --value true
   ```

2. Wait for active sessions to complete:
   ```bash
   python manage.py active_sessions --wait-for-completion
   ```

3. Stop services:
   ```bash
   docker-compose down
   # or
   kubectl scale deployment --all --replicas=0 -n watchdog
   ```

4. Perform maintenance tasks

5. Restart services:
   ```bash
   docker-compose up -d
   # or
   kubectl scale deployment --all --replicas=1 -n watchdog
   ```

6. Disable maintenance mode:
   ```bash
   python manage.py set_feature_flag --name "global.maintenance_mode" --value false
   ```

## Support

For additional support, contact:

- Technical support: support@watchdogai.com
- Documentation: https://docs.watchdogai.com
- GitHub repository: https://github.com/watchdogai/v3-platform