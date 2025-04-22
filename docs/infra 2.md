## Data Retention Policies

Watchdog AI enforces strict data retention policies to comply with GDPR and CCPA requirements.

### Audit Logs
- **Retention Period**: 90 days (via Redis TTL)
- **Storage**: Redis + S3 archive
- **Environment Variable**: `AUDIT_LOG_TTL_SECONDS=7776000` (90 days in seconds)
- **Fallback**: System defaults to 90 days if not configured

### Session Data
- **Retention Period**: 365 days (via Redis TTL)
- **Storage**: Redis only
- **Environment Variable**: `SESSION_TTL_SECONDS=31536000` (365 days in seconds)
- **Fallback**: System defaults to 365 days if not configured

### Example .env Configuration
```
# Data Retention Settings
AUDIT_LOG_TTL_SECONDS=7776000  # 90 days
SESSION_TTL_SECONDS=31536000    # 365 days
```

Note: These TTLs are enforced automatically on all Redis key writes in the respective modules. 