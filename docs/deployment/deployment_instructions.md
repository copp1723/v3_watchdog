# Column Mapping Pipeline Deployment Instructions

## Overview

This document provides instructions for deploying the LLM-driven column mapping pipeline to production, now that it has been successfully tested in staging.

## Deployment Checklist

### 1. Pre-Deployment Preparation

- [x] Merge feature branch to main (completed)
- [x] Deploy to staging environment (completed)
- [x] Verify functionality in staging (completed)
- [x] Collect performance metrics (completed)
- [x] Gather user feedback (completed)
- [x] Update documentation (completed)
- [ ] Schedule production deployment

### 2. Configuration Settings

Ensure the following environment variables are set for the production deployment:

```
# Redis settings for column mapping cache
REDIS_CACHE_ENABLED=true
REDIS_HOST=<production_redis_host>
REDIS_PORT=6379
REDIS_DB=0
COLUMN_MAPPING_CACHE_TTL=86400
COLUMN_MAPPING_CACHE_PREFIX=watchdog:column_mapping:

# Column mapping behavior
DROP_UNMAPPED_COLUMNS=false  # Keep false for production as default
MIN_CONFIDENCE_TO_AUTOMAP=0.7  # Keep 0.7 as it has optimal performance
```

### 3. Deployment Steps

1. Pull the latest changes from the main branch:
   ```bash
   git checkout main
   git pull origin main
   ```

2. Deploy the application to production:
   ```bash
   ./deploy.sh production
   ```

3. Verify the deployment by checking the application logs:
   ```bash
   ./check_logs.sh production
   ```

4. Run smoke tests against the production environment:
   ```bash
   python verify_column_mapping.py --env=production
   ```

### 4. Post-Deployment Verification

- [ ] Verify the application is running correctly
- [ ] Confirm Redis caching is working properly
- [ ] Perform end-to-end test with a real dataset
- [ ] Check AgentOps metrics to ensure performance meets expectations
- [ ] Monitor error rates for the first 24 hours

### 5. Rollback Plan

If issues are detected during or after deployment, follow these steps to rollback:

1. Revert to the previous stable version:
   ```bash
   ./deploy.sh production --version=<previous_version>
   ```

2. Verify the rollback was successful:
   ```bash
   ./check_logs.sh production
   ```

3. Document the issues that led to the rollback.

## Documentation

All documentation has been updated and is available at:

- API Documentation: `docs/api.md`
- Prompt Versioning: `docs/prompt_versions.md`
- Deployment Guide: `docs/deployment.md`
- Performance Summary: `performance_summary.md`
- User Feedback: `user_feedback_summary.md`

## Support

For any issues during or after deployment, contact:

- Engineering Lead: eng-lead@watchdogai.com
- DevOps Team: devops@watchdogai.com