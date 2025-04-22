# Schema Evolution System Handoff Document

## Overview

The Schema Evolution System is designed to implement self-healing data mapping and schema adaptation capabilities for the Watchdog AI platform. This document provides a comprehensive guide for the next phase of development, focusing on the AdaptiveSchema Engine, security & compliance, and observability components.

## Current State

The current implementation includes:
- Basic column mapping functionality with Redis caching
- Option to drop unmapped columns
- Initial documentation and testing framework

## Next Phase Requirements

### 1. AdaptiveSchema Engine

#### 1.1 Confidence Scoring System

The confidence scoring system will use multiple algorithms to determine the reliability of column mappings:

```python
class ConfidenceCalculator:
    def calculate_confidence(self, original: str, canonical: str, context: dict) -> float:
        """
        Calculate mapping confidence using multiple algorithms:
        - Levenshtein distance for string similarity
        - Jaccard similarity for set-based comparison
        - Feedback-based weighting from historical mappings
        """
        levenshtein_score = self._calculate_levenshtein_score(original, canonical)
        jaccard_score = self._calculate_jaccard_score(original, canonical)
        feedback_score = self._calculate_feedback_score(original, canonical, context)
        
        # Load weights from configuration
        weights = self._load_confidence_weights()
        
        # Weighted combination of scores
        return (
            weights['levenshtein'] * levenshtein_score +
            weights['jaccard'] * jaccard_score +
            weights['feedback'] * feedback_score
        )
```

#### 1.2 Batch Mapping Correction UI/API

Implement a REST API and UI for batch mapping corrections:

```python
@router.post("/api/v1/mappings/batch")
async def batch_update_mappings(
    request: BatchMappingRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Update multiple column mappings in a single request.
    Requires admin privileges.
    """
    # Validate request
    # Apply mappings
    # Log changes
    # Return results
```

#### 1.3 Rollback System

Implement a rollback mechanism using a remapping log:

```python
class MappingRollback:
    def __init__(self):
        self.redis_client = Redis()
        self.nova_client = NovaClient()  # For admin authentication
        self.local_token_cache = LocalTokenCache()  # For resilience during Nova outages
        
    async def rollback_mapping(
        self,
        mapping_id: str,
        admin_token: str,
        reason: str
    ) -> bool:
        """
        Roll back a mapping to its previous state.
        Requires admin authentication via Nova.
        """
        # Verify admin token (with local cache fallback)
        # Get mapping history
        # Apply rollback
        # Log rollback event
```

### 2. Security & Compliance

#### 2.1 API Rate Limiting

Implement rate limiting for the mapping UI:

```python
class MappingRateLimiter:
    def __init__(self):
        self.redis_client = Redis()
        
    async def check_rate_limit(
        self,
        user_id: str,
        endpoint: str
    ) -> bool:
        """
        Check if user has exceeded rate limits for mapping endpoints.
        Different limits for different endpoints:
        - UI: 100 requests per minute
        - API: 1000 requests per minute
        """
```

#### 2.2 PII Audit System

Implement PII auditing for mapping logs:

```python
class PIIAuditor:
    def __init__(self):
        self.pii_patterns = self._load_pii_patterns()
        
    def audit_mapping_logs(self, logs: List[Dict]) -> Dict[str, List[str]]:
        """
        Audit mapping logs for PII exposure.
        Returns dict of PII types found and their locations.
        """
```

### 3. Observability

#### 3.1 Structured Logging

Implement structured JSON logging for mapping changes:

```python
class MappingLogger:
    def __init__(self):
        self.logger = logging.getLogger("mapping")
        
    def log_mapping_change(
        self,
        event_type: str,
        mapping: Dict,
        user: str,
        context: Dict
    ):
        """
        Log mapping changes in structured JSON format.
        Includes:
        - Event type
        - Mapping details
        - User info
        - Context
        - Timestamp
        """
```

#### 3.2 Alerting System

Implement alerting for high-frequency mapping failures:

```python
class MappingAlertManager:
    def __init__(self):
        self.redis_client = Redis()
        self.alert_threshold = 10  # failures per minute
        self.alert_cooldown = 300  # seconds between alerts for same error type
        
    async def check_mapping_failures(self):
        """
        Monitor mapping failure rate and trigger alerts if threshold exceeded.
        Alerts include:
        - Failure rate
        - Affected columns
        - Time window
        - Suggested actions
        """
```

## Risk Matrix & Mitigation Plan

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Misclassified PII | Medium | High | Run shadow-mode audits before enforcing redaction |
| Mapping drift due to bad user feedback | High | Medium | Add user trust weighting (new users = lower impact) |
| Redis or Nova token outages | Medium | High | Add local token cache with TTL |
| Alert spam from early mapping errors | High | Medium | Throttle alerts, batch by error type |
| Performance degradation with large mapping sets | Medium | High | Implement pagination and lazy loading |
| Security vulnerabilities in admin override | Low | Critical | Regular security audits, least privilege access |

## Migration Plan for Existing Mappings

### Legacy Mapping Migration

1. **Migration Script Development**
   ```python
   class LegacyMappingMigrator:
       def __init__(self, legacy_source, mapping_store):
           self.legacy_source = legacy_source
           self.mapping_store = mapping_store
           
       def migrate_mappings(self):
           """Import legacy mappings into the new system"""
           legacy_mappings = self._load_legacy_mappings()
           
           for mapping in legacy_mappings:
               # Assign default confidence scores
               confidence = self._calculate_initial_confidence(mapping)
               
               # Tag as legacy
               mapping['metadata'] = {
                   'legacy_source': True,
                   'migration_date': datetime.now().isoformat(),
                   'original_source': self.legacy_source
               }
               
               # Store in new system
               self.mapping_store.save_mapping(
                   original=mapping['original'],
                   canonical=mapping['canonical'],
                   confidence=confidence,
                   metadata=mapping['metadata']
               )
   ```

2. **Migration Process**
   - Run in dry-run mode first to validate
   - Execute migration in batches
   - Verify migrated mappings
   - Keep legacy system running in parallel during transition
   - Switch over once validation is complete

3. **Rollback Plan**
   - Maintain backup of all legacy mappings
   - Create rollback script to revert if issues arise
   - Define clear criteria for rollback decision

## Governance for Mapping Confidence Weights

### Weight Configuration Management

1. **Configuration Storage**
   - Store weights in `config/mapping_weights.yaml`:
   ```yaml
   confidence_weights:
     levenshtein: 0.4
     jaccard: 0.3
     feedback: 0.3
   
   user_trust_weights:
     new_user: 0.5
     established_user: 0.8
     admin_user: 1.0
   ```

2. **Governance Process**
   - Weights can be adjusted by data science team
   - Changes require approval from technical lead
   - All changes must be documented with rationale
   - A/B testing required for significant changes

3. **Monitoring Impact**
   - Track mapping success rate before/after weight changes
   - Compare confidence score distribution
   - Monitor user feedback on mapping quality

## Feature Flag Strategy

### Implementation

1. **Feature Flag Configuration**
   ```python
   class FeatureFlags:
       def __init__(self):
           self.redis_client = Redis()
           
       def is_enabled(self, feature: str, user_id: str = None, org_id: str = None) -> bool:
           """
           Check if a feature is enabled for a user/organization
           """
           # Check global flag
           if not self._is_globally_enabled(feature):
               return False
               
           # Check organization-specific flag
           if org_id and not self._is_enabled_for_org(feature, org_id):
               return False
               
           # Check user-specific flag
           if user_id and not self._is_enabled_for_user(feature, user_id):
               return False
               
           return True
   ```

2. **Feature Flags to Implement**
   - `adaptive_schema_engine`: Enable/disable the confidence scoring system
   - `batch_mapping_api`: Enable/disable the batch mapping endpoint
   - `rollback_system`: Enable/disable the rollback functionality
   - `pii_audit`: Enable/disable PII auditing
   - `advanced_alerting`: Enable/disable the alerting system

3. **Rollout Strategy**
   - Enable for internal users first
   - Gradually enable for beta customers
   - Enable for all users once stable
   - Maintain ability to disable features if issues arise

## User-Facing Documentation

### 1. Admin UI Guide

#### Mapping Management
- How to view all current mappings
- How to search and filter mappings
- How to edit mappings manually
- How to approve/reject suggested mappings

#### Rollback Operations
- How to view mapping history
- How to roll back to a previous mapping
- How to view rollback logs
- How to handle rollback conflicts

#### Audit and Compliance
- How to view PII audit reports
- How to export mapping logs
- How to configure audit settings
- How to respond to compliance alerts

### 2. End-User Mapping Explainer UI

#### Mapping Visualization
- Show original column name
- Show mapped canonical name
- Display confidence score with visual indicator
- Show contributing factors to the confidence score

#### Explanation Components
- Similarity score (e.g., "87% similar to canonical name")
- Historical usage (e.g., "Used successfully 42 times")
- User feedback impact (e.g., "+2 positive confirmations")
- Context match (e.g., "Matches expected pattern for date fields")

#### User Actions
- How to confirm a mapping
- How to reject a mapping
- How to suggest an alternative
- How to request manual review

## Implementation Plan

### Phase 1: Core Engine (Week 1-2)
1. Implement confidence scoring algorithms
2. Set up batch mapping API endpoints
3. Create basic UI components
4. Implement rollback mechanism
5. Develop migration script for legacy mappings

### Phase 2: Security & Compliance (Week 3)
1. Implement rate limiting
2. Set up PII auditing
3. Add admin authentication
4. Create security documentation
5. Implement feature flags

### Phase 3: Observability (Week 4)
1. Implement structured logging
2. Set up alerting system
3. Create monitoring dashboards
4. Add performance metrics
5. Develop user-facing documentation

## Testing Strategy

### Unit Tests
- Test each confidence scoring algorithm
- Test rate limiting logic
- Test PII detection
- Test rollback functionality
- Test feature flag logic

### Integration Tests
- Test full mapping workflow
- Test batch operations
- Test alerting system
- Test admin overrides
- Test migration process

### Performance Tests
- Test rate limiting under load
- Test batch processing performance
- Test logging performance
- Test alert system responsiveness
- Test with large mapping sets

## Monitoring & Maintenance

### Key Metrics
1. Mapping success rate
2. Confidence score distribution
3. API response times
4. Alert frequency
5. PII detection rate
6. Feature flag usage statistics

### Maintenance Tasks
1. Regular confidence score recalibration
2. PII pattern updates
3. Rate limit adjustments
4. Log rotation and archival
5. Feature flag cleanup

## Dependencies

### Critical Dependencies
1. Nova credential storage for admin overrides
2. Redis for caching and rate limiting
3. Elasticsearch for log storage
4. Alert management system
5. Feature flag service

### Optional Dependencies
1. Grafana for dashboards
2. Prometheus for metrics
3. Sentry for error tracking

## Rollout Strategy

1. Deploy to staging environment
2. Run comprehensive test suite
3. Migrate legacy mappings
4. Enable feature flags for internal users
5. Gradual rollout to production
6. Monitor performance and errors
7. Gather user feedback
8. Iterate based on feedback

## Documentation Requirements

1. API documentation
2. UI user guide
3. Admin guide
4. Monitoring guide
5. Troubleshooting guide
6. End-user mapping explainer

## Success Criteria

1. 95% mapping success rate
2. < 1% false positive PII detection
3. < 100ms average API response time
4. < 1% alert false positive rate
5. 100% admin override success rate
6. 100% legacy mapping migration success

## Support & Escalation

### Level 1 Support
- Basic mapping issues
- UI problems
- Rate limit questions

### Level 2 Support
- API issues
- Performance problems
- Alert investigation

### Level 3 Support
- Admin override issues
- Security incidents
- System-wide problems

## Contact Information

- Technical Lead: [Name]
- Security Team: [Contact]
- Operations Team: [Contact]
- Documentation Team: [Contact]

## Next Steps

1. Review and approve implementation plan
2. Set up development environment
3. Begin Phase 1 implementation
4. Schedule regular progress reviews
5. Plan user acceptance testing 