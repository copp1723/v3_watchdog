# Redis Caching Implementation for Watchdog AI

## Overview

This document describes the implementation of Redis-based caching for the Watchdog AI ingestion pipeline. The caching system uses Redis to store parsed and normalized DataFrames, significantly speeding up repeated data uploads and analysis operations.

## Implementation Details

### Core Components

1. **Redis Cache Client (`src/utils/cache.py`)**
   - Implements `DataFrameCache` class for managing DataFrame serialization/deserialization
   - Handles Redis connection and error recovery
   - Creates cache keys based on file content hashes and normalization rules version
   - Provides consistent get/set/invalidate interfaces

2. **Enhanced Data I/O (`src/utils/data_io.py`)**
   - Modified `load_data()` to check cache before parsing files
   - Stores parsed and normalized DataFrames in cache after processing
   - Added Redis caching to `compute_lead_gross()` and `validate_data()`
   - Maintains Streamlit's caching for fallback and compatibility

3. **Test Suite (`tests/test_redis_cache.py`)**
   - Unit tests for cache functionality
   - Integration tests with mocked dependencies
   - Tests for both cache hits and misses

### Redis Configuration

The Redis cache client uses the following configuration sources (in order of precedence):

1. Environment variables:
   - `REDIS_HOST` (default: 'localhost')
   - `REDIS_PORT` (default: 6379)
   - `REDIS_DB` (default: 0)
   - `CACHE_TTL_HOURS` (default: 24)

2. Docker Compose:
   - Redis service runs on standard port 6379
   - Web service depends on Redis availability

### Caching Flow

1. **File Upload:**
   - When a file is uploaded, a cache key is generated from the file content hash and normalization rules version
   - The system checks if this key exists in Redis
   - On cache hit: Return the cached DataFrame directly, avoiding parse/normalize steps
   - On cache miss: Process the file normally and cache the result

2. **Metrics Computation:**
   - When computing metrics (e.g., lead gross), cache based on DataFrame content
   - Store computation results to avoid recalculation

3. **Data Validation:**
   - Cache validation results for DataFrames
   - Use special serialization format for validation summaries

### Cache Key Strategy

Cache keys are designed to be unique based on:
- SHA-256 hash of file content (first 10MB used for very large files)
- Normalization rules version to ensure cache invalidation when rules change
- Operation type for computation caches (e.g., "compute_lead_gross", "validate_data")

### Monitoring & Instrumentation

The implementation includes comprehensive instrumentation:

1. **Sentry Tags:**
   - `cache_check`: "enabled"/"disabled"
   - `cache_result`: "hit"/"miss"
   - `metrics_cache_result`: "hit"/"miss"
   - `validation_cache_result`: "hit"/"miss"
   - `normalization_rules_version`: rule version used

2. **Sentry Metrics:**
   - `cache.hits` counter with tags for cache type and operation
   - `cache.writes` counter for cache writes
   - Exception tracking for cache errors

3. **Logging:**
   - Cache initialization status
   - Cache hit/miss events
   - Cache error conditions

## Benefits

- **Performance Improvement:** Significantly faster data processing for repeated uploads of the same file
- **Resource Efficiency:** Reduces CPU and memory usage for file parsing and DataFrame operations
- **Persistence:** Cache survives application restarts (unlike Streamlit's in-memory cache)
- **Scalability:** Redis can be scaled independently of the application
- **Monitoring:** Comprehensive instrumentation for tracking cache efficiency

## Limitations & Future Improvements

- **Space Efficiency:** Could implement compression for larger DataFrames
- **Cache Eviction:** Currently relies on TTL; could implement LRU or other eviction strategies
- **Advanced Caching:** Could add more sophisticated caching for other application operations
- **Circuit Breaker:** Could add more robust handling for Redis connection failures
- **Cache Management:** Could add admin endpoints to view/manage cache contents

## Testing and Validation

The Redis caching system has been tested with:
- Unit tests for all core functionality
- Integration tests for cache hits and misses
- Empty DataFrame edge cases
- Redis connection failure scenarios

## Usage Notes

The caching system is designed to be transparent to users. It enhances performance without requiring any changes to application usage. The implementation maintains backward compatibility with Streamlit's native caching.