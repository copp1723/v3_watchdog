"""
Query caching system for Watchdog AI insights.
Provides caching and retrieval of query results with versioning.
"""

import redis
import json
import hashlib
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class CacheStats:
    """Statistics about cache usage."""
    hits: int = 0
    misses: int = 0
    total_queries: int = 0
    hit_rate: float = 0.0

@dataclass
class QueryResult:
    """Represents a cached query result."""
    query_hash: str
    result: Dict[str, Any]
    sub_queries: List[str]
    timestamp: str
    version: str
    trace_id: str
    metrics: Dict[str, Any]

class QueryCache:
    """Caches and retrieves query results."""
    
    def __init__(self, redis_url: Optional[str] = None, ttl: int = 60 * 60 * 24):
        """
        Initialize the query cache.
        
        Args:
            redis_url: Optional Redis URL for persistence
            ttl: Cache TTL in seconds (default 24 hours)
        """
        self.ttl = ttl
        self.stats = CacheStats()
        
        try:
            self.redis = redis.from_url(redis_url) if redis_url else redis.Redis()
            self.redis.ping()
            logger.info("Redis cache initialized successfully")
        except Exception as e:
            logger.warning(f"Redis connection failed, using in-memory cache: {e}")
            self.redis = None
            self._memory_cache = {}
    
    def _compute_hash(self, query: str, context: Dict[str, Any]) -> str:
        """Compute a stable hash for the query and context."""
        # Sort context keys for stable hashing
        sorted_context = {k: context[k] for k in sorted(context.keys())}
        
        # Create hash input
        hash_input = f"{query}:{json.dumps(sorted_context)}"
        
        # Generate hash
        return hashlib.sha256(hash_input.encode()).hexdigest()
    
    def get_query_result(self, query: str, context: Dict[str, Any]) -> Optional[QueryResult]:
        """
        Get cached result for a query.
        
        Args:
            query: Query string
            context: Query context
            
        Returns:
            Cached QueryResult or None if not found
        """
        self.stats.total_queries += 1
        query_hash = self._compute_hash(query, context)
        
        try:
            if self.redis:
                data = self.redis.get(f"query:{query_hash}")
                if data:
                    self.stats.hits += 1
                    return QueryResult(**json.loads(data))
            else:
                if query_hash in self._memory_cache:
                    self.stats.hits += 1
                    return QueryResult(**self._memory_cache[query_hash])
            
            self.stats.misses += 1
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving from cache: {e}")
            self.stats.misses += 1
            return None
        finally:
            # Update hit rate
            self.stats.hit_rate = self.stats.hits / self.stats.total_queries if self.stats.total_queries > 0 else 0
    
    def save_query_result(self, query: str, context: Dict[str, Any],
                         result: Dict[str, Any], sub_queries: List[str],
                         trace_id: str, metrics: Dict[str, Any]) -> None:
        """
        Save a query result to cache.
        
        Args:
            query: Query string
            context: Query context
            result: Query result
            sub_queries: List of sub-query hashes
            trace_id: Trace ID for the query
            metrics: Execution metrics
        """
        query_hash = self._compute_hash(query, context)
        
        cache_entry = QueryResult(
            query_hash=query_hash,
            result=result,
            sub_queries=sub_queries,
            timestamp=datetime.now().isoformat(),
            version="1.0.0",
            trace_id=trace_id,
            metrics=metrics
        )
        
        try:
            if self.redis:
                self.redis.setex(
                    f"query:{query_hash}",
                    self.ttl,
                    json.dumps(cache_entry.__dict__)
                )
            else:
                self._memory_cache[query_hash] = cache_entry.__dict__
                
            logger.info(f"Cached query result with hash {query_hash}")
            
        except Exception as e:
            logger.error(f"Error saving to cache: {e}")
    
    def get_cache_stats(self) -> CacheStats:
        """Get current cache statistics."""
        return self.stats
    
    def clear_cache(self) -> None:
        """Clear all cached results."""
        try:
            if self.redis:
                keys = self.redis.keys("query:*")
                if keys:
                    self.redis.delete(*keys)
            else:
                self._memory_cache.clear()
            
            # Reset stats
            self.stats = CacheStats()
            
            logger.info("Cache cleared successfully")
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")