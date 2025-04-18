"""
Redis-based caching for Watchdog AI.

Provides a cache interface for storing and retrieving DataFrames using Redis.
"""

import json
import hashlib
import logging
import pandas as pd
import redis
from typing import Optional, Dict, Any, Tuple
import os
import sentry_sdk
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Default configuration with fallbacks
DEFAULT_REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
DEFAULT_REDIS_PORT = int(os.environ.get('REDIS_PORT', 6379))
DEFAULT_REDIS_DB = int(os.environ.get('REDIS_DB', 0))
DEFAULT_CACHE_TTL = int(os.environ.get('CACHE_TTL_HOURS', 24)) * 3600  # Convert hours to seconds

class DataFrameCache:
    """
    Redis-based cache for storing and retrieving DataFrame objects.
    
    Serializes DataFrames to JSON for storage and deserialization on retrieval.
    Uses SHA-256 hash of file contents and rules version as cache keys.
    """
    
    def __init__(self, 
                 host: str = DEFAULT_REDIS_HOST, 
                 port: int = DEFAULT_REDIS_PORT, 
                 db: int = DEFAULT_REDIS_DB,
                 ttl: int = DEFAULT_CACHE_TTL):
        """
        Initialize the DataFrame cache with Redis connection.
        
        Args:
            host: Redis server hostname
            port: Redis server port
            db: Redis database number
            ttl: Cache TTL in seconds
        """
        self.host = host
        self.port = port
        self.db = db
        self.ttl = ttl
        self._client = None
        self._prefix = "watchdog:df:"
        self._stats = {"hits": 0, "misses": 0}
        
        # Initialize Redis client
        try:
            self._client = redis.Redis(host=host, port=port, db=db, socket_timeout=5)
            self._client.ping()  # Test connection
            logger.info(f"Redis cache initialized at {host}:{port}/{db}")
        except Exception as e:
            logger.warning(f"Redis connection failed: {str(e)}. Will proceed without caching.")
            self._client = None
            try:
                sentry_sdk.capture_exception(e)
            except Exception:
                pass  # Sentry not available
            
    @property
    def is_available(self) -> bool:
        """Check if Redis is available."""
        if not self._client:
            return False
        try:
            return self._client.ping()
        except:
            return False
        
    def create_key(self, file_content: bytes, rules_version: Any) -> str:
        """
        Create a cache key from file content and rules version.
        
        Args:
            file_content: Raw bytes of the file
            rules_version: The version of normalization rules
            
        Returns:
            Cache key string
        """
        # Use first 10MB max for large files to avoid excessive hashing
        content_hash = hashlib.sha256(file_content[:10_000_000]).hexdigest()
        # Convert rules_version to string reliably
        rules_str = str(rules_version) if rules_version is not None else "default"
        # Create a composite key with prefix
        return f"{self._prefix}{content_hash}:{rules_str}"
        
    def get(self, key: str) -> Optional[pd.DataFrame]:
        """
        Retrieve DataFrame from cache.
        
        Args:
            key: Cache key
            
        Returns:
            DataFrame if cache hit, None on miss/error
        """
        if not self.is_available:
            return None
            
        try:
            # Set a tag for monitoring
            sentry_sdk.set_tag("cache_operation", "get")
            
            # Try to get from cache
            serialized = self._client.get(key)
            if not serialized:
                # Tag as cache miss
                sentry_sdk.set_tag("cache_result", "miss")
                self._stats["misses"] += 1
                return None
                
            # Deserialize the cached data
            cached_data = json.loads(serialized)
            
            # Create DataFrame from the serialized data
            df = pd.DataFrame.from_dict(cached_data["data"])
            
            # Restore proper types if available
            if "dtypes" in cached_data:
                for col, dtype in cached_data["dtypes"].items():
                    if col in df.columns:
                        try:
                            if dtype.startswith("datetime"):
                                df[col] = pd.to_datetime(df[col])
                            else:
                                df[col] = df[col].astype(dtype)
                        except Exception as e:
                            logger.warning(f"Failed to convert column {col} to {dtype}: {str(e)}")
            
            # Tag as cache hit
            sentry_sdk.set_tag("cache_result", "hit")
            self._stats["hits"] += 1
            
            # Track hit in metric
            try:
                sentry_sdk.capture_message("Cache hit", level="info")
            except Exception:
                pass  # Sentry not configured
            
            logger.info(f"Cache hit for key {key[:20]}... (size: {len(df)} rows)")
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving from cache: {str(e)}")
            sentry_sdk.capture_exception(e)
            return None
            
    def set(self, key: str, df: pd.DataFrame) -> bool:
        """
        Store DataFrame in cache.
        
        Args:
            key: Cache key
            df: DataFrame to cache
            
        Returns:
            True on success, False on failure
        """
        if not self.is_available or df is None:
            return False
            
        try:
            # Set a tag for monitoring
            sentry_sdk.set_tag("cache_operation", "set")
            
            # Prepare data for serialization
            # Extract dtype information for proper restoration
            dtypes = {}
            for col, dtype in df.dtypes.items():
                dtypes[col] = str(dtype)
                
            serialized_data = {
                "data": df.to_dict(orient="list"),
                "dtypes": dtypes,
                "cached_at": datetime.now().isoformat(),
                "row_count": len(df),
                "column_count": len(df.columns)
            }
            
            # Serialize and store
            serialized = json.dumps(serialized_data)
            self._client.setex(key, self.ttl, serialized)
            
            # Track cache write in metric
            try:
                sentry_sdk.capture_message("Cache write", level="info")
            except Exception:
                pass  # Sentry not configured
            
            logger.info(f"Cached DataFrame with key {key[:20]}... ({len(df)} rows, {len(df.columns)} columns)")
            return True
            
        except Exception as e:
            logger.error(f"Error writing to cache: {str(e)}")
            sentry_sdk.capture_exception(e)
            return False
            
    def invalidate(self, key: str) -> bool:
        """
        Invalidate a specific cache entry.
        
        Args:
            key: Cache key to invalidate
            
        Returns:
            True if invalidated, False otherwise
        """
        if not self.is_available:
            return False
            
        try:
            return bool(self._client.delete(key))
        except Exception as e:
            logger.error(f"Error invalidating cache: {str(e)}")
            return False
            
    def invalidate_all(self) -> int:
        """
        Invalidate all cache entries with this cache's prefix.
        
        Returns:
            Number of invalidated keys
        """
        if not self.is_available:
            return 0
            
        try:
            # Find all keys with our prefix
            pattern = f"{self._prefix}*"
            all_keys = self._client.keys(pattern)
            
            if not all_keys:
                return 0
                
            # Delete all matching keys
            return self._client.delete(*all_keys)
            
        except Exception as e:
            logger.error(f"Error invalidating all cache entries: {str(e)}")
            return 0
            
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = self._stats.copy()
        stats["available"] = self.is_available
        
        if self.is_available:
            # Add info about number of keys
            try:
                stats["cached_items"] = len(self._client.keys(f"{self._prefix}*"))
                info = self._client.info()
                stats["redis_used_memory"] = info.get("used_memory_human", "N/A")
                stats["redis_clients"] = info.get("connected_clients", "N/A")
            except:
                stats["cached_items"] = "N/A"
                
        return stats

# Create a global cache instance
cache = DataFrameCache()

def get_cache() -> DataFrameCache:
    """
    Get the global cache instance.
    
    Returns:
        The global DataFrameCache instance
    """
    return cache