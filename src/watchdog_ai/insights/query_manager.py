"""
Query management and trending insights tracking for Watchdog AI.
"""

import yaml
import os
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
from collections import Counter
import json

logger = logging.getLogger(__name__)

class QueryManager:
    """Manages example queries and tracks trending insights."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize QueryManager.
        
        Args:
            config_path: Optional path to example queries config
        """
        self.config_path = config_path or os.path.join(
            os.path.dirname(__file__), 
            "..", 
            "config",
            "example_queries.yaml"
        )
        self.example_queries = self._load_example_queries()
        self.query_history = []
        self.trending_cache = {}
        self.cache_timestamp = None
        self.CACHE_DURATION = timedelta(minutes=15)
        
    def _load_example_queries(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load example queries from config file."""
        try:
            if not os.path.exists(self.config_path):
                logger.warning(f"Example queries config not found at {self.config_path}")
                return {}
                
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
                
        except Exception as e:
            logger.error(f"Error loading example queries: {e}")
            return {}
            
    def get_example_queries(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get example queries, optionally filtered by category.
        
        Args:
            category: Optional category to filter by
            
        Returns:
            List of query dictionaries
        """
        try:
            if not category:
                # Return all queries flattened
                return [
                    query for category_queries in self.example_queries.values()
                    for query in category_queries
                ]
                
            return self.example_queries.get(category, [])
            
        except Exception as e:
            logger.error(f"Error getting example queries: {e}")
            return []
            
    def get_categories(self) -> List[str]:
        """Get list of available query categories."""
        return list(self.example_queries.keys())
        
    def track_query(self, query: str, metadata: Optional[Dict] = None) -> None:
        """
        Track a query execution for trending analysis.
        
        Args:
            query: The query string
            metadata: Optional metadata about the query execution
        """
        try:
            self.query_history.append({
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata or {}
            })
            
            # Clear trending cache so it will be recalculated
            self.trending_cache = {}
            self.cache_timestamp = None
            
        except Exception as e:
            logger.error(f"Error tracking query: {e}")
            
    def get_trending_insights(self, 
                            days: int = 7,
                            min_count: int = 2) -> List[Dict[str, Any]]:
        """
        Get trending insights based on query history.
        
        Args:
            days: Number of days to analyze
            min_count: Minimum query count to be considered trending
            
        Returns:
            List of trending queries with metadata
        """
        try:
            # Check cache
            if (self.cache_timestamp and 
                datetime.now() - self.cache_timestamp < self.CACHE_DURATION):
                return self.trending_cache
                
            # Calculate trending window
            cutoff = datetime.now() - timedelta(days=days)
            
            # Filter recent queries
            recent_queries = [
                item for item in self.query_history
                if datetime.fromisoformat(item["timestamp"]) > cutoff
            ]
            
            # Count query occurrences
            query_counts = Counter(item["query"] for item in recent_queries)
            
            # Get trending queries
            trending = [
                {
                    "query": query,
                    "count": count,
                    "last_used": max(
                        datetime.fromisoformat(item["timestamp"])
                        for item in recent_queries
                        if item["query"] == query
                    ).isoformat(),
                    "metadata": next(
                        (item["metadata"] for item in reversed(recent_queries)
                         if item["query"] == query),
                        {}
                    )
                }
                for query, count in query_counts.items()
                if count >= min_count
            ]
            
            # Sort by count and recency
            trending.sort(
                key=lambda x: (x["count"], x["last_used"]),
                reverse=True
            )
            
            # Update cache
            self.trending_cache = trending
            self.cache_timestamp = datetime.now()
            
            return trending
            
        except Exception as e:
            logger.error(f"Error getting trending insights: {e}")
            return []
            
    def save_query_history(self, filepath: str) -> bool:
        """
        Save query history to file.
        
        Args:
            filepath: Path to save history to
            
        Returns:
            bool: Success status
        """
        try:
            with open(filepath, 'w') as f:
                json.dump(self.query_history, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving query history: {e}")
            return False
            
    def load_query_history(self, filepath: str) -> bool:
        """
        Load query history from file.
        
        Args:
            filepath: Path to load history from
            
        Returns:
            bool: Success status
        """
        try:
            if not os.path.exists(filepath):
                return False
                
            with open(filepath, 'r') as f:
                self.query_history = json.load(f)
            return True
        except Exception as e:
            logger.error(f"Error loading query history: {e}")
            return False 