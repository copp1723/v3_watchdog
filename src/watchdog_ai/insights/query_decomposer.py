"""
Query decomposition system for Watchdog AI insights.
Breaks down complex queries into atomic sub-queries.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime
import uuid
import pandas as pd
from pydantic import BaseModel, Field

from .query_cache import QueryCache

logger = logging.getLogger(__name__)

@dataclass
class SubQuery:
    """Represents a decomposed sub-query."""
    query: str
    intent: str
    dependencies: List[str]
    metrics: List[str]
    filters: Dict[str, Any]
    context: Dict[str, Any]

class QueryDecomposer:
    """Decomposes complex queries into atomic sub-queries."""
    
    def __init__(self, cache: QueryCache):
        """
        Initialize the query decomposer.
        
        Args:
            cache: QueryCache instance for result caching
        """
        self.cache = cache
    
    def split_query(self, query: str, df: pd.DataFrame) -> List[SubQuery]:
        """
        Split a complex query into atomic sub-queries.
        
        Args:
            query: Complex query to decompose
            df: Input DataFrame
            
        Returns:
            List of SubQuery objects
        """
        # Detect comparison keywords
        comparisons = re.findall(r'(compare|vs\.|versus|difference between|higher than|lower than)', query.lower())
        
        # Detect temporal keywords
        temporal = re.findall(r'(trend|over time|monthly|yearly|weekly|change in|growth)', query.lower())
        
        # Detect aggregation keywords
        aggregations = re.findall(r'(average|total|sum of|mean|max|min|highest|lowest)', query.lower())
        
        # Initialize sub-queries list
        sub_queries = []
        
        # Handle comparison queries
        if comparisons:
            # Split on comparison keywords
            parts = re.split('|'.join(comparisons), query, flags=re.IGNORECASE)
            
            # Create sub-queries for each part
            for i, part in enumerate(parts):
                if part.strip():
                    sub_queries.append(
                        SubQuery(
                            query=part.strip(),
                            intent="comparison" if i > 0 else "base",
                            dependencies=[f"part_{j}" for j in range(i)],
                            metrics=self._extract_metrics(part, df.columns),
                            filters=self._extract_filters(part),
                            context={"part_index": i}
                        )
                    )
        
        # Handle temporal queries
        elif temporal:
            # Create time series sub-query
            sub_queries.append(
                SubQuery(
                    query=query,
                    intent="trend",
                    dependencies=[],
                    metrics=self._extract_metrics(query, df.columns),
                    filters=self._extract_filters(query),
                    context={"temporal": True}
                )
            )
        
        # Handle aggregation queries
        elif aggregations:
            # Create aggregation sub-query
            sub_queries.append(
                SubQuery(
                    query=query,
                    intent="aggregation",
                    dependencies=[],
                    metrics=self._extract_metrics(query, df.columns),
                    filters=self._extract_filters(query),
                    context={"aggregation": aggregations[0]}
                )
            )
        
        # Default to single query
        else:
            sub_queries.append(
                SubQuery(
                    query=query,
                    intent="direct",
                    dependencies=[],
                    metrics=self._extract_metrics(query, df.columns),
                    filters=self._extract_filters(query),
                    context={}
                )
            )
        
        return sub_queries
    
    def _extract_metrics(self, query: str, columns: List[str]) -> List[str]:
        """Extract metric column names from query."""
        metrics = []
        
        # Common metric keywords
        metric_keywords = [
            'sales', 'revenue', 'profit', 'gross', 'margin',
            'volume', 'count', 'amount', 'price'
        ]
        
        # Look for column names in query
        for col in columns:
            if col.lower() in query.lower():
                metrics.append(col)
        
        # Look for metric keywords
        for keyword in metric_keywords:
            if keyword in query.lower():
                # Find closest matching column
                matches = [col for col in columns if keyword in col.lower()]
                if matches:
                    metrics.extend(matches)
        
        return list(set(metrics))
    
    def _extract_filters(self, query: str) -> Dict[str, Any]:
        """Extract filter conditions from query."""
        filters = {}
        
        # Extract date ranges
        date_ranges = re.findall(r'(from|between|in|during)\s+([^\s]+)\s+(to|and|through)\s+([^\s]+)', query, re.IGNORECASE)
        if date_ranges:
            filters['date_range'] = {
                'start': date_ranges[0][1],
                'end': date_ranges[0][3]
            }
        
        # Extract numeric comparisons
        numeric = re.findall(r'(greater than|less than|above|below|over|under)\s+(\d+(?:\.\d+)?)', query, re.IGNORECASE)
        if numeric:
            filters['numeric'] = {
                'operator': numeric[0][0],
                'value': float(numeric[0][1])
            }
        
        # Extract categories
        categories = re.findall(r'(for|in|by)\s+([^\s]+)', query, re.IGNORECASE)
        if categories:
            filters['category'] = categories[0][1]
        
        return filters
    
    def reconstruct_answer(self, sub_queries: List[SubQuery], 
                          results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reconstruct final answer from sub-query results.
        
        Args:
            sub_queries: List of executed sub-queries
            results: Dictionary of sub-query results
            
        Returns:
            Combined final result
        """
        if not sub_queries or not results:
            return {
                "error": "No results to reconstruct",
                "sub_queries": len(sub_queries),
                "results": len(results)
            }
        
        # Handle single direct query
        if len(sub_queries) == 1 and sub_queries[0].intent == "direct":
            return results[sub_queries[0].query]
        
        # Initialize combined result
        combined = {
            "summary": "",
            "metrics": {},
            "breakdown": [],
            "recommendations": [],
            "confidence": "high"
        }
        
        # Handle comparison queries
        if any(q.intent == "comparison" for q in sub_queries):
            base_query = next(q for q in sub_queries if q.intent == "base")
            comp_queries = [q for q in sub_queries if q.intent == "comparison"]
            
            base_result = results[base_query.query]
            comp_results = [results[q.query] for q in comp_queries]
            
            # Combine metrics
            combined["metrics"] = {
                "base": base_result["metrics"],
                "comparison": [r["metrics"] for r in comp_results]
            }
            
            # Build comparison summary
            combined["summary"] = self._build_comparison_summary(
                base_result, comp_results
            )
            
            # Combine breakdowns
            combined["breakdown"] = (
                base_result["breakdown"] +
                [item for r in comp_results for item in r["breakdown"]]
            )
            
            # Combine recommendations
            combined["recommendations"] = (
                base_result["recommendations"] +
                [rec for r in comp_results for rec in r["recommendations"]]
            )
        
        # Handle trend queries
        elif any(q.intent == "trend" for q in sub_queries):
            trend_query = next(q for q in sub_queries if q.intent == "trend")
            trend_result = results[trend_query.query]
            
            combined.update(trend_result)
            
            # Add trend-specific metrics
            if "trend_data" in trend_result:
                combined["metrics"]["trend"] = trend_result["trend_data"]
        
        # Handle aggregation queries
        elif any(q.intent == "aggregation" for q in sub_queries):
            agg_query = next(q for q in sub_queries if q.intent == "aggregation")
            agg_result = results[agg_query.query]
            
            combined.update(agg_result)
            
            # Add aggregation context
            combined["metrics"]["aggregation_type"] = agg_query.context["aggregation"]
        
        return combined
    
    def _build_comparison_summary(self, base_result: Dict[str, Any],
                                comp_results: List[Dict[str, Any]]) -> str:
        """Build a natural language summary of the comparison."""
        summary_parts = [base_result["summary"]]
        
        for comp in comp_results:
            summary_parts.append(f"In comparison, {comp['summary'].lower()}")
        
        return " ".join(summary_parts)