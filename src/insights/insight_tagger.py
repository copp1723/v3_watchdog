"""
Insight Tagging and Classification Module for V3 Watchdog AI.

Provides functionality for classifying and prioritizing insights based on business value and urgency.
"""

import re
import time
import json
import logging
import math
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple, Set
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Priority levels for insights
class InsightPriority:
    """Priority levels for insights."""
    CRITICAL = "critical"
    RECOMMENDED = "recommended"
    OPTIONAL = "optional"
    INFORMATIONAL = "informational"  # Lowest priority, just FYI


class InsightType:
    """Types of insights."""
    ANOMALY = "anomaly"  # Something unusual or unexpected
    TREND = "trend"      # Pattern over time
    CORRELATION = "correlation"  # Relationship between variables
    FORECAST = "forecast"  # Prediction of future values
    SUMMARY = "summary"  # Simple summary of data
    COMPARISON = "comparison"  # Comparison between groups/periods
    BREAKDOWN = "breakdown"  # Detailed breakdowns of metrics
    ALERT = "alert"  # Critical alert


class InsightTagger:
    """Classifies insights by priority, type, and tags."""
    
    def __init__(self, thresholds: Dict[str, Any] = None, model_name: str = None):
        """
        Initialize the insight tagger with customizable thresholds.
        
        Args:
            thresholds: Optional custom thresholds for classification
            model_name: Optional name of the sentence transformer model
        """
        # Default thresholds for priority classification
        self.thresholds = {
            # Profit thresholds - critical if change > 10%
            "profit_delta_pct": 10.0,
            # Sales thresholds - critical if change > 20%
            "sales_delta_pct": 20.0,
            # Lead volume thresholds - critical if change > 25%
            "lead_delta_pct": 25.0,
            # Inventory thresholds - critical if > 60 days
            "aged_inventory_days": 60,
            # Conversion rate thresholds - critical if change > 5 percentage points
            "conversion_delta_pts": 5.0,
            # Service thresholds
            "service_delta_pct": 15.0,
            # General threshold for numeric outliers (z-score)
            "outlier_z_score": 2.5,
            # Similarity threshold for tag suggestion and grouping
            "similarity_threshold": 0.75,
        }
        
        # Update with custom thresholds if provided
        if thresholds:
            self.thresholds.update(thresholds)
        
        # Regular expressions for keyword matching
        self.keyword_patterns = {
            InsightPriority.CRITICAL: [
                r'significant\s+decrease', r'sharp\s+decline', r'rapidly\s+decreasing',
                r'urgent', r'critical', r'immediate\s+attention', r'severely',
                r'dramatic\s+drop', r'major\s+issue', r'serious\s+problem',
                r'substantial\s+loss', r'risk', r'danger', r'harm', r'emergency'
            ],
            InsightPriority.RECOMMENDED: [
                r'recommend', r'opportunity', r'improve', r'increase', r'enhance',
                r'optimize', r'boost', r'grow', r'expand', r'strengthen',
                r'benefit', r'advantage', r'gain', r'profit', r'promising'
            ],
            InsightPriority.OPTIONAL: [
                r'consider', r'might', r'could', r'option', r'alternative',
                r'possibly', r'potential', r'experiment', r'test', r'try',
                r'explore', r'investigate', r'review', r'examine'
            ],
            InsightPriority.INFORMATIONAL: [
                r'note', r'observe', r'notice', r'see', r'find', r'discover',
                r'learn', r'know', r'understand', r'recognize', r'aware',
                r'fyi', r'information', r'data', r'statistic'
            ]
        }
        
        # Compile all regex patterns for efficiency
        for priority, patterns in self.keyword_patterns.items():
            self.keyword_patterns[priority] = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
        
        # Common tags for different business areas
        self.tag_categories = {
            "sales": [
                "sales", "revenue", "profit", "margin", "deal", "conversion", 
                "closing", "pipeline", "forecast", "opportunity"
            ],
            "inventory": [
                "inventory", "stock", "supply", "demand", "age", "days", 
                "turnover", "allocation", "ordering", "availability"
            ],
            "lead": [
                "lead", "prospect", "inquiry", "source", "campaign", "traffic", 
                "marketing", "referral", "website", "digital"
            ],
            "service": [
                "service", "repair", "maintenance", "parts", "warranty", "recall", 
                "satisfaction", "retention", "scheduling", "appointment"
            ],
            "finance": [
                "finance", "loan", "credit", "funding", "cash", "expense", 
                "budget", "accounting", "cost", "investment"
            ],
            "customer": [
                "customer", "satisfaction", "experience", "feedback", "survey", 
                "review", "loyalty", "retention", "demographic", "segment"
            ],
            "performance": [
                "performance", "benchmark", "target", "goal", "kpi", "metric", 
                "measurement", "trend", "growth", "decline"
            ],
            "alert": [
                "alert", "warning", "issue", "problem", "error", "concern", 
                "attention", "critical", "urgent", "important"
            ]
        }
        
        # Initialize sentence transformer model for similarity calculations if available
        self.model = None
        self.model_name = model_name or "all-MiniLM-L6-v2"
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Initialized sentence transformer model: {self.model_name}")
        except Exception as e:
            logger.warning(f"Failed to initialize sentence transformer model: {e}")
            logger.warning("Tag similarity features will be limited")
    
    def tag_insight(self, insight: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tag an insight with priority, type, and suggested tags.
        
        Args:
            insight: The insight to tag
            
        Returns:
            The tagged insight with priority, type, and tags added
        """
        if not insight:
            return insight
        
        # Make a copy to avoid modifying the original
        tagged_insight = insight.copy()
        
        # Extract key fields for analysis
        summary = insight.get('summary', '')
        metrics = insight.get('metrics', {})
        recommendations = insight.get('recommendations', [])
        
        # Determine priority and type
        priority = self._determine_priority(summary, metrics, recommendations)
        insight_type = self._determine_type(summary, metrics, recommendations)
        
        # Generate tags
        tags = self._generate_tags(summary, metrics, recommendations, insight_type, priority)
        
        # Add tags to the insight
        tagged_insight['priority'] = priority
        tagged_insight['type'] = insight_type
        tagged_insight['tags'] = tags
        
        # Add audit metadata
        tagged_insight['audit'] = self._create_audit_metadata(insight)
        
        # Add timestamp for when it was tagged
        tagged_insight['tagged_at'] = datetime.now().isoformat()
        
        return tagged_insight
    
    def _determine_priority(self, summary: str, metrics: Dict[str, Any], recommendations: List[str]) -> str:
        """
        Determine the priority of an insight.
        
        Args:
            summary: Summary text
            metrics: Insight metrics
            recommendations: Recommendations list
            
        Returns:
            Priority level
        """
        # Start with lowest priority, escalate as needed
        priority = InsightPriority.INFORMATIONAL
        
        # Check metrics for critical changes
        if metrics:
            # Check for profit-related metrics
            profit_metrics = self._find_metrics_by_pattern(metrics, ['profit', 'margin', 'revenue', 'sales'])
            if profit_metrics and self._has_significant_change(profit_metrics, self.thresholds["profit_delta_pct"]):
                return InsightPriority.CRITICAL
            
            # Check for lead volume metrics
            lead_metrics = self._find_metrics_by_pattern(metrics, ['lead', 'prospect', 'inquiry'])
            if lead_metrics and self._has_significant_change(lead_metrics, self.thresholds["lead_delta_pct"]):
                priority = max(priority, InsightPriority.RECOMMENDED)
            
            # Check for inventory metrics
            inventory_metrics = self._find_metrics_by_pattern(metrics, ['inventory', 'stock', 'age', 'days'])
            if inventory_metrics and self._exceeds_threshold(inventory_metrics, self.thresholds["aged_inventory_days"]):
                priority = max(priority, InsightPriority.RECOMMENDED)
            
            # Check for extreme outliers in any metric
            if self._has_outlier(metrics, self.thresholds["outlier_z_score"]):
                priority = max(priority, InsightPriority.RECOMMENDED)
        
        # Check summary for keywords
        for candidate_priority, patterns in self.keyword_patterns.items():
            for pattern in patterns:
                if pattern.search(summary):
                    # For critical, require multiple conditions or strong evidence
                    if candidate_priority == InsightPriority.CRITICAL:
                        # Text evidence alone isn't enough for critical, need metrics support
                        if metrics and len(recommendations) > 0:
                            priority = max(priority, candidate_priority, key=self._priority_order)
                    else:
                        priority = max(priority, candidate_priority, key=self._priority_order)
        
        # Check recommendations - their presence escalates priority
        if recommendations:
            if len(recommendations) >= 3:
                priority = max(priority, InsightPriority.RECOMMENDED, key=self._priority_order)
            elif len(recommendations) > 0:
                priority = max(priority, InsightPriority.OPTIONAL, key=self._priority_order)
        
        return priority
    
    def _determine_type(self, summary: str, metrics: Dict[str, Any], recommendations: List[str]) -> str:
        """
        Determine the type of an insight.
        
        Args:
            summary: Summary text
            metrics: Insight metrics
            recommendations: Recommendations list
            
        Returns:
            Insight type
        """
        # Default type
        insight_type = InsightType.SUMMARY
        
        # Type patterns
        type_patterns = {
            InsightType.ANOMALY: [r'unusual', r'unexpected', r'anomaly', r'irregular', r'outlier', r'abnormal', r'strange'],
            InsightType.TREND: [r'trend', r'increasing', r'decreasing', r'growing', r'declining', r'over time', r'pattern'],
            InsightType.CORRELATION: [r'correlation', r'relationship', r'connected', r'associated', r'linked', r'together'],
            InsightType.FORECAST: [r'forecast', r'predict', r'projection', r'future', r'expect', r'anticipate'],
            InsightType.COMPARISON: [r'compared', r'comparison', r'versus', r'against', r'higher than', r'lower than', r'relative'],
            InsightType.BREAKDOWN: [r'breakdown', r'segment', r'category', r'division', r'group', r'type', r'detailed'],
            InsightType.ALERT: [r'alert', r'warning', r'attention', r'urgent', r'critical', r'important', r'immediately']
        }
        
        # Compile patterns
        compiled_patterns = {t: [re.compile(p, re.IGNORECASE) for p in patterns] 
                            for t, patterns in type_patterns.items()}
        
        # Check summary for type keywords
        for candidate_type, patterns in compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(summary):
                    insight_type = candidate_type
                    
                    # If alert is detected with critical signals, return immediately
                    if candidate_type == InsightType.ALERT and self._has_critical_signals(summary, metrics):
                        return InsightType.ALERT
        
        return insight_type
    
    def _generate_tags(self, summary: str, metrics: Dict[str, Any], 
                      recommendations: List[str], insight_type: str, priority: str) -> List[str]:
        """
        Generate tags for the insight based on content.
        
        Args:
            summary: Summary text
            metrics: Insight metrics
            recommendations: Recommendations list
            insight_type: Type of insight
            priority: Priority level
            
        Returns:
            List of tags
        """
        tags = set()
        
        # Add type and priority as tags
        tags.add(insight_type)
        tags.add(priority)
        
        # Check all tag categories for matches in summary
        for category, keywords in self.tag_categories.items():
            for keyword in keywords:
                pattern = r'\b' + re.escape(keyword) + r'\b'
                if re.search(pattern, summary, re.IGNORECASE):
                    tags.add(category)
                    break
        
        # Check metric names for additional tags
        if metrics:
            for metric_name in metrics.keys():
                for category, keywords in self.tag_categories.items():
                    for keyword in keywords:
                        if keyword.lower() in metric_name.lower():
                            tags.add(category)
                            break
        
        # Use semantic analysis for additional tag suggestions if model is available
        if self.model and summary:
            additional_tags = self._suggest_tags_by_similarity(summary)
            tags.update(additional_tags)
        
        return sorted(list(tags))
    
    def _suggest_tags_by_similarity(self, text: str) -> Set[str]:
        """
        Suggest tags based on semantic similarity.
        
        Args:
            text: Text to analyze
            
        Returns:
            Set of suggested tags
        """
        suggested_tags = set()
        
        if not self.model:
            return suggested_tags
        
        try:
            # Encode the input text
            text_embedding = self.model.encode(text)
            
            # Create category descriptions for comparison
            category_descriptions = {
                "sales": "Sales, revenue, profit, deals, and sales pipeline information",
                "inventory": "Inventory levels, stock, supply, and vehicle availability",
                "lead": "Lead generation, prospects, marketing campaigns, and lead sources",
                "service": "Service department, repairs, maintenance, and warranty work",
                "finance": "Financial information, loans, credit, and accounting",
                "customer": "Customer experience, satisfaction, and demographics",
                "performance": "Performance metrics, KPIs, goals, and benchmarks",
                "alert": "Important alerts, warnings, and issues requiring attention"
            }
            
            # Encode all category descriptions
            categories = list(category_descriptions.keys())
            descriptions = [category_descriptions[cat] for cat in categories]
            category_embeddings = self.model.encode(descriptions)
            
            # Calculate similarities
            similarities = np.dot(text_embedding, category_embeddings.T)
            
            # Add categories that exceed the threshold
            threshold = self.thresholds.get("similarity_threshold", 0.75)
            for i, similarity in enumerate(similarities):
                if similarity >= threshold:
                    suggested_tags.add(categories[i])
        
        except Exception as e:
            logger.error(f"Error in semantic tag suggestion: {e}")
        
        return suggested_tags
    
    def _create_audit_metadata(self, insight: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create audit metadata for an insight.
        
        Args:
            insight: The insight to create metadata for
            
        Returns:
            Dictionary with audit metadata
        """
        # Generate standard audit fields
        audit = {
            "created_at": datetime.now().isoformat(),
            "created_by": insight.get("created_by", "system"),
            "origin_dataset": insight.get("dataset", "unknown"),
            "tag_history": [],
            "version": 1
        }
        
        # Add existing tags to history if present
        if "tags" in insight:
            audit["tag_history"].append({
                "timestamp": datetime.now().isoformat(),
                "tags": insight["tags"],
                "action": "initial"
            })
        
        return audit
    
    def find_similar_insights(self, 
                             insight: Dict[str, Any], 
                             other_insights: List[Dict[str, Any]],
                             min_similarity: float = None) -> List[Dict[str, Any]]:
        """
        Find similar insights based on embedding similarity.
        
        Args:
            insight: Target insight to compare against
            other_insights: List of insights to check for similarity
            min_similarity: Minimum similarity threshold (0-1)
            
        Returns:
            List of similar insights with similarity scores
        """
        if not self.model or not insight or not other_insights:
            return []
        
        # Use default threshold if not specified
        if min_similarity is None:
            min_similarity = self.thresholds.get("similarity_threshold", 0.75)
        
        similar_insights = []
        
        try:
            # Get text from the target insight
            target_text = insight.get("summary", "")
            if not target_text:
                return []
            
            # Encode the target insight
            target_embedding = self.model.encode(target_text)
            
            # Compare with each other insight
            for other in other_insights:
                # Skip comparing with self
                if other.get("id") == insight.get("id"):
                    continue
                
                # Get text from the other insight
                other_text = other.get("summary", "")
                if not other_text:
                    continue
                
                # Encode and calculate similarity
                other_embedding = self.model.encode(other_text)
                similarity = np.dot(target_embedding, other_embedding) / (
                    np.linalg.norm(target_embedding) * np.linalg.norm(other_embedding)
                )
                
                # Add to results if above threshold
                if similarity >= min_similarity:
                    similar_insights.append({
                        "insight": other,
                        "similarity": similarity
                    })
            
            # Sort by similarity (highest first)
            similar_insights.sort(key=lambda x: x["similarity"], reverse=True)
        
        except Exception as e:
            logger.error(f"Error finding similar insights: {e}")
        
        return similar_insights
    
    def group_insights_by_similarity(self, 
                                   insights: List[Dict[str, Any]],
                                   min_similarity: float = None) -> List[Dict[str, Any]]:
        """
        Group insights by semantic similarity.
        
        Args:
            insights: List of insights to group
            min_similarity: Minimum similarity threshold (0-1)
            
        Returns:
            List of insight groups with similarity information
        """
        if not self.model or not insights:
            return []
        
        # Use default threshold if not specified
        if min_similarity is None:
            min_similarity = self.thresholds.get("similarity_threshold", 0.75)
        
        groups = []
        remaining = insights.copy()
        
        try:
            while remaining:
                # Take the first insight as a seed for a new group
                seed = remaining.pop(0)
                
                # Find similar insights
                similar = self.find_similar_insights(seed, remaining, min_similarity)
                
                # Create a new group
                group = {
                    "seed": seed,
                    "members": [
                        {"insight": similar_item["insight"], "similarity": similar_item["similarity"]}
                        for similar_item in similar
                    ],
                    "size": 1 + len(similar)
                }
                
                # Remove the grouped insights from remaining
                remaining = [
                    insight for insight in remaining
                    if insight.get("id") not in [m["insight"].get("id") for m in group["members"]]
                ]
                
                # Add the group to the result
                groups.append(group)
        
        except Exception as e:
            logger.error(f"Error grouping insights: {e}")
        
        # Sort groups by size (largest first)
        groups.sort(key=lambda x: x["size"], reverse=True)
        
        return groups
    
    def _find_metrics_by_pattern(self, metrics: Dict[str, Any], patterns: List[str]) -> Dict[str, Any]:
        """
        Find metrics that match any of the given patterns.
        
        Args:
            metrics: Metrics dictionary
            patterns: List of patterns to match
            
        Returns:
            Dictionary of matching metrics
        """
        matching_metrics = {}
        
        for key, value in metrics.items():
            if any(pattern in key.lower() for pattern in patterns):
                matching_metrics[key] = value
        
        return matching_metrics
    
    def _has_significant_change(self, metrics: Dict[str, Any], threshold_pct: float) -> bool:
        """
        Check if any metric shows a significant percentage change.
        
        Args:
            metrics: Metrics dictionary
            threshold_pct: Percentage threshold for significant change
            
        Returns:
            True if any metric exceeds the threshold, False otherwise
        """
        for key, value in metrics.items():
            # Check if there's a change or delta field
            if '_change' in key or '_delta' in key or '_pct' in key:
                try:
                    # Remove % sign and convert to float if necessary
                    str_value = str(value).replace('%', '')
                    numeric_value = float(str_value)
                    
                    # Check absolute value against threshold
                    if abs(numeric_value) >= threshold_pct:
                        return True
                except (ValueError, TypeError):
                    # Not a numeric value or couldn't convert
                    continue
        
        return False
    
    def _exceeds_threshold(self, metrics: Dict[str, Any], threshold: float) -> bool:
        """
        Check if any metric exceeds a given threshold.
        
        Args:
            metrics: Metrics dictionary
            threshold: Threshold value to check against
            
        Returns:
            True if any metric exceeds the threshold, False otherwise
        """
        for key, value in metrics.items():
            try:
                numeric_value = float(value)
                if numeric_value >= threshold:
                    return True
            except (ValueError, TypeError):
                # Not a numeric value or couldn't convert
                continue
        
        return False
    
    def _has_outlier(self, metrics: Dict[str, Any], z_score_threshold: float) -> bool:
        """
        Check if any metric is an outlier based on z-score.
        
        Args:
            metrics: Metrics dictionary
            z_score_threshold: Z-score threshold for outlier detection
            
        Returns:
            True if any metric is an outlier, False otherwise
        """
        # Extract numeric values
        numeric_values = []
        for value in metrics.values():
            try:
                # Handle percentage values
                str_value = str(value).replace('%', '')
                numeric_values.append(float(str_value))
            except (ValueError, TypeError):
                # Not a numeric value or couldn't convert
                continue
        
        if len(numeric_values) < 2:
            # Not enough data for z-score calculation
            return False
        
        # Calculate mean and standard deviation
        mean = np.mean(numeric_values)
        std = np.std(numeric_values)
        
        if std == 0:
            # All values are the same, no outliers
            return False
        
        # Calculate z-scores and check for outliers
        z_scores = [(x - mean) / std for x in numeric_values]
        return any(abs(z) >= z_score_threshold for z in z_scores)
    
    def _has_critical_signals(self, summary: str, metrics: Dict[str, Any]) -> bool:
        """
        Check if the insight contains critical signals.
        
        Args:
            summary: Summary text
            metrics: Metrics dictionary
            
        Returns:
            True if critical signals are present, False otherwise
        """
        # Check for critical keywords
        critical_keywords = ['urgent', 'critical', 'immediate', 'severe', 'danger', 'risk', 'emergency']
        if any(re.search(r'\b' + keyword + r'\b', summary, re.IGNORECASE) for keyword in critical_keywords):
            return True
        
        # Check for large negative changes in metrics
        for key, value in metrics.items():
            if '_change' in key or '_delta' in key or '_pct' in key:
                try:
                    # Remove % sign and convert to float if necessary
                    str_value = str(value).replace('%', '')
                    numeric_value = float(str_value)
                    
                    # Very large negative change
                    if numeric_value <= -25:
                        return True
                except (ValueError, TypeError):
                    # Not a numeric value or couldn't convert
                    continue
        
        return False
    
    def _priority_order(self, p: str) -> int:
        """
        Helper function to determine the ordering of priorities.
        
        Args:
            p: Priority string
            
        Returns:
            Integer representing priority order (higher = more important)
        """
        order = {
            InsightPriority.CRITICAL: 4,
            InsightPriority.RECOMMENDED: 3,
            InsightPriority.OPTIONAL: 2,
            InsightPriority.INFORMATIONAL: 1
        }
        return order.get(p, 0)


class InsightStore:
    """Store for tagged insights with persistence and retrieval functionality."""
    
    def __init__(self, storage_path: str = None):
        """
        Initialize the insight store.
        
        Args:
            storage_path: Optional path to store insights
        """
        self.storage_path = storage_path or 'insights.json'
        self.audit_log_path = os.path.join(os.path.dirname(self.storage_path), 'audit_log.json')
        self.insights = []
        self._load_insights()
    
    def _load_insights(self) -> None:
        """Load insights from storage."""
        try:
            with open(self.storage_path, 'r') as f:
                self.insights = json.load(f)
                logger.info(f"Loaded {len(self.insights)} insights from {self.storage_path}")
        except (FileNotFoundError, json.JSONDecodeError):
            self.insights = []
            logger.info(f"No existing insights found at {self.storage_path}, initialized empty store")
    
    def _save_insights(self) -> None:
        """Save insights to storage."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            
            with open(self.storage_path, 'w') as f:
                json.dump(self.insights, f, indent=2)
                logger.info(f"Saved {len(self.insights)} insights to {self.storage_path}")
        except Exception as e:
            logger.error(f"Failed to save insights: {e}")
    
    def add_insight(self, insight: Dict[str, Any]) -> str:
        """
        Add a tagged insight to the store.
        
        Args:
            insight: Tagged insight to add
            
        Returns:
            ID of the added insight
        """
        # Ensure the insight has an ID
        if 'id' not in insight:
            insight['id'] = f"insight_{int(time.time())}_{len(self.insights)}"
        
        # Add timestamp if not present
        if 'timestamp' not in insight:
            insight['timestamp'] = datetime.now().isoformat()
        
        # Add to insights list
        self.insights.append(insight)
        
        # Save to storage
        self._save_insights()
        
        # Add to audit log
        self._record_audit("add_insight", insight)
        
        return insight['id']
    
    def get_insight(self, insight_id: str) -> Optional[Dict[str, Any]]:
        """
        Get an insight by ID.
        
        Args:
            insight_id: ID of the insight to retrieve
            
        Returns:
            The insight if found, None otherwise
        """
        for insight in self.insights:
            if insight.get('id') == insight_id:
                return insight
        return None
    
    def get_insights_by_priority(self, priority: str) -> List[Dict[str, Any]]:
        """
        Get all insights with a specific priority.
        
        Args:
            priority: Priority level to filter by
            
        Returns:
            List of matching insights
        """
        return [insight for insight in self.insights if insight.get('priority') == priority]
    
    def get_insights_by_type(self, insight_type: str) -> List[Dict[str, Any]]:
        """
        Get all insights of a specific type.
        
        Args:
            insight_type: Type to filter by
            
        Returns:
            List of matching insights
        """
        return [insight for insight in self.insights if insight.get('type') == insight_type]
    
    def get_insights_by_tag(self, tag: str) -> List[Dict[str, Any]]:
        """
        Get all insights with a specific tag.
        
        Args:
            tag: Tag to filter by
            
        Returns:
            List of matching insights
        """
        return [insight for insight in self.insights 
                if 'tags' in insight and tag in insight.get('tags', [])]
    
    def get_insights_by_tags(self, tags: List[str], match_all: bool = False) -> List[Dict[str, Any]]:
        """
        Get insights matching multiple tags.
        
        Args:
            tags: List of tags to filter by
            match_all: Whether all tags must match (AND) or any tag (OR)
            
        Returns:
            List of matching insights
        """
        if not tags:
            return []
        
        if match_all:
            # All tags must match (AND)
            return [insight for insight in self.insights 
                    if 'tags' in insight and all(tag in insight.get('tags', []) for tag in tags)]
        else:
            # Any tag must match (OR)
            return [insight for insight in self.insights 
                    if 'tags' in insight and any(tag in insight.get('tags', []) for tag in tags)]
    
    def get_recent_insights(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        Get insights from the last N days.
        
        Args:
            days: Number of days to look back
            
        Returns:
            List of recent insights
        """
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        return [insight for insight in self.insights 
                if insight.get('timestamp', '0') >= cutoff_date]
    
    def update_tags(self, insight_id: str, tags: List[str]) -> bool:
        """
        Update the tags for an insight.
        
        Args:
            insight_id: ID of the insight to update
            tags: New tags list
            
        Returns:
            True if updated, False if not found
        """
        for i, insight in enumerate(self.insights):
            if insight.get('id') == insight_id:
                # Get the old tags
                old_tags = insight.get('tags', [])
                
                # Update the tags
                self.insights[i]['tags'] = tags
                
                # Update tag history in audit metadata
                if 'audit' not in self.insights[i]:
                    self.insights[i]['audit'] = {
                        "created_at": datetime.now().isoformat(),
                        "tag_history": []
                    }
                
                # Add to tag history
                self.insights[i]['audit'].setdefault('tag_history', []).append({
                    "timestamp": datetime.now().isoformat(),
                    "previous": old_tags,
                    "current": tags,
                    "action": "update"
                })
                
                # Increment version
                if 'version' in self.insights[i]['audit']:
                    self.insights[i]['audit']['version'] += 1
                else:
                    self.insights[i]['audit']['version'] = 1
                
                # Save to storage
                self._save_insights()
                
                # Add to audit log
                self._record_audit("update_tags", insight, {"previous_tags": old_tags, "new_tags": tags})
                
                return True
        
        return False
    
    def delete_insight(self, insight_id: str) -> bool:
        """
        Delete an insight by ID.
        
        Args:
            insight_id: ID of the insight to delete
            
        Returns:
            True if deleted, False if not found
        """
        deleted_insight = None
        initial_count = len(self.insights)
        
        # Find the insight to be deleted (for audit log)
        for insight in self.insights:
            if insight.get('id') == insight_id:
                deleted_insight = insight
                break
        
        # Remove the insight
        self.insights = [insight for insight in self.insights 
                        if insight.get('id') != insight_id]
        
        if len(self.insights) < initial_count:
            # Save to storage
            self._save_insights()
            
            # Add to audit log
            if deleted_insight:
                self._record_audit("delete_insight", deleted_insight)
            
            return True
        
        return False
    
    def _record_audit(self, action: str, insight: Dict[str, Any], details: Dict[str, Any] = None) -> None:
        """
        Record an audit log entry.
        
        Args:
            action: The action performed
            insight: The insight affected
            details: Additional details about the action
        """
        try:
            # Create audit log entry
            audit_entry = {
                "timestamp": datetime.now().isoformat(),
                "action": action,
                "insight_id": insight.get("id", "unknown"),
                "insight_type": insight.get("type", "unknown"),
                "user": "system",  # This could be updated with actual user info
                "details": details or {}
            }
            
            # Load existing audit log
            existing_log = []
            try:
                if os.path.exists(self.audit_log_path):
                    with open(self.audit_log_path, 'r') as f:
                        existing_log = json.load(f)
            except Exception:
                existing_log = []
            
            # Add new entry
            existing_log.append(audit_entry)
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.audit_log_path), exist_ok=True)
            
            # Save updated log
            with open(self.audit_log_path, 'w') as f:
                json.dump(existing_log, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to record audit log: {e}")


def tag_insight(insight: Dict[str, Any], thresholds: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Tag a single insight (convenience function).
    
    Args:
        insight: Insight to tag
        thresholds: Optional custom thresholds
        
    Returns:
        Tagged insight with priority, type, and tags added
    """
    tagger = InsightTagger(thresholds)
    return tagger.tag_insight(insight)


def tag_insights(insights: List[Dict[str, Any]], 
                thresholds: Dict[str, Any] = None,
                save: bool = False,
                storage_path: str = None) -> List[Dict[str, Any]]:
    """
    Tag multiple insights and optionally save them.
    
    Args:
        insights: List of insights to tag
        thresholds: Optional custom thresholds
        save: Whether to save insights to storage
        storage_path: Path to storage file if saving
        
    Returns:
        List of tagged insights
    """
    tagger = InsightTagger(thresholds)
    tagged_insights = [tagger.tag_insight(insight) for insight in insights]
    
    if save:
        store = InsightStore(storage_path)
        for insight in tagged_insights:
            store.add_insight(insight)
    
    return tagged_insights


def get_critical_insights(storage_path: str = None, days: int = 7) -> List[Dict[str, Any]]:
    """
    Get recent critical insights.
    
    Args:
        storage_path: Path to storage file
        days: Number of days to look back
        
    Returns:
        List of critical insights from the recent period
    """
    store = InsightStore(storage_path)
    recent = store.get_recent_insights(days)
    return [insight for insight in recent if insight.get('priority') == InsightPriority.CRITICAL]


def get_insights_by_tags(tags: List[str], match_all: bool = False, storage_path: str = None) -> List[Dict[str, Any]]:
    """
    Get insights with specified tags.
    
    Args:
        tags: List of tags to filter by
        match_all: Whether all tags must match (AND) or any tag (OR)
        storage_path: Path to storage file
        
    Returns:
        List of insights matching the tag criteria
    """
    store = InsightStore(storage_path)
    return store.get_insights_by_tags(tags, match_all)


def group_similar_insights(storage_path: str = None, min_similarity: float = 0.75) -> List[Dict[str, Any]]:
    """
    Group similar insights from storage.
    
    Args:
        storage_path: Path to storage file
        min_similarity: Minimum similarity threshold (0-1)
        
    Returns:
        List of insight groups with similarity information
    """
    store = InsightStore(storage_path)
    tagger = InsightTagger()
    return tagger.group_insights_by_similarity(store.insights, min_similarity)