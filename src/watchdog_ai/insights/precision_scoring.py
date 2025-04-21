"""
Precision scoring engine for query validation.
"""

import logging
from typing import Dict, Any, List
import re

logger = logging.getLogger(__name__)

class PrecisionScoringEngine:
    """Evaluates query precision and provides confidence scoring."""
    
    def __init__(self):
        """Initialize the scoring engine."""
        self.metric_patterns = [
            r'profit', r'revenue', r'sales', r'gross', r'performance',
            r'amount', r'total', r'average', r'count'
        ]
        
        self.dimension_patterns = [
            r'sales\s*rep', r'representative', r'agent', r'employee',
            r'lead\s*source', r'source', r'make', r'model', r'year'
        ]
        
        self.time_patterns = [
            r'today', r'yesterday', r'last\s*week', r'last\s*month',
            r'this\s*month', r'year', r'quarter'
        ]
        
        # Special case patterns for common business questions
        self.business_patterns = [
            (r'top\s*performing\s*(sales\s*rep|representative|agent)', 0.8),
            (r'best\s*(sales\s*rep|representative|agent)', 0.8),
            (r'highest\s*(profit|revenue|sales)', 0.8),
            (r'lowest\s*(profit|revenue|sales)', 0.8),
            (r'negative\s*profit', 0.9),
            (r'performance\s*(by|of)\s*(sales\s*rep|representative|agent)', 0.8)
        ]

    def predict_precision(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict the precision score for a query.
        
        Args:
            query: The user's query
            context: Query context including columns and data quality metrics
            
        Returns:
            Dict containing precision score and reasoning
        """
        query = query.lower()
        score = 0.0
        reasons = []
        
        # Check for special business question patterns first
        for pattern, base_score in self.business_patterns:
            if re.search(pattern, query):
                score = max(score, base_score)
                reasons.append(f"Matches business pattern: {pattern}")
        
        # If no business pattern matched, do detailed analysis
        if score == 0.0:
            # Check for metric mentions
            metric_matches = [p for p in self.metric_patterns if re.search(p, query)]
            if metric_matches:
                score += 0.3
                reasons.append(f"Found metrics: {', '.join(metric_matches)}")
            
            # Check for dimension mentions
            dimension_matches = [p for p in self.dimension_patterns if re.search(p, query)]
            if dimension_matches:
                score += 0.3
                reasons.append(f"Found dimensions: {', '.join(dimension_matches)}")
            
            # Check for time references
            time_matches = [p for p in self.time_patterns if re.search(p, query)]
            if time_matches:
                score += 0.2
                reasons.append(f"Found time references: {', '.join(time_matches)}")
        
        # Check column matches
        columns = context.get('columns', [])
        column_matches = []
        for col in columns:
            col_pattern = re.escape(col.lower().replace('_', ' '))
            if re.search(col_pattern, query):
                column_matches.append(col)
        
        if column_matches:
            score += 0.2
            reasons.append(f"Matches columns: {', '.join(column_matches)}")
        
        # Adjust for data quality
        nan_percentage = context.get('nan_percentage', 0)
        if nan_percentage > 20:
            score *= 0.8
            reasons.append(f"Data quality concern: {nan_percentage:.1f}% missing values")
        
        # Determine confidence level
        if score >= 0.7:
            confidence_level = "high"
        elif score >= 0.4:
            confidence_level = "medium"
        else:
            confidence_level = "low"
        
        return {
            "score": score,
            "confidence_level": confidence_level,
            "reasons": reasons
        }