"""
Intent management system for Watchdog AI.
Provides centralized intent registration and matching.
"""

from typing import List, Dict, Any, Optional
import logging
import pandas as pd
from datetime import datetime
from .models import InsightResult
from .intents import Intent, TopMetricIntent, BottomMetricIntent, CountMetricIntent, HighestCountIntent

logger = logging.getLogger(__name__)

class IntentManager:
    """Manages registration and matching of intents."""
    
    def __init__(self):
        """Initialize with core intents."""
        self.intents: List[Intent] = []
        self._register_core_intents()
    
    def _register_core_intents(self):
        """Register the core set of intents."""
        self.intents = [
            TopMetricIntent(),
            BottomMetricIntent(),
            CountMetricIntent(),
            HighestCountIntent()
        ]
        logger.info(f"Registered {len(self.intents)} core intents")
    
    def find_matching_intent(self, prompt: str) -> Optional[Intent]:
        """
        Find the first intent that matches the given prompt.
        
        Args:
            prompt: The user's prompt to match
            
        Returns:
            The matching intent or None if no match found
        """
        logger.debug(f"Finding intent match for prompt: {prompt[:50]}...")
        
        for intent in self.intents:
            try:
                if intent.matches(prompt):
                    logger.info(f"Matched intent: {type(intent).__name__}")
                    return intent
            except Exception as e:
                logger.error(f"Error matching intent {type(intent).__name__}: {e}")
                continue
        
        logger.info("No matching intent found")
        return None
    
    def generate_insight(self, prompt: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate an insight using the matching intent.
        
        Args:
            prompt: The user's prompt
            df: The DataFrame to analyze
            
        Returns:
            The insight result
        """
        logger.info(f"Generating insight for prompt: {prompt[:50]}...")
        
        # Find matching intent
        intent = self.find_matching_intent(prompt)
        if not intent:
            logger.info("No matching intent, returning fallback")
            return self._generate_fallback_response(prompt, df)
        
        try:
            # Generate insight using the matched intent
            result = intent.analyze(df, prompt)
            
            # Convert InsightResult to dict format expected by UI
            response = {
                "summary": result.summary,
                "value_insights": [result.title] + result.recommendations,
                "actionable_flags": result.recommendations,
                "confidence": result.confidence,
                "chart_data": result.chart_data.to_dict('records') if result.chart_data is not None else None,
                "chart_encoding": result.chart_encoding,
                "supporting_data": result.supporting_data.to_dict('records') if result.supporting_data is not None else None,
                "timestamp": datetime.now().isoformat(),
                "is_error": result.error is not None,
                "error": result.error,
                "title": result.title,
                "is_direct_calculation": True
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating insight: {e}")
            return {
                "summary": "Failed to generate insight",
                "error": str(e),
                "error_type": "analysis_error",
                "timestamp": datetime.now().isoformat(),
                "is_error": True,
                "is_direct_calculation": True
            }
    
    def _generate_fallback_response(self, prompt: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate a helpful fallback response when no intent matches."""
        # Get available columns for examples
        columns = df.columns.tolist()
        
        # Generate example queries based on available columns
        example_queries = []
        
        # Look for metric columns
        metric_cols = [col for col in columns if any(term in col.lower() for term in ['gross', 'profit', 'revenue', 'price', 'cost'])]
        if metric_cols:
            example_queries.append(f"What was the highest {metric_cols[0]}?")
            example_queries.append(f"Show me the lowest {metric_cols[0]}")
        
        # Look for category columns
        category_cols = [col for col in columns if any(term in col.lower() for term in ['rep', 'source', 'make', 'model'])]
        if category_cols:
            example_queries.append(f"Which {category_cols[0]} had the most deals?")
            if metric_cols:
                example_queries.append(f"What {category_cols[0]} had the highest {metric_cols[0]}?")
        
        return {
            "title": "I'm Not Sure About That",
            "summary": (
                "I'm not sure how to analyze that specific question. "
                "Here are some examples of questions I can help with:"
            ),
            "value_insights": example_queries,
            "actionable_flags": [],
            "confidence": "low",
            "timestamp": datetime.now().isoformat(),
            "is_error": False,
            "is_direct_calculation": True
        }

# Create a singleton instance
intent_manager = IntentManager()