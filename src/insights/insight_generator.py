"""
Insight Generator module for direct data analysis.
"""

import pandas as pd
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
import redis
from .intent_manager import intent_manager
from .models import InsightResult, FeedbackEntry

logger = logging.getLogger(__name__)

# Initialize Redis client
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

class InsightGenerator:
    """Generates insights using intent-based analysis."""
    
    def __init__(self):
        """Initialize the insight generator."""
        self.intent_manager = intent_manager
    
    def generate_insight(self, prompt: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate an insight based on the prompt and data.
        
        Args:
            prompt: The user's prompt
            df: DataFrame to analyze
            
        Returns:
            Dictionary containing the insight result
        """
        try:
            logger.info(f"Generating insight for prompt: {prompt[:50]}...")
            
            # Input validation
            if not prompt or not isinstance(prompt, str):
                error_msg = "Invalid prompt: Prompt must be a non-empty string"
                logger.error(error_msg)
                return {
                    "summary": "Failed to generate insight",
                    "error": error_msg,
                    "error_type": "input_validation",
                    "timestamp": datetime.now().isoformat(),
                    "is_error": True,
                    "is_direct_calculation": True
                }
            
            if df is None or df.empty:
                error_msg = "No data available for analysis"
                logger.error(error_msg)
                return {
                    "summary": error_msg,
                    "error": error_msg,
                    "error_type": "missing_data",
                    "timestamp": datetime.now().isoformat(),
                    "is_error": True,
                    "is_direct_calculation": True
                }
            
            # Generate insight using intent manager
            response = self.intent_manager.generate_insight(prompt, df)
            
            # Add timestamp if not present
            if "timestamp" not in response:
                response["timestamp"] = datetime.now().isoformat()
            
            return response
            
        except Exception as e:
            error_msg = f"Error generating insight: {str(e)}"
            logger.error(error_msg)
            return {
                "summary": "Failed to generate insight",
                "error": str(e),
                "error_type": "unknown",
                "timestamp": datetime.now().isoformat(),
                "is_error": True,
                "is_direct_calculation": True
            }
    
    def _generate_example_queries(self, df: pd.DataFrame) -> List[str]:
        """
        Generate example queries based on available columns.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            List of example queries
        """
        examples = []
        columns = df.columns.tolist()
        
        # Look for metric columns
        metric_cols = [col for col in columns if any(term in col.lower() for term in ['gross', 'profit', 'revenue', 'price', 'cost'])]
        if metric_cols:
            examples.append(f"What was the highest {metric_cols[0]}?")
            examples.append(f"Show me the lowest {metric_cols[0]}")
            examples.append(f"How many deals had negative {metric_cols[0]}?")
        
        # Look for category columns
        category_cols = [col for col in columns if any(term in col.lower() for term in ['rep', 'source', 'make', 'model'])]
        if category_cols:
            examples.append(f"Which {category_cols[0]} had the most deals?")
            if metric_cols:
                examples.append(f"What {category_cols[0]} had the highest {metric_cols[0]}?")
        
        return examples

# Create a singleton instance
insight_generator = InsightGenerator()