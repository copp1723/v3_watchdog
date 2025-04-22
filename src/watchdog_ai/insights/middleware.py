"""
Middleware for insight processing.
Provides hooks for pre-processing and post-processing of data.
"""

import pandas as pd
from typing import Dict, Any, Optional

class InsightMiddleware:
    """
    Middleware for insight processing.
    Provides hooks for pre-processing and post-processing of data.
    """
    
    def __init__(self):
        """Initialize the middleware."""
        pass
    
    def pre_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pre-process the dataframe before analysis.
        
        Args:
            df: The pandas DataFrame to process
            
        Returns:
            The processed DataFrame
        """
        # Check if DataFrame is None
        if df is None:
            return None
            
        # Make a copy to avoid modifying the original
        processed_df = df.copy()
        
        # Add any pre-processing logic here
        
        return processed_df
    
    def post_process(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Post-process the analysis result.
        
        Args:
            result: The analysis result to process
            
        Returns:
            The processed result
        """
        # Check if result is None
        if result is None:
            return None
            
        # Make a copy to avoid modifying the original
        processed_result = result.copy()
        
        # Add any post-processing logic here
        
        return processed_result 