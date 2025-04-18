"""
Base Insight module for Watchdog AI.

Provides base classes and abstract interfaces for insight generation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import logging
import sentry_sdk
from abc import ABC, abstractmethod
import time

# Configure logger
logger = logging.getLogger(__name__)

class InsightBase(ABC):
    """
    Base class for all insight generators.
    
    This abstract class provides the foundation for creating insight
    generators with common functionality like data validation, error handling,
    instrumentation, and standardized return structures.
    """
    
    def __init__(self, insight_type: str):
        """
        Initialize the insight generator.
        
        Args:
            insight_type: String identifier for the insight type
        """
        self.insight_type = insight_type
    
    def generate(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Generate an insight from the provided DataFrame.
        
        Args:
            df: DataFrame to analyze
            **kwargs: Additional parameters for specific insight types
            
        Returns:
            Dictionary containing insight data
        """
        start_time = time.time()
        
        try:
            # Track insight generation in Sentry
            sentry_sdk.set_tag("insight_type", self.insight_type)
            sentry_sdk.set_tag("data_rows", len(df))
            sentry_sdk.set_tag("insight_version", self.get_version())
            
            # Input validation
            validation_result = self.validate_input(df, **kwargs)
            if "error" in validation_result:
                return self._create_error_response(validation_result["error"])
            
            # Preprocess the data
            preprocessed_df = self.preprocess_data(df, **kwargs)
            
            # Execute the insight calculation
            insight_result = self.compute_insight(preprocessed_df, **kwargs)
            
            # Post-process the result
            final_result = self.postprocess_result(insight_result, df, **kwargs)
            
            # Add metadata
            final_result["insight_type"] = self.insight_type
            final_result["generated_at"] = datetime.now().isoformat()
            final_result["execution_time_ms"] = int((time.time() - start_time) * 1000)
            
            # Log successful generation
            logger.info(f"Successfully generated {self.insight_type} insight in {final_result['execution_time_ms']}ms")
            sentry_sdk.capture_message(f"Insight: {self.insight_type} generated successfully", level="info")
            
            # Send performance metrics to Sentry
            self._record_performance_metrics(start_time, final_result)
            
            return final_result
            
        except Exception as e:
            execution_time_ms = int((time.time() - start_time) * 1000)
            error_msg = f"Error generating {self.insight_type} insight: {str(e)}"
            logger.error(error_msg)
            sentry_sdk.capture_exception(e)
            
            # Return error information
            return {
                "insight_type": self.insight_type,
                "generated_at": datetime.now().isoformat(),
                "error": str(e),
                "success": False,
                "execution_time_ms": execution_time_ms
            }
    
    def validate_input(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Validate the input data for this insight.
        
        Args:
            df: DataFrame to validate
            **kwargs: Additional parameters to validate
            
        Returns:
            Dict with error message if validation fails, empty dict if successful
        """
        if df is None or df.empty:
            return {"error": "No data available for analysis"}
        
        # Check for minimum row count
        if len(df) < self.get_minimum_rows():
            return {"error": f"Insufficient data for analysis. At least {self.get_minimum_rows()} rows required."}
        
        # Perform insight-specific validation
        return self._validate_insight_input(df, **kwargs)
    
    def _validate_insight_input(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Insight-specific input validation.
        
        Override this method in child classes to add specific validation.
        
        Args:
            df: DataFrame to validate
            **kwargs: Additional parameters to validate
            
        Returns:
            Dict with error message if validation fails, empty dict if successful
        """
        return {}
    
    def preprocess_data(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Preprocess the data for analysis.
        
        Args:
            df: DataFrame to preprocess
            **kwargs: Additional parameters for preprocessing
            
        Returns:
            Preprocessed DataFrame
        """
        # Return a copy of the DataFrame to avoid modifying the original
        return df.copy()
    
    @abstractmethod
    def compute_insight(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Compute the insight from the preprocessed data.
        
        This method must be implemented by all concrete insight classes.
        
        Args:
            df: Preprocessed DataFrame
            **kwargs: Additional parameters for insight computation
            
        Returns:
            Dictionary with insight data
        """
        pass
    
    def postprocess_result(self, insight_result: Dict[str, Any], original_df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Post-process the insight result.
        
        Args:
            insight_result: Result from compute_insight
            original_df: The original DataFrame
            **kwargs: Additional parameters for postprocessing
            
        Returns:
            Postprocessed insight data
        """
        # Default implementation just returns the insight result
        return insight_result
    
    def get_version(self) -> str:
        """
        Get the version of this insight implementation.
        
        Returns:
            Version string
        """
        return "1.0.0"
    
    def get_minimum_rows(self) -> int:
        """
        Get the minimum number of rows required for this insight.
        
        Returns:
            Minimum row count
        """
        return 5  # Default minimum
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """
        Create a standardized error response.
        
        Args:
            error_message: The error message
            
        Returns:
            Error response dictionary
        """
        return {
            "insight_type": self.insight_type,
            "generated_at": datetime.now().isoformat(),
            "error": error_message,
            "success": False
        }
    
    def _record_performance_metrics(self, start_time: float, result: Dict[str, Any]) -> None:
        """
        Record performance metrics in Sentry.
        
        Args:
            start_time: Start time of the insight generation
            result: The insight result
        """
        try:
            execution_time = time.time() - start_time
            sentry_sdk.set_tag("execution_time_ms", int(execution_time * 1000))
            
            # Add breadcrumb with performance data
            sentry_sdk.add_breadcrumb(
                category="performance",
                message=f"Generated {self.insight_type} insight",
                data={
                    "execution_time_ms": int(execution_time * 1000),
                    "data_rows": result.get("data_rows", 0),
                    "insight_type": self.insight_type
                },
                level="info"
            )
        except Exception as e:
            logger.warning(f"Error recording performance metrics: {str(e)}")

class ChartableInsight(InsightBase):
    """
    Base class for insights that include chart data.
    
    Extends InsightBase with methods for chart data creation and formatting.
    """
    
    def postprocess_result(self, insight_result: Dict[str, Any], original_df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Post-process the insight result adding chart data.
        
        Args:
            insight_result: Result from compute_insight
            original_df: The original DataFrame
            **kwargs: Additional parameters for postprocessing
            
        Returns:
            Postprocessed insight data with chart information
        """
        # Get base postprocessing
        result = super().postprocess_result(insight_result, original_df, **kwargs)
        
        # Add chart data if not already present
        if "chart_data" not in result:
            chart_data = self.create_chart_data(result, original_df, **kwargs)
            if chart_data is not None:
                result["chart_data"] = chart_data
                
                # Add chart encoding if not present
                if "chart_encoding" not in result:
                    chart_encoding = self.create_chart_encoding(result, chart_data, **kwargs)
                    if chart_encoding is not None:
                        result["chart_encoding"] = chart_encoding
        
        return result
    
    def create_chart_data(self, insight_result: Dict[str, Any], original_df: pd.DataFrame, **kwargs) -> Optional[pd.DataFrame]:
        """
        Create chart data for this insight.
        
        Args:
            insight_result: The insight result
            original_df: The original DataFrame
            **kwargs: Additional parameters for chart creation
            
        Returns:
            DataFrame for chart visualization or None if no chart is available
        """
        # Default implementation returns None - subclasses should override
        return None
    
    def create_chart_encoding(self, insight_result: Dict[str, Any], chart_data: pd.DataFrame, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Create chart encoding specification for the chart data.
        
        Args:
            insight_result: The insight result
            chart_data: The chart data DataFrame
            **kwargs: Additional parameters for encoding creation
            
        Returns:
            Chart encoding dictionary or None if default encoding should be used
        """
        # Default implementation returns None - subclasses should override
        return None

def find_column_by_pattern(df: pd.DataFrame, patterns: List[str], error_if_missing: bool = True) -> Optional[str]:
    """
    Find a column in the DataFrame matching one of the patterns.
    
    Args:
        df: DataFrame to search in
        patterns: List of patterns to match (case-insensitive)
        error_if_missing: Whether to log an error if no matching column is found
        
    Returns:
        Name of the first matching column or None if not found
    """
    matching_cols = [col for col in df.columns if any(pattern.lower() in col.lower() for pattern in patterns)]
    
    if matching_cols:
        return matching_cols[0]
    
    if error_if_missing:
        pattern_str = ", ".join(patterns)
        error_msg = f"No column matching patterns: {pattern_str} found in DataFrame"
        logger.error(error_msg)
    
    return None