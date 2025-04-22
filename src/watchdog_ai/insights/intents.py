"""
Intent detection and processing for insights.
"""

from typing import Dict, Any, Optional
import pandas as pd

class Intent:
    """Base class for all intents."""
    
    def matches(self, query: str) -> bool:
        """Check if this intent matches the given query."""
        raise NotImplementedError()
    
    def analyze(self, data: pd.DataFrame, query: str) -> Dict[str, Any]:
        """Analyze the data according to this intent."""
        raise NotImplementedError()

class TopMetricIntent(Intent):
    """Intent for finding the highest value of a metric."""
    
    def matches(self, query: str) -> bool:
        """Check if the query is asking for a highest/top value."""
        query = query.lower()
        return any(word in query for word in ["highest", "top", "best", "most"])
    
    def analyze(self, data: pd.DataFrame, query: str) -> Dict[str, Any]:
        """Find the highest value in the data."""
        # This is a simplified implementation
        if "Total_Gross" not in data.columns:
            return {
                "title": "Error",
                "summary": "Could not find Total_Gross column",
                "confidence": "low"
            }
        
        # Get the top value
        top_idx = data["Total_Gross"].idxmax()
        top_value = data.loc[top_idx]
        
        return {
            "title": "Top Rep by Gross",
            "summary": f"{top_value['SalesRepName']} had the highest gross at ${top_value['Total_Gross']:,.0f}",
            "recommendations": ["Analyze what made this performance successful"],
            "chart_data": data[["SalesRepName", "Total_Gross"]].to_dict(),
            "confidence": "high"
        }

class BottomMetricIntent(Intent):
    """Intent for finding the lowest value of a metric."""
    
    def matches(self, query: str) -> bool:
        """Check if the query is asking for a lowest/bottom value."""
        query = query.lower()
        return any(word in query for word in ["lowest", "bottom", "worst", "least"])
    
    def analyze(self, data: pd.DataFrame, query: str) -> Dict[str, Any]:
        """Find the lowest value in the data."""
        # This is a simplified implementation
        if "Total_Gross" not in data.columns:
            return {
                "title": "Error",
                "summary": "Could not find Total_Gross column",
                "confidence": "low"
            }
        
        # Get the bottom value
        bottom_idx = data["Total_Gross"].idxmin()
        bottom_value = data.loc[bottom_idx]
        
        return {
            "title": "Bottom Rep by Gross",
            "summary": f"{bottom_value['SalesRepName']} had the lowest gross at ${bottom_value['Total_Gross']:,.0f}",
            "recommendations": ["Investigate if additional support is needed"],
            "chart_data": data[["SalesRepName", "Total_Gross"]].to_dict(),
            "confidence": "high"
        }

class AverageMetricIntent(Intent):
    """Intent for finding the average value of a metric."""
    
    def matches(self, query: str) -> bool:
        """Check if the query is asking for an average value."""
        query = query.lower()
        return any(word in query for word in ["average", "mean", "typical"])
    
    def analyze(self, data: pd.DataFrame, query: str) -> Dict[str, Any]:
        """Find the average value in the data."""
        # This is a simplified implementation
        if "Total_Gross" not in data.columns:
            return {
                "title": "Error",
                "summary": "Could not find Total_Gross column",
                "confidence": "low"
            }
        
        # Calculate average
        avg_value = data["Total_Gross"].mean()
        
        return {
            "title": "Average Gross",
            "summary": f"The average gross is ${avg_value:,.0f}",
            "recommendations": ["Compare individual performance to this baseline"],
            "chart_data": data[["SalesRepName", "Total_Gross"]].to_dict(),
            "confidence": "high"
        }