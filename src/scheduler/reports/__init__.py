"""
Report Generators package for V3 Watchdog AI.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd

class ReportGenerator(ABC):
    """Abstract base class for all report generators."""
    
    @abstractmethod
    def generate(self, data: pd.DataFrame, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a report from data.
        
        Args:
            data: DataFrame containing the data for the report
            parameters: Additional parameters for report generation
            
        Returns:
            Dictionary with report content
        """
        pass