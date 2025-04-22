"""Base test utilities and helper functions."""

import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

class TestBase:
    """Base class for test cases with common utilities."""
    
    @staticmethod
    def create_sample_df(rows: int = 100) -> pd.DataFrame:
        """Create a sample DataFrame for testing."""
        return pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=rows),
            'value': np.random.randn(rows)
        })
    
    @staticmethod
    def compare_dataframes(df1: pd.DataFrame, df2: pd.DataFrame) -> bool:
        """Compare two DataFrames for equality."""
        if df1.shape != df2.shape:
            return False
        return df1.equals(df2)
    
    @staticmethod
    def get_test_file_path(filename: str) -> str:
        """Get full path to a test file."""
        return os.path.join(os.path.dirname(__file__), 'data', filename)
    
    @staticmethod
    def create_test_insight_data() -> Dict[str, Any]:
        """Create test insight data."""
        return {
            'summary': 'Test insight summary',
            'metrics': {'value': 100},
            'chart_data': pd.DataFrame({'x': range(5), 'y': range(5)}),
            'recommendations': ['Test recommendation']
        }

