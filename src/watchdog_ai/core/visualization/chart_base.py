
"""
Base chart interfaces and types for Watchdog AI.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from enum import Enum
import pandas as pd

class ChartType(str, Enum):
    """Chart type enumeration."""
    BAR = 'bar'
    LINE = 'line'
    PIE = 'pie'
    SCATTER = 'scatter'
    AREA = 'area'
    HEATMAP = 'heatmap'
    TABLE = 'table'
    UNKNOWN = 'unknown'
    
    @classmethod
    def from_string(cls, chart_type: str) -> 'ChartType':
        """Convert string to ChartType enum."""
        try:
            return cls(chart_type.lower())
        except ValueError:
            return cls.UNKNOWN

class ChartConfig:
    """Configuration for chart generation."""
    
    def __init__(
        self,
        chart_type: Union[ChartType, str] = ChartType.BAR,
        title: Optional[str] = None,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        x_column: Optional[str] = None,
        y_column: Optional[str] = None,
        color_column: Optional[str] = None,
        size_column: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ):
        """
        Initialize chart configuration.
        
        Args:
            chart_type: Type of chart to generate
            title: Chart title
            x_label: Label for x-axis
            y_label: Label for y-axis
            x_column: Column name for x-axis data
            y_column: Column name for y-axis data
            color_column: Column name for color encoding
            size_column: Column name for size encoding
            width: Chart width in pixels
            height: Chart height in pixels
        """
        self.chart_type = chart_type if isinstance(chart_type, ChartType) else ChartType.from_string(chart_type)
        self.title = title
        self.x_label = x_label
        self.y_label = y_label
        self.x_column = x_column
        self.y_column = y_column
        self.color_column = color_column
        self.size_column = size_column
        self.width = width
        self.height = height
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'chart_type': self.chart_type.value,
            'title': self.title,
            'x_label': self.x_label,
            'y_label': self.y_label,
            'x_column': self.x_column,
            'y_column': self.y_column,
            'color_column': self.color_column,
            'size_column': self.size_column,
            'width': self.width,
            'height': self.height
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChartConfig':
        """Create ChartConfig from dictionary."""
        return cls(
            chart_type=data.get('chart_type', ChartType.BAR),
            title=data.get('title'),
            x_label=data.get('x_label'),
            y_label=data.get('y_label'),
            x_column=data.get('x_column'),
            y_column=data.get('y_column'),
            color_column=data.get('color_column'),
            size_column=data.get('size_column'),
            width=data.get('width'),
            height=data.get('height')
        )

class ChartBase(ABC):
    """Base class for all chart implementations."""
    
    def __init__(self, config: Optional[ChartConfig] = None):
        """
        Initialize chart with configuration.
        
        Args:
            config: Chart configuration object
        """
        self.config = config or ChartConfig()
    
    @abstractmethod
    def from_dataframe(self, df: pd.DataFrame) -> Any:
        """
        Generate chart from DataFrame.
        
        Args:
            df: DataFrame containing chart data
            
        Returns:
            Chart object
        """
        pass
    
    @abstractmethod
    def from_dict(self, data: Dict[str, Any]) -> Any:
        """
        Generate chart from dictionary data.
        
        Args:
            data: Dictionary containing chart data
            
        Returns:
            Chart object
        """
        pass
    
    @abstractmethod
    def render(self) -> Any:
        """
        Render the chart.
        
        Returns:
            Rendered chart object
        """
        pass
    
    @staticmethod
    def get_chart_for_type(chart_type: Union[ChartType, str]) -> 'ChartBase':
        """
        Factory method to get chart implementation for specified type.
        
        Args:
            chart_type: Type of chart to create
            
        Returns:
            ChartBase implementation for the specified type
        """
        # This will be implemented by each chart type
        # with specific imports for each implementation
        from .chart_implementations import (
            BarChart, LineChart, PieChart, 
            ScatterChart, AreaChart, HeatmapChart
        )
        
        type_map = {
            ChartType.BAR: BarChart,
            ChartType.LINE: LineChart,
            ChartType.PIE: PieChart,
            ChartType.SCATTER: ScatterChart,
            ChartType.AREA: AreaChart,
            ChartType.HEATMAP: HeatmapChart
        }
        
        chart_type_enum = chart_type if isinstance(chart_type, ChartType) else ChartType.from_string(chart_type)
        
        if chart_type_enum in type_map:
            return type_map[chart_type_enum]()
        
        # Default to bar chart
        return BarChart()

