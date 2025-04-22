
"""
Concrete chart implementations for Watchdog AI.

This module provides specific chart implementations for various chart types,
using appropriate visualization libraries (Altair, Plotly).
"""

import altair as alt
import pandas as pd
import logging
from typing import Dict, Any, Optional, Union, List

from .chart_base import ChartBase, ChartConfig, ChartType

logger = logging.getLogger(__name__)

class BarChart(ChartBase):
    """Bar chart implementation."""
    
    def from_dataframe(self, df: pd.DataFrame) -> alt.Chart:
        """
        Generate bar chart from DataFrame.
        
        Args:
            df: DataFrame containing chart data
            
        Returns:
            Altair Chart object
        """
        try:
            # Determine x and y columns
            x_column = self.config.x_column or self._find_x_column(df)
            y_column = self.config.y_column or self._find_y_column(df)
            
            if not x_column or not y_column:
                logger.warning("Could not determine x or y columns")
                return self._create_error_chart()
            
            # Create the base chart
            chart = alt.Chart(df).mark_bar()
            
            # Add encodings
            encoding = {
                'x': alt.X(x_column, title=self.config.x_label or x_column),
                'y': alt.Y(y_column, title=self.config.y_label or y_column)
            }
            
            # Add color if specified
            if self.config.color_column and self.config.color_column in df.columns:
                encoding['color'] = alt.Color(self.config.color_column)
                
            chart = chart.encode(**encoding)
            
            # Add title if specified
            if self.config.title:
                chart = chart.properties(title=self.config.title)
                
            # Set width and height if specified
            properties = {}
            if self.config.width:
                properties['width'] = self.config.width
            if self.config.height:
                properties['height'] = self.config.height
                
            if properties:
                chart = chart.properties(**properties)
                
            return chart
            
        except Exception as e:
            logger.error(f"Error creating bar chart: {str(e)}")
            return self._create_error_chart()
    
    def from_dict(self, data: Dict[str, Any]) -> alt.Chart:
        """
        Generate bar chart from dictionary data.
        
        Args:
            data: Dictionary containing chart data
            
        Returns:
            Altair Chart object
        """
        try:
            # Convert dictionary to DataFrame
            # Handle different formats
            if 'x' in data and 'y' in data:
                df = pd.DataFrame({
                    'x': data['x'],
                    'y': data['y']
                })
                self.config.x_column = 'x'
                self.config.y_column = 'y'
            elif isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                logger.warning("Unsupported data format")
                return self._create_error_chart()
                
            return self.from_dataframe(df)
            
        except Exception as e:
            logger.error(f"Error creating bar chart from dict: {str(e)}")
            return self._create_error_chart()
    
    def render(self) -> alt.Chart:
        """
        Render the chart.
        
        Returns:
            Rendered chart object
        """
        # This is a placeholder as rendering is typically handled by the charting library
        # with Streamlit integration
        return None
        
    def _find_x_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find appropriate column for x-axis."""
        # Prefer time/date columns for x-axis in line charts
        # Check for date/time columns
        date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
        if date_cols:
            return date_cols[0]
            
        # Look for columns with time/date in the name
        time_cols = [col for col in df.columns if any(term in col.lower() for term in ['time', 'date', 'year', 'month', 'day'])]
        if time_cols:
            return time_cols[0]
            
        # If no time columns, use the first column
        return df.columns[0] if len(df.columns) > 0 else None
        
    def _find_y_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find appropriate column for y-axis."""
        # Prefer numeric columns for y-axis
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols:
            return numeric_cols[0]
            
        # If no numeric columns, use the second column or first column if only one
        if len(df.columns) > 1:
            return df.columns[1]
        elif len(df.columns) > 0:
            return df.columns[0]
        return None
        
    def _create_error_chart(self) -> alt.Chart:
        """Create a fallback chart for error cases."""
        df = pd.DataFrame({'x': [0, 1, 2], 'y': [0, 0, 0]})
        return (
            alt.Chart(df)
            .mark_line()
            .encode(x='x', y='y')
            .properties(title="Error creating chart")
        )


class PieChart(ChartBase):
    """Pie chart implementation."""
    
    def from_dataframe(self, df: pd.DataFrame) -> alt.Chart:
        """
        Generate pie chart from DataFrame.
        
        Args:
            df: DataFrame containing chart data
            
        Returns:
            Altair Chart object
        """
        try:
            # Determine category and value columns
            category_column = self.config.x_column or self._find_category_column(df)
            value_column = self.config.y_column or self._find_value_column(df)
            
            if not category_column or not value_column:
                logger.warning("Could not determine category or value columns")
                return self._create_error_chart()
            
            # Create the pie chart using Arc marks
            chart = alt.Chart(df).mark_arc().encode(
                theta=alt.Theta(field=value_column, type="quantitative"),
                color=alt.Color(field=category_column, type="nominal", 
                                title=self.config.x_label or category_column)
            )
            
            # Add title if specified
            if self.config.title:
                chart = chart.properties(title=self.config.title)
                
            # Set width and height if specified
            properties = {}
            if self.config.width:
                properties['width'] = self.config.width
            if self.config.height:
                properties['height'] = self.config.height
                
            if properties:
                chart = chart.properties(**properties)
                
            return chart
            
        except Exception as e:
            logger.error(f"Error creating pie chart: {str(e)}")
            return self._create_error_chart()
    
    def from_dict(self, data: Dict[str, Any]) -> alt.Chart:
        """
        Generate pie chart from dictionary data.
        
        Args:
            data: Dictionary containing chart data
            
        Returns:
            Altair Chart object
        """
        try:
            # Convert dictionary to DataFrame
            # Handle different formats
            if 'labels' in data and 'values' in data:
                df = pd.DataFrame({
                    'category': data['labels'],
                    'value': data['values']
                })
                self.config.x_column = 'category'
                self.config.y_column = 'value'
            elif isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                logger.warning("Unsupported data format")
                return self._create_error_chart()
                
            return self.from_dataframe(df)
            
        except Exception as e:
            logger.error(f"Error creating pie chart from dict: {str(e)}")
            return self._create_error_chart()
    
    def render(self) -> alt.Chart:
        """
        Render the chart.
        
        Returns:
            Rendered chart object
        """
        # This is a placeholder as rendering is typically handled by the charting library
        return None
        
    def _find_category_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find appropriate column for categories."""
        # Prefer categorical/string columns for categories
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            return categorical_cols[0]
            
        # If no categorical columns, use the first column
        return df.columns[0] if len(df.columns) > 0 else None
        
    def _find_value_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find appropriate column for values."""
        # Prefer numeric columns for values
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols:
            return numeric_cols[0]
            
        # If no numeric columns, use the second column or first column if only one
        if len(df.columns) > 1:
            return df.columns[1]
        elif len(df.columns) > 0:
            return df.columns[0]
        return None
        
    def _create_error_chart(self) -> alt.Chart:
        """Create a fallback chart for error cases."""
        df = pd.DataFrame({'category': ['Error'], 'value': [1]})
        return (
            alt.Chart(df)
            .mark_arc()
            .encode(theta='value', color='category')
            .properties(title="Error creating chart")
        )


class ScatterChart(ChartBase):
    """Scatter chart implementation."""
    
    def from_dataframe(self, df: pd.DataFrame) -> alt.Chart:
        """
        Generate scatter chart from DataFrame.
        
        Args:
            df: DataFrame containing chart data
            
        Returns:
            Altair Chart object
        """
        try:
            # Determine x and y columns
            x_column = self.config.x_column or self._find_x_column(df)
            y_column = self.config.y_column or self._find_y_column(df)
            
            if not x_column or not y_column:
                logger.warning("Could not determine x or y columns")
                return self._create_error_chart()
            
            # Create the base chart
            chart = alt.Chart(df).mark_circle()
            
            # Add encodings
            encoding = {
                'x': alt.X(x_column, title=self.config.x_label or x_column),
                'y': alt.Y(y_column, title=self.config.y_label or y_column),
            }
            
            # Add color if specified
            if self.config.color_column and self.config.color_column in df.columns:
                encoding['color'] = alt.Color(self.config.color_column)
                
            # Add size if specified
            if self.config.size_column and self.config.size_column in df.columns:
                encoding['size'] = alt.Size(self.config.size_column)
                
            chart = chart.encode(**encoding)
            
            # Add title if specified
            if self.config.title:
                chart = chart.properties(title=self.config.title)
                
            # Set width and height if specified
            properties = {}
            if self.config.width:
                properties['width'] = self.config.width
            if self.config.height:
                properties['height'] = self.config.height
                
            if properties:
                chart = chart.properties(**properties)
                
            return chart
            
        except Exception as e:
            logger.error(f"Error creating scatter chart: {str(e)}")
            return self._create_error_chart()
    
    def from_dict(self, data: Dict[str, Any]) -> alt.Chart:
        """
        Generate scatter chart from dictionary data.
        
        Args:
            data: Dictionary containing chart data
            
        Returns:
            Altair Chart object
        """
        try:
            # Convert dictionary to DataFrame
            # Handle different formats
            if 'x' in data and 'y' in data:
                df_data = {
                    'x': data['x'],
                    'y': data['y']
                }
                # Add size if available
                if 'size' in data and isinstance(data['size'], list):
                    df_data['size'] = data['size']
                    self.config.size_column = 'size'
                    
                # Add color if available
                if 'color' in data and isinstance(data['color'], list):
                    df_data['color'] = data['color']
                    self.config.color_column = 'color'
                    
                df = pd.DataFrame(df_data)
                self.config.x_column = 'x'
                self.config.y_column = 'y'
            elif isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                logger.warning("Unsupported data format")
                return self._create_error_chart()
                
            return self.from_dataframe(df)
            
        except Exception as e:
            logger.error(f"Error creating scatter chart from dict: {str(e)}")
            return self._create_error_chart()
    
    def render(self) -> alt.Chart:
        """
        Render the chart.
        
        Returns:
            Rendered chart object
        """
        # This is a placeholder as rendering is typically handled by the charting library
        return None
        
    def _find_x_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find appropriate column for x-axis."""
        # Prefer numeric columns for scatter plots
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if len(numeric_cols) >= 2:
            return numeric_cols[0]
        elif numeric_cols:
            return numeric_cols[0]
            
        # If no numeric columns, use the first column
        return df.columns[0] if len(df.columns) > 0 else None
        
    def _find_y_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find appropriate column for y-axis."""
        # Prefer numeric columns for y-axis
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if len(numeric_cols) >= 2:
            return numeric_cols[1]
        elif numeric_cols:
            return numeric_cols[0]
            
        # If no numeric columns, use the second column or first column if only one
        if len(df.columns) > 1:
            return df.columns[1]
        elif len(df.columns) > 0:
            return df.columns[0]
        return None
        
    def _create_error_chart(self) -> alt.Chart:
        """Create a fallback chart for error cases."""
        df = pd.DataFrame({'x': [0], 'y': [0]})
        return (
            alt.Chart(df)
            .mark_circle()
            .encode(x='x', y='y')
            .properties(title="Error creating chart")
        )


class AreaChart(ChartBase):
    """Area chart implementation."""
    
    def from_dataframe(self, df: pd.DataFrame) -> alt.Chart:
        """
        Generate area chart from DataFrame.
        
        Args:
            df: DataFrame containing chart data
            
        Returns:
            Altair Chart object
        """
        try:
            # Determine x and y columns
            x_column = self.config.x_column or self._find_x_column(df)
            y_column = self.config.y_column or self._find_y_column(df)
            
            if not x_column or not y_column:
                logger.warning("Could not determine x or y columns")
                return self._create_error_chart()
            
            # Create the base chart
            chart = alt.Chart(df).mark_area()
            
            # Add encodings
            encoding = {
                'x': alt.X(x_column, title=self.config.x_label or x_column),
                'y': alt.Y(y_column, title=self.config.y_label or y_column)
            }
            
            # Add color if specified
            if self.config.color_column and self.config.color_column in df.columns:
                encoding['color'] = alt.Color(self.config.color_column)
                
            chart = chart.encode(**encoding)
            
            # Add title if specified
            if self.config.title:
                chart = chart.properties(title=self.config.title)
                
            # Set width and height if specified
            properties = {}
            if self.config.width:
                properties['width'] = self.config.width
            if self.config.height:
                properties['height'] = self.config.height
                
            if properties:
                chart = chart.properties(**properties)
                
            return chart
            
        except Exception as e:
            logger.error(f"Error creating area chart: {str(e)}")
            return self._create_error_chart()
    
    def from_dict(self, data: Dict[str, Any]) -> alt.Chart:
        """
        Generate area chart from dictionary data.
        
        Args:
            data: Dictionary containing chart data
            
        Returns:
            Altair Chart object
        """
        try:
            # Convert dictionary to DataFrame
            # Handle different formats
            if 'x' in data and 'y' in data:
                df = pd.DataFrame({
                    'x': data['x'],
                    'y': data['y']
                })
                self.config.x_column = 'x'
                self.config.y_column = 'y'
            elif isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                logger.warning("Unsupported data format")
                return self._create_error_chart()
                
            return self.from_dataframe(df)
            
        except Exception as e:
            logger.error(f"Error creating area chart from dict: {str(e)}")
            return self._create_error_chart()
    
    def render(self) -> alt.Chart:
        """
        Render the chart.
        
        Returns:
            Rendered chart object
        """
        # This is a placeholder as rendering is typically handled by the charting library
        return None
        
    def
        """
        Render the chart.
        
        Returns:
            Rendered chart object
        """
        # This is a placeholder as rendering is typically handled by the charting library
        # with Streamlit integration
        return None
        
    def _find_x_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find appropriate column for x-axis."""
        # Prefer categorical/string columns for x-axis in bar charts
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            return categorical_cols[0]
            
        # If no categorical columns, use the first column
        return df.columns[0] if len(df.columns) > 0 else None
        
    def _find_y_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find appropriate column for y-axis."""
        # Prefer numeric columns for y-axis
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols:
            return numeric_cols[0]
            
        # If no numeric columns, use the second column or first column if only one
        if len(df.columns) > 1:
            return df.columns[1]
        elif len(df.columns) > 0:
            return df.columns[0]
        return None
        
    def _create_error_chart(self) -> alt.Chart:
        """Create a fallback chart for error cases."""
        df = pd.DataFrame({'x': ['Error'], 'y': [0]})
        return (
            alt.Chart(df)
            .mark_bar()
            .encode(x='x', y='y')
            .properties(title="Error creating chart")
        )


class LineChart(ChartBase):
    """Line chart implementation."""
    
    def from_dataframe(self, df: pd.DataFrame) -> alt.Chart:
        """
        Generate line chart from DataFrame.
        
        Args:
            df: DataFrame containing chart data
            
        Returns:
            Altair Chart object
        """
        try:
            # Determine x and y columns
            x_column = self.config.x_column or self._find_x_column(df)
            y_column = self.config.y_column or self._find_y_column(df)
            
            if not x_column or not y_column:
                logger.warning("Could not determine x or y columns")
                return self._create_error_chart()
            
            # Create the base chart
            chart = alt.Chart(df).mark_line()
            
            # Add encodings
            encoding = {
                'x': alt.X(x_column, title=self.config.x_label or x_column),
                'y': alt.Y(y_column, title=self.config.y_label or y_column)
            }
            
            # Add color if specified
            if self.config.color_column and self.config.color_column in df.columns:
                encoding['color'] = alt.Color(self.config.color_column)
                
            chart = chart.encode(**encoding)
            
            # Add title if specified
            if self.config.title:
                chart = chart.properties(title=self.config.title)
                
            # Set width and height if specified
            properties = {}
            if self.config.width:
                properties['width'] = self.config.width
            if self.config.height:
                properties['height'] = self.config.height
                
            if properties:
                chart = chart.properties(**properties)
                
            return chart
            
        except Exception as e:
            logger.error(f"Error creating line chart: {str(e)}")
            return self._create_error_chart()
    
    def from_dict(self, data: Dict[str, Any]) -> alt.Chart:
        """
        Generate line chart from dictionary data.
        
        Args:
            data: Dictionary containing chart data
            
        Returns:
            Altair Chart object
        """
        try:
            # Convert dictionary to DataFrame
            # Handle different formats
            if 'x' in data and 'y' in data:
                df = pd.DataFrame({
                    'x': data['x'],
                    'y': data['y']
                })
                self.config.x_column = 'x'
                self.config.y_column = 'y'
            elif isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                logger.warning("Unsupported data format")
                return self._create_error_chart()
                
            return self.from_dataframe(df)
            
        except Exception as e:
            logger.error(f"Error creating line chart from dict: {str(e)}")
            return self._create_error_chart()
    
    def render(self) -> alt.Chart:

