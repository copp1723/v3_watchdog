"""
Analytics Engine for Watchdog AI.

This module provides data analysis capabilities including sales trends,
profit analysis, and period-over-period comparisons. It standardizes data
loading and processing while providing a unified analysis API.
"""

import logging
import os
from datetime import datetime
from typing import Dict, Any, List, Tuple

import pandas as pd

from watchdog_ai.core.constants import (
    COLUMN_MAPPINGS,
    DEFAULT_COLUMN_TYPES,
    DEFAULT_REQUIRED_COLUMNS,
    NAN_WARNING_THRESHOLD,
    NAN_SEVERE_THRESHOLD,
    ERR_NO_DATA,
    ERR_COLUMN_NOT_FOUND
)
from watchdog_ai.core.data_utils import (
    find_matching_column,
    clean_numeric_data,
    analyze_data_quality,
    format_metric_value,
    normalize_boolean_column
)
from watchdog_ai.insights.insight_functions import InsightFunctions

logger = logging.getLogger(__name__)


class AnalyticsEngine:
    """Core analytics engine for processing and analyzing sales data."""

    def __init__(self):
        """Initialize the analytics engine."""
        self.insights = InsightFunctions()

    def load_csv(self, file_path: str) -> pd.DataFrame:
        """
        Load a CSV file into a pandas DataFrame.

        Args:
            file_path: Path to the CSV file

        Returns:
            Loaded DataFrame

        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the file is empty or not a valid CSV
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                raise ValueError(ERR_NO_DATA)
            return df
        except Exception as e:
            logger.error(f"Error loading CSV file {file_path}: {e}")
            raise ValueError(f"Failed to load CSV file: {str(e)}")

    def _standardize_dataframe(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Standardize DataFrame columns and data types.

        Args:
            df: Input DataFrame to standardize

        Returns:
            Tuple containing:
            - Standardized DataFrame
            - List of warning messages

        Raises:
            ValueError: If required columns are missing or data is invalid
        """
        # Initialize
        warnings = []
        result = df.copy()

        # Check for empty DataFrame
        if result.empty:
            raise ValueError(ERR_NO_DATA)
        
        # Step 1: Standardize column names
        for standard_col, variants in COLUMN_MAPPINGS.items():
            if standard_col not in result.columns:
                found = False
                for variant in variants:
                    if variant in result.columns:
                        result.rename(columns={variant: standard_col}, inplace=True)
                        warnings.append(f"Renamed column '{variant}' to '{standard_col}'")
                        found = True
                        break
        
        # Step 2: Check for missing required columns
        missing_cols = [col for col in DEFAULT_REQUIRED_COLUMNS if col not in result.columns]
        if missing_cols:
            raise ValueError(ERR_COLUMN_NOT_FOUND.format(", ".join(missing_cols)))
        
        # Step 3: Enforce data types with proper error handling
        for col, dtype in DEFAULT_COLUMN_TYPES.items():
            if col in result.columns:
                try:
                    if dtype == 'datetime64[ns]':
                        # First try with coerce to see if we have invalid dates
                        temp_col = pd.to_datetime(result[col], errors='coerce')
                        if temp_col.isna().any() and not result[col].isna().any():
                            # Found invalid dates that were coerced to NaT
                            invalid_date = result[col][temp_col.isna() & ~result[col].isna()].iloc[0]
                            raise ValueError(f"Column {col} contains invalid date format: '{invalid_date}'")
                        result[col] = temp_col
                    elif dtype in ('float64', 'int64'):
                        # Try strict conversion first to detect problems
                        try:
                            result[col] = pd.to_numeric(result[col], errors='raise')
                        except Exception:
                            # Clean and try again, but raise error if still fails after cleaning
                            result[col] = clean_numeric_data(result, col)[col]
                            if result[col].isna().all():
                                raise ValueError(f"Column {col} contains non-numeric values that can't be converted")
                    elif dtype == 'int64' and col == 'IsSale':
                        # Special handling for boolean columns
                        result[col] = normalize_boolean_column(result[col])
                except Exception as e:
                    # For critical data types, raise an error instead of just warning
                    if col in ['SaleDate', 'TotalGross']:
                        raise ValueError(f"Invalid data type in column {col}: {str(e)}")
                    warnings.append(
                        f"Could not convert {col} to {dtype}: {str(e)}"
                    )

        # Step 4: Check data quality
        for col in result.columns:
            quality = analyze_data_quality(result, col)
            nan_pct = quality['nan_percentage']
            if nan_pct >= NAN_SEVERE_THRESHOLD * 100:
                warnings.append(
                    f"Severe: Column '{col}' has {nan_pct:.1f}% missing values"
                )
            elif nan_pct >= NAN_WARNING_THRESHOLD * 100:
                warnings.append(
                    f"Warning: Column '{col}' has {nan_pct:.1f}% missing values"
                )

        return result, warnings
    def calculate_sales_trends(
        self, df: pd.DataFrame, date_col: str = 'SaleDate'
    ) -> Dict[str, Any]:
        """
        Calculate sales trends over time.

        Args:
            df: Input DataFrame
            date_col: Name of date column to use

        Returns:
            Dictionary containing trend metrics and time series data
        """
        # Make sure we work on a clean copy without temporary columns
        df = df.copy()
        if 'period' in df.columns:
            df = df.drop('period', axis=1)
        if date_col not in df.columns:
            return {
                "error": ERR_COLUMN_NOT_FOUND.format(date_col),
                "trends": []
            }

        try:
            # Ensure date column is datetime
            df[date_col] = pd.to_datetime(df[date_col])
            # Calculate daily sales counts
            daily_sales = df.groupby(
                pd.Grouper(key=date_col, freq='D')
            ).size().reset_index(name='count')
            daily_sales = daily_sales.sort_values(date_col)
            # Calculate rolling averages
            daily_sales['7d_avg'] = daily_sales['count'].rolling(7).mean()
            daily_sales['30d_avg'] = daily_sales['count'].rolling(30).mean()
            # Calculate month-to-date and year-to-date sales
            current_month = datetime.now().month
            current_year = datetime.now().year
            mtd_sales = df[
                (df[date_col].dt.month == current_month) &
                (df[date_col].dt.year == current_year)
            ].shape[0]
            ytd_sales = df[df[date_col].dt.year == current_year].shape[0]
            # Format for JSON
            trends = []
            for _, row in daily_sales.iterrows():
                trends.append({
                    "date": row[date_col].strftime('%Y-%m-%d'),
                    "count": int(row['count']),
                    "7d_avg": (float(row['7d_avg'])
                               if not pd.isna(row['7d_avg']) else None),
                    "30d_avg": (float(row['30d_avg'])
                                if not pd.isna(row['30d_avg']) else None)
                })
            # Calculate summary metrics
            total_sales = daily_sales['count'].sum()
            avg_daily = daily_sales['count'].mean()
            return {
                "total_sales": int(total_sales),
                "average_daily_sales": round(avg_daily, 2),
                "mtd_sales": int(mtd_sales),
                "ytd_sales": int(ytd_sales),
                "trends": trends
            }
        except Exception as e:
            logger.error(f"Error calculating sales trends: {e}")
            return {"error": str(e), "trends": []}
    def calculate_gross_profit_by_source(
        self,
        df: pd.DataFrame,
        source_col: str = 'LeadSource',
        profit_col: str = 'TotalGross'
    ) -> Dict[str, Any]:
        """
        Calculate gross profit aggregated by lead source.

        Args:
            df: Input DataFrame
            source_col: Lead source column name
            profit_col: Gross profit column name

        Returns:
            Dictionary containing profit metrics by source
        """
        # Make sure we work on a clean copy without temporary columns
        df = df.copy()
        if 'period' in df.columns:
            df = df.drop('period', axis=1)
        required_cols = [source_col, profit_col]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            return {
                "error": ERR_COLUMN_NOT_FOUND.format(", ".join(missing)),
                "breakdown": []
            }

        try:
            # Group by source and calculate metrics
            grouped = df.groupby(source_col).agg({
                profit_col: ['sum', 'mean', 'count']
            }).reset_index()
            # Flatten column names
            grouped.columns = [
                source_col, 'total_profit', 'avg_profit', 'sale_count'
            ]
            # Sort by total profit
            grouped = grouped.sort_values('total_profit', ascending=False)
            # Calculate percentages
            total_profit = grouped['total_profit'].sum()
            grouped['profit_percentage'] = (
                grouped['total_profit'] / total_profit * 100
            )
            # Format values for display
            breakdown = []
            for _, row in grouped.iterrows():
                breakdown.append({
                    "source": row[source_col],
                    "total_profit": format_metric_value(
                        row['total_profit'], 'profit'
                    ),
                    "avg_profit": format_metric_value(
                        row['avg_profit'], 'profit'
                    ),
                    "sale_count": int(row['sale_count']),
                    "percentage": f"{row['profit_percentage']:.1f}%"
                })
            return {
                "total_profit": format_metric_value(total_profit, 'profit'),
                "breakdown": breakdown
            }
        except Exception as e:
            logger.error(f"Error calculating profit by source: {e}")
            return {"error": str(e), "breakdown": []}

    def calculate_yoy_comparison(
        self,
        df: pd.DataFrame,
        date_col: str = 'SaleDate',
        metric_col: str = 'TotalGross'
    ) -> Dict[str, Any]:
        """
        Calculate year-over-year comparison.

        Args:
            df: Input DataFrame
            date_col: Date column name
            metric_col: Metric to compare

        Returns:
            Dictionary containing year-over-year comparisons
        """
        return self._calculate_period_comparison(df, date_col, metric_col, 'Y')

    def calculate_mom_comparison(
        self,
        df: pd.DataFrame,
        date_col: str = 'SaleDate',
        metric_col: str = 'TotalGross'
    ) -> Dict[str, Any]:
        """
        Calculate month-over-month comparison.

        Args:
            df: Input DataFrame
            date_col: Date column name
            metric_col: Metric to compare

        Returns:
            Dictionary containing month-over-month comparisons
        """
        return self._calculate_period_comparison(df, date_col, metric_col, 'M')

    def _calculate_period_comparison(
        self,
        df: pd.DataFrame,
        date_col: str = 'SaleDate',
        metric_col: str = 'TotalGross',
        period: str = 'Y'
    ) -> Dict[str, Any]:
        """
        Calculate year-over-year or month-over-month comparisons.

        Args:
            df: Input DataFrame
            date_col: Date column name
            metric_col: Metric to compare
            period: 'Y' for year-over-year, 'M' for month-over-month

        Returns:
            Dictionary containing period-over-period comparisons
        """
        required_cols = [date_col, metric_col]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            return {
                "error": ERR_COLUMN_NOT_FOUND.format(", ".join(missing)),
                "comparisons": []
            }

        try:
            # Work on a copy to avoid modifying original
            temp_df = df.copy()
            # Ensure date column is datetime
            temp_df[date_col] = pd.to_datetime(temp_df[date_col])
            # Group by period
            if period == 'Y':
                temp_df['period'] = temp_df[date_col].dt.year
                # Period is represented as year
                period_name = "Year"
            else:  # 'M'
                temp_df['period'] = temp_df[date_col].dt.strftime('%Y-%m')
                period_name = "Month"

            # Calculate metrics by period
            grouped = temp_df.groupby('period').agg({
                metric_col: ['sum', 'mean', 'count']
            }).reset_index()

            # Flatten column names
            grouped.columns = ['period'] + [
                f"{metric_col}_{op}" for op in ['sum', 'mean', 'count']
            ]

            # Calculate period-over-period changes
            grouped = grouped.sort_values('period')
            grouped['prev_sum'] = grouped[f"{metric_col}_sum"].shift(1)
            grouped['change'] = grouped[f"{metric_col}_sum"] - grouped['prev_sum']
            grouped['change_pct'] = grouped.apply(
                lambda row: ((row['change'] / row['prev_sum'] * 100)
                             if row['prev_sum'] and row['prev_sum'] != 0 else 0),
                axis=1
            )

            # Format results
            comparisons = []
            for _, row in grouped.iterrows():
                if pd.isna(row['prev_sum']):
                    continue
                comparisons.append({
                    "period": str(row['period']),
                    "period_name": period_name,
                    "current_value": format_metric_value(row[f"{metric_col}_sum"], metric_col),
                    "previous_value": format_metric_value(row['prev_sum'], metric_col),
                    "change": format_metric_value(row['change'], metric_col),
                    "change_percentage": f"{row['change_pct']:.1f}%"
                })
            
            # Clean up temporary columns
            if 'period' in temp_df.columns:
                temp_df.drop(columns=['period'], inplace=True)
                
            return {
                "period_type": period,
                "period_name": period_name,
                "comparisons": comparisons
            }
        except Exception as e:
            logger.error(f"Error calculating period comparison: {e}")
            return {"error": str(e), "comparisons": []}

    def _analyze_data_quality(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Analyze data quality for all columns in the DataFrame.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary mapping column names to quality metrics
        """
        quality_metrics = {}
        for col in df.columns:
            quality = analyze_data_quality(df, col)
            quality_metrics[col] = {
                "missing_percentage": quality["nan_percentage"],
                "warning_level": quality["warning_level"]
            }
        return quality_metrics

    def run_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run a comprehensive analysis on the provided DataFrame.

        Args:
            df: Input DataFrame to analyze

        Returns:
            Dictionary containing analysis results ready for JSON serialization

        Raises:
            ValueError: If input data is invalid or required columns are
                missing
        """
        try:
            # Standardize the data
            standardized_df, warnings = self._standardize_dataframe(df)
            
            # Calculate all metrics
            sales_trends = self.calculate_sales_trends(standardized_df)
            profit_analysis = self.calculate_gross_profit_by_source(standardized_df)
            yoy_comparison = self.calculate_yoy_comparison(standardized_df)
            mom_comparison = self.calculate_mom_comparison(standardized_df)
            
            # Check for errors in individual analyses and collect additional warnings
            analyses = [sales_trends, profit_analysis, yoy_comparison, mom_comparison]
            for analysis in analyses:
                if 'error' in analysis:
                    warnings.append(analysis['error'])
            
            # Build result dictionary
            result = {
                'success': True,
                'timestamp': datetime.now().isoformat(),
                'metadata': {
                    'row_count': len(standardized_df),
                    'column_count': len(standardized_df.columns)
                },
                'warnings': warnings,
                'data_quality': self._analyze_data_quality(standardized_df),
                'sales_trends': sales_trends,
                'profit_by_source': profit_analysis,
                'year_over_year': yoy_comparison,
                'month_over_month': mom_comparison
            }
            
            return result
        except Exception as e:
            logger.error(f"Unexpected error in run_analysis: {e}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
