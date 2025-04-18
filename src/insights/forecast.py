"""
Predictive Sales Forecast for Watchdog AI.

This module provides classes for generating time-series forecasts
of sales metrics and inventory performance.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
import sentry_sdk
from dataclasses import dataclass
from abc import ABC, abstractmethod
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

# Configure logger
logger = logging.getLogger(__name__)

@dataclass
class ForecastConfig:
    """Configuration for forecast generation."""
    periods: int = 30  # Number of periods to forecast
    confidence_level: float = 0.95  # Confidence level for intervals
    min_history_periods: int = 60  # Minimum periods needed for forecasting
    seasonality_test_periods: int = 90  # Periods to test for seasonality

class ForecastResult:
    """Container for forecast results."""
    def __init__(
        self,
        forecast: pd.Series,
        confidence_intervals: pd.DataFrame,
        metrics: Dict[str, Any],
        model_info: Dict[str, Any]
    ):
        self.forecast = forecast
        self.confidence_intervals = confidence_intervals
        self.metrics = metrics
        self.model_info = model_info
        self.generated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert forecast result to dictionary."""
        return {
            "forecast": self.forecast.to_dict(),
            "confidence_intervals": self.confidence_intervals.to_dict(),
            "metrics": self.metrics,
            "model_info": self.model_info,
            "generated_at": self.generated_at.isoformat()
        }

class ForecastGenerator(ABC):
    """
    Base class for forecast generation.
    
    Provides common functionality for time series analysis
    and forecast generation.
    """
    
    def __init__(self, config: Optional[ForecastConfig] = None):
        """
        Initialize the forecast generator.
        
        Args:
            config: Optional configuration for forecasting behavior
        """
        self.config = config or ForecastConfig()
    
    def generate(self, df: pd.DataFrame, **kwargs) -> ForecastResult:
        """
        Generate a forecast from historical data.
        
        Args:
            df: Historical DataFrame to forecast from
            **kwargs: Additional parameters for specific forecasters
            
        Returns:
            ForecastResult containing predictions and metadata
        """
        try:
            # Track forecasting in Sentry
            sentry_sdk.set_tag("forecast_generator", self.__class__.__name__)
            
            # Validate and prepare data
            prepared_df = self._prepare_data(df)
            
            if len(prepared_df) < self.config.min_history_periods:
                raise ValueError(
                    f"Insufficient history for forecasting. Need at least "
                    f"{self.config.min_history_periods} periods, got {len(prepared_df)}"
                )
            
            # Check for seasonality
            seasonality_info = self._analyze_seasonality(prepared_df)
            
            # Generate forecast
            forecast, intervals = self._generate_forecast(
                prepared_df, seasonality_info, **kwargs
            )
            
            # Calculate accuracy metrics
            metrics = self._calculate_metrics(prepared_df, forecast)
            
            # Compile model info
            model_info = {
                "seasonality": seasonality_info,
                "data_points": len(prepared_df),
                "forecast_periods": self.config.periods,
                "confidence_level": self.config.confidence_level
            }
            
            return ForecastResult(
                forecast=forecast,
                confidence_intervals=intervals,
                metrics=metrics,
                model_info=model_info
            )
            
        except Exception as e:
            logger.error(f"Error generating forecast: {str(e)}")
            sentry_sdk.capture_exception(e)
            raise
    
    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for forecasting.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Prepared DataFrame
        """
        # Subclasses should implement specific preparation
        return df
    
    def _analyze_seasonality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze time series for seasonal patterns.
        
        Args:
            df: Time series DataFrame
            
        Returns:
            Dictionary with seasonality information
        """
        try:
            # Use recent data for seasonality analysis
            recent_data = df.iloc[-self.config.seasonality_test_periods:]
            
            # Perform seasonal decomposition
            decomposition = seasonal_decompose(
                recent_data,
                period=7,  # Start with weekly seasonality
                extrapolate_trend=True
            )
            
            # Calculate strength of seasonality
            seasonal_strength = np.std(decomposition.seasonal) / np.std(recent_data)
            
            # Test for stationarity
            adf_test = adfuller(recent_data)
            
            return {
                "seasonal_strength": float(seasonal_strength),
                "has_seasonality": seasonal_strength > 0.1,
                "period": 7,
                "stationary_pvalue": float(adf_test[1])
            }
            
        except Exception as e:
            logger.warning(f"Error analyzing seasonality: {str(e)}")
            return {
                "seasonal_strength": 0.0,
                "has_seasonality": False,
                "period": None,
                "stationary_pvalue": None
            }
    
    @abstractmethod
    def _generate_forecast(
        self,
        df: pd.DataFrame,
        seasonality_info: Dict[str, Any],
        **kwargs
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Generate the forecast and confidence intervals.
        
        Args:
            df: Prepared DataFrame
            seasonality_info: Dictionary with seasonality analysis
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (forecast Series, confidence intervals DataFrame)
        """
        pass
    
    def _calculate_metrics(
        self,
        historical: pd.DataFrame,
        forecast: pd.Series
    ) -> Dict[str, float]:
        """
        Calculate forecast accuracy metrics.
        
        Args:
            historical: Historical data
            forecast: Generated forecast
            
        Returns:
            Dictionary of metric names and values
        """
        # Calculate basic metrics
        metrics = {
            "mean": float(historical.mean()),
            "std": float(historical.std()),
            "min": float(historical.min()),
            "max": float(historical.max())
        }
        
        # Add forecast-specific metrics
        metrics.update({
            "forecast_mean": float(forecast.mean()),
            "forecast_std": float(forecast.std()),
            "forecast_min": float(forecast.min()),
            "forecast_max": float(forecast.max())
        })
        
        return metrics


class SalesPerformanceForecaster(ForecastGenerator):
    """Generates forecasts for sales performance metrics."""
    
    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare sales data for forecasting."""
        try:
            # Ensure required columns exist
            date_col = next(
                col for col in df.columns
                if any(term in col.lower() for term in ['date', 'time', 'day'])
            )
            
            metric_col = next(
                col for col in df.columns
                if any(term in col.lower() for term in ['gross', 'revenue', 'sales'])
            )
            
            # Convert to datetime
            df[date_col] = pd.to_datetime(df[date_col])
            
            # Ensure metric is numeric
            df[metric_col] = pd.to_numeric(
                df[metric_col].astype(str).str.replace(r'[\$,]', '', regex=True),
                errors='coerce'
            )
            
            # Group by date and calculate daily totals
            daily_data = df.groupby(df[date_col].dt.date)[metric_col].sum()
            
            # Sort by date
            daily_data = daily_data.sort_index()
            
            return daily_data
            
        except Exception as e:
            logger.error(f"Error preparing sales data: {str(e)}")
            raise
    
    def _generate_forecast(
        self,
        df: pd.DataFrame,
        seasonality_info: Dict[str, Any],
        **kwargs
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """Generate sales performance forecast."""
        try:
            # Determine ARIMA parameters based on seasonality
            if seasonality_info.get("has_seasonality"):
                # Use seasonal ARIMA
                order = (1, 1, 1)
                seasonal_order = (1, 1, 1, seasonality_info["period"])
                
                model = ARIMA(
                    df,
                    order=order,
                    seasonal_order=seasonal_order
                )
            else:
                # Use regular ARIMA
                model = ARIMA(df, order=(1, 1, 1))
            
            # Fit model
            fitted = model.fit()
            
            # Generate forecast
            forecast = fitted.forecast(self.config.periods)
            
            # Get confidence intervals
            intervals = fitted.get_forecast(self.config.periods).conf_int(
                alpha=1 - self.config.confidence_level
            )
            
            return forecast, intervals
            
        except Exception as e:
            logger.error(f"Error generating sales forecast: {str(e)}")
            raise


class InventoryTurnoverForecaster(ForecastGenerator):
    """Generates forecasts for inventory turnover and aging."""
    
    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare inventory data for forecasting."""
        try:
            # Find relevant columns
            days_col = next(
                col for col in df.columns
                if any(term in col.lower() for term in ['days', 'age', 'aging'])
            )
            
            date_col = next(
                col for col in df.columns
                if any(term in col.lower() for term in ['date', 'time', 'day'])
            )
            
            # Convert to datetime
            df[date_col] = pd.to_datetime(df[date_col])
            
            # Ensure days is numeric
            df[days_col] = pd.to_numeric(df[days_col], errors='coerce')
            
            # Calculate daily average days on lot
            daily_aging = df.groupby(df[date_col].dt.date)[days_col].mean()
            
            # Sort by date
            daily_aging = daily_aging.sort_index()
            
            return daily_aging
            
        except Exception as e:
            logger.error(f"Error preparing inventory data: {str(e)}")
            raise
    
    def _generate_forecast(
        self,
        df: pd.DataFrame,
        seasonality_info: Dict[str, Any],
        **kwargs
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """Generate inventory turnover forecast."""
        try:
            # Use simpler model for aging data
            model = ARIMA(df, order=(1, 0, 1))
            
            # Fit model
            fitted = model.fit()
            
            # Generate forecast
            forecast = fitted.forecast(self.config.periods)
            
            # Get confidence intervals
            intervals = fitted.get_forecast(self.config.periods).conf_int(
                alpha=1 - self.config.confidence_level
            )
            
            return forecast, intervals
            
        except Exception as e:
            logger.error(f"Error generating inventory forecast: {str(e)}")
            raise


class MarginTrendForecaster(ForecastGenerator):
    """Generates forecasts for gross margin trends."""
    
    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare margin data for forecasting."""
        try:
            # Find relevant columns
            gross_col = next(
                col for col in df.columns
                if any(term in col.lower() for term in ['gross', 'profit'])
            )
            
            price_col = next(
                (col for col in df.columns
                if any(term in col.lower() for term in ['price', 'revenue', 'sale'])),
                None
            )
            
            date_col = next(
                col for col in df.columns
                if any(term in col.lower() for term in ['date', 'time', 'day'])
            )
            
            # Convert to datetime
            df[date_col] = pd.to_datetime(df[date_col])
            
            # Calculate margins
            if price_col:
                # Clean numeric columns
                df[gross_col] = pd.to_numeric(
                    df[gross_col].astype(str).str.replace(r'[\$,]', '', regex=True),
                    errors='coerce'
                )
                df[price_col] = pd.to_numeric(
                    df[price_col].astype(str).str.replace(r'[\$,]', '', regex=True),
                    errors='coerce'
                )
                
                # Calculate margin percentage
                df['margin'] = df[gross_col] / df[price_col]
            else:
                # Use gross directly if no price column
                df['margin'] = pd.to_numeric(
                    df[gross_col].astype(str).str.replace(r'[\$,]', '', regex=True),
                    errors='coerce'
                )
            
            # Calculate daily average margin
            daily_margin = df.groupby(df[date_col].dt.date)['margin'].mean()
            
            # Sort by date
            daily_margin = daily_margin.sort_index()
            
            return daily_margin
            
        except Exception as e:
            logger.error(f"Error preparing margin data: {str(e)}")
            raise
    
    def _generate_forecast(
        self,
        df: pd.DataFrame,
        seasonality_info: Dict[str, Any],
        **kwargs
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """Generate margin trend forecast."""
        try:
            # Use ARIMA model with differencing for margin data
            model = ARIMA(df, order=(1, 1, 1))
            
            # Fit model
            fitted = model.fit()
            
            # Generate forecast
            forecast = fitted.forecast(self.config.periods)
            
            # Get confidence intervals
            intervals = fitted.get_forecast(self.config.periods).conf_int(
                alpha=1 - self.config.confidence_level
            )
            
            return forecast, intervals
            
        except Exception as e:
            logger.error(f"Error generating margin forecast: {str(e)}")
            raise