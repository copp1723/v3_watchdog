"""
Insight predictor module for Watchdog AI.

This module provides trend detection, anomaly detection,
and event triggering for personalized insights.
"""

import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from enum import Enum
import statsmodels.api as sm
from scipy import stats
from dataclasses import dataclass, field, asdict

from ..utils.logging_config import get_logger

logger = get_logger(__name__)

# Define insight types
class InsightType(str, Enum):
    TREND = "trend"
    ANOMALY = "anomaly"
    FORECAST = "forecast"
    EVENT = "event"

# Define insight scopes
class InsightScope(str, Enum):
    INVENTORY = "inventory"
    SALES = "sales"
    LEADS = "leads"
    SERVICE = "service"
    OVERALL = "overall"

# Define trend directions
class TrendDirection(str, Enum):
    UP = "up"
    DOWN = "down"
    STABLE = "stable"
    VOLATILE = "volatile"

# Define roles for targeting
class UserRole(str, Enum):
    GENERAL_MANAGER = "General Manager"
    SALES_MANAGER = "Sales Manager"
    INVENTORY_MANAGER = "Inventory Manager"
    MARKETING_MANAGER = "Marketing Manager"
    FINANCE_MANAGER = "Finance Manager"
    SALES_REPRESENTATIVE = "Sales Representative"
    SERVICE_MANAGER = "Service Manager"

@dataclass
class InsightResult:
    """Standardized output for insights."""
    insight_type: InsightType
    scope: InsightScope
    subject: str
    summary: str
    confidence: float  # 0.0 to 1.0
    target_roles: List[UserRole]
    tags: List[str]
    trend_direction: Optional[TrendDirection] = None
    magnitude: Optional[float] = None
    chart_config: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    dealer_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = asdict(self)
        # Convert enums to strings
        result['insight_type'] = self.insight_type.value
        result['scope'] = self.scope.value
        if self.trend_direction:
            result['trend_direction'] = self.trend_direction.value
        result['target_roles'] = [role.value for role in self.target_roles]
        return result
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

class InsightPredictor:
    """
    Detects trends, anomalies, and events in dealership data.
    
    This class analyzes normalized data from the ingestion pipeline
    to generate personalized insights for different user roles.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the insight predictor.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Set default thresholds
        self.trend_confidence_threshold = self.config.get('trend_confidence_threshold', 0.7)
        self.anomaly_z_threshold = self.config.get('anomaly_z_threshold', 2.0)
        self.anomaly_iqr_multiplier = self.config.get('anomaly_iqr_multiplier', 1.5)
        self.min_data_points = self.config.get('min_data_points', 5)
        self.trend_period_days = self.config.get('trend_period_days', 14)  # 2 weeks default
        
        # Role mapping for insights
        self.role_mapping = {
            InsightScope.INVENTORY: [
                UserRole.GENERAL_MANAGER,
                UserRole.INVENTORY_MANAGER
            ],
            InsightScope.SALES: [
                UserRole.GENERAL_MANAGER,
                UserRole.SALES_MANAGER,
                UserRole.SALES_REPRESENTATIVE
            ],
            InsightScope.LEADS: [
                UserRole.GENERAL_MANAGER,
                UserRole.SALES_MANAGER,
                UserRole.MARKETING_MANAGER
            ],
            InsightScope.SERVICE: [
                UserRole.GENERAL_MANAGER,
                UserRole.SERVICE_MANAGER
            ],
            InsightScope.OVERALL: [
                UserRole.GENERAL_MANAGER
            ]
        }
        
        logger.info(
            "Initialized InsightPredictor",
            extra={
                "component": "insight_predictor",
                "action": "initialize",
                "config": self.config
            }
        )
    
    def detect_trends(self, df: pd.DataFrame, date_column: str, value_column: str, 
                      scope: InsightScope, subject: str) -> List[InsightResult]:
        """
        Detect time-based trends in the data.
        
        Args:
            df: DataFrame with time series data
            date_column: Name of the date column
            value_column: Name of the value column to analyze
            scope: Scope of the insight (e.g., INVENTORY, SALES)
            subject: Subject of the trend (e.g., "Used Cars Inventory")
            
        Returns:
            List of InsightResult objects
        """
        try:
            # Ensure we have enough data
            if len(df) < self.min_data_points:
                logger.warning(
                    f"Insufficient data for trend detection: {len(df)} points < {self.min_data_points}",
                    extra={
                        "component": "insight_predictor",
                        "action": "detect_trends",
                        "data_points": len(df),
                        "subject": subject
                    }
                )
                return []
            
            # Ensure date column is datetime
            df[date_column] = pd.to_datetime(df[date_column])
            
            # Sort by date
            df = df.sort_values(date_column)
            
            # Check for dealer_id column
            dealer_id = None
            if 'dealer_id' in df.columns and len(df['dealer_id'].unique()) == 1:
                dealer_id = df['dealer_id'].iloc[0]
            
            insights = []
            
            # 1. Calculate Rolling Averages (7-day)
            if len(df) >= 7:
                df['rolling_7day'] = df[value_column].rolling(window=7, min_periods=3).mean()
                
                # Get the first and last valid rolling average
                first_valid = df['rolling_7day'].first_valid_index()
                last_valid = df['rolling_7day'].last_valid_index()
                
                if first_valid is not None and last_valid is not None:
                    start_avg = df.loc[first_valid, 'rolling_7day']
                    end_avg = df.loc[last_valid, 'rolling_7day']
                    
                    # Calculate percentage change
                    pct_change = ((end_avg - start_avg) / start_avg) * 100 if start_avg != 0 else 0
                    
                    # Determine direction
                    if abs(pct_change) < 5:  # Less than 5% change is considered stable
                        direction = TrendDirection.STABLE
                    elif pct_change > 0:
                        direction = TrendDirection.UP
                    else:
                        direction = TrendDirection.DOWN
                    
                    # Calculate confidence based on consistency and data points
                    confidence = min(0.9, 0.5 + (abs(pct_change) / 100) + (len(df) / 100))
                    
                    if confidence >= self.trend_confidence_threshold:
                        insights.append(InsightResult(
                            insight_type=InsightType.TREND,
                            scope=scope,
                            subject=subject,
                            summary=f"{subject} has {direction.value}ed by {abs(pct_change):.1f}% over the past 7 days.",
                            confidence=confidence,
                            target_roles=self.role_mapping[scope],
                            tags=["rolling_average", "7day", scope.value, direction.value],
                            trend_direction=direction,
                            magnitude=abs(pct_change),
                            chart_config={
                                "type": "line",
                                "data": {
                                    "x": df[date_column].dt.strftime('%Y-%m-%d').tolist()[-7:],
                                    "y": df[value_column].tolist()[-7:],
                                    "trend": df['rolling_7day'].dropna().tolist()[-7:]
                                }
                            },
                            dealer_id=dealer_id
                        ))
            
            # 2. Linear Regression for Trajectory
            # Convert dates to numeric values for regression
            df['date_numeric'] = (df[date_column] - df[date_column].min()).dt.total_seconds()
            
            # Prepare X and y for regression
            X = sm.add_constant(df['date_numeric'])
            y = df[value_column]
            
            # Fit regression model
            model = sm.OLS(y, X).fit()
            
            # Get slope and its significance
            slope = model.params[1]
            slope_pvalue = model.pvalues[1]
            
            # Calculate average value for context
            avg_value = df[value_column].mean()
            
            # Normalize slope to percentage per day
            seconds_per_day = 86400
            daily_pct_change = (slope * seconds_per_day / avg_value) * 100
            
            # Determine direction based on slope
            if abs(daily_pct_change) < 0.5:  # Less than 0.5% change per day is considered stable
                direction = TrendDirection.STABLE
            elif daily_pct_change > 0:
                direction = TrendDirection.UP
            else:
                direction = TrendDirection.DOWN
            
            # Calculate confidence based on p-value and R-squared
            confidence = (1 - slope_pvalue) * 0.8 + (model.rsquared * 0.2)
            
            if confidence >= self.trend_confidence_threshold:
                # Calculate total percentage change over trend period
                total_pct_change = daily_pct_change * self.trend_period_days
                
                insights.append(InsightResult(
                    insight_type=InsightType.TREND,
                    scope=scope,
                    subject=subject,
                    summary=f"{subject} shows a {direction.value}ward trend of {abs(daily_pct_change):.2f}% per day ({abs(total_pct_change):.1f}% over {self.trend_period_days} days).",
                    confidence=confidence,
                    target_roles=self.role_mapping[scope],
                    tags=["regression", "trajectory", scope.value, direction.value],
                    trend_direction=direction,
                    magnitude=abs(total_pct_change),
                    chart_config={
                        "type": "line",
                        "data": {
                            "x": df[date_column].dt.strftime('%Y-%m-%d').tolist(),
                            "y": df[value_column].tolist(),
                            "trend": model.predict(X).tolist()
                        }
                    },
                    metadata={
                        "r_squared": model.rsquared,
                        "p_value": float(slope_pvalue),
                        "daily_change": float(daily_pct_change)
                    },
                    dealer_id=dealer_id
                ))
            
            # 3. Week-over-Week Analysis (if we have at least 2 weeks of data)
            if (df[date_column].max() - df[date_column].min()).days >= 14:
                # Get last 7 days data
                last_week_mask = df[date_column] >= (df[date_column].max() - timedelta(days=7))
                last_week_df = df[last_week_mask]
                last_week_avg = last_week_df[value_column].mean()
                
                # Get previous 7 days data
                prev_week_mask = (
                    (df[date_column] < (df[date_column].max() - timedelta(days=7))) & 
                    (df[date_column] >= (df[date_column].max() - timedelta(days=14)))
                )
                prev_week_df = df[prev_week_mask]
                prev_week_avg = prev_week_df[value_column].mean()
                
                # Calculate WoW change
                if prev_week_avg != 0:
                    wow_pct_change = ((last_week_avg - prev_week_avg) / prev_week_avg) * 100
                    
                    # Determine direction
                    if abs(wow_pct_change) < 5:  # Less than 5% change is considered stable
                        direction = TrendDirection.STABLE
                    elif wow_pct_change > 0:
                        direction = TrendDirection.UP
                    else:
                        direction = TrendDirection.DOWN
                    
                    # Higher confidence for larger changes and more data points
                    confidence = min(0.95, 0.6 + (abs(wow_pct_change) / 100) + (len(last_week_df) / 20))
                    
                    insights.append(InsightResult(
                        insight_type=InsightType.TREND,
                        scope=scope,
                        subject=subject,
                        summary=f"{subject} is {direction.value} {abs(wow_pct_change):.1f}% compared to previous week.",
                        confidence=confidence,
                        target_roles=self.role_mapping[scope],
                        tags=["week_over_week", "wow", scope.value, direction.value],
                        trend_direction=direction,
                        magnitude=abs(wow_pct_change),
                        chart_config={
                            "type": "bar",
                            "data": {
                                "labels": ["Previous Week", "Last Week"],
                                "values": [float(prev_week_avg), float(last_week_avg)]
                            }
                        },
                        dealer_id=dealer_id
                    ))
            
            return insights
            
        except Exception as e:
            logger.error(
                f"Error in trend detection: {str(e)}",
                extra={
                    "component": "insight_predictor",
                    "action": "detect_trends",
                    "error": str(e),
                    "subject": subject
                }
            )
            return []
    
    def detect_anomalies(self, df: pd.DataFrame, value_column: str, 
                        scope: InsightScope, subject: str) -> List[InsightResult]:
        """
        Detect anomalies in the data using z-scores and IQR methods.
        
        Args:
            df: DataFrame with data
            value_column: Name of the value column to analyze
            scope: Scope of the insight (e.g., INVENTORY, SALES)
            subject: Subject of the anomaly (e.g., "Used Cars Sales")
            
        Returns:
            List of InsightResult objects
        """
        try:
            # Ensure we have enough data
            if len(df) < self.min_data_points:
                return []
            
            # Check for dealer_id column
            dealer_id = None
            if 'dealer_id' in df.columns and len(df['dealer_id'].unique()) == 1:
                dealer_id = df['dealer_id'].iloc[0]
            
            insights = []
            
            # 1. Z-Score Method
            mean = df[value_column].mean()
            std = df[value_column].std()
            
            if std > 0:  # Avoid division by zero
                df['z_score'] = (df[value_column] - mean) / std
                
                # Identify anomalies
                high_anomalies = df[df['z_score'] > self.anomaly_z_threshold]
                low_anomalies = df[df['z_score'] < -self.anomaly_z_threshold]
                
                # Process high anomalies
                if not high_anomalies.empty:
                    # Get the most extreme anomaly
                    extreme = high_anomalies.loc[high_anomalies['z_score'].idxmax()]
                    z_score = extreme['z_score']
                    
                    # Calculate confidence based on how extreme the z-score is
                    confidence = min(0.99, 0.7 + (z_score - self.anomaly_z_threshold) / 10)
                    
                    # Create insight for high anomaly
                    date_str = ""
                    if 'date' in extreme:
                        date_str = f" on {extreme['date'].strftime('%Y-%m-%d')}"
                    
                    insights.append(InsightResult(
                        insight_type=InsightType.ANOMALY,
                        scope=scope,
                        subject=subject,
                        summary=f"Unusually high {subject} detected{date_str} ({extreme[value_column]:.1f}, {z_score:.1f} standard deviations above average).",
                        confidence=confidence,
                        target_roles=self.role_mapping[scope],
                        tags=["anomaly", "high", "z_score", scope.value],
                        trend_direction=TrendDirection.UP,
                        magnitude=float(z_score),
                        chart_config={
                            "type": "scatter",
                            "data": {
                                "x": list(range(len(df))),
                                "y": df[value_column].tolist(),
                                "anomalies": high_anomalies.index.tolist()
                            }
                        },
                        metadata={
                            "method": "z_score",
                            "threshold": self.anomaly_z_threshold,
                            "mean": float(mean),
                            "std": float(std)
                        },
                        dealer_id=dealer_id
                    ))
                
                # Process low anomalies
                if not low_anomalies.empty:
                    # Get the most extreme anomaly
                    extreme = low_anomalies.loc[low_anomalies['z_score'].idxmin()]
                    z_score = abs(extreme['z_score'])
                    
                    # Calculate confidence based on how extreme the z-score is
                    confidence = min(0.99, 0.7 + (z_score - self.anomaly_z_threshold) / 10)
                    
                    # Create insight for low anomaly
                    date_str = ""
                    if 'date' in extreme:
                        date_str = f" on {extreme['date'].strftime('%Y-%m-%d')}"
                    
                    insights.append(InsightResult(
                        insight_type=InsightType.ANOMALY,
                        scope=scope,
                        subject=subject,
                        summary=f"Unusually low {subject} detected{date_str} ({extreme[value_column]:.1f}, {z_score:.1f} standard deviations below average).",
                        confidence=confidence,
                        target_roles=self.role_mapping[scope],
                        tags=["anomaly", "low", "z_score", scope.value],
                        trend_direction=TrendDirection.DOWN,
                        magnitude=float(z_score),
                        chart_config={
                            "type": "scatter",
                            "data": {
                                "x": list(range(len(df))),
                                "y": df[value_column].tolist(),
                                "anomalies": low_anomalies.index.tolist()
                            }
                        },
                        metadata={
                            "method": "z_score",
                            "threshold": self.anomaly_z_threshold,
                            "mean": float(mean),
                            "std": float(std)
                        },
                        dealer_id=dealer_id
                    ))
            
            # 2. IQR Method
            Q1 = df[value_column].quantile(0.25)
            Q3 = df[value_column].quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR > 0:  # Avoid division by zero
                # Identify anomalies
                high_anomalies = df[df[value_column] > (Q3 + self.anomaly_iqr_multiplier * IQR)]
                low_anomalies = df[df[value_column] < (Q1 - self.anomaly_iqr_multiplier * IQR)]
                
                # Process high anomalies (IQR method)
                if not high_anomalies.empty:
                    # Get the most extreme anomaly
                    extreme = high_anomalies.loc[high_anomalies[value_column].idxmax()]
                    
                    # Calculate how many IQRs away from Q3
                    iqrs_away = (extreme[value_column] - Q3) / IQR
                    
                    # Calculate confidence based on how extreme the value is
                    confidence = min(0.95, 0.7 + (iqrs_away - self.anomaly_iqr_multiplier) / 5)
                    
                    # Create insight for high anomaly (IQR method)
                    date_str = ""
                    if 'date' in extreme:
                        date_str = f" on {extreme['date'].strftime('%Y-%m-%d')}"
                    
                    insights.append(InsightResult(
                        insight_type=InsightType.ANOMALY,
                        scope=scope,
                        subject=subject,
                        summary=f"Outlier detected in {subject}{date_str} ({extreme[value_column]:.1f}, {iqrs_away:.1f}x IQR above normal range).",
                        confidence=confidence,
                        target_roles=self.role_mapping[scope],
                        tags=["anomaly", "high", "iqr", scope.value],
                        trend_direction=TrendDirection.UP,
                        magnitude=float(iqrs_away),
                        chart_config={
                            "type": "boxplot",
                            "data": {
                                "values": df[value_column].tolist(),
                                "outliers": high_anomalies[value_column].tolist()
                            }
                        },
                        metadata={
                            "method": "iqr",
                            "threshold": self.anomaly_iqr_multiplier,
                            "q1": float(Q1),
                            "q3": float(Q3),
                            "iqr": float(IQR)
                        },
                        dealer_id=dealer_id
                    ))
                
                # Process low anomalies (IQR method)
                if not low_anomalies.empty:
                    # Get the most extreme anomaly
                    extreme = low_anomalies.loc[low_anomalies[value_column].idxmin()]
                    
                    # Calculate how many IQRs away from Q1
                    iqrs_away = (Q1 - extreme[value_column]) / IQR
                    
                    # Calculate confidence based on how extreme the value is
                    confidence = min(0.95, 0.7 + (iqrs_away - self.anomaly_iqr_multiplier) / 5)
                    
                    # Create insight for low anomaly (IQR method)
                    date_str = ""
                    if 'date' in extreme:
                        date_str = f" on {extreme['date'].strftime('%Y-%m-%d')}"
                    
                    insights.append(InsightResult(
                        insight_type=InsightType.ANOMALY,
                        scope=scope,
                        subject=subject,
                        summary=f"Outlier detected in {subject}{date_str} ({extreme[value_column]:.1f}, {iqrs_away:.1f}x IQR below normal range).",
                        confidence=confidence,
                        target_roles=self.role_mapping[scope],
                        tags=["anomaly", "low", "iqr", scope.value],
                        trend_direction=TrendDirection.DOWN,
                        magnitude=float(iqrs_away),
                        chart_config={
                            "type": "boxplot",
                            "data": {
                                "values": df[value_column].tolist(),
                                "outliers": low_anomalies[value_column].tolist()
                            }
                        },
                        metadata={
                            "method": "iqr",
                            "threshold": self.anomaly_iqr_multiplier,
                            "q1": float(Q1),
                            "q3": float(Q3),
                            "iqr": float(IQR)
                        },
                        dealer_id=dealer_id
                    ))
            
            return insights
            
        except Exception as e:
            logger.error(
                f"Error in anomaly detection: {str(e)}",
                extra={
                    "component": "insight_predictor",
                    "action": "detect_anomalies",
                    "error": str(e),
                    "subject": subject
                }
            )
            return []
    
    def detect_events(self, df: pd.DataFrame, date_column: str, value_column: str, 
                     scope: InsightScope, subject: str, event_config: Dict[str, Any]) -> List[InsightResult]:
        """
        Detect notable events in the data based on thresholds.
        
        Args:
            df: DataFrame with data
            date_column: Name of the date column
            value_column: Name of the value column to analyze
            scope: Scope of the insight (e.g., INVENTORY, SALES)
            subject: Subject of the event (e.g., "Used Cars Sales")
            event_config: Configuration for event detection
            
        Returns:
            List of InsightResult objects
        """
        try:
            if len(df) == 0:
                return []
            
            # Check for dealer_id column
            dealer_id = None
            if 'dealer_id' in df.columns and len(df['dealer_id'].unique()) == 1:
                dealer_id = df['dealer_id'].iloc[0]
            
            insights = []
            
            # Ensure date column is datetime
            df[date_column] = pd.to_datetime(df[date_column])
            
            # Sort by date
            df = df.sort_values(date_column)
            
            # 1. Zero Value Days
            threshold = event_config.get('zero_threshold', 0)
            zero_days = df[df[value_column] <= threshold]
            
            if not zero_days.empty:
                # Get the most recent zero day
                recent_zero = zero_days.iloc[-1]
                
                # Check if it's recent (within the last 7 days)
                is_recent = (df[date_column].max() - recent_zero[date_column]).days <= 7
                
                if is_recent:
                    insights.append(InsightResult(
                        insight_type=InsightType.EVENT,
                        scope=scope,
                        subject=subject,
                        summary=f"Zero activity detected for {subject} on {recent_zero[date_column].strftime('%Y-%m-%d')}.",
                        confidence=0.95,  # High confidence for zero detection
                        target_roles=self.role_mapping[scope],
                        tags=["event", "zero_value", scope.value],
                        trend_direction=TrendDirection.DOWN,
                        magnitude=0.0,
                        chart_config={
                            "type": "bar",
                            "data": {
                                "x": df[date_column].dt.strftime('%Y-%m-%d').tolist()[-14:],  # Last 14 days
                                "y": df[value_column].tolist()[-14:],
                                "zero_days": zero_days[date_column].dt.strftime('%Y-%m-%d').tolist()
                            }
                        },
                        metadata={
                            "threshold": threshold,
                            "zero_days_count": len(zero_days),
                            "recent_date": recent_zero[date_column].isoformat()
                        },
                        dealer_id=dealer_id
                    ))
            
            # 2. Below Threshold Events
            min_threshold = event_config.get('min_threshold')
            if min_threshold is not None:
                below_threshold = df[df[value_column] < min_threshold]
                
                if not below_threshold.empty:
                    # Get the most recent below-threshold day
                    recent_below = below_threshold.iloc[-1]
                    
                    # Check if it's recent (within the last 7 days)
                    is_recent = (df[date_column].max() - recent_below[date_column]).days <= 7
                    
                    if is_recent:
                        insights.append(InsightResult(
                            insight_type=InsightType.EVENT,
                            scope=scope,
                            subject=subject,
                            summary=f"{subject} fell below minimum threshold ({min_threshold}) on {recent_below[date_column].strftime('%Y-%m-%d')} with value {recent_below[value_column]:.1f}.",
                            confidence=0.9,
                            target_roles=self.role_mapping[scope],
                            tags=["event", "below_threshold", scope.value],
                            trend_direction=TrendDirection.DOWN,
                            magnitude=float(min_threshold - recent_below[value_column]),
                            chart_config={
                                "type": "line",
                                "data": {
                                    "x": df[date_column].dt.strftime('%Y-%m-%d').tolist()[-14:],  # Last 14 days
                                    "y": df[value_column].tolist()[-14:],
                                    "threshold": [min_threshold] * 14,
                                    "below_days": below_threshold[date_column].dt.strftime('%Y-%m-%d').tolist()
                                }
                            },
                            metadata={
                                "threshold_type": "minimum",
                                "threshold_value": min_threshold,
                                "below_days_count": len(below_threshold),
                                "recent_date": recent_below[date_column].isoformat()
                            },
                            dealer_id=dealer_id
                        ))
            
            # 3. Above Threshold Events
            max_threshold = event_config.get('max_threshold')
            if max_threshold is not None:
                above_threshold = df[df[value_column] > max_threshold]
                
                if not above_threshold.empty:
                    # Get the most recent above-threshold day
                    recent_above = above_threshold.iloc[-1]
                    
                    # Check if it's recent (within the last 7 days)
                    is_recent = (df[date_column].max() - recent_above[date_column]).days <= 7
                    
                    if is_recent:
                        insights.append(InsightResult(
                            insight_type=InsightType.EVENT,
                            scope=scope,
                            subject=subject,
                            summary=f"{subject} exceeded maximum threshold ({max_threshold}) on {recent_above[date_column].strftime('%Y-%m-%d')} with value {recent_above[value_column]:.1f}.",
                            confidence=0.9,
                            target_roles=self.role_mapping[scope],
                            tags=["event", "above_threshold", scope.value],
                            trend_direction=TrendDirection.UP,
                            magnitude=float(recent_above[value_column] - max_threshold),
                            chart_config={
                                "type": "line",
                                "data": {
                                    "x": df[date_column].dt.strftime('%Y-%m-%d').tolist()[-14:],  # Last 14 days
                                    "y": df[value_column].tolist()[-14:],
                                    "threshold": [max_threshold] * 14,
                                    "above_days": above_threshold[date_column].dt.strftime('%Y-%m-%d').tolist()
                                }
                            },
                            metadata={
                                "threshold_type": "maximum",
                                "threshold_value": max_threshold,
                                "above_days_count": len(above_threshold),
                                "recent_date": recent_above[date_column].isoformat()
                            },
                            dealer_id=dealer_id
                        ))
            
            return insights
            
        except Exception as e:
            logger.error(
                f"Error in event detection: {str(e)}",
                extra={
                    "component": "insight_predictor",
                    "action": "detect_events",
                    "error": str(e),
                    "subject": subject
                }
            )
            return []
    
    def analyze_data(self, df: pd.DataFrame, config: Dict[str, Any]) -> List[InsightResult]:
        """
        Analyze data and generate insights using all detection methods.
        
        Args:
            df: DataFrame with data
            config: Configuration for analysis
            
        Returns:
            List of InsightResult objects
        """
        scope = InsightScope(config.get("scope", InsightScope.SALES.value))
        subject = config.get("subject", "Sales")
        date_column = config.get("date_column", "date")
        value_column = config.get("value_column", "value")
        
        all_insights = []
        
        # Run trend detection
        if config.get("detect_trends", True):
            trend_insights = self.detect_trends(df, date_column, value_column, scope, subject)
            all_insights.extend(trend_insights)
            
            logger.info(
                f"Generated {len(trend_insights)} trend insights",
                extra={
                    "component": "insight_predictor",
                    "action": "analyze_data",
                    "insight_type": "trend",
                    "count": len(trend_insights),
                    "subject": subject
                }
            )
        
        # Run anomaly detection
        if config.get("detect_anomalies", True):
            anomaly_insights = self.detect_anomalies(df, value_column, scope, subject)
            all_insights.extend(anomaly_insights)
            
            logger.info(
                f"Generated {len(anomaly_insights)} anomaly insights",
                extra={
                    "component": "insight_predictor",
                    "action": "analyze_data",
                    "insight_type": "anomaly",
                    "count": len(anomaly_insights),
                    "subject": subject
                }
            )
        
        # Run event detection
        if config.get("detect_events", True):
            event_config = config.get("event_config", {})
            event_insights = self.detect_events(df, date_column, value_column, scope, subject, event_config)
            all_insights.extend(event_insights)
            
            logger.info(
                f"Generated {len(event_insights)} event insights",
                extra={
                    "component": "insight_predictor",
                    "action": "analyze_data",
                    "insight_type": "event",
                    "count": len(event_insights),
                    "subject": subject
                }
            )
        
        # Sort insights by confidence (highest first)
        all_insights.sort(key=lambda x: x.confidence, reverse=True)
        
        return all_insights

# Example usage
if __name__ == "__main__":
    # Sample data
    dates = pd.date_range(start='2023-01-01', periods=30)
    values = [100, 105, 103, 110, 112, 108, 115, 120, 125, 130,
              132, 135, 140, 145, 150, 148, 155, 2, 165, 170,
              175, 180, 185, 190, 195, 200, 205, 210, 215, 220]
    
    df = pd.DataFrame({
        'date': dates,
        'sales': values
    })
    
    # Create predictor
    predictor = InsightPredictor()
    
    # Analyze data
    insights = predictor.analyze_data(df, {
        "scope": InsightScope.SALES.value,
        "subject": "Vehicle Sales",
        "date_column": "date",
        "value_column": "sales",
        "event_config": {
            "min_threshold": 100,
            "max_threshold": 200
        }
    })
    
    # Print insights
    for insight in insights:
        print("\n" + "="*80)
        print(f"Type: {insight.insight_type.value}, Direction: {insight.trend_direction.value if insight.trend_direction else 'N/A'}")
        print(f"Subject: {insight.subject}")
        print(f"Summary: {insight.summary}")
        print(f"Confidence: {insight.confidence:.2f}")
        print(f"Target roles: {[role.value for role in insight.target_roles]}")
        print(f"Tags: {insight.tags}")