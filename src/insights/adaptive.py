"""
Adaptive Threshold Learning for Watchdog AI.

This module provides classes for learning and adapting insight thresholds
based on user feedback and historical data.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
import sentry_sdk
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .models import FeedbackEntry

# Configure logger
logger = logging.getLogger(__name__)

@dataclass
class ThresholdConfig:
    """Configuration for threshold learning."""
    min_feedback_count: int = 10  # Minimum feedback entries needed for adaptation
    learning_rate: float = 0.1  # How quickly thresholds adapt to new feedback
    feedback_window_days: int = 90  # How far back to consider feedback
    min_threshold: float = 0.0  # Minimum allowed threshold value
    max_threshold: float = float('inf')  # Maximum allowed threshold value

class ThresholdLearner(ABC):
    """
    Base class for adaptive threshold learning.
    
    Learns from user feedback to dynamically adjust thresholds
    used in insight generation.
    """
    
    def __init__(self, config: Optional[ThresholdConfig] = None):
        """
        Initialize the threshold learner.
        
        Args:
            config: Optional configuration for learning behavior
        """
        self.config = config or ThresholdConfig()
        self.last_update = None
        self.current_threshold = None
        self.feedback_history: List[FeedbackEntry] = []
    
    def fit(self, historical_df: pd.DataFrame, feedback_entries: List[FeedbackEntry]) -> None:
        """
        Fit the learner to historical data and feedback.
        
        Args:
            historical_df: Historical DataFrame to learn from
            feedback_entries: List of feedback entries to consider
        """
        try:
            # Track fitting in Sentry
            sentry_sdk.set_tag("threshold_learner", self.__class__.__name__)
            sentry_sdk.set_tag("feedback_count", len(feedback_entries))
            
            # Filter recent feedback
            recent_feedback = self._filter_recent_feedback(feedback_entries)
            
            if len(recent_feedback) < self.config.min_feedback_count:
                logger.info(
                    f"Insufficient feedback ({len(recent_feedback)} < "
                    f"{self.config.min_feedback_count}) for threshold adaptation"
                )
                # Initialize with default threshold
                self.current_threshold = self._calculate_default_threshold(historical_df)
                return
            
            # Calculate feedback-weighted threshold
            self.current_threshold = self._calculate_adaptive_threshold(
                historical_df, recent_feedback
            )
            
            # Apply bounds
            self.current_threshold = max(
                min(self.current_threshold, self.config.max_threshold),
                self.config.min_threshold
            )
            
            self.last_update = datetime.now()
            self.feedback_history = recent_feedback
            
            logger.info(f"Updated threshold to {self.current_threshold:.2f}")
            
        except Exception as e:
            logger.error(f"Error fitting threshold learner: {str(e)}")
            sentry_sdk.capture_exception(e)
            # Fallback to default threshold
            self.current_threshold = self._calculate_default_threshold(historical_df)
    
    def predict_threshold(self, current_df: pd.DataFrame) -> float:
        """
        Predict the appropriate threshold for new data.
        
        Args:
            current_df: Current DataFrame to analyze
            
        Returns:
            Predicted threshold value
        """
        if self.current_threshold is None:
            # No learned threshold yet, calculate default
            return self._calculate_default_threshold(current_df)
        
        # Apply any final adjustments based on current data
        return self._adjust_threshold_for_current_data(
            self.current_threshold, current_df
        )
    
    def _filter_recent_feedback(self, feedback_entries: List[FeedbackEntry]) -> List[FeedbackEntry]:
        """Filter feedback entries to recent window."""
        cutoff = datetime.now() - timedelta(days=self.config.feedback_window_days)
        return [
            entry for entry in feedback_entries
            if entry.timestamp >= cutoff
        ]
    
    def _calculate_adaptive_threshold(
        self, 
        historical_df: pd.DataFrame,
        feedback_entries: List[FeedbackEntry]
    ) -> float:
        """
        Calculate threshold based on feedback and historical data.
        
        Args:
            historical_df: Historical DataFrame
            feedback_entries: List of feedback entries
            
        Returns:
            Adapted threshold value
        """
        # Get current default threshold
        base_threshold = self._calculate_default_threshold(historical_df)
        
        # Calculate feedback-based adjustment
        adjustment = self._calculate_feedback_adjustment(feedback_entries)
        
        # Apply learning rate to adjustment
        return base_threshold + (adjustment * self.config.learning_rate)
    
    def _calculate_feedback_adjustment(self, feedback_entries: List[FeedbackEntry]) -> float:
        """
        Calculate threshold adjustment based on feedback.
        
        Args:
            feedback_entries: List of feedback entries
            
        Returns:
            Adjustment value for threshold
        """
        # Convert feedback types to numeric scores
        scores = []
        for entry in feedback_entries:
            if entry.feedback_type == "helpful":
                scores.append(0.0)  # No adjustment needed
            elif entry.feedback_type == "not_helpful":
                if "too_many_alerts" in (entry.metadata or {}):
                    scores.append(1.0)  # Increase threshold
                elif "missed_alerts" in (entry.metadata or {}):
                    scores.append(-1.0)  # Decrease threshold
                else:
                    scores.append(0.0)  # No clear direction
        
        if not scores:
            return 0.0
        
        # Weight recent feedback more heavily
        weights = np.linspace(0.5, 1.0, len(scores))
        weighted_avg = np.average(scores, weights=weights)
        
        return weighted_avg
    
    def _adjust_threshold_for_current_data(
        self,
        base_threshold: float,
        current_df: pd.DataFrame
    ) -> float:
        """
        Make final adjustments to threshold based on current data.
        
        Args:
            base_threshold: Base threshold value
            current_df: Current DataFrame
            
        Returns:
            Adjusted threshold value
        """
        # By default, return base threshold
        # Subclasses can override for specific adjustments
        return base_threshold
    
    @abstractmethod
    def _calculate_default_threshold(self, df: pd.DataFrame) -> float:
        """
        Calculate default threshold from data.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Default threshold value
        """
        pass


class InventoryAgingLearner(ThresholdLearner):
    """Learns thresholds for inventory aging anomalies."""
    
    def __init__(self, config: Optional[ThresholdConfig] = None):
        """Initialize inventory aging learner."""
        if config is None:
            config = ThresholdConfig(
                min_threshold=15.0,  # Minimum 15 days over average
                max_threshold=120.0,  # Maximum 120 days over average
                learning_rate=0.15  # Slightly more aggressive learning
            )
        super().__init__(config)
    
    def _calculate_default_threshold(self, df: pd.DataFrame) -> float:
        """Calculate default aging threshold."""
        try:
            # Find days on lot column
            days_col = next(
                col for col in df.columns 
                if any(term in col.lower() for term in ['days', 'age', 'aging'])
            )
            
            # Calculate threshold as 1.5 standard deviations above mean
            days = pd.to_numeric(df[days_col], errors='coerce')
            mean_days = days.mean()
            std_days = days.std()
            
            if pd.isna(mean_days) or pd.isna(std_days):
                return 30.0  # Default fallback
            
            return mean_days + (1.5 * std_days)
            
        except Exception as e:
            logger.warning(f"Error calculating default aging threshold: {str(e)}")
            return 30.0  # Conservative default
    
    def _adjust_threshold_for_current_data(
        self,
        base_threshold: float,
        current_df: pd.DataFrame
    ) -> float:
        """Adjust threshold based on current inventory mix."""
        try:
            # Find model/type column
            model_col = next(
                col for col in current_df.columns
                if any(term in col.lower() for term in ['model', 'type', 'category'])
            )
            
            # Calculate average days by model
            days_col = next(
                col for col in current_df.columns 
                if any(term in col.lower() for term in ['days', 'age', 'aging'])
            )
            
            model_days = current_df.groupby(model_col)[days_col].mean()
            
            # If high variance between models, increase threshold
            if model_days.std() > model_days.mean() * 0.5:
                return base_threshold * 1.2
            
            return base_threshold
            
        except Exception as e:
            logger.warning(f"Error adjusting threshold for current data: {str(e)}")
            return base_threshold


class GrossMarginLearner(ThresholdLearner):
    """Learns thresholds for gross margin anomalies."""
    
    def __init__(self, config: Optional[ThresholdConfig] = None):
        """Initialize gross margin learner."""
        if config is None:
            config = ThresholdConfig(
                min_threshold=0.05,  # Minimum 5% margin
                max_threshold=0.40,  # Maximum 40% margin
                learning_rate=0.08  # Conservative learning rate
            )
        super().__init__(config)
    
    def _calculate_default_threshold(self, df: pd.DataFrame) -> float:
        """Calculate default margin threshold."""
        try:
            # Find gross and price/cost columns
            gross_col = next(
                col for col in df.columns
                if any(term in col.lower() for term in ['gross', 'profit', 'margin'])
            )
            
            price_col = next(
                (col for col in df.columns
                if any(term in col.lower() for term in ['price', 'revenue', 'sale'])),
                None
            )
            
            if price_col:
                # Calculate margins
                margins = pd.to_numeric(df[gross_col], errors='coerce') / \
                         pd.to_numeric(df[price_col], errors='coerce')
                
                # Use 25th percentile as threshold
                threshold = margins.quantile(0.25)
                
                if pd.isna(threshold):
                    return 0.15  # Default fallback
                
                return max(threshold, self.config.min_threshold)
            
            return 0.15  # Default fallback
            
        except Exception as e:
            logger.warning(f"Error calculating default margin threshold: {str(e)}")
            return 0.15  # Conservative default
    
    def _adjust_threshold_for_current_data(
        self,
        base_threshold: float,
        current_df: pd.DataFrame
    ) -> float:
        """Adjust threshold based on current sales mix."""
        try:
            # Find model/type column
            model_col = next(
                col for col in current_df.columns
                if any(term in col.lower() for term in ['model', 'type', 'category'])
            )
            
            # Calculate average margin by model
            gross_col = next(
                col for col in current_df.columns
                if any(term in col.lower() for term in ['gross', 'profit', 'margin'])
            )
            
            model_margins = current_df.groupby(model_col)[gross_col].mean()
            
            # If high variance between models, decrease threshold
            if model_margins.std() > model_margins.mean() * 0.4:
                return base_threshold * 0.8
            
            return base_threshold
            
        except Exception as e:
            logger.warning(f"Error adjusting threshold for current data: {str(e)}")
            return base_threshold


class LeadConversionLearner(ThresholdLearner):
    """Learns thresholds for lead conversion anomalies."""
    
    def __init__(self, config: Optional[ThresholdConfig] = None):
        """Initialize lead conversion learner."""
        if config is None:
            config = ThresholdConfig(
                min_threshold=0.08,  # Minimum 8% conversion rate
                max_threshold=0.35,  # Maximum 35% conversion rate
                learning_rate=0.12  # Moderate learning rate
            )
        super().__init__(config)
    
    def _calculate_default_threshold(self, df: pd.DataFrame) -> float:
        """Calculate default conversion threshold."""
        try:
            # Find conversion status column
            status_col = next(
                col for col in df.columns
                if any(term in col.lower() for term in ['status', 'converted', 'sold'])
            )
            
            # Calculate overall conversion rate
            if df[status_col].dtype == bool:
                conversion_rate = df[status_col].mean()
            else:
                # Assume text status, look for success indicators
                success_terms = ['sold', 'converted', 'won', 'success']
                conversion_rate = df[status_col].str.lower().isin(success_terms).mean()
            
            if pd.isna(conversion_rate):
                return 0.15  # Default fallback
            
            # Use 80% of average as threshold
            return max(conversion_rate * 0.8, self.config.min_threshold)
            
        except Exception as e:
            logger.warning(f"Error calculating default conversion threshold: {str(e)}")
            return 0.15  # Conservative default
    
    def _adjust_threshold_for_current_data(
        self,
        base_threshold: float,
        current_df: pd.DataFrame
    ) -> float:
        """Adjust threshold based on current lead mix."""
        try:
            # Find source column
            source_col = next(
                col for col in current_df.columns
                if any(term in col.lower() for term in ['source', 'lead', 'channel'])
            )
            
            # Find conversion column
            status_col = next(
                col for col in current_df.columns
                if any(term in col.lower() for term in ['status', 'converted', 'sold'])
            )
            
            # Calculate conversion by source
            if current_df[status_col].dtype == bool:
                source_conv = current_df.groupby(source_col)[status_col].mean()
            else:
                success_terms = ['sold', 'converted', 'won', 'success']
                source_conv = current_df.groupby(source_col)[status_col].apply(
                    lambda x: x.str.lower().isin(success_terms).mean()
                )
            
            # If high variance between sources, decrease threshold
            if source_conv.std() > source_conv.mean() * 0.5:
                return base_threshold * 0.85
            
            return base_threshold
            
        except Exception as e:
            logger.warning(f"Error adjusting threshold for current data: {str(e)}")
            return base_threshold