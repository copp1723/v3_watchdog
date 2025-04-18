"""
Feedback-driven prompt tuning system.

Automatically adjusts LLM prompts based on user feedback to improve
insight relevance and quality.
"""

import re
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime, timedelta
import sentry_sdk
from dataclasses import dataclass
from .models import FeedbackEntry

# Configure logger
logger = logging.getLogger(__name__)

@dataclass
class TuningConfig:
    """Configuration for prompt tuning."""
    min_feedback_count: int = 10  # Minimum feedback needed for tuning
    feedback_window_days: int = 30  # How far back to consider feedback
    max_emphasis_boost: float = 2.0  # Maximum multiplier for emphasis
    min_confidence: float = 0.7  # Minimum confidence threshold

class PromptTuner:
    """
    Tunes prompts based on user feedback.
    
    Uses feedback history to adjust prompt templates for improved
    insight relevance and quality.
    """
    
    def __init__(self, config: Optional[TuningConfig] = None):
        """
        Initialize the prompt tuner.
        
        Args:
            config: Optional configuration for tuning behavior
        """
        self.config = config or TuningConfig()
        self.last_tune = None
        self.feedback_history: List[FeedbackEntry] = []
        
        # Metric importance weights (adjusted by feedback)
        self.metric_weights = {
            'gross_profit': 1.0,
            'deal_count': 1.0,
            'days_on_lot': 1.0,
            'lead_source': 1.0,
            'sales_rep': 1.0
        }
        
        # Format preferences (adjusted by feedback)
        self.format_weights = {
            'bullet_points': 1.0,
            'paragraphs': 1.0,
            'table': 1.0,
            'chart': 1.0
        }
    
    def tune_prompt(self, template: str, feedback: List[FeedbackEntry]) -> str:
        """
        Tune a prompt template based on feedback.
        
        Args:
            template: Original prompt template
            feedback: List of feedback entries
            
        Returns:
            Tuned prompt template
        """
        try:
            # Track tuning in Sentry
            sentry_sdk.set_tag("prompt_tuner", "active")
            
            # Filter recent feedback
            recent_feedback = self._filter_recent_feedback(feedback)
            
            if len(recent_feedback) < self.config.min_feedback_count:
                logger.info(
                    f"Insufficient feedback ({len(recent_feedback)} < "
                    f"{self.config.min_feedback_count}) for tuning"
                )
                sentry_sdk.set_tag("prompt_tuned", False)
                return template
            
            # Update weights based on feedback
            self._update_weights(recent_feedback)
            
            # Apply tuning strategies
            tuned_template = template
            tuned_template = self._adjust_emphasis(tuned_template)
            tuned_template = self._enhance_context(tuned_template)
            tuned_template = self._optimize_format(tuned_template)
            tuned_template = self._calibrate_confidence(tuned_template)
            
            self.last_tune = datetime.now()
            self.feedback_history = recent_feedback
            
            # Track successful tuning
            sentry_sdk.set_tag("prompt_tuned", True)
            sentry_sdk.set_tag("feedback_count", len(recent_feedback))
            
            return tuned_template
            
        except Exception as e:
            logger.error(f"Error tuning prompt: {str(e)}")
            sentry_sdk.capture_exception(e)
            return template
    
    def _filter_recent_feedback(self, feedback: List[FeedbackEntry]) -> List[FeedbackEntry]:
        """Filter feedback entries to recent window."""
        cutoff = datetime.now() - timedelta(days=self.config.feedback_window_days)
        return [
            entry for entry in feedback
            if entry.timestamp >= cutoff
        ]
    
    def _update_weights(self, feedback: List[FeedbackEntry]) -> None:
        """Update metric and format weights based on feedback."""
        try:
            # Reset weights
            for key in self.metric_weights:
                self.metric_weights[key] = 1.0
            for key in self.format_weights:
                self.format_weights[key] = 1.0
            
            # Analyze feedback
            for entry in feedback:
                # Skip entries without metadata
                if not entry.metadata:
                    continue
                
                # Update metric weights
                if entry.feedback_type == "helpful":
                    # Boost metrics mentioned in helpful insights
                    for metric in entry.metadata.get("metrics_used", []):
                        if metric in self.metric_weights:
                            self.metric_weights[metric] *= 1.1
                
                elif entry.feedback_type == "not_helpful":
                    # Reduce weight of problematic metrics
                    for metric in entry.metadata.get("metrics_used", []):
                        if metric in self.metric_weights:
                            self.metric_weights[metric] *= 0.9
                
                # Update format weights
                format_used = entry.metadata.get("format_used")
                if format_used in self.format_weights:
                    if entry.feedback_type == "helpful":
                        self.format_weights[format_used] *= 1.1
                    elif entry.feedback_type == "not_helpful":
                        self.format_weights[format_used] *= 0.9
            
            # Normalize weights
            self._normalize_weights(self.metric_weights)
            self._normalize_weights(self.format_weights)
            
        except Exception as e:
            logger.error(f"Error updating weights: {str(e)}")
            sentry_sdk.capture_exception(e)
    
    def _normalize_weights(self, weights: Dict[str, float]) -> None:
        """Normalize weights to prevent extreme values."""
        # Get max weight
        max_weight = max(weights.values())
        
        # Scale all weights
        if max_weight > self.config.max_emphasis_boost:
            scale = self.config.max_emphasis_boost / max_weight
            for key in weights:
                weights[key] *= scale
    
    def _adjust_emphasis(self, template: str) -> str:
        """Adjust metric emphasis based on weights."""
        try:
            # Add emphasis to important metrics
            for metric, weight in self.metric_weights.items():
                if weight > 1.1:  # Only emphasize significantly boosted metrics
                    # Add importance marker to metric references
                    template = re.sub(
                        f"({metric})",
                        r"**\1**",  # Bold important metrics
                        template,
                        flags=re.IGNORECASE
                    )
            
            return template
            
        except Exception as e:
            logger.error(f"Error adjusting emphasis: {str(e)}")
            return template
    
    def _enhance_context(self, template: str) -> str:
        """Enhance prompt context based on feedback."""
        try:
            # Add context about preferred metrics
            important_metrics = [
                metric for metric, weight in self.metric_weights.items()
                if weight > 1.1
            ]
            
            if important_metrics:
                context = (
                    "\nFocus on these key metrics: " +
                    ", ".join(important_metrics)
                )
                template += context
            
            return template
            
        except Exception as e:
            logger.error(f"Error enhancing context: {str(e)}")
            return template
    
    def _optimize_format(self, template: str) -> str:
        """Optimize output format based on preferences."""
        try:
            # Determine preferred format
            preferred_format = max(
                self.format_weights.items(),
                key=lambda x: x[1]
            )[0]
            
            # Add format preference
            if preferred_format == "bullet_points":
                template = template.replace(
                    "Respond in JSON format",
                    "Respond in JSON format. Use bullet points for lists."
                )
            elif preferred_format == "paragraphs":
                template = template.replace(
                    "Respond in JSON format",
                    "Respond in JSON format. Use clear paragraphs for explanations."
                )
            
            return template
            
        except Exception as e:
            logger.error(f"Error optimizing format: {str(e)}")
            return template
    
    def _calibrate_confidence(self, template: str) -> str:
        """Calibrate confidence requirements based on feedback."""
        try:
            # Calculate average confidence from helpful insights
            helpful_feedback = [
                f for f in self.feedback_history
                if f.feedback_type == "helpful"
                and f.metadata
                and "confidence" in f.metadata
            ]
            
            if helpful_feedback:
                avg_confidence = sum(
                    float(f.metadata["confidence"])
                    for f in helpful_feedback
                ) / len(helpful_feedback)
                
                # Add confidence threshold if high confidence correlates with helpfulness
                if avg_confidence > self.config.min_confidence:
                    template += f"\nMaintain confidence level above {avg_confidence:.1f}."
            
            return template
            
        except Exception as e:
            logger.error(f"Error calibrating confidence: {str(e)}")
            return template