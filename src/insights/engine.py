"""
Insight Engine for Watchdog AI.

Orchestrates the full insight generation pipeline from data loading
through statistical analysis and LLM summarization.
"""

import pandas as pd
from datetime import datetime
import logging
import sentry_sdk
from typing import Dict, Any, List, Optional
from ..utils.data_io import load_data, validate_data
from ..utils.errors import InsightGenerationError
from .base_insight import InsightBase
from .insight_functions import (
    MonthlyGrossMarginInsight,
    LeadConversionRateInsight
)
from .summarizer import Summarizer
from .models import FeedbackEntry, FeedbackStats
from .feedback import feedback_manager
from .adaptive import (
    InventoryAgingLearner,
    GrossMarginLearner,
    LeadConversionLearner
)
from .forecast import (
    SalesPerformanceForecaster,
    InventoryTurnoverForecaster,
    MarginTrendForecaster
)

# Configure logger
logger = logging.getLogger(__name__)

class InsightEngine:
    """
    Orchestrates the insight generation pipeline.
    
    Coordinates data loading, insight generation, and LLM summarization
    to produce executive-ready insights from dealership data.
    """
    
    def __init__(self, llm_client: Any):
        """
        Initialize the insight engine.
        
        Args:
            llm_client: LLM client for text generation
        """
        self.llm_client = llm_client
        self.summarizer = Summarizer(llm_client)
        
        # Register available insights
        self.insights: Dict[str, InsightBase] = {
            "monthly_gross_margin": MonthlyGrossMarginInsight(),
            "lead_conversion_rate": LeadConversionRateInsight()
        }
        
        # Track pipeline metadata
        self.rules_version = "1.0.0"
        self.session_id = None
        
        # Add feedback manager
        self.feedback_manager = feedback_manager
        
        # Initialize threshold learners
        self.threshold_learners = {
            "inventory_aging": InventoryAgingLearner(),
            "gross_margin": GrossMarginLearner(),
            "lead_conversion": LeadConversionLearner()
        }
        
        # Initialize forecasters
        self.forecasters = {
            "sales_performance": SalesPerformanceForecaster(),
            "inventory_turnover": InventoryTurnoverForecaster(),
            "margin_trend": MarginTrendForecaster()
        }
    
    def run(self, uploaded_file: Any) -> Dict[str, Any]:
        """
        Run the full insight pipeline on uploaded data.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            Dictionary containing processed data and insights
            
        Raises:
            InsightGenerationError: If pipeline execution fails
        """
        try:
            # Initialize session tracking
            self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            sentry_sdk.set_tag("session_id", self.session_id)
            
            # Step 1: Load & validate data
            sentry_sdk.set_tag("pipeline_step", "load_data")
            logger.info("Loading and validating data...")
            
            df = load_data(uploaded_file)
            df, validation_summary = validate_data(df)
            
            sentry_sdk.set_tag("data_rows", len(df))
            sentry_sdk.set_tag("data_quality_score", validation_summary["quality_score"])
            
            # Step 2: Generate insights
            sentry_sdk.set_tag("pipeline_step", "insight_generation")
            logger.info("Generating insights...")
            
            insights = []
            for insight_type, generator in self.insights.items():
                try:
                    # Update thresholds if we have a learner
                    if insight_type in self.threshold_learners:
                        learner = self.threshold_learners[insight_type]
                        
                        # Get feedback for this insight type
                        feedback = self.feedback_manager.get_feedback(
                            insight_id=insight_type
                        )
                        
                        # Fit learner and get threshold
                        learner.fit(df, [
                            FeedbackEntry.from_dict(entry)
                            for entry in feedback
                        ])
                        
                        # Add threshold to generator kwargs
                        threshold = learner.predict_threshold(df)
                        kwargs = {"threshold": threshold}
                        
                        # Add threshold context for LLM
                        if learner.last_update:
                            kwargs["threshold_context"] = (
                                f"Using learned threshold of {threshold:.2f} "
                                f"based on user feedback"
                            )
                    else:
                        kwargs = {}
                    
                    # Generate raw insight
                    raw_insight = generator.generate(df, **kwargs)
                    
                    if "error" in raw_insight:
                        logger.warning(f"Error generating {insight_type}: {raw_insight['error']}")
                        continue
                    
                    # Step 3: Summarize with LLM
                    sentry_sdk.set_tag("pipeline_step", "summarization")
                    sentry_sdk.set_tag("summarization_type", insight_type)
                    
                    # Get feedback stats for this insight type
                    feedback_stats = self.get_feedback_stats(insight_type)
                    
                    # Format metrics table for LLM
                    metrics_table = self._format_metrics_table(raw_insight)
                    
                    # Add feedback context to prompt if available
                    context = {
                        "entity_name": "Dealership",  # TODO: Make configurable
                        "date_range": self._get_date_range(df),
                        "metrics_table": metrics_table
                    }
                    
                    if feedback_stats and feedback_stats.total_feedback > 0:
                        context["feedback_context"] = (
                            f"Previous feedback on this insight type: "
                            f"{feedback_stats.total_feedback} ratings, "
                            f"{feedback_stats.feedback_percentages.get('helpful', 0):.1f}% helpful"
                        )
                    
                    # Generate summary
                    summary = self.summarizer.summarize(
                        f"{insight_type}_prompt.tpl",
                        **context
                    )
                    
                    # Add summary and feedback stats to insight
                    raw_insight["summary"] = summary
                    if feedback_stats:
                        raw_insight["feedback_stats"] = {
                            "total_feedback": feedback_stats.total_feedback,
                            "helpful_percentage": feedback_stats.feedback_percentages.get("helpful", 0),
                            "average_rating": feedback_stats.average_rating
                        }
                    
                    insights.append(raw_insight)
                    
                    # Generate forecasts if we have a forecaster
                    forecast_results = {}
                    if insight_type in self.forecasters:
                        forecaster = self.forecasters[insight_type]
                        
                        try:
                            # Generate forecast
                            forecast = forecaster.generate(df)
                            
                            # Add forecast context
                            forecast_results = {
                                "forecast": forecast.forecast.to_dict(),
                                "confidence_intervals": forecast.confidence_intervals.to_dict(),
                                "metrics": forecast.metrics
                            }
                            
                            # Add forecast context to kwargs
                            kwargs["forecast_context"] = (
                                f"Based on historical patterns, next period forecast: "
                                f"${forecast.forecast.mean():,.2f} "
                                f"(±${forecast.confidence_intervals.std().mean():,.2f})"
                            )
                        except Exception as forecast_error:
                            logger.warning(
                                f"Error generating forecast for {insight_type}: {str(forecast_error)}"
                            )
                            sentry_sdk.capture_exception(forecast_error)
                    
                    # Add forecast results if available
                    if forecast_results:
                        raw_insight["forecast"] = forecast_results
                    
                except Exception as e:
                    logger.error(f"Error processing {insight_type}: {str(e)}")
                    sentry_sdk.capture_exception(e)
            
            # Step 4: Return final payload
            sentry_sdk.set_tag("pipeline_step", "complete")
            
            return {
                "data": df,
                "insights": insights,
                "metadata": {
                    "session_id": self.session_id,
                    "run_timestamp": datetime.now().isoformat(),
                    "rules_version": self.rules_version,
                    "validation_summary": validation_summary
                }
            }
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            sentry_sdk.capture_exception(e)
            raise InsightGenerationError(f"Pipeline execution failed: {str(e)}")
    
    def _format_metrics_table(self, insight: Dict[str, Any]) -> str:
        """Format insight metrics as a markdown table for LLM."""
        table_rows = ["| Metric | Value |", "|--------|--------|"]
        
        # Add overall stats if present
        if "overall_stats" in insight:
            for metric, value in insight["overall_stats"].items():
                if isinstance(value, (int, float)):
                    formatted_value = f"${value:,.2f}" if "gross" in metric.lower() else f"{value:,}"
                    table_rows.append(f"| {metric.replace('_', ' ').title()} | {formatted_value} |")
        
        # Add benchmarks if present
        if "benchmarks" in insight:
            for metric, value in insight["benchmarks"].items():
                if isinstance(value, (int, float)):
                    formatted_value = f"${value:,.2f}" if "gross" in metric.lower() else f"{value:,}"
                    table_rows.append(f"| {metric.replace('_', ' ').title()} | {formatted_value} |")
        
        return "\n".join(table_rows)
    
    def _get_date_range(self, df: pd.DataFrame) -> str:
        """Get the date range covered by the data."""
        date_cols = ["date", "sale_date", "transaction_date"]
        
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
                min_date = df[col].min()
                max_date = df[col].max()
                
                if pd.notnull(min_date) and pd.notnull(max_date):
                    return f"{min_date.strftime('%b %Y')} - {max_date.strftime('%b %Y')}"
        
        return "All Time"  # Fallback if no valid date column found
    
    def get_feedback_stats(self, insight_id: str) -> Optional[FeedbackStats]:
        """
        Get aggregated feedback statistics for an insight type.
        
        Args:
            insight_id: The insight type ID
            
        Returns:
            FeedbackStats object if feedback exists, None otherwise
        """
        try:
            # Get feedback entries for this insight
            entries = [
                FeedbackEntry.from_dict(entry)
                for entry in self.feedback_manager.get_feedback(insight_id=insight_id)
            ]
            
            if not entries:
                return None
                
            return FeedbackStats.from_entries(entries)
            
        except Exception as e:
            logger.warning(f"Error getting feedback stats: {str(e)}")
            return None