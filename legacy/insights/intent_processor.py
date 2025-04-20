"""
Intent processing for insight generation.
"""

from typing import Dict, Any, Optional
import pandas as pd
import re
from .models import InsightResult
from .direct_analysis import (
    analyze_negative_profits,
    analyze_by_lead_source,
    find_metric_column,
    find_category_column
)

class IntentProcessor:
    """Processes different types of insight intents."""
    
    @staticmethod
    def process_count_intent(df: pd.DataFrame, prompt: str, condition: str) -> InsightResult:
        """Process counting-related intents."""
        if "negative" in prompt.lower() and any(term in prompt.lower() for term in ["profit", "gross"]):
            # Find gross profit column
            gross_col = find_metric_column(df.columns, "gross")
            if not gross_col:
                return InsightResult(
                    title="Missing Data",
                    summary="Could not find gross profit column in the data.",
                    recommendations=[],
                    error="Missing gross profit column"
                )
            
            # Analyze negative profits
            metrics, viz_data = analyze_negative_profits(df, gross_col)
            
            # Create response
            title = "Negative Profit Analysis"
            summary = (
                f"Found {metrics['count']} deals ({metrics['percentage']:.1f}%) with negative gross profit, "
                f"totaling ${abs(metrics['total_loss']):,.2f} in losses."
            )
            
            recommendations = [
                f"Review {metrics['count']} deals with negative profit for process improvements",
                f"Average loss per negative deal is ${abs(metrics['avg_loss']):,.2f}",
                "Consider implementing pre-deal profit checks",
                "Analyze common factors in negative profit deals"
            ]
            
            # Add lead source analysis if available
            source_col = find_category_column(df.columns, "source")
            if source_col:
                source_analysis = analyze_by_lead_source(df[df[gross_col] < 0], gross_col, source_col)
                if source_analysis:
                    top_source = max(source_analysis, key=lambda x: x[f'{gross_col}_count'])
                    recommendations.append(
                        f"Most negative profits ({top_source[f'{gross_col}_count']} deals) "
                        f"came from {top_source[source_col]}"
                    )
            
            return InsightResult(
                title=title,
                summary=summary,
                recommendations=recommendations,
                chart_data=viz_data,
                chart_encoding={
                    "x": "Category",
                    "y": "Count",
                    "tooltip": ["Category", "Count"]
                }
            )
        
        return InsightResult(
            title="Analysis Error",
            summary="Could not determine what to count from the prompt",
            recommendations=[],
            error="Unclear count metric"
        )