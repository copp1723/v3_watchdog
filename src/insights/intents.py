"""
Intent-based insight generation system.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
import pandas as pd
import re
from .models import InsightResult
from ..utils.columns import find_metric_column, find_category_column, normalize_lead_source

class Intent(ABC):
    """Base class for all insight intents."""
    
    @abstractmethod
    def matches(self, prompt: str) -> bool:
        """
        Check if this intent matches the given prompt.
        
        Args:
            prompt: User's question/prompt
            
        Returns:
            True if this intent can handle the prompt
        """
        pass
    
    @abstractmethod
    def analyze(self, df: pd.DataFrame, prompt: str) -> InsightResult:
        """
        Analyze the data according to this intent.
        
        Args:
            df: DataFrame to analyze
            prompt: User's question/prompt
            
        Returns:
            Analysis result
        """
        pass

class CountMetricIntent(Intent):
    """Intent for counting/aggregating specific data points."""
    
    def matches(self, prompt: str) -> bool:
        """Check if the prompt is asking for a count."""
        prompt_lower = prompt.lower()
        count_terms = ["how many", "count of", "number of", "total number"]
        return any(term in prompt_lower for term in count_terms)

    def analyze(self, df: pd.DataFrame, prompt: str) -> InsightResult:
        """Analyze the data to get counts."""
        prompt_lower = prompt.lower()
        
        # Look for what we're counting
        if "negative" in prompt_lower and any(term in prompt_lower for term in ["profit", "gross", "margin"]):
            # Find gross profit column
            gross_col = find_metric_column(df.columns, "gross")
            if not gross_col:
                return InsightResult(
                    title="Missing Data",
                    summary="Could not find gross profit column in the data.",
                    recommendations=[],
                    error="Missing gross profit column"
                )
            
            # Calculate negative profit metrics
            df[gross_col] = pd.to_numeric(df[gross_col].astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce')
            negative_profits = df[df[gross_col] < 0]
            count = len(negative_profits)
            total_loss = negative_profits[gross_col].sum()
            avg_loss = total_loss / count if count > 0 else 0
            
            # Create summary
            title = "Negative Profit Analysis"
            summary = f"Found {count} deals with negative gross profit, totaling ${abs(total_loss):,.2f} in losses."
            
            # Add recommendations
            recommendations = [
                f"Review {count} deals with negative profit for process improvements",
                f"Average loss per negative deal is ${abs(avg_loss):,.2f}",
                "Consider implementing pre-deal profit checks",
                "Analyze common factors in negative profit deals"
            ]
            
            # Prepare chart data
            chart_data = pd.DataFrame({
                'Category': ['Negative Profit Deals', 'Other Deals'],
                'Count': [count, len(df) - count]
            })
            
            return InsightResult(
                title=title,
                summary=summary,
                recommendations=recommendations,
                chart_data=chart_data,
                chart_encoding={
                    "x": "Category",
                    "y": "Count",
                    "tooltip": ["Category", "Count"]
                },
                confidence="high"
            )
        
        return InsightResult(
            title="Analysis Error",
            summary="Could not determine what to count from the prompt",
            recommendations=[],
            error="Unclear count metric"
        )

class TopMetricIntent(Intent):
    """Intent for finding highest/top values of metrics."""
    
    def matches(self, prompt: str) -> bool:
        prompt_lower = prompt.lower()
        has_top = any(word in prompt_lower for word in ["highest", "top", "best", "most"])
        return has_top

    def analyze(self, df: pd.DataFrame, prompt: str) -> InsightResult:
        prompt_lower = prompt.lower()
        
        # Determine metric and category columns
        metric_type = None
        for metric in ["gross", "price", "revenue", "cost"]:
            if metric in prompt_lower:
                metric_type = metric
                break
        
        category_type = None
        for category in ["rep", "source", "make", "model"]:
            if category in prompt_lower:
                category_type = category
                break
        
        if not metric_type or not category_type:
            return InsightResult(
                title="Analysis Error",
                summary="Could not determine metric or category from prompt",
                recommendations=[],
                error="Missing metric or category specification"
            )
        
        # Find actual column names
        metric_col = find_metric_column(df.columns, metric_type)
        category_col = find_category_column(df.columns, category_type)
        
        if not metric_col or not category_col:
            missing = []
            if not metric_col:
                missing.append(f"{metric_type} metric")
            if not category_col:
                missing.append(f"{category_type} category")
            return InsightResult(
                title="Missing Data",
                summary=f"Could not find columns for: {', '.join(missing)}",
                recommendations=[],
                error=f"Missing columns: {', '.join(missing)}"
            )
        
        try:
            # Clean numeric data
            df[metric_col] = pd.to_numeric(df[metric_col].astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce')
            
            # Group and analyze
            grouped = df.groupby(category_col)[metric_col].agg(['sum', 'mean', 'count']).reset_index()
            top_row = grouped.nlargest(1, 'sum').iloc[0]
            
            # Create result
            title = f"Top {category_type.title()} by {metric_type.title()}"
            summary = (
                f"{top_row[category_col]} leads with ${top_row['sum']:,.2f} total {metric_type}, "
                f"averaging ${top_row['mean']:,.2f} across {top_row['count']} transactions."
            )
            
            # Add recommendations
            recommendations = [
                f"Study {top_row[category_col]}'s successful strategies for team training.",
                f"Analyze their approach to identify best practices.",
                f"Consider having them mentor other team members."
            ]
            
            # Prepare chart data
            chart_data = grouped.sort_values('sum', ascending=False).head(5)
            chart_encoding = {
                "x": {"field": category_col, "type": "nominal"},
                "y": {"field": "sum", "type": "quantitative", "title": f"Total {metric_type.title()}"},
                "tooltip": [
                    {"field": category_col, "type": "nominal"},
                    {"field": "sum", "type": "quantitative", "format": "$,.2f"},
                    {"field": "mean", "type": "quantitative", "format": "$,.2f"},
                    {"field": "count", "type": "quantitative"}
                ]
            }
            
            return InsightResult(
                title=title,
                summary=summary,
                recommendations=recommendations,
                chart_data=chart_data,
                chart_encoding=chart_encoding,
                supporting_data=df[[category_col, metric_col]],
                confidence="high"
            )
            
        except Exception as e:
            return InsightResult(
                title="Analysis Error",
                summary=f"Error analyzing data: {str(e)}",
                recommendations=[],
                error=str(e)
            )

class BottomMetricIntent(Intent):
    """Intent for finding lowest/bottom values of metrics."""
    
    def matches(self, prompt: str) -> bool:
        prompt_lower = prompt.lower()
        has_bottom = any(word in prompt_lower for word in ["lowest", "bottom", "worst", "least"])
        return has_bottom

    def analyze(self, df: pd.DataFrame, prompt: str) -> InsightResult:
        # Implementation similar to TopMetricIntent but using nsmallest instead of nlargest
        # ... (similar implementation with opposite sorting)
        pass

class AverageMetricIntent(Intent):
    """Intent for calculating averages/means of metrics."""
    
    def matches(self, prompt: str) -> bool:
        prompt_lower = prompt.lower()
        has_average = any(word in prompt_lower for word in ["average", "mean", "typical"])
        return has_average

    def analyze(self, df: pd.DataFrame, prompt: str) -> InsightResult:
        # Implementation for calculating and displaying averages
        # ... (implementation focusing on means and distributions)
        pass

class NegativeProfitIntent(Intent):
    """Intent for analyzing negative profit transactions."""
    
    def matches(self, prompt: str) -> bool:
        """Check if the prompt is about negative profits."""
        prompt_lower = prompt.lower()
        return "negative" in prompt_lower and any(
            term in prompt_lower for term in ["profit", "gross", "margin"]
        )

    def analyze(self, df: pd.DataFrame, prompt: str) -> InsightResult:
        """Analyze negative profit transactions."""
        # Find gross profit column
        gross_col = find_metric_column(df.columns, "gross")
        if not gross_col:
            return InsightResult(
                title="Missing Data",
                summary="Could not find gross profit column in the data.",
                recommendations=[],
                error="Missing gross profit column"
            )
        
        try:
            # Clean and convert gross profit values
            df[gross_col] = pd.to_numeric(df[gross_col].astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce')
            
            # Analyze negative profits
            negative_profits = df[df[gross_col] < 0]
            count = len(negative_profits)
            total_loss = negative_profits[gross_col].sum()
            avg_loss = total_loss / count if count > 0 else 0
            
            # Calculate percentage
            total_deals = len(df)
            percentage = (count / total_deals) * 100 if total_deals > 0 else 0
            
            # Create detailed summary
            title = "Negative Profit Analysis"
            summary = (
                f"Found {count} deals ({percentage:.1f}%) with negative gross profit, "
                f"totaling ${abs(total_loss):,.2f} in losses."
            )
            
            # Add detailed recommendations
            recommendations = [
                f"Review {count} deals with negative profit for process improvements",
                f"Average loss per negative deal is ${abs(avg_loss):,.2f}",
                "Consider implementing pre-deal profit checks",
                "Analyze common factors in negative profit deals"
            ]
            
            # Add lead source analysis if available
            lead_source_col = find_category_column(df.columns, "source")
            if lead_source_col:
                lead_source_counts = negative_profits[lead_source_col].value_counts()
                if not lead_source_counts.empty:
                    top_source = lead_source_counts.index[0]
                    source_count = lead_source_counts.iloc[0]
                    recommendations.append(
                        f"Most negative profits ({source_count} deals) came from {top_source}"
                    )
            
            # Prepare visualization data
            chart_data = pd.DataFrame({
                'Category': ['Negative Profit Deals', 'Other Deals'],
                'Count': [count, len(df) - count]
            })
            
            return InsightResult(
                title=title,
                summary=summary,
                recommendations=recommendations,
                chart_data=chart_data,
                chart_encoding={
                    "x": "Category",
                    "y": "Count",
                    "tooltip": ["Category", "Count"]
                },
                confidence="high"
            )
            
        except Exception as e:
            return InsightResult(
                title="Analysis Error",
                summary=f"Error analyzing negative profits: {str(e)}",
                recommendations=[],
                error=str(e)
            )

class HighestCountIntent(Intent):
    """Intent for finding entities with highest count/frequency."""
    
    def matches(self, prompt: str) -> bool:
        """Check if the prompt is asking about highest count/frequency."""
        prompt_lower = prompt.lower()
        count_terms = ["most", "highest number", "highest count", "most deals", "most sales"]
        return any(term in prompt_lower for term in count_terms)

    def analyze(self, df: pd.DataFrame, prompt: str) -> InsightResult:
        """Analyze the data to find highest count by category."""
        prompt_lower = prompt.lower()
        
        # Determine category to count
        category_type = None
        for category in ["rep", "source", "make", "model"]:
            if category in prompt_lower:
                category_type = category
                break
        
        if not category_type:
            return InsightResult(
                title="Analysis Error",
                summary="Could not determine what to count from prompt",
                recommendations=[],
                error="Missing category specification"
            )
        
        # Find category column
        category_col = find_category_column(df.columns, category_type)
        if not category_col:
            return InsightResult(
                title="Missing Data",
                summary=f"Could not find column for {category_type}",
                recommendations=[],
                error=f"Missing column: {category_type}"
            )
        
        try:
            # Count by category
            counts = df[category_col].value_counts()
            top_category = counts.index[0]
            top_count = counts.iloc[0]
            
            # Create summary
            title = f"Top {category_type.title()} by Deal Count"
            summary = f"{top_category} leads with {top_count} deals"
            
            # Add recommendations
            recommendations = [
                f"Analyze {top_category}'s successful strategies",
                f"Consider having {top_category} share best practices",
                "Review deal sources and closing techniques"
            ]
            
            # Prepare chart data
            chart_data = pd.DataFrame({
                'Category': counts.index[:5],  # Top 5 for visualization
                'Count': counts.values[:5]
            })
            
            return InsightResult(
                title=title,
                summary=summary,
                recommendations=recommendations,
                chart_data=chart_data,
                chart_encoding={
                    "x": "Category",
                    "y": "Count",
                    "tooltip": ["Category", "Count"]
                },
                confidence="high"
            )
            
        except Exception as e:
            return InsightResult(
                title="Analysis Error",
                summary=f"Error analyzing counts: {str(e)}",
                recommendations=[],
                error=str(e)
            )

class LeadSourceIntent(Intent):
    """Intent for analyzing specific lead sources."""
    
    def matches(self, prompt: str) -> bool:
        """Check if the prompt is about a specific lead source."""
        prompt_lower = prompt.lower()
        
        # Check for lead source mentions
        has_source = any(term in prompt_lower for term in ["lead source", "leadsource", "source"])
        
        # Check for specific sources
        source_names = ["cargurus", "car gurus", "car guru", "autotrader", "cars.com", "facebook"]
        has_specific_source = any(source in prompt_lower for source in source_names)
        
        return has_source or has_specific_source

    def analyze(self, df: pd.DataFrame, prompt: str) -> InsightResult:
        """Analyze data for a specific lead source."""
        prompt_lower = prompt.lower()
        
        # Find lead source column
        source_col = find_category_column(df.columns, "source")
        if not source_col:
            return InsightResult(
                title="Missing Data",
                summary="Could not find lead source column in the data.",
                recommendations=[],
                error="Missing lead source column"
            )
        
        try:
            # Clean and standardize lead source values
            df[source_col] = df[source_col].fillna("Unknown")
            df[source_col] = df[source_col].apply(normalize_lead_source)
            
            # Identify target lead source from prompt
            target_source_name = None
            if "cargurus" in prompt_lower or "car gurus" in prompt_lower or "car guru" in prompt_lower:
                target_source_name = "CarGurus"
            elif "autotrader" in prompt_lower or "auto trader" in prompt_lower:
                target_source_name = "AutoTrader"
            elif "cars.com" in prompt_lower or "cars com" in prompt_lower:
                target_source_name = "Cars.com"
            elif "facebook" in prompt_lower or "fb" in prompt_lower:
                target_source_name = "Facebook"
            elif "dealer website" in prompt_lower or "website" in prompt_lower:
                target_source_name = "Dealer Website"
            elif "walk" in prompt_lower and "in" in prompt_lower:
                target_source_name = "Walk In"
            
            if not target_source_name:
                return InsightResult(
                    title="Lead Source Not Found",
                    summary="Could not identify the specific lead source in your question.",
                    recommendations=[
                        "Try specifying a lead source like CarGurus, AutoTrader, etc.",
                        "Or ask about all lead sources to see an overview."
                    ],
                    confidence="medium"
                )
            
            # Filter for target source
            target_source = df[df[source_col] == target_source_name]
            
            if len(target_source) == 0:
                return InsightResult(
                    title=f"No Data for {target_source_name}",
                    summary=f"No deals found from {target_source_name}.",
                    recommendations=[
                        "Verify the lead source name is correct",
                        "Check if this is a new lead source",
                        f"Available sources: {', '.join(df[source_col].unique())}"
                    ],
                    confidence="high"
                )
            
            # Calculate metrics
            total_deals = len(target_source)
            total_all_deals = len(df)
            percentage = (total_deals / total_all_deals) * 100
            
            # Calculate gross metrics if available
            gross_col = find_metric_column(df.columns, "gross")
            if gross_col:
                # Clean gross values
                df[gross_col] = pd.to_numeric(df[gross_col].astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce')
                total_gross = target_source[gross_col].sum()
                avg_gross = total_gross / total_deals if total_deals > 0 else 0
                
                title = f"{target_source_name} Performance"
                summary = (
                    f"{target_source_name} generated {total_deals} deals ({percentage:.1f}% of total) "
                    f"with ${total_gross:,.2f} total gross"
                )
                
                recommendations = [
                    f"Average gross per deal: ${avg_gross:,.2f}",
                    f"This source represents {percentage:.1f}% of all deals",
                    "Monitor conversion rates from this source",
                    "Compare ROI with other lead sources"
                ]
                
                # Prepare visualization data
                chart_data = pd.DataFrame({
                    'Metric': ['Deal Count', 'Total Gross'],
                    'Value': [total_deals, total_gross],
                    'Percentage': [percentage, (total_gross / df[gross_col].sum()) * 100]
                })
                
            else:
                title = f"{target_source_name} Deal Count"
                summary = f"{target_source_name} generated {total_deals} deals ({percentage:.1f}% of total)"
                
                recommendations = [
                    f"This source represents {percentage:.1f}% of all deals",
                    "Monitor conversion rates from this source",
                    "Consider tracking gross profit for better analysis"
                ]
                
                # Prepare visualization data
                chart_data = pd.DataFrame({
                    'Category': [target_source_name, 'Other Sources'],
                    'Count': [total_deals, total_all_deals - total_deals]
                })
            
            return InsightResult(
                title=title,
                summary=summary,
                recommendations=recommendations,
                chart_data=chart_data,
                chart_encoding={
                    "x": "Category" if "Category" in chart_data.columns else "Metric",
                    "y": "Count" if "Count" in chart_data.columns else "Value",
                    "tooltip": ["Category", "Count"] if "Category" in chart_data.columns else ["Metric", "Value", "Percentage"]
                },
                confidence="high"
            )
            
        except Exception as e:
            return InsightResult(
                title="Analysis Error",
                summary=f"Error analyzing lead source: {str(e)}",
                recommendations=[],
                error=str(e)
            )

class IntentManager:
    """Manages the collection of available intents."""
    
    def __init__(self):
        self.intents = [
            LeadSourceIntent(),  # Add LeadSourceIntent first for priority
            TopMetricIntent(),
            BottomMetricIntent(),
            CountMetricIntent(),
            HighestCountIntent()
        ]

# Create a default instance for easy import
intent_manager = IntentManager()